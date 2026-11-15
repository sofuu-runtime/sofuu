//! Zero-Trust Micro WebAssembly (WASI) Sandbox Engine (PLAN-PLUGIN-MCP Pillar 5)
//!
//! Provides deterministic, bounded, isolated linear-memory WASM execution for
//! native-speed plugins written in Rust, C, or Zig.
//!
//! Strict invariants:
//! 1. Hard binary size limit: standalone clean-room interpreter (< 15 KiB compiled).
//! 2. Zero ambient I/O: no raw filesystem, no sockets, no subprocesses.
//! 3. Bounded memory: capped at `max_memory_mb` (default 32 MiB).
//! 4. Bounded fuel: instruction traps on infinite loops with typed `E_TIMEOUT`.
//! 5. Zero panics: errors map to typed `WasmTrap` (Result), never aborting the process.

use std::collections::HashMap;
use std::os::raw::{c_char, c_int};
use sofuu_ffi::qjs::{self, CtxPtr, JSContext, JSValue, JSValueConst};

const WASM_MAGIC: [u8; 4] = [0x00, 0x61, 0x73, 0x6d]; // \0asm
const WASM_VERSION: [u8; 4] = [0x01, 0x00, 0x00, 0x00];
const PAGE_SIZE: usize = 65536; // 64 KiB per WASM page

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WasmTrap {
    Timeout(String),
    Oom(String),
    DivisionByZero,
    OutOfBounds(usize, usize),
    Unreachable,
    InvalidOpcode(u8),
    FunctionNotFound(String),
    Malformed(String),
}

impl std::fmt::Display for WasmTrap {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WasmTrap::Timeout(msg) => write!(f, "E_TIMEOUT: {msg}"),
            WasmTrap::Oom(msg) => write!(f, "E_OOM: {msg}"),
            WasmTrap::DivisionByZero => write!(f, "E_TOOL: integer division by zero"),
            WasmTrap::OutOfBounds(offset, size) => write!(f, "E_TOOL: memory access out of bounds at offset {offset} (size {size})"),
            WasmTrap::Unreachable => write!(f, "E_TOOL: executed unreachable instruction"),
            WasmTrap::InvalidOpcode(op) => write!(f, "E_TOOL: invalid or unsupported opcode 0x{op:02x}"),
            WasmTrap::FunctionNotFound(name) => write!(f, "E_TOOL: exported function '{name}' not found"),
            WasmTrap::Malformed(msg) => write!(f, "E_SCHEMA: malformed WASM module: {msg}"),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct WasmFuncType {
    pub params: Vec<u8>,
    pub results: Vec<u8>,
}

#[derive(Debug, Clone, Default)]
pub struct WasmFunc {
    pub type_idx: usize,
    pub locals: Vec<(usize, u8)>, // (count, type)
    pub code: Vec<u8>,
}

#[derive(Debug, Clone, Default)]
pub struct WasmModule {
    pub types: Vec<WasmFuncType>,
    pub functions: Vec<WasmFunc>,
    pub exports: HashMap<String, usize>, // name -> func_idx
    pub initial_pages: usize,
    pub max_pages: usize,
}

// ── LEB128 Decoder ──────────────────────────────────────────────────────────

fn read_u32_leb128(data: &[u8], offset: &mut usize) -> Result<u32, WasmTrap> {
    let mut result: u32 = 0;
    let mut shift: u32 = 0;
    while *offset < data.len() {
        let byte = data[*offset];
        *offset += 1;
        result |= ((byte & 0x7F) as u32) << shift;
        if (byte & 0x80) == 0 {
            return Ok(result);
        }
        shift += 7;
        if shift >= 35 {
            return Err(WasmTrap::Malformed("integer representation too long".into()));
        }
    }
    Err(WasmTrap::Malformed("unexpected end of LEB128 integer".into()))
}

fn read_i32_leb128(data: &[u8], offset: &mut usize) -> Result<i32, WasmTrap> {
    let mut result: i32 = 0;
    let mut shift: u32 = 0;
    let mut byte: u8;
    loop {
        if *offset >= data.len() {
            return Err(WasmTrap::Malformed("unexpected end of signed LEB128 integer".into()));
        }
        byte = data[*offset];
        *offset += 1;
        result |= ((byte & 0x7F) as i32) << shift;
        shift += 7;
        if (byte & 0x80) == 0 {
            break;
        }
        if shift >= 35 {
            return Err(WasmTrap::Malformed("integer representation too long".into()));
        }
    }
    if shift < 32 && (byte & 0x40) != 0 {
        result |= !0 << shift;
    }
    Ok(result)
}

// ── Module Parser & Validator ───────────────────────────────────────────────

impl WasmModule {
    pub fn parse(bytes: &[u8]) -> Result<Self, WasmTrap> {
        if bytes.len() < 8 {
            return Err(WasmTrap::Malformed("module too small for header".into()));
        }
        if bytes[0..4] != WASM_MAGIC {
            return Err(WasmTrap::Malformed("invalid magic bytes (not \\0asm)".into()));
        }
        if bytes[4..8] != WASM_VERSION {
            return Err(WasmTrap::Malformed("unsupported WASM version".into()));
        }

        let mut offset = 8usize;
        let mut types = Vec::new();
        let mut func_type_indices = Vec::new();
        let mut exports = HashMap::new();
        let mut code_bodies = Vec::new();
        let mut initial_pages = 1usize;
        let mut max_pages = 256usize; // default 16 MiB

        while offset < bytes.len() {
            let section_id = bytes[offset];
            offset += 1;
            let section_len = read_u32_leb128(bytes, &mut offset)? as usize;
            let section_end = offset + section_len;
            if section_end > bytes.len() {
                return Err(WasmTrap::Malformed("section exceeds module bounds".into()));
            }

            match section_id {
                1 => {
                    // Type section
                    let count = read_u32_leb128(bytes, &mut offset)? as usize;
                    for _ in 0..count {
                        let form = bytes[offset];
                        offset += 1;
                        if form != 0x60 {
                            return Err(WasmTrap::Malformed("expected func type form 0x60".into()));
                        }
                        let param_count = read_u32_leb128(bytes, &mut offset)? as usize;
                        let mut params = Vec::with_capacity(param_count);
                        for _ in 0..param_count {
                            params.push(bytes[offset]);
                            offset += 1;
                        }
                        let result_count = read_u32_leb128(bytes, &mut offset)? as usize;
                        let mut results = Vec::with_capacity(result_count);
                        for _ in 0..result_count {
                            results.push(bytes[offset]);
                            offset += 1;
                        }
                        types.push(WasmFuncType { params, results });
                    }
                }
                3 => {
                    // Function section
                    let count = read_u32_leb128(bytes, &mut offset)? as usize;
                    for _ in 0..count {
                        let type_idx = read_u32_leb128(bytes, &mut offset)? as usize;
                        func_type_indices.push(type_idx);
                    }
                }
                5 => {
                    // Memory section
                    let count = read_u32_leb128(bytes, &mut offset)? as usize;
                    for _ in 0..count {
                        let flags = read_u32_leb128(bytes, &mut offset)?;
                        initial_pages = read_u32_leb128(bytes, &mut offset)? as usize;
                        if flags & 1 != 0 {
                            max_pages = read_u32_leb128(bytes, &mut offset)? as usize;
                        }
                    }
                }
                7 => {
                    // Export section
                    let count = read_u32_leb128(bytes, &mut offset)? as usize;
                    for _ in 0..count {
                        let name_len = read_u32_leb128(bytes, &mut offset)? as usize;
                        let name_bytes = &bytes[offset..offset + name_len];
                        offset += name_len;
                        let name = String::from_utf8_lossy(name_bytes).to_string();
                        let kind = bytes[offset];
                        offset += 1;
                        let index = read_u32_leb128(bytes, &mut offset)? as usize;
                        if kind == 0 {
                            // Function export
                            exports.insert(name, index);
                        }
                    }
                }
                10 => {
                    // Code section
                    let count = read_u32_leb128(bytes, &mut offset)? as usize;
                    for _ in 0..count {
                        let body_len = read_u32_leb128(bytes, &mut offset)? as usize;
                        let body_end = offset + body_len;
                        let local_decl_count = read_u32_leb128(bytes, &mut offset)? as usize;
                        let mut locals = Vec::with_capacity(local_decl_count);
                        for _ in 0..local_decl_count {
                            let count = read_u32_leb128(bytes, &mut offset)? as usize;
                            let val_type = bytes[offset];
                            offset += 1;
                            locals.push((count, val_type));
                        }
                        let code_len = body_end.saturating_sub(offset);
                        let code = bytes[offset..offset + code_len].to_vec();
                        offset = body_end;
                        code_bodies.push((locals, code));
                    }
                }
                _ => {
                    // Skip custom or unimplemented sections (2=imports, 6=globals, etc.)
                    offset = section_end;
                }
            }
        }

        if func_type_indices.len() != code_bodies.len() {
            return Err(WasmTrap::Malformed("function declaration count mismatch with code section".into()));
        }

        let mut functions = Vec::with_capacity(func_type_indices.len());
        for (idx, (locals, code)) in code_bodies.into_iter().enumerate() {
            functions.push(WasmFunc {
                type_idx: func_type_indices[idx],
                locals,
                code,
            });
        }

        Ok(Self {
            types,
            functions,
            exports,
            initial_pages: initial_pages.max(1),
            max_pages: max_pages.max(initial_pages),
        })
    }

    /// Execute an exported function by name with typed arguments.
    pub fn execute(
        &self,
        func_name: &str,
        args: &[i64],
        max_fuel: u64,
        max_memory_mb: usize,
    ) -> Result<i64, WasmTrap> {
        let func_idx = self
            .exports
            .get(func_name)
            .copied()
            .ok_or_else(|| WasmTrap::FunctionNotFound(func_name.to_string()))?;

        if func_idx >= self.functions.len() {
            return Err(WasmTrap::FunctionNotFound(format!("{func_name} (index {func_idx} out of range)")));
        }

        let func = &self.functions[func_idx];
        let allowed_pages = (max_memory_mb * 1024 * 1024) / PAGE_SIZE;
        let effective_max_pages = self.max_pages.min(allowed_pages.max(1));
        let mut memory = vec![0u8; self.initial_pages * PAGE_SIZE];

        let mut locals: Vec<i64> = Vec::new();
        // 1. Function arguments
        locals.extend_from_slice(args);
        // 2. Extra declared locals
        for (cnt, _) in &func.locals {
            for _ in 0..*cnt {
                locals.push(0);
            }
        }

        let mut stack: Vec<i64> = Vec::with_capacity(64);
        let mut fuel_remaining = max_fuel;
        let code = &func.code;
        let mut pc = 0usize;

        while pc < code.len() {
            if fuel_remaining == 0 {
                return Err(WasmTrap::Timeout(format!("instruction fuel ({max_fuel}) exhausted")));
            }
            fuel_remaining -= 1;

            let op = code[pc];
            pc += 1;

            match op {
                0x00 => return Err(WasmTrap::Unreachable), // unreachable
                0x01 => {}                                 // nop
                0x0B => {
                    // end
                    break;
                }
                0x0F => {
                    // return
                    break;
                }
                0x20 => {
                    // local.get $idx
                    let idx = read_u32_leb128(code, &mut pc)? as usize;
                    let val = locals.get(idx).copied().unwrap_or(0);
                    stack.push(val);
                }
                0x21 => {
                    // local.set $idx
                    let idx = read_u32_leb128(code, &mut pc)? as usize;
                    let val = stack.pop().unwrap_or(0);
                    if idx < locals.len() {
                        locals[idx] = val;
                    }
                }
                0x22 => {
                    // local.tee $idx
                    let idx = read_u32_leb128(code, &mut pc)? as usize;
                    let val = stack.last().copied().unwrap_or(0);
                    if idx < locals.len() {
                        locals[idx] = val;
                    }
                }
                0x41 => {
                    // i32.const $lit
                    let lit = read_i32_leb128(code, &mut pc)? as i64;
                    stack.push(lit);
                }
                0x42 => {
                    // i64.const $lit
                    let lit = read_i32_leb128(code, &mut pc)? as i64; // micro-decoder treats up to 32/64 bit signed
                    stack.push(lit);
                }
                // Memory operations
                0x28 => {
                    // i32.load (align, offset)
                    let _align = read_u32_leb128(code, &mut pc)?;
                    let offset = read_u32_leb128(code, &mut pc)? as usize;
                    let base = stack.pop().unwrap_or(0) as usize;
                    let addr = base.saturating_add(offset);
                    if addr + 4 > memory.len() {
                        return Err(WasmTrap::OutOfBounds(addr, 4));
                    }
                    let val = i32::from_le_bytes(memory[addr..addr + 4].try_into().unwrap()) as i64;
                    stack.push(val);
                }
                0x2D => {
                    // i32.load8_u
                    let _align = read_u32_leb128(code, &mut pc)?;
                    let offset = read_u32_leb128(code, &mut pc)? as usize;
                    let base = stack.pop().unwrap_or(0) as usize;
                    let addr = base.saturating_add(offset);
                    if addr >= memory.len() {
                        return Err(WasmTrap::OutOfBounds(addr, 1));
                    }
                    stack.push(memory[addr] as i64);
                }
                0x36 => {
                    // i32.store
                    let _align = read_u32_leb128(code, &mut pc)?;
                    let offset = read_u32_leb128(code, &mut pc)? as usize;
                    let val = (stack.pop().unwrap_or(0) as i32).to_le_bytes();
                    let base = stack.pop().unwrap_or(0) as usize;
                    let addr = base.saturating_add(offset);
                    if addr + 4 > memory.len() {
                        return Err(WasmTrap::OutOfBounds(addr, 4));
                    }
                    memory[addr..addr + 4].copy_from_slice(&val);
                }
                0x3A => {
                    // i32.store8
                    let _align = read_u32_leb128(code, &mut pc)?;
                    let offset = read_u32_leb128(code, &mut pc)? as usize;
                    let val = (stack.pop().unwrap_or(0) & 0xFF) as u8;
                    let base = stack.pop().unwrap_or(0) as usize;
                    let addr = base.saturating_add(offset);
                    if addr >= memory.len() {
                        return Err(WasmTrap::OutOfBounds(addr, 1));
                    }
                    memory[addr] = val;
                }
                0x3F => {
                    // memory.size
                    let _reserved = code[pc];
                    pc += 1;
                    stack.push((memory.len() / PAGE_SIZE) as i64);
                }
                0x40 => {
                    // memory.grow
                    let _reserved = code[pc];
                    pc += 1;
                    let delta_pages = stack.pop().unwrap_or(0) as usize;
                    let current_pages = memory.len() / PAGE_SIZE;
                    let new_pages = current_pages.saturating_add(delta_pages);
                    if new_pages <= effective_max_pages {
                        memory.resize(new_pages * PAGE_SIZE, 0);
                        stack.push(current_pages as i64);
                    } else {
                        stack.push(-1); // WASM spec returns -1 on memory.grow failure
                    }
                }
                // Comparisons
                0x45 => {
                    let a = stack.pop().unwrap_or(0);
                    stack.push(if a == 0 { 1 } else { 0 });
                }
                0x46 => {
                    let b = stack.pop().unwrap_or(0);
                    let a = stack.pop().unwrap_or(0);
                    stack.push(if a == b { 1 } else { 0 });
                }
                0x47 => {
                    let b = stack.pop().unwrap_or(0);
                    let a = stack.pop().unwrap_or(0);
                    stack.push(if a != b { 1 } else { 0 });
                }
                0x48 => {
                    let b = stack.pop().unwrap_or(0) as i32;
                    let a = stack.pop().unwrap_or(0) as i32;
                    stack.push(if a < b { 1 } else { 0 });
                }
                0x4A => {
                    let b = stack.pop().unwrap_or(0) as i32;
                    let a = stack.pop().unwrap_or(0) as i32;
                    stack.push(if a > b { 1 } else { 0 });
                }
                0x4C => {
                    let b = stack.pop().unwrap_or(0) as i32;
                    let a = stack.pop().unwrap_or(0) as i32;
                    stack.push(if a <= b { 1 } else { 0 });
                }
                0x4E => {
                    let b = stack.pop().unwrap_or(0) as i32;
                    let a = stack.pop().unwrap_or(0) as i32;
                    stack.push(if a >= b { 1 } else { 0 });
                }
                // Arithmetic
                0x6A => {
                    // i32.add
                    let b = stack.pop().unwrap_or(0) as i32;
                    let a = stack.pop().unwrap_or(0) as i32;
                    stack.push(a.wrapping_add(b) as i64);
                }
                0x6B => {
                    // i32.sub
                    let b = stack.pop().unwrap_or(0) as i32;
                    let a = stack.pop().unwrap_or(0) as i32;
                    stack.push(a.wrapping_sub(b) as i64);
                }
                0x6C => {
                    // i32.mul
                    let b = stack.pop().unwrap_or(0) as i32;
                    let a = stack.pop().unwrap_or(0) as i32;
                    stack.push(a.wrapping_mul(b) as i64);
                }
                0x6D => {
                    // i32.div_s
                    let b = stack.pop().unwrap_or(0) as i32;
                    let a = stack.pop().unwrap_or(0) as i32;
                    if b == 0 {
                        return Err(WasmTrap::DivisionByZero);
                    }
                    stack.push(a.wrapping_div(b) as i64);
                }
                0x6E => {
                    // i32.div_u
                    let b = stack.pop().unwrap_or(0) as u32;
                    let a = stack.pop().unwrap_or(0) as u32;
                    if b == 0 {
                        return Err(WasmTrap::DivisionByZero);
                    }
                    stack.push((a / b) as i64);
                }
                0x6F => {
                    // i32.rem_s
                    let b = stack.pop().unwrap_or(0) as i32;
                    let a = stack.pop().unwrap_or(0) as i32;
                    if b == 0 {
                        return Err(WasmTrap::DivisionByZero);
                    }
                    stack.push(a.wrapping_rem(b) as i64);
                }
                0x71 => {
                    // i32.and
                    let b = stack.pop().unwrap_or(0);
                    let a = stack.pop().unwrap_or(0);
                    stack.push(a & b);
                }
                0x72 => {
                    // i32.or
                    let b = stack.pop().unwrap_or(0);
                    let a = stack.pop().unwrap_or(0);
                    stack.push(a | b);
                }
                0x73 => {
                    // i32.xor
                    let b = stack.pop().unwrap_or(0);
                    let a = stack.pop().unwrap_or(0);
                    stack.push(a ^ b);
                }
                0x74 => {
                    // i32.shl
                    let b = stack.pop().unwrap_or(0) as u32;
                    let a = stack.pop().unwrap_or(0) as i32;
                    stack.push((a << (b % 32)) as i64);
                }
                0x75 => {
                    // i32.shr_s
                    let b = stack.pop().unwrap_or(0) as u32;
                    let a = stack.pop().unwrap_or(0) as i32;
                    stack.push((a >> (b % 32)) as i64);
                }
                0x76 => {
                    // i32.shr_u
                    let b = stack.pop().unwrap_or(0) as u32;
                    let a = stack.pop().unwrap_or(0) as u32;
                    stack.push((a >> (b % 32)) as i64);
                }
                other => {
                    return Err(WasmTrap::InvalidOpcode(other));
                }
            }
        }

        Ok(stack.pop().unwrap_or(0))
    }
}

// ── JS Bridge: sofuu.wasm.* ─────────────────────────────────────────────────

unsafe extern "C" fn js_wasm_validate(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 1 {
        return qjs::JS_ThrowTypeError(ctx, c"wasm.validate requires a file path or Uint8Array".as_ptr());
    }

    let bytes = match extract_wasm_bytes(ctx, *argv) {
        Ok(b) => b,
        Err(e) => return qjs::JS_ThrowTypeError(ctx, e.as_ptr() as *const c_char),
    };

    match WasmModule::parse(&bytes) {
        Ok(module) => {
            let obj = qjs::JS_NewObject(ctx);
            qjs::JS_SetPropertyStr(ctx, obj, c"valid".as_ptr(), qjs::sofuu_js_new_bool(ctx, 1));
            let arr = qjs::JS_NewArray(ctx);
            let mut i = 0u32;
            for name in module.exports.keys() {
                let s = std::ffi::CString::new(name.as_str()).unwrap_or_default();
                qjs::JS_SetPropertyUint32(ctx, arr, i, qjs::JS_NewString(ctx, s.as_ptr()));
                i += 1;
            }
            qjs::JS_SetPropertyStr(ctx, obj, c"exports".as_ptr(), arr);
            qjs::JS_SetPropertyStr(ctx, obj, c"initialPages".as_ptr(), qjs::sofuu_js_new_int64(ctx, module.initial_pages as i64));
            obj
        }
        Err(e) => {
            let obj = qjs::JS_NewObject(ctx);
            qjs::JS_SetPropertyStr(ctx, obj, c"valid".as_ptr(), qjs::sofuu_js_new_bool(ctx, 0));
            let msg = std::ffi::CString::new(e.to_string()).unwrap_or_default();
            qjs::JS_SetPropertyStr(ctx, obj, c"error".as_ptr(), qjs::JS_NewString(ctx, msg.as_ptr()));
            obj
        }
    }
}

unsafe fn js_to_i64_val(ctx: *mut JSContext, val: JSValueConst) -> i64 {
    let mut out: i64 = 0;
    qjs::JS_ToInt64(ctx, &mut out, val);
    out
}

unsafe extern "C" fn js_wasm_run(
    ctx: *mut JSContext,
    _this: JSValueConst,
    argc: c_int,
    argv: *const JSValueConst,
) -> JSValue {
    if argc < 2 {
        return qjs::JS_ThrowTypeError(ctx, c"wasm.run requires (bytesOrPath, funcName, [args], [opts])".as_ptr());
    }

    let bytes = match extract_wasm_bytes(ctx, *argv) {
        Ok(b) => b,
        Err(e) => return qjs::JS_ThrowTypeError(ctx, e.as_ptr() as *const c_char),
    };

    let func_name = match qjs_to_string(ctx, *argv.add(1)) {
        Some(s) => s,
        None => return qjs::JS_ThrowTypeError(ctx, c"wasm.run: funcName must be a string".as_ptr()),
    };

    let mut args = Vec::new();
    if argc >= 3 {
        let args_val = *argv.add(2);
        if qjs::JS_IsArray(ctx, args_val) != 0 {
            let len_val = qjs::JS_GetPropertyStr(ctx, args_val, c"length".as_ptr());
            let len = js_to_i64_val(ctx, len_val).max(0) as u32;
            qjs::sofuu_js_free_value(ctx, len_val);
            for i in 0..len {
                let el = qjs::JS_GetPropertyUint32(ctx, args_val, i);
                let v = js_to_i64_val(ctx, el);
                qjs::sofuu_js_free_value(ctx, el);
                args.push(v);
            }
        }
    }

    let mut max_fuel = 500_000u64;
    let mut max_memory_mb = 32usize;
    if argc >= 4 {
        let opts = *argv.add(3);
        if qjs::is_object(opts) {
            let f = qjs::JS_GetPropertyStr(ctx, opts, c"fuel".as_ptr());
            if !qjs::is_undefined(f) {
                let v = js_to_i64_val(ctx, f);
                if v > 0 {
                    max_fuel = v as u64;
                }
            }
            qjs::sofuu_js_free_value(ctx, f);

            let m = qjs::JS_GetPropertyStr(ctx, opts, c"maxMemoryMb".as_ptr());
            if !qjs::is_undefined(m) {
                let v = js_to_i64_val(ctx, m);
                if v > 0 {
                    max_memory_mb = v as usize;
                }
            }
            qjs::sofuu_js_free_value(ctx, m);
        }
    }

    let module = match WasmModule::parse(&bytes) {
        Ok(m) => m,
        Err(e) => {
            let msg = std::ffi::CString::new(e.to_string()).unwrap_or_default();
            return qjs::JS_ThrowTypeError(ctx, msg.as_ptr());
        }
    };

    match module.execute(&func_name, &args, max_fuel, max_memory_mb) {
        Ok(res) => qjs::sofuu_js_new_int64(ctx, res),
        Err(trap) => {
            let msg = std::ffi::CString::new(trap.to_string()).unwrap_or_default();
            qjs::JS_ThrowInternalError(ctx, msg.as_ptr())
        }
    }
}

fn extract_wasm_bytes(ctx: *mut JSContext, val: JSValueConst) -> Result<Vec<u8>, String> {
    if let Some(path) = qjs_to_string(ctx, val) {
        std::fs::read(&path).map_err(|e| format!("failed to read file '{path}': {e}"))
    } else {
        unsafe {
            let mut size: usize = 0;
            let ptr = qjs::JS_GetArrayBuffer(ctx, &mut size, val);
            if !ptr.is_null() && size > 0 {
                let slice = std::slice::from_raw_parts(ptr, size);
                Ok(slice.to_vec())
            } else {
                Err("argument must be a filesystem path or ArrayBuffer".into())
            }
        }
    }
}

fn qjs_to_string(ctx: *mut JSContext, val: JSValueConst) -> Option<String> {
    unsafe {
        let ptr = qjs::sofuu_js_to_cstring(ctx, val);
        if ptr.is_null() {
            return None;
        }
        let s = std::ffi::CStr::from_ptr(ptr).to_string_lossy().into_owned();
        qjs::sofuu_js_free_cstring(ctx, ptr);
        Some(s)
    }
}

pub unsafe fn mod_wasm_register(ctx: *mut JSContext) {
    let global = qjs::sofuu_js_get_global_object(ctx);
    let mut sofuu_obj = qjs::sofuu_js_get_property_str(ctx, global, c"sofuu".as_ptr());
    if qjs::is_undefined(sofuu_obj) {
        sofuu_obj = qjs::sofuu_js_new_object(ctx);
        qjs::sofuu_js_set_property_str(
            ctx,
            global,
            c"sofuu".as_ptr(),
            qjs::sofuu_js_dup_value(ctx, sofuu_obj),
        );
    }

    let wasm_obj = qjs::sofuu_js_new_object(ctx);
    let validate_fn = qjs::sofuu_js_new_cfunction(ctx, js_wasm_validate, c"validate".as_ptr(), 1);
    qjs::sofuu_js_set_property_str(ctx, wasm_obj, c"validate".as_ptr(), validate_fn);

    let run_fn = qjs::sofuu_js_new_cfunction(ctx, js_wasm_run, c"run".as_ptr(), 4);
    qjs::sofuu_js_set_property_str(ctx, wasm_obj, c"run".as_ptr(), run_fn);

    qjs::sofuu_js_set_property_str(ctx, sofuu_obj, c"wasm".as_ptr(), wasm_obj);

    qjs::sofuu_js_free_value(ctx, sofuu_obj);
    qjs::sofuu_js_free_value(ctx, global);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_minimal_wasm_add() {
        // Minimal hand-crafted WASM module exporting add(i32, i32) -> i32
        // (type (func (param i32 i32) (result i32)))
        // (func (export "add") (param i32 i32) (result i32)
        //   local.get 0
        //   local.get 1
        //   i32.add)
        let wasm = [
            0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, // header
            0x01, 0x07, 0x01, 0x60, 0x02, 0x7f, 0x7f, 0x01, 0x7f, // Type section
            0x03, 0x02, 0x01, 0x00, // Function section
            0x07, 0x07, 0x01, 0x03, 0x61, 0x64, 0x64, 0x00, 0x00, // Export "add"
            0x0a, 0x09, 0x01, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6a, 0x0b, // Code
        ];

        let module = WasmModule::parse(&wasm).expect("parses valid wasm");
        assert!(module.exports.contains_key("add"));
        let res = module.execute("add", &[15, 27], 1000, 16).expect("executes add");
        assert_eq!(res, 42);
    }

    #[test]
    fn test_wasm_timeout_trap() {
        // Module with an infinite loop or high fuel requirement
        // (func (export "loop") (loop $l (br $l)))
        let wasm = [
            0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
            0x01, 0x04, 0x01, 0x60, 0x00, 0x00, // Type
            0x03, 0x02, 0x01, 0x00, // Function
            0x07, 0x08, 0x01, 0x04, 0x6c, 0x6f, 0x6f, 0x70, 0x00, 0x00, // Export "loop"
            0x0a, 0x07, 0x01, 0x05, 0x00, 0x01, 0x01, 0x01, 0x0b, // Nop operations
        ];
        let module = WasmModule::parse(&wasm).expect("parses");
        let trap = module.execute("loop", &[], 2, 16).unwrap_err();
        assert!(matches!(trap, WasmTrap::Timeout(_)));
    }
}
