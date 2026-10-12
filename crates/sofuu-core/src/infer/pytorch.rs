// crates/sofuu-core/src/infer/pytorch.rs — Zero-copy PyTorch .pt / .pth checkpoint parser.
//
// Compliant with standard PyTorch 1.6+ zipfile checkpoint structure.
// Uses a lightweight, pure-Rust mini-pickle interpreter to extract state_dict
// tensor shapes, strides, and dtypes, mapping uncompressed tensor storages directly
// into memory with zero data copies.

use std::collections::HashMap;
use std::path::Path;

use super::mmap::MmapFile;
use super::npy::TensorDType;
use super::zip_reader::ZipArchive;

#[derive(Debug, Clone)]
pub struct PyTorchTensorMeta {
    pub name: String,
    pub storage_id: String,
    pub storage_type: String,
    pub storage_offset: usize,
    pub shape: Vec<usize>,
    pub stride: Vec<usize>,
    pub dtype: TensorDType,
}

#[derive(Debug, Clone)]
pub struct PyTorchTensorView {
    pub dtype: TensorDType,
    pub shape: Vec<usize>,
    pub data_offset: usize, // absolute offset inside mmap
    pub size_bytes: usize,
}

pub struct PyTorchCheckpoint {
    pub mmap: MmapFile,
    pub tensors: HashMap<String, PyTorchTensorView>,
}

#[derive(Debug, Clone)]
enum PickleValue {
    None,
    Bool(bool),
    Int(i64),
    String(String),
    Tuple(Vec<PickleValue>),
    List(Vec<PickleValue>),
    Dict(Vec<(PickleValue, PickleValue)>),
    Global(String, String),
    Tensor(PyTorchTensorMeta),
    Mark,
}

impl PickleValue {
    fn as_string(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    fn as_int(&self) -> Option<i64> {
        match self {
            Self::Int(i) => Some(*i),
            _ => None,
        }
    }
}

fn storage_class_to_dtype(name: &str) -> TensorDType {
    match name {
        "FloatStorage" => TensorDType::F32,
        "HalfStorage" => TensorDType::F16,
        "BFloat16Storage" => TensorDType::BF16,
        "DoubleStorage" => TensorDType::F64,
        "LongStorage" => TensorDType::I64,
        "IntStorage" => TensorDType::I32,
        "ShortStorage" => TensorDType::I16,
        "CharStorage" | "Int8Storage" => TensorDType::I8,
        "ByteStorage" | "UInt8Storage" => TensorDType::U8,
        "BoolStorage" => TensorDType::Bool,
        _ => TensorDType::Unknown,
    }
}

pub fn parse_pytorch_pickle(bytes: &[u8]) -> Result<Vec<(String, PyTorchTensorMeta)>, String> {
    let mut stack: Vec<PickleValue> = Vec::with_capacity(128);
    let mut memo: HashMap<usize, PickleValue> = HashMap::new();
    let mut cursor = 0;

    while cursor < bytes.len() {
        let opcode = bytes[cursor];
        cursor += 1;

        match opcode {
            0x80 => {
                // PROTO: 1 byte protocol version
                if cursor >= bytes.len() {
                    return Err("unexpected EOF in PROTO".to_string());
                }
                cursor += 1;
            }
            0x95 => {
                // FRAME: 8 byte frame length (protocol 4 hint, skip)
                if cursor + 8 > bytes.len() {
                    return Err("unexpected EOF in FRAME".to_string());
                }
                cursor += 8;
            }
            b'}' => {
                // EMPTY_DICT
                stack.push(PickleValue::Dict(Vec::new()));
            }
            b')' => {
                // EMPTY_TUPLE
                stack.push(PickleValue::Tuple(Vec::new()));
            }
            b']' => {
                // EMPTY_LIST
                stack.push(PickleValue::List(Vec::new()));
            }
            b'(' => {
                // MARK
                stack.push(PickleValue::Mark);
            }
            b'0' => {
                // POP
                stack.pop();
            }
            b'1' => {
                // POP_MARK
                while let Some(top) = stack.pop() {
                    if matches!(top, PickleValue::Mark) {
                        break;
                    }
                }
            }
            b'2' => {
                // DUP
                if let Some(top) = stack.last().cloned() {
                    stack.push(top);
                }
            }
            b'N' => {
                // NONE
                stack.push(PickleValue::None);
            }
            0x88 => {
                // NEWTRUE
                stack.push(PickleValue::Bool(true));
            }
            0x89 => {
                // NEWFALSE
                stack.push(PickleValue::Bool(false));
            }
            b'K' => {
                // BININT1: 1 byte unsigned int
                if cursor >= bytes.len() {
                    return Err("unexpected EOF in BININT1".to_string());
                }
                let val = bytes[cursor] as i64;
                cursor += 1;
                stack.push(PickleValue::Int(val));
            }
            b'M' => {
                // BININT2: 2 bytes unsigned int LE
                if cursor + 2 > bytes.len() {
                    return Err("unexpected EOF in BININT2".to_string());
                }
                let val = u16::from_le_bytes(bytes[cursor..cursor + 2].try_into().unwrap()) as i64;
                cursor += 2;
                stack.push(PickleValue::Int(val));
            }
            b'J' => {
                // BININT: 4 bytes signed int LE
                if cursor + 4 > bytes.len() {
                    return Err("unexpected EOF in BININT".to_string());
                }
                let val = i32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as i64;
                cursor += 4;
                stack.push(PickleValue::Int(val));
            }
            0x8a => {
                // LONG1: 1 byte length followed by LE signed bytes
                if cursor >= bytes.len() {
                    return Err("unexpected EOF in LONG1".to_string());
                }
                let len = bytes[cursor] as usize;
                cursor += 1;
                if cursor + len > bytes.len() {
                    return Err("unexpected EOF in LONG1 payload".to_string());
                }
                let mut val = 0i64;
                for (i, &b) in bytes[cursor..cursor + len].iter().enumerate() {
                    if i < 8 {
                        val |= (b as i64) << (i * 8);
                    }
                }
                cursor += len;
                stack.push(PickleValue::Int(val));
            }
            b'X' => {
                // BINUNICODE: 4 bytes LE length, then UTF-8 string
                if cursor + 4 > bytes.len() {
                    return Err("unexpected EOF in BINUNICODE".to_string());
                }
                let len = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
                cursor += 4;
                if cursor + len > bytes.len() {
                    return Err("unexpected EOF in BINUNICODE payload".to_string());
                }
                let s = String::from_utf8_lossy(&bytes[cursor..cursor + len]).to_string();
                cursor += len;
                stack.push(PickleValue::String(s));
            }
            0x8c => {
                // SHORT_BINUNICODE: 1 byte length, then UTF-8 string
                if cursor >= bytes.len() {
                    return Err("unexpected EOF in SHORT_BINUNICODE".to_string());
                }
                let len = bytes[cursor] as usize;
                cursor += 1;
                if cursor + len > bytes.len() {
                    return Err("unexpected EOF in SHORT_BINUNICODE payload".to_string());
                }
                let s = String::from_utf8_lossy(&bytes[cursor..cursor + len]).to_string();
                cursor += len;
                stack.push(PickleValue::String(s));
            }
            b'c' => {
                // GLOBAL: two newline-terminated strings
                let rem = &bytes[cursor..];
                let nl1 = rem.iter().position(|&b| b == b'\n').ok_or("malformed GLOBAL (nl1)")?;
                let mod_name = String::from_utf8_lossy(&rem[..nl1]).to_string();
                let rem2 = &rem[nl1 + 1..];
                let nl2 = rem2.iter().position(|&b| b == b'\n').ok_or("malformed GLOBAL (nl2)")?;
                let class_name = String::from_utf8_lossy(&rem2[..nl2]).to_string();
                cursor += nl1 + 1 + nl2 + 1;
                stack.push(PickleValue::Global(mod_name, class_name));
            }
            0x93 => {
                // STACK_GLOBAL: pops name, pops module
                let name = match stack.pop() {
                    Some(PickleValue::String(s)) => s,
                    _ => return Err("expected string for STACK_GLOBAL name".to_string()),
                };
                let module = match stack.pop() {
                    Some(PickleValue::String(s)) => s,
                    _ => return Err("expected string for STACK_GLOBAL module".to_string()),
                };
                stack.push(PickleValue::Global(module, name));
            }
            b'q' => {
                // BINPUT: 1 byte memo index
                if cursor >= bytes.len() {
                    return Err("unexpected EOF in BINPUT".to_string());
                }
                let idx = bytes[cursor] as usize;
                cursor += 1;
                if let Some(top) = stack.last() {
                    memo.insert(idx, top.clone());
                }
            }
            b'r' => {
                // LONG_BINPUT: 4 bytes memo index LE
                if cursor + 4 > bytes.len() {
                    return Err("unexpected EOF in LONG_BINPUT".to_string());
                }
                let idx = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
                cursor += 4;
                if let Some(top) = stack.last() {
                    memo.insert(idx, top.clone());
                }
            }
            0x94 => {
                // MEMOIZE
                let idx = memo.len();
                if let Some(top) = stack.last() {
                    memo.insert(idx, top.clone());
                }
            }
            b'h' => {
                // BINGET: 1 byte memo index
                if cursor >= bytes.len() {
                    return Err("unexpected EOF in BINGET".to_string());
                }
                let idx = bytes[cursor] as usize;
                cursor += 1;
                let val = memo.get(&idx).ok_or_else(|| format!("memo index {} not found", idx))?;
                stack.push(val.clone());
            }
            b'j' => {
                // LONG_BINGET: 4 bytes memo index LE
                if cursor + 4 > bytes.len() {
                    return Err("unexpected EOF in LONG_BINGET".to_string());
                }
                let idx = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
                cursor += 4;
                let val = memo.get(&idx).ok_or_else(|| format!("memo index {} not found", idx))?;
                stack.push(val.clone());
            }
            b't' => {
                // TUPLE: pop down to MARK
                let mut items = Vec::new();
                while let Some(val) = stack.pop() {
                    if matches!(val, PickleValue::Mark) {
                        break;
                    }
                    items.push(val);
                }
                items.reverse();
                stack.push(PickleValue::Tuple(items));
            }
            0x85 => {
                // TUPLE1
                let a = stack.pop().ok_or("stack underflow in TUPLE1")?;
                stack.push(PickleValue::Tuple(vec![a]));
            }
            0x86 => {
                // TUPLE2
                let b = stack.pop().ok_or("stack underflow in TUPLE2")?;
                let a = stack.pop().ok_or("stack underflow in TUPLE2")?;
                stack.push(PickleValue::Tuple(vec![a, b]));
            }
            0x87 => {
                // TUPLE3
                let c = stack.pop().ok_or("stack underflow in TUPLE3")?;
                let b = stack.pop().ok_or("stack underflow in TUPLE3")?;
                let a = stack.pop().ok_or("stack underflow in TUPLE3")?;
                stack.push(PickleValue::Tuple(vec![a, b, c]));
            }
            b'l' => {
                // LIST: pop down to MARK
                let mut items = Vec::new();
                while let Some(val) = stack.pop() {
                    if matches!(val, PickleValue::Mark) {
                        break;
                    }
                    items.push(val);
                }
                items.reverse();
                stack.push(PickleValue::List(items));
            }
            b'a' => {
                // APPEND: pop item, append to list
                let val = stack.pop().ok_or("stack underflow in APPEND")?;
                if let Some(PickleValue::List(list)) = stack.last_mut() {
                    list.push(val);
                }
            }
            b'e' => {
                // APPENDS: pop down to mark, append to list
                let mut items = Vec::new();
                while let Some(val) = stack.pop() {
                    if matches!(val, PickleValue::Mark) {
                        break;
                    }
                    items.push(val);
                }
                items.reverse();
                if let Some(PickleValue::List(list)) = stack.last_mut() {
                    list.extend(items);
                }
            }
            b'd' => {
                // DICT: pop down to MARK in key-value pairs
                let mut kvs = Vec::new();
                let mut items = Vec::new();
                while let Some(val) = stack.pop() {
                    if matches!(val, PickleValue::Mark) {
                        break;
                    }
                    items.push(val);
                }
                items.reverse();
                for chunk in items.chunks_exact(2) {
                    kvs.push((chunk[0].clone(), chunk[1].clone()));
                }
                stack.push(PickleValue::Dict(kvs));
            }
            b's' => {
                // SETITEM: pop val, pop key, set into dict
                let val = stack.pop().ok_or("stack underflow in SETITEM")?;
                let key = stack.pop().ok_or("stack underflow in SETITEM")?;
                if let Some(PickleValue::Dict(dict)) = stack.last_mut() {
                    dict.push((key, val));
                }
            }
            b'u' => {
                // SETITEMS: pop down to MARK, set into dict
                let mut items = Vec::new();
                while let Some(val) = stack.pop() {
                    if matches!(val, PickleValue::Mark) {
                        break;
                    }
                    items.push(val);
                }
                items.reverse();
                if let Some(PickleValue::Dict(dict)) = stack.last_mut() {
                    for chunk in items.chunks_exact(2) {
                        dict.push((chunk[0].clone(), chunk[1].clone()));
                    }
                }
            }
            b'Q' => {
                // BINPERSID: persistent load ('storage', storage_class, key, location, numel)
                let pid = stack.pop().ok_or("stack underflow in BINPERSID")?;
                stack.push(pid);
            }
            b'P' => {
                // PERSID: newline-terminated string
                let rem = &bytes[cursor..];
                let nl = rem.iter().position(|&b| b == b'\n').ok_or("malformed PERSID")?;
                let s = String::from_utf8_lossy(&rem[..nl]).to_string();
                cursor += nl + 1;
                stack.push(PickleValue::String(s));
            }
            b'R' => {
                // REDUCE: pop args tuple, pop callable, reduce
                let args = stack.pop().ok_or("stack underflow in REDUCE (args)")?;
                let callable = stack.pop().ok_or("stack underflow in REDUCE (callable)")?;

                match callable {
                    PickleValue::Global(ref module, ref name) => {
                        if (module == "torch._utils" && (name == "_rebuild_tensor_v2" || name == "_rebuild_tensor"))
                            || (module == "torch" && name == "_rebuild_tensor_v2")
                        {
                            // Args: (storage, storage_offset, size, stride, requires_grad, backward_hooks)
                            if let PickleValue::Tuple(ref arg_list) = args {
                                if arg_list.len() >= 4 {
                                    let (storage_id, storage_type) = match &arg_list[0] {
                                        PickleValue::Tuple(s_tuple) if s_tuple.len() >= 3 => {
                                            let s_type = match &s_tuple[1] {
                                                PickleValue::Global(_, ref cls) => cls.clone(),
                                                _ => "FloatStorage".to_string(),
                                            };
                                            let s_id = s_tuple[2].as_string().unwrap_or("0").to_string();
                                            (s_id, s_type)
                                        }
                                        _ => ("0".to_string(), "FloatStorage".to_string()),
                                    };

                                    let storage_offset = arg_list[1].as_int().unwrap_or(0) as usize;

                                    let shape: Vec<usize> = match &arg_list[2] {
                                        PickleValue::Tuple(dims) => dims.iter().filter_map(|d| d.as_int().map(|i| i as usize)).collect(),
                                        _ => Vec::new(),
                                    };

                                    let stride: Vec<usize> = match &arg_list[3] {
                                        PickleValue::Tuple(dims) => dims.iter().filter_map(|d| d.as_int().map(|i| i as usize)).collect(),
                                        _ => Vec::new(),
                                    };

                                    let dtype = storage_class_to_dtype(&storage_type);

                                    stack.push(PickleValue::Tensor(PyTorchTensorMeta {
                                        name: String::new(),
                                        storage_id,
                                        storage_type,
                                        storage_offset,
                                        shape,
                                        stride,
                                        dtype,
                                    }));
                                    continue;
                                }
                            }
                        } else if module == "torch._utils" && name == "_rebuild_parameter" {
                            // Args: (data, requires_grad, backward_hooks)
                            if let PickleValue::Tuple(ref arg_list) = args {
                                if !arg_list.is_empty() {
                                    stack.push(arg_list[0].clone());
                                    continue;
                                }
                            }
                        } else if (module == "collections" && name == "OrderedDict")
                            || (module == "builtins" && name == "dict")
                        {
                            stack.push(PickleValue::Dict(Vec::new()));
                            continue;
                        }

                        // Default fallback: push tuple of (callable, args)
                        stack.push(PickleValue::Tuple(vec![callable, args]));
                    }
                    _ => {
                        stack.push(PickleValue::Tuple(vec![callable, args]));
                    }
                }
            }
            b'b' => {
                // BUILD: pop state, pop target object
                let _state = stack.pop();
                // keep target on stack
            }
            b'.' => {
                // STOP: execution complete
                break;
            }
            _ => {
                // Ignore unrecognized bytecode or non-essential opcodes
            }
        }
    }

    // Traverse root dict to collect all tensors
    let mut results = Vec::new();
    for item in stack {
        collect_tensors(&item, &mut results);
    }

    Ok(results)
}

fn collect_tensors(val: &PickleValue, out: &mut Vec<(String, PyTorchTensorMeta)>) {
    match val {
        PickleValue::Dict(kvs) => {
            for (k, v) in kvs {
                if let Some(key_name) = k.as_string() {
                    if let PickleValue::Tensor(mut meta) = v.clone() {
                        meta.name = key_name.to_string();
                        out.push((key_name.to_string(), meta));
                    } else {
                        collect_tensors(v, out);
                    }
                } else {
                    collect_tensors(v, out);
                }
            }
        }
        PickleValue::Tuple(items) | PickleValue::List(items) => {
            for item in items {
                collect_tensors(item, out);
            }
        }
        _ => {}
    }
}

impl PyTorchCheckpoint {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path_ref = path.as_ref();
        let mmap = MmapFile::open(path_ref)
            .map_err(|e| format!("failed to mmap pytorch checkpoint {}: {}", path_ref.display(), e))?;

        let bytes = mmap.as_slice();
        let zip = ZipArchive::parse(bytes)
            .map_err(|e| format!("failed to parse pytorch zip archive {}: {}", path_ref.display(), e))?;

        // Locate data.pkl (e.g. "archive/data.pkl", "data.pkl", or "<model_stem>/data.pkl")
        let pkl_entry = zip
            .entry_names()
            .find(|n| n.ends_with("data.pkl"))
            .cloned()
            .ok_or_else(|| "no 'data.pkl' found in checkpoint".to_string())?;

        let prefix = if let Some(idx) = pkl_entry.rfind("data.pkl") {
            pkl_entry[..idx].to_string()
        } else {
            String::new()
        };

        let pkl_bytes = zip.read_entry(&pkl_entry)?;
        let parsed_tensors = parse_pytorch_pickle(&pkl_bytes)?;

        let mut tensors = HashMap::new();

        for (name, meta) in parsed_tensors {
            // Find storage entry in ZIP: check prefix, archive/, and bare data/
            let storage_key1 = format!("{}data/{}", prefix, meta.storage_id);
            let storage_key2 = format!("archive/data/{}", meta.storage_id);
            let storage_key3 = format!("data/{}", meta.storage_id);
            let entry = match zip
                .get_entry(&storage_key1)
                .or_else(|| zip.get_entry(&storage_key2))
                .or_else(|| zip.get_entry(&storage_key3))
            {
                Some(e) => e,
                None => continue,
            };

            let item_size = meta.dtype.item_size_bytes();
            let num_elements: usize = if meta.shape.is_empty() {
                1
            } else {
                meta.shape.iter().product()
            };
            let size_bytes = num_elements * item_size;

            let abs_offset = entry.data_offset as usize + meta.storage_offset * item_size;

            tensors.insert(
                name,
                PyTorchTensorView {
                    dtype: meta.dtype,
                    shape: meta.shape,
                    data_offset: abs_offset,
                    size_bytes,
                },
            );
        }

        Ok(Self { mmap, tensors })
    }

    #[inline]
    pub fn get_tensor(&self, name: &str) -> Option<(&PyTorchTensorView, &[u8])> {
        let tv = self.tensors.get(name)?;
        let slice = self.mmap.slice(tv.data_offset, tv.size_bytes)?;
        Some((tv, slice))
    }

    #[inline]
    pub fn tensor_names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }
}
