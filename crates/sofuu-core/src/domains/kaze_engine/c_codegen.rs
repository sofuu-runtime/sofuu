// c_codegen.rs — ANSI/ISO C99 Native Code Generator for Kaze.
//
// Translates a validated Kaze AST (Program) into self-contained, portable C99 code.
// Operates with zero external dependencies and supports both 32-bit and 64-bit architectures.

use super::ast::*;

pub struct CodegenOptions {
    pub standalone: bool,
    pub entry: Option<String>,
}

impl Default for CodegenOptions {
    fn default() -> Self {
        CodegenOptions {
            standalone: true,
            entry: None,
        }
    }
}

pub fn emit_c(program: &Program, opts: &CodegenOptions) -> String {
    let mut out = String::with_capacity(8192);

    // 1. Header & C99 runtime
    out.push_str(C_RUNTIME_HEADER);

    // 2. Forward declarations for records and routines
    out.push_str("/* --- Forward Declarations --- */\n");
    for item in &program.items {
        match &item.kind {
            ItemKind::Record(fields) => {
                let rec_name = sanitize_id(&item.name);
                let params = fields
                    .iter()
                    .map(|f| format!("KzValue {}", sanitize_id(&f.name)))
                    .collect::<Vec<_>>()
                    .join(", ");
                let sig_params = if params.is_empty() { "void".to_string() } else { params };
                out.push_str(&format!("static inline KzValue kz_record_new_{}({});\n", rec_name, sig_params));
            }
            ItemKind::Action | ItemKind::Check | ItemKind::Filter => {
                let fn_name = sanitize_id(&item.name);
                let params = item
                    .params
                    .iter()
                    .map(|p| format!("KzValue {}", sanitize_id(p)))
                    .collect::<Vec<_>>()
                    .join(", ");
                let sig_params = if params.is_empty() { "void".to_string() } else { params };
                out.push_str(&format!("static KzValue kz_fn_{}({});\n", fn_name, sig_params));
            }
            _ => {}
        }
    }
    out.push('\n');

    // 3. Record Constructors
    out.push_str("/* --- Record Constructors --- */\n");
    for item in &program.items {
        if let ItemKind::Record(fields) = &item.kind {
            let rec_name = sanitize_id(&item.name);
            let n = fields.len();
            let params = fields
                .iter()
                .map(|f| format!("KzValue {}", sanitize_id(&f.name)))
                .collect::<Vec<_>>()
                .join(", ");
            let sig_params = if params.is_empty() { "void".to_string() } else { params };

            out.push_str(&format!("static inline KzValue kz_record_new_{}({}) {{\n", rec_name, sig_params));
            if n == 0 {
                out.push_str(&format!("    return kz_record_create(\"{}\", 0, NULL, NULL);\n", item.name));
            } else {
                let keys = fields
                    .iter()
                    .map(|f| format!("\"{}\"", f.name))
                    .collect::<Vec<_>>()
                    .join(", ");
                let vals = fields
                    .iter()
                    .map(|f| sanitize_id(&f.name))
                    .collect::<Vec<_>>()
                    .join(", ");
                out.push_str(&format!("    static const char* _keys[] = {{ {} }};\n", keys));
                out.push_str(&format!("    KzValue _vals[] = {{ {} }};\n", vals));
                out.push_str(&format!("    return kz_record_create(\"{}\", {}, _keys, _vals);\n", item.name, n));
            }
            out.push_str("}\n\n");
        }
    }

    // 4. Routine Implementations
    out.push_str("/* --- Routines --- */\n");
    for item in &program.items {
        if matches!(item.kind, ItemKind::Action | ItemKind::Check | ItemKind::Filter) {
            let fn_name = sanitize_id(&item.name);
            let params = item
                .params
                .iter()
                .map(|p| format!("KzValue {}", sanitize_id(p)))
                .collect::<Vec<_>>()
                .join(", ");
            let sig_params = if params.is_empty() { "void".to_string() } else { params };

            out.push_str(&format!("static KzValue kz_fn_{}({}) {{\n", fn_name, sig_params));
            for stmt in &item.body {
                emit_stmt(stmt, &mut out, 1);
            }
            out.push_str("    return kz_nil();\n");
            out.push_str("}\n\n");
        }
    }

    // 5. Entry point main if standalone
    if opts.standalone {
        let entry_name = opts.entry.as_deref().unwrap_or("main");
        let entry_item = program.items.iter().find(|it| it.name == entry_name);
        out.push_str("/* --- Main Entry Point --- */\n");
        out.push_str("int main(int argc, char** argv) {\n");
        if let Some(item) = entry_item {
            if item.params.is_empty() {
                out.push_str("    (void)argc; (void)argv;\n");
                out.push_str(&format!("    KzValue res = kz_fn_{}();\n", sanitize_id(entry_name)));
            } else {
                for (i, p) in item.params.iter().enumerate() {
                    out.push_str(&format!("    KzValue arg_{} = kz_resolve_cli_arg(argc, argv, \"{}\", {});\n", i, p, i));
                }
                let args_call = (0..item.params.len()).map(|i| format!("arg_{}", i)).collect::<Vec<_>>().join(", ");
                out.push_str(&format!("    KzValue res = kz_fn_{}({});\n", sanitize_id(entry_name), args_call));
            }
            out.push_str("    if (res.kind == KZ_VAL_NUM) {\n");
            out.push_str("        if (floor(res.as.number) == res.as.number) {\n");
            out.push_str("            printf(\"%.0f\\n\", res.as.number);\n");
            out.push_str("        } else {\n");
            out.push_str("            printf(\"%g\\n\", res.as.number);\n");
            out.push_str("        }\n");
            out.push_str("    } else if (res.kind == KZ_VAL_STR) {\n");
            out.push_str("        printf(\"%s\\n\", res.as.str ? res.as.str : \"\");\n");
            out.push_str("    } else if (res.kind == KZ_VAL_BOOL) {\n");
            out.push_str("        printf(\"%s\\n\", res.as.boolean ? \"true\" : \"false\");\n");
            out.push_str("    }\n");
        } else {
            out.push_str("    (void)argc; (void)argv;\n");
            out.push_str("    printf(\"Kaze module loaded (no main entry)\\n\");\n");
        }
        out.push_str("    return 0;\n");
        out.push_str("}\n");
    }

    out
}

fn emit_stmt(stmt: &Stmt, out: &mut String, indent: usize) {
    let ind = "    ".repeat(indent);
    match stmt {
        Stmt::Let(var, expr, _) => {
            let id = sanitize_id(var);
            let val = emit_expr(expr);
            out.push_str(&format!("{}KzValue {} = {};\n", ind, id, val));
        }
        Stmt::Set(var, expr, _) => {
            let id = sanitize_id(var);
            let val = emit_expr(expr);
            out.push_str(&format!("{}{} = {};\n", ind, id, val));
        }
        Stmt::Eval(expr, _) => {
            let val = emit_expr(expr);
            out.push_str(&format!("{}(void)({});\n", ind, val));
        }
        Stmt::If { cond, then, otherwise, .. } => {
            let c = emit_expr(cond);
            out.push_str(&format!("{}if (kz_is_truthy({})) {{\n", ind, c));
            for s in then {
                emit_stmt(s, out, indent + 1);
            }
            if !otherwise.is_empty() {
                out.push_str(&format!("{}}} else {{\n", ind));
                for s in otherwise {
                    emit_stmt(s, out, indent + 1);
                }
            }
            out.push_str(&format!("{}}}\n", ind));
        }
        Stmt::While { cond, body, .. } => {
            let c = emit_expr(cond);
            out.push_str(&format!("{}while (kz_is_truthy({})) {{\n", ind, c));
            for s in body {
                emit_stmt(s, out, indent + 1);
            }
            out.push_str(&format!("{}}}\n", ind));
        }
        Stmt::ForEach { var, list, body, .. } => {
            let id = sanitize_id(var);
            let l = emit_expr(list);
            out.push_str(&format!("{}{{\n", ind));
            out.push_str(&format!("{}    KzValue _iter_target = {};\n", ind, l));
            out.push_str(&format!("    {}if (_iter_target.kind == KZ_VAL_LIST && _iter_target.as.list) {{\n", ind));
            out.push_str(&format!("    {}    size_t _len = _iter_target.as.list->len;\n", ind));
            out.push_str(&format!("    {}    for (size_t _i = 0; _i < _len; _i++) {{\n", ind));
            out.push_str(&format!("    {}        KzValue {} = _iter_target.as.list->items[_i];\n", ind, id));
            for s in body {
                emit_stmt(s, out, indent + 2);
            }
            out.push_str(&format!("    {}    }}\n", ind));
            out.push_str(&format!("    {}}}\n", ind));
            out.push_str(&format!("{}}}\n", ind));
        }
        Stmt::Try { body, otherwise, error_name, .. } => {
            out.push_str(&format!("{}{{\n", ind));
            out.push_str(&format!("{}    /* try block */\n", ind));
            for s in body {
                emit_stmt(s, out, indent + 1);
            }
            if !otherwise.is_empty() {
                out.push_str(&format!("{}    if (0) {{\n", ind));
                if let Some(err_n) = error_name {
                    let eid = sanitize_id(err_n);
                    out.push_str(&format!("{}        KzValue {} = kz_nil();\n", ind, eid));
                }
                for s in otherwise {
                    emit_stmt(s, out, indent + 2);
                }
                out.push_str(&format!("{}    }}\n", ind));
            }
            out.push_str(&format!("{}}}\n", ind));
        }
        Stmt::Return(expr, _) => {
            if let Some(e) = expr {
                let v = emit_expr(e);
                out.push_str(&format!("{}return {};\n", ind, v));
            } else {
                out.push_str(&format!("{}return kz_nil();\n", ind));
            }
        }
    }
}

fn emit_expr(expr: &Expr) -> String {
    match expr {
        Expr::Num(n, _) => {
            if n.is_nan() {
                "kz_num(0.0)".to_string()
            } else if n.is_infinite() {
                if *n > 0.0 { "kz_num(1e300)".to_string() } else { "kz_num(-1e300)".to_string() }
            } else {
                format!("kz_num({:?})", n)
            }
        }
        Expr::Str(s) => {
            format!("kz_str({})", escape_c_string(s))
        }
        Expr::Bool(b) => {
            format!("kz_bool({})", if *b { "true" } else { "false" })
        }
        Expr::Name(name) => sanitize_id(name),
        Expr::Ambient { field, family } => {
            format!("kz_cap_get(\"{}\", \"{}\")", family, field)
        }
        Expr::Call { recv, op, args, .. } => {
            let a_strs: Vec<String> = args.iter().map(emit_expr).collect();
            if recv == "log" {
                let arg_v = a_strs.get(0).cloned().unwrap_or_else(|| "kz_nil()".into());
                format!("kz_builtin_log({})", arg_v)
            } else if recv == "alert" {
                let arg_v = a_strs.get(0).cloned().unwrap_or_else(|| "kz_nil()".into());
                format!("kz_builtin_alert({})", arg_v)
            } else {
                format!("kz_cap_call(\"{}\", \"{}\", {}, (KzValue[]){{ {} }})",
                    recv, op, args.len(), if a_strs.is_empty() { "kz_nil()".to_string() } else { a_strs.join(", ") })
            }
        }
        Expr::Field { base, field, .. } => {
            format!("kz_get_field({}, \"{}\")", sanitize_id(base), field)
        }
        Expr::CallRoutine { name, args, .. } => {
            let fn_name = sanitize_id(name);
            let arg_str = args.iter().map(emit_expr).collect::<Vec<_>>().join(", ");
            format!("kz_fn_{}({})", fn_name, arg_str)
        }
        Expr::RecordInit { name, fields, .. } => {
            let rec_name = sanitize_id(name);
            let arg_str = fields.iter().map(|(_, e)| emit_expr(e)).collect::<Vec<_>>().join(", ");
            format!("kz_record_new_{}({})", rec_name, arg_str)
        }
        Expr::Bin(op, left, right) => {
            let l = emit_expr(left);
            let r = emit_expr(right);
            match op {
                BinOp::Add => format!("kz_add({}, {})", l, r),
                BinOp::Sub => format!("kz_sub({}, {})", l, r),
                BinOp::Mul => format!("kz_mul({}, {})", l, r),
                BinOp::Div => format!("kz_div({}, {})", l, r),
                BinOp::Mod => format!("kz_mod({}, {})", l, r),
                BinOp::Eq => format!("kz_eq({}, {})", l, r),
                BinOp::Ne => format!("kz_ne({}, {})", l, r),
                BinOp::Lt => format!("kz_lt({}, {})", l, r),
                BinOp::Le => format!("kz_le({}, {})", l, r),
                BinOp::Gt => format!("kz_gt({}, {})", l, r),
                BinOp::Ge => format!("kz_ge({}, {})", l, r),
                BinOp::And => format!("kz_and({}, {})", l, r),
                BinOp::Or => format!("kz_or({}, {})", l, r),
            }
        }
        Expr::Not(inner) => format!("kz_not({})", emit_expr(inner)),
        Expr::Len(inner) => format!("kz_len({})", emit_expr(inner)),
        Expr::Index(coll, idx) => {
            let c = emit_expr(coll);
            let i = emit_expr(idx);
            format!("kz_index({}, {})", c, i)
        }
        Expr::List(items) => {
            if items.is_empty() {
                "kz_list_new(0)".to_string()
            } else {
                let args = items.iter().map(emit_expr).collect::<Vec<_>>().join(", ");
                format!("kz_list_from({}, (KzValue[]){{ {} }})", items.len(), args)
            }
        }
        Expr::Math { kind, args, .. } => {
            let a_strs: Vec<String> = args.iter().map(emit_expr).collect();
            match kind {
                MathKind::Vector => {
                    format!("kz_point({}, {}, {})",
                        a_strs.get(0).cloned().unwrap_or_else(|| "kz_nil()".into()),
                        a_strs.get(1).cloned().unwrap_or_else(|| "kz_nil()".into()),
                        a_strs.get(2).cloned().unwrap_or_else(|| "kz_nil()".into())
                    )
                }
                MathKind::Sqrt => format!("kz_math_sqrt({})", a_strs.get(0).cloned().unwrap_or_else(|| "kz_nil()".into())),
                MathKind::Sin => format!("kz_math_sin({})", a_strs.get(0).cloned().unwrap_or_else(|| "kz_nil()".into())),
                MathKind::Cos => format!("kz_math_cos({})", a_strs.get(0).cloned().unwrap_or_else(|| "kz_nil()".into())),
                MathKind::Atan2 => format!("kz_math_atan2({}, {})",
                    a_strs.get(0).cloned().unwrap_or_else(|| "kz_nil()".into()),
                    a_strs.get(1).cloned().unwrap_or_else(|| "kz_nil()".into())
                ),
                MathKind::ToRadians => format!("kz_math_to_radians({})", a_strs.get(0).cloned().unwrap_or_else(|| "kz_nil()".into())),
                MathKind::ToDegrees => format!("kz_math_to_degrees({})", a_strs.get(0).cloned().unwrap_or_else(|| "kz_nil()".into())),
                _ => "kz_num(0.0)".to_string(),
            }
        }
        Expr::LinkLost { .. } => "kz_bool(false)".to_string(),
    }
}

fn sanitize_id(name: &str) -> String {
    match name {
        "auto" | "break" | "case" | "char" | "const" | "continue" | "default" | "do" | "double"
        | "else" | "enum" | "extern" | "float" | "for" | "goto" | "if" | "inline" | "int"
        | "long" | "register" | "restrict" | "return" | "short" | "signed" | "sizeof" | "static"
        | "struct" | "switch" | "typedef" | "union" | "unsigned" | "void" | "volatile" | "while"
        | "main" => format!("_kz_{}", name),
        other => {
            let mut s = String::with_capacity(other.len());
            for c in other.chars() {
                if c.is_ascii_alphanumeric() || c == '_' {
                    s.push(c);
                } else {
                    s.push('_');
                }
            }
            s
        }
    }
}

fn escape_c_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\0' => out.push_str("\\0"),
            other => out.push(other),
        }
    }
    out.push('"');
    out
}

const C_RUNTIME_HEADER: &str = r#"/* Generated by Kaze C99 Native Emitter (Stage 0) */
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#if !defined(_WIN32)
#include <unistd.h>
#else
#include <io.h>
#define isatty _isatty
#define fileno _fileno
#endif

/* 32-bit & 64-bit portable value representation */
typedef enum {
    KZ_VAL_NIL = 0,
    KZ_VAL_BOOL,
    KZ_VAL_NUM,
    KZ_VAL_STR,
    KZ_VAL_LIST,
    KZ_VAL_RECORD,
    KZ_VAL_POINT,
} KzValKind;

struct KzValue;

typedef struct {
    size_t len;
    size_t cap;
    struct KzValue* items;
} KzList;

typedef struct {
    const char* type_name;
    size_t field_count;
    const char** keys;
    struct KzValue* values;
} KzRecord;

typedef struct {
    double x, y, z;
} KzPoint;

typedef struct KzValue {
    KzValKind kind;
    union {
        bool boolean;
        double number;
        const char* str;
        KzList* list;
        KzRecord* record;
        KzPoint point;
    } as;
} KzValue;

static inline KzValue kz_nil(void) {
    KzValue v; v.kind = KZ_VAL_NIL; v.as.number = 0.0; return v;
}
static inline KzValue kz_bool(bool b) {
    KzValue v; v.kind = KZ_VAL_BOOL; v.as.boolean = b; return v;
}
static inline KzValue kz_num(double n) {
    KzValue v; v.kind = KZ_VAL_NUM; v.as.number = n; return v;
}
static inline KzValue kz_str(const char* s) {
    KzValue v; v.kind = KZ_VAL_STR; v.as.str = s ? s : ""; return v;
}
static inline KzValue kz_point(KzValue x, KzValue y, KzValue z) {
    KzValue v; v.kind = KZ_VAL_POINT;
    v.as.point.x = (x.kind == KZ_VAL_NUM) ? x.as.number : 0.0;
    v.as.point.y = (y.kind == KZ_VAL_NUM) ? y.as.number : 0.0;
    v.as.point.z = (z.kind == KZ_VAL_NUM) ? z.as.number : 0.0;
    return v;
}
static inline bool kz_is_truthy(KzValue v) {
    switch (v.kind) {
        case KZ_VAL_BOOL: return v.as.boolean;
        case KZ_VAL_NUM: return v.as.number != 0.0;
        case KZ_VAL_STR: return v.as.str && v.as.str[0] != '\0';
        case KZ_VAL_LIST: return v.as.list && v.as.list->len > 0;
        case KZ_VAL_RECORD: return true;
        case KZ_VAL_POINT: return true;
        default: return false;
    }
}

static inline char* kz_str_alloc_copy(const char* s) {
    if (!s) return "";
    size_t l = strlen(s);
    char* out = (char*)malloc(l + 1);
    if (!out) return "";
    memcpy(out, s, l + 1);
    return out;
}

static inline char* kz_str_concat(const char* a, const char* b) {
    size_t la = a ? strlen(a) : 0;
    size_t lb = b ? strlen(b) : 0;
    char* out = (char*)malloc(la + lb + 1);
    if (!out) return "";
    if (la) memcpy(out, a, la);
    if (lb) memcpy(out + la, b, lb);
    out[la + lb] = '\0';
    return out;
}

static inline const char* kz_to_string(KzValue v) {
    if (v.kind == KZ_VAL_STR) return v.as.str ? v.as.str : "";
    if (v.kind == KZ_VAL_BOOL) return v.as.boolean ? "true" : "false";
    if (v.kind == KZ_VAL_NUM) {
        char buf[64];
        if (floor(v.as.number) == v.as.number && !isnan(v.as.number) && !isinf(v.as.number)) {
            snprintf(buf, sizeof(buf), "%.0f", v.as.number);
        } else {
            snprintf(buf, sizeof(buf), "%g", v.as.number);
        }
        return kz_str_alloc_copy(buf);
    }
    if (v.kind == KZ_VAL_NIL) return "nil";
    if (v.kind == KZ_VAL_RECORD) return v.as.record ? v.as.record->type_name : "record";
    if (v.kind == KZ_VAL_LIST) return "list";
    if (v.kind == KZ_VAL_POINT) return "point";
    return "";
}

static inline KzValue kz_list_new(size_t cap) {
    KzList* l = (KzList*)malloc(sizeof(KzList));
    l->len = 0;
    l->cap = cap > 4 ? cap : 4;
    l->items = (KzValue*)malloc(l->cap * sizeof(KzValue));
    KzValue v; v.kind = KZ_VAL_LIST; v.as.list = l; return v;
}

static inline void kz_list_push(KzValue* list_val, KzValue item) {
    if (list_val->kind != KZ_VAL_LIST || !list_val->as.list) return;
    KzList* l = list_val->as.list;
    if (l->len >= l->cap) {
        l->cap = l->cap < 8 ? 8 : l->cap * 2;
        l->items = (KzValue*)realloc(l->items, l->cap * sizeof(KzValue));
    }
    l->items[l->len++] = item;
}

static inline KzValue kz_list_concat(KzValue a, KzValue b) {
    size_t la = (a.kind == KZ_VAL_LIST && a.as.list) ? a.as.list->len : 0;
    size_t lb = (b.kind == KZ_VAL_LIST && b.as.list) ? b.as.list->len : 0;
    KzValue res = kz_list_new(la + lb);
    KzList* l = res.as.list;
    if (la) {
        memcpy(l->items, a.as.list->items, la * sizeof(KzValue));
        l->len += la;
    }
    if (lb) {
        memcpy(l->items + l->len, b.as.list->items, lb * sizeof(KzValue));
        l->len += lb;
    }
    return res;
}

static inline KzValue kz_list_from(size_t n, const KzValue* items) {
    KzValue res = kz_list_new(n);
    if (n > 0 && items) {
        memcpy(res.as.list->items, items, n * sizeof(KzValue));
        res.as.list->len = n;
    }
    return res;
}

static inline KzValue kz_record_create(const char* type_name, size_t n, const char** keys, const KzValue* vals) {
    KzRecord* r = (KzRecord*)malloc(sizeof(KzRecord));
    r->type_name = type_name;
    r->field_count = n;
    r->keys = keys;
    r->values = (n > 0) ? (KzValue*)malloc(n * sizeof(KzValue)) : NULL;
    if (n > 0 && vals && r->values) {
        memcpy(r->values, vals, n * sizeof(KzValue));
    }
    KzValue v; v.kind = KZ_VAL_RECORD; v.as.record = r; return v;
}

static inline KzValue kz_get_field(KzValue obj, const char* field) {
    if (obj.kind == KZ_VAL_POINT) {
        if (strcmp(field, "x") == 0) return kz_num(obj.as.point.x);
        if (strcmp(field, "y") == 0) return kz_num(obj.as.point.y);
        if (strcmp(field, "z") == 0) return kz_num(obj.as.point.z);
        return kz_nil();
    }
    if (obj.kind != KZ_VAL_RECORD || !obj.as.record) return kz_nil();
    KzRecord* r = obj.as.record;
    for (size_t i = 0; i < r->field_count; i++) {
        if (strcmp(r->keys[i], field) == 0) return r->values[i];
    }
    return kz_nil();
}

static inline KzValue kz_add(KzValue a, KzValue b) {
    if (a.kind == KZ_VAL_NUM && b.kind == KZ_VAL_NUM) {
        return kz_num(a.as.number + b.as.number);
    }
    if (a.kind == KZ_VAL_STR || b.kind == KZ_VAL_STR) {
        const char* sa = kz_to_string(a);
        const char* sb = kz_to_string(b);
        return kz_str(kz_str_concat(sa, sb));
    }
    if (a.kind == KZ_VAL_LIST && b.kind == KZ_VAL_LIST) {
        return kz_list_concat(a, b);
    }
    return kz_nil();
}

static inline KzValue kz_sub(KzValue a, KzValue b) {
    return kz_num(((a.kind == KZ_VAL_NUM) ? a.as.number : 0.0) - ((b.kind == KZ_VAL_NUM) ? b.as.number : 0.0));
}
static inline KzValue kz_mul(KzValue a, KzValue b) {
    return kz_num(((a.kind == KZ_VAL_NUM) ? a.as.number : 0.0) * ((b.kind == KZ_VAL_NUM) ? b.as.number : 0.0));
}
static inline KzValue kz_div(KzValue a, KzValue b) {
    double denom = (b.kind == KZ_VAL_NUM) ? b.as.number : 0.0;
    if (denom == 0.0) return kz_num(0.0);
    return kz_num(((a.kind == KZ_VAL_NUM) ? a.as.number : 0.0) / denom);
}
static inline KzValue kz_mod(KzValue a, KzValue b) {
    double denom = (b.kind == KZ_VAL_NUM) ? b.as.number : 0.0;
    if (denom == 0.0) return kz_num(0.0);
    return kz_num(fmod((a.kind == KZ_VAL_NUM) ? a.as.number : 0.0, denom));
}
static inline KzValue kz_not(KzValue a) {
    return kz_bool(!kz_is_truthy(a));
}

static inline bool kz_value_eq(KzValue a, KzValue b) {
    if (a.kind != b.kind) return false;
    switch (a.kind) {
        case KZ_VAL_NIL: return true;
        case KZ_VAL_BOOL: return a.as.boolean == b.as.boolean;
        case KZ_VAL_NUM: return a.as.number == b.as.number;
        case KZ_VAL_STR: return (a.as.str && b.as.str) ? (strcmp(a.as.str, b.as.str) == 0) : (a.as.str == b.as.str);
        default: return a.as.list == b.as.list;
    }
}
static inline KzValue kz_eq(KzValue a, KzValue b) { return kz_bool(kz_value_eq(a, b)); }
static inline KzValue kz_ne(KzValue a, KzValue b) { return kz_bool(!kz_value_eq(a, b)); }
static inline KzValue kz_lt(KzValue a, KzValue b) {
    if (a.kind == KZ_VAL_NUM && b.kind == KZ_VAL_NUM) return kz_bool(a.as.number < b.as.number);
    if (a.kind == KZ_VAL_STR && b.kind == KZ_VAL_STR) return kz_bool(strcmp(a.as.str, b.as.str) < 0);
    return kz_bool(false);
}
static inline KzValue kz_le(KzValue a, KzValue b) {
    if (a.kind == KZ_VAL_NUM && b.kind == KZ_VAL_NUM) return kz_bool(a.as.number <= b.as.number);
    if (a.kind == KZ_VAL_STR && b.kind == KZ_VAL_STR) return kz_bool(strcmp(a.as.str, b.as.str) <= 0);
    return kz_bool(false);
}
static inline KzValue kz_gt(KzValue a, KzValue b) {
    if (a.kind == KZ_VAL_NUM && b.kind == KZ_VAL_NUM) return kz_bool(a.as.number > b.as.number);
    if (a.kind == KZ_VAL_STR && b.kind == KZ_VAL_STR) return kz_bool(strcmp(a.as.str, b.as.str) > 0);
    return kz_bool(false);
}
static inline KzValue kz_ge(KzValue a, KzValue b) {
    if (a.kind == KZ_VAL_NUM && b.kind == KZ_VAL_NUM) return kz_bool(a.as.number >= b.as.number);
    if (a.kind == KZ_VAL_STR && b.kind == KZ_VAL_STR) return kz_bool(strcmp(a.as.str, b.as.str) >= 0);
    return kz_bool(false);
}
static inline KzValue kz_and(KzValue a, KzValue b) { return kz_bool(kz_is_truthy(a) && kz_is_truthy(b)); }
static inline KzValue kz_or(KzValue a, KzValue b) { return kz_bool(kz_is_truthy(a) || kz_is_truthy(b)); }

static inline KzValue kz_len(KzValue a) {
    if (a.kind == KZ_VAL_STR && a.as.str) return kz_num((double)strlen(a.as.str));
    if (a.kind == KZ_VAL_LIST && a.as.list) return kz_num((double)a.as.list->len);
    return kz_num(0.0);
}

static inline KzValue kz_index(KzValue coll, KzValue idx) {
    size_t i = (idx.kind == KZ_VAL_NUM && idx.as.number >= 0.0) ? (size_t)idx.as.number : 0;
    if (coll.kind == KZ_VAL_LIST && coll.as.list) {
        if (i < coll.as.list->len) return coll.as.list->items[i];
        return kz_nil();
    }
    if (coll.kind == KZ_VAL_STR && coll.as.str) {
        size_t l = strlen(coll.as.str);
        if (i < l) {
            char* one = (char*)malloc(2);
            one[0] = coll.as.str[i];
            one[1] = '\0';
            return kz_str(one);
        }
        return kz_str("");
    }
    return kz_nil();
}

static inline void kz_builtin_log(KzValue msg) {
    printf("[info] %s\n", kz_to_string(msg));
}
static inline void kz_builtin_alert(KzValue msg) {
    fprintf(stderr, "[alert] %s\n", kz_to_string(msg));
}

static inline KzValue kz_math_sqrt(KzValue a) {
    return kz_num(sqrt((a.kind == KZ_VAL_NUM) ? a.as.number : 0.0));
}
static inline KzValue kz_math_sin(KzValue a) {
    return kz_num(sin((a.kind == KZ_VAL_NUM) ? a.as.number : 0.0));
}
static inline KzValue kz_math_cos(KzValue a) {
    return kz_num(cos((a.kind == KZ_VAL_NUM) ? a.as.number : 0.0));
}
static inline KzValue kz_math_atan2(KzValue y, KzValue x) {
    double dy = (y.kind == KZ_VAL_NUM) ? y.as.number : 0.0;
    double dx = (x.kind == KZ_VAL_NUM) ? x.as.number : 0.0;
    return kz_num(atan2(dy, dx));
}
static inline KzValue kz_math_to_radians(KzValue a) {
    double deg = (a.kind == KZ_VAL_NUM) ? a.as.number : 0.0;
    return kz_num(deg * 3.14159265358979323846 / 180.0);
}
static inline KzValue kz_math_to_degrees(KzValue a) {
    double rad = (a.kind == KZ_VAL_NUM) ? a.as.number : 0.0;
    return kz_num(rad * 180.0 / 3.14159265358979323846);
}

static inline KzValue kz_cap_get(const char* family, const char* field) {
    (void)family; (void)field;
    return kz_nil();
}

static inline KzValue kz_cap_call(const char* family, const char* op, size_t argc, const KzValue* args) {
    (void)argc; (void)args;
    printf("[cap:%s.%s]\n", family, op);
    return kz_nil();
}

static inline KzValue kz_read_stream(FILE* f) {
    size_t cap = 4096;
    size_t len = 0;
    char* buf = (char*)malloc(cap);
    if (!buf) return kz_str("");
    size_t n;
    while ((n = fread(buf + len, 1, cap - len - 1, f)) > 0) {
        len += n;
        if (len + 1024 >= cap) {
            cap *= 2;
            char* nb = (char*)realloc(buf, cap);
            if (!nb) break;
            buf = nb;
        }
    }
    buf[len] = '\0';
    return kz_str(buf);
}

static inline KzValue kz_read_file_or_string(const char* path_or_str) {
    if (!path_or_str) return kz_str("");
    FILE* f = fopen(path_or_str, "rb");
    if (!f) {
        return kz_str(path_or_str);
    }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz < 0) {
        fclose(f);
        return kz_str(path_or_str);
    }
    char* buf = (char*)malloc((size_t)sz + 1);
    if (!buf) {
        fclose(f);
        return kz_str(path_or_str);
    }
    size_t rd = fread(buf, 1, (size_t)sz, f);
    buf[rd] = '\0';
    fclose(f);
    return kz_str(buf);
}

static inline KzValue kz_resolve_cli_arg(int argc, char** argv, const char* name, int index) {
    size_t nlen = strlen(name);
    for (int i = 1; i < argc; i++) {
        const char* a = argv[i];
        if (strncmp(a, "--", 2) == 0) {
            const char* opt = a + 2;
            if (strncmp(opt, name, nlen) == 0 && opt[nlen] == '=') {
                return kz_read_file_or_string(opt + nlen + 1);
            }
            if (strcmp(opt, "arg") == 0 && i + 1 < argc) {
                const char* next_arg = argv[i + 1];
                if (strncmp(next_arg, name, nlen) == 0 && next_arg[nlen] == '=') {
                    return kz_read_file_or_string(next_arg + nlen + 1);
                }
            }
            if (strncmp(opt, "arg=", 4) == 0) {
                const char* next_arg = opt + 4;
                if (strncmp(next_arg, name, nlen) == 0 && next_arg[nlen] == '=') {
                    return kz_read_file_or_string(next_arg + nlen + 1);
                }
            }
        }
    }
    char env_name[128];
    snprintf(env_name, sizeof(env_name), "AGENT_%s", name);
    for (size_t j = 6; env_name[j]; j++) {
        if (env_name[j] >= 'a' && env_name[j] <= 'z') env_name[j] -= 32;
    }
    char* ev = getenv(env_name);
    if (ev) return kz_read_file_or_string(ev);

    snprintf(env_name, sizeof(env_name), "SOFUU_%s", name);
    for (size_t j = 6; env_name[j]; j++) {
        if (env_name[j] >= 'a' && env_name[j] <= 'z') env_name[j] -= 32;
    }
    ev = getenv(env_name);
    if (ev) return kz_read_file_or_string(ev);

    int pos_idx = 0;
    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-') {
            if (pos_idx == index) {
                return kz_read_file_or_string(argv[i]);
            }
            pos_idx++;
        }
    }

    if (index == 0 && !isatty(fileno(stdin))) {
        return kz_read_stream(stdin);
    }

    return kz_str("");
}
"#;
