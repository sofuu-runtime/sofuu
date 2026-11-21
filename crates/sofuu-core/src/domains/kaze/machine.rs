// src/machine.rs — Machine syntax (prefix-tree / S-expression) parser and emitter.
//
// Implements the machine-oriented syntax specified in PLAN-KAZE-DUAL-SYNTAX.md.
// Provides bidirectional conversion between human syntax and machine syntax,
// parsing directly into the shared AST (Program).

use crate::sfs::ast::*;
use crate::sfs::parser::ParseError;

const MAX_INPUT_BYTES: usize = 10 * 1024 * 1024; // 10 MiB
const MAX_NESTING_DEPTH: usize = 256;

#[derive(Clone, Debug, PartialEq)]
pub enum SExpr {
    Atom(String, u32, u32),     // value, line, col
    String(String, u32, u32),   // value, line, col
    List(Vec<SExpr>, u32, u32), // children, line, col
}

impl SExpr {
    pub fn line(&self) -> u32 {
        match self {
            SExpr::Atom(_, l, _) | SExpr::String(_, l, _) | SExpr::List(_, l, _) => *l,
        }
    }

    pub fn col(&self) -> u32 {
        match self {
            SExpr::Atom(_, _, c) | SExpr::String(_, _, c) | SExpr::List(_, _, c) => *c,
        }
    }
}

/// Returns true if the source starts with the machine-syntax header `(kaze ...`.
pub fn is_machine_syntax(src: &str) -> bool {
    let mut in_comment = false;
    let mut chars = src.chars().peekable();
    while let Some(&ch) = chars.peek() {
        if in_comment {
            if ch == '\n' {
                in_comment = false;
            }
            chars.next();
            continue;
        }
        if ch.is_whitespace() {
            chars.next();
            continue;
        }
        if ch == ';' || ch == '#' {
            in_comment = true;
            chars.next();
            continue;
        }
        if ch == '/' {
            chars.next();
            if chars.peek() == Some(&'/') {
                in_comment = true;
                chars.next();
                continue;
            }
            return false;
        }
        if ch == '(' {
            let mut prefix = String::new();
            while let Some(&c) = chars.peek() {
                if c.is_whitespace() || c == ')' {
                    break;
                }
                prefix.push(c);
                chars.next();
            }
            return prefix == "(kaze";
        }
        return false;
    }
    false
}

fn err<T>(line: u32, col: u32, msg: impl Into<String>) -> Result<T, ParseError> {
    Err(ParseError {
        line,
        col,
        msg: msg.into(),
        code: "E_KAZE_PARSE",
    })
}

/// Tokenize and parse raw prefix-tree text into an SExpr hierarchy.
pub fn parse_sexpr(src: &str) -> Result<SExpr, ParseError> {
    if src.len() > MAX_INPUT_BYTES {
        return err(1, 1, "machine syntax input exceeds maximum allowed size (10 MB)");
    }

    let mut line: u32 = 1;
    let mut col: u32 = 0;
    let mut stack: Vec<(Vec<SExpr>, u32, u32)> = Vec::new();
    let mut chars = src.char_indices().peekable();

    while let Some(&(_i, ch)) = chars.peek() {
        chars.next();
        if ch == '\n' {
            line += 1;
            col = 0;
            continue;
        }
        col += 1;

        if ch.is_whitespace() {
            continue;
        }

        // Line comments: ';' or '#'
        if ch == ';' || ch == '#' {
            while let Some(&(_, next_ch)) = chars.peek() {
                if next_ch == '\n' {
                    break;
                }
                chars.next();
            }
            continue;
        }

        // Open parenthesis
        if ch == '(' {
            if stack.len() >= MAX_NESTING_DEPTH {
                return err(line, col, "machine syntax nesting depth exceeds maximum (256)");
            }
            stack.push((Vec::new(), line, col));
            continue;
        }

        // Close parenthesis
        if ch == ')' {
            if let Some((children, open_line, open_col)) = stack.pop() {
                let list = SExpr::List(children, open_line, open_col);
                if let Some(parent) = stack.last_mut() {
                    parent.0.push(list);
                } else {
                    // Check for trailing non-whitespace
                    while let Some(&(_, remaining)) = chars.peek() {
                        if remaining == '\n' {
                            line += 1;
                            col = 0;
                            chars.next();
                            continue;
                        }
                        if remaining.is_whitespace() {
                            col += 1;
                            chars.next();
                            continue;
                        }
                        if remaining == ';' || remaining == '#' {
                            while let Some(&(_, c)) = chars.peek() {
                                if c == '\n' {
                                    break;
                                }
                                chars.next();
                            }
                            continue;
                        }
                        return err(line, col, "unexpected tokens after root closing parenthesis");
                    }
                    return Ok(list);
                }
            } else {
                return err(line, col, "unexpected ')' without matching '('");
            }
            continue;
        }

        // String literal
        if ch == '"' {
            let start_line = line;
            let start_col = col;
            let mut s = String::new();
            let mut closed = false;

            while let Some(&(_, sc)) = chars.peek() {
                chars.next();
                if sc == '\n' {
                    line += 1;
                    col = 0;
                } else {
                    col += 1;
                }

                if sc == '"' {
                    closed = true;
                    break;
                }
                if sc == '\\' {
                    if let Some(&(_, esc)) = chars.peek() {
                        chars.next();
                        col += 1;
                        match esc {
                            'n' => s.push('\n'),
                            't' => s.push('\t'),
                            'r' => s.push('\r'),
                            '\\' => s.push('\\'),
                            '"' => s.push('"'),
                            _ => {
                                s.push('\\');
                                s.push(esc);
                            }
                        }
                    } else {
                        return err(line, col, "unexpected end of input in string escape");
                    }
                } else {
                    s.push(sc);
                }
            }

            if !closed {
                return err(start_line, start_col, "unterminated string literal");
            }

            let expr = SExpr::String(s, start_line, start_col);
            if let Some(parent) = stack.last_mut() {
                parent.0.push(expr);
            } else {
                return err(start_line, start_col, "root element must be a list '(kaze ...)'");
            }
            continue;
        }

        // Atom (identifiers, numbers, symbols)
        let start_line = line;
        let start_col = col;
        let mut atom = String::new();
        atom.push(ch);

        while let Some(&(_, next_ch)) = chars.peek() {
            if next_ch.is_whitespace() || next_ch == '(' || next_ch == ')' || next_ch == ';' || next_ch == '#' {
                break;
            }
            atom.push(next_ch);
            chars.next();
            col += 1;
        }

        let expr = SExpr::Atom(atom, start_line, start_col);
        if let Some(parent) = stack.last_mut() {
            parent.0.push(expr);
        } else {
            return err(start_line, start_col, "root element must be a list '(kaze ...)'");
        }
    }

    if !stack.is_empty() {
        let (_, open_line, open_col) = stack.last().unwrap();
        return err(*open_line, *open_col, "unclosed '(' at end of input");
    }

    err(1, 1, "empty machine syntax input; expected '(kaze ...)'")
}

/// Parse machine syntax into a shared AST Program.
pub fn parse_machine(src: &str) -> Result<Program, ParseError> {
    let root = parse_sexpr(src)?;
    let elements = match &root {
        SExpr::List(elems, _, _) => elems,
        _ => return err(root.line(), root.col(), "machine syntax root must be '(kaze ...)'"),
    };

    if elements.is_empty() {
        return err(root.line(), root.col(), "empty '(kaze)' root form");
    }

    match &elements[0] {
        SExpr::Atom(tag, l, c) if tag == "kaze" => {}
        _ => return err(root.line(), root.col(), "expected '(kaze ...)' at root"),
    }

    let mut prog = Program::default();
    let mut format_verified = false;

    for item in &elements[1..] {
        match item {
            SExpr::List(children, l, c) if !children.is_empty() => {
                let tag = match &children[0] {
                    SExpr::Atom(t, _, _) => t.as_str(),
                    _ => return err(*l, *c, "expected tag symbol at head of form"),
                };

                match tag {
                    "format" => {
                        if children.len() != 2 {
                            return err(*l, *c, "(format <version>) requires exactly 1 version argument");
                        }
                        let ver_num = match &children[1] {
                            SExpr::Atom(v, _, _) => v.parse::<u32>().map_err(|_| ()),
                            SExpr::String(v, _, _) => v.parse::<u32>().map_err(|_| ()),
                            _ => Err(()),
                        };
                        match ver_num {
                            Ok(1) => {
                                format_verified = true;
                            }
                            Ok(v) => return err(*l, *c, format!("unsupported machine syntax format version: {}", v)),
                            Err(_) => return err(*l, *c, "machine syntax format version must be an integer"),
                        }
                    }
                    "needs" => {
                        for cap in &children[1..] {
                            let cap_str = match cap {
                                SExpr::Atom(s, _, _) | SExpr::String(s, _, _) => s.clone(),
                                _ => return err(cap.line(), cap.col(), "capability in 'needs' must be an identifier or string"),
                            };
                            if !prog.needs.contains(&cap_str) {
                                prog.needs.push(cap_str);
                            }
                        }
                    }
                    "use" => {
                        if children.len() != 2 {
                            return err(*l, *c, "(use \"<path>\") takes exactly one string argument");
                        }
                        let path = match &children[1] {
                            SExpr::String(s, _, _) | SExpr::Atom(s, _, _) => s.clone(),
                            _ => return err(children[1].line(), children[1].col(), "import path must be a string"),
                        };
                        prog.uses.push(Use { path, line: *l });
                    }
                    "action" | "check" | "filter" | "when" | "every" => {
                        let parsed_item = lower_item(tag, children, *l)?;
                        prog.items.push(parsed_item);
                    }
                    unknown => {
                        return err(*l, *c, format!("unknown top-level declaration tag: '{}'", unknown));
                    }
                }
            }
            _ => return err(item.line(), item.col(), "top-level items inside '(kaze ...)' must be lists"),
        }
    }

    if !format_verified {
        // Format 1 default
    }

    Ok(prog)
}

fn lower_item(tag: &str, children: &[SExpr], line: u32) -> Result<Item, ParseError> {
    match tag {
        "action" | "check" | "filter" => {
            if children.len() < 2 {
                return err(line, 1, format!("'({})' requires a name", tag));
            }
            let name = match &children[1] {
                SExpr::Atom(s, _, _) | SExpr::String(s, _, _) => s.clone(),
                _ => return err(line, 1, format!("name of '{}' must be an identifier", tag)),
            };

            let kind = match tag {
                "action" => ItemKind::Action,
                "check" => ItemKind::Check,
                "filter" => ItemKind::Filter,
                _ => unreachable!(),
            };

            let mut params = Vec::new();
            let mut body = Vec::new();

            for child in &children[2..] {
                match child {
                    SExpr::List(form, fl, fc) if !form.is_empty() => {
                        let subtag = match &form[0] {
                            SExpr::Atom(t, _, _) => t.as_str(),
                            _ => return err(*fl, *fc, "expected tag symbol"),
                        };
                        match subtag {
                            "params" => {
                                for p in &form[1..] {
                                    match p {
                                        SExpr::Atom(s, _, _) | SExpr::String(s, _, _) => params.push(s.clone()),
                                        _ => return err(p.line(), p.col(), "parameter name must be an identifier"),
                                    }
                                }
                            }
                            "body" => {
                                for stmt_sexpr in &form[1..] {
                                    body.push(lower_stmt(stmt_sexpr)?);
                                }
                            }
                            unknown => return err(*fl, *fc, format!("unexpected form '{}' in item '{}'", unknown, name)),
                        }
                    }
                    _ => return err(child.line(), child.col(), "item forms must be lists, e.g. '(params ...)' or '(body ...)'"),
                }
            }

            Ok(Item {
                name,
                kind,
                params,
                body,
                line,
                origin: String::new(),
            })
        }
        "when" => {
            if children.len() < 3 {
                return err(line, 1, "'(when <cond> (body ...))' requires a condition and a body");
            }
            let cond = lower_expr(&children[1])?;
            let mut body = Vec::new();
            for child in &children[2..] {
                match child {
                    SExpr::List(form, fl, fc) if !form.is_empty() => {
                        let subtag = match &form[0] {
                            SExpr::Atom(t, _, _) => t.as_str(),
                            _ => return err(*fl, *fc, "expected tag symbol"),
                        };
                        if subtag == "body" {
                            for stmt_sexpr in &form[1..] {
                                body.push(lower_stmt(stmt_sexpr)?);
                            }
                        } else {
                            return err(*fl, *fc, format!("unexpected form '{}' in when hook", subtag));
                        }
                    }
                    _ => return err(child.line(), child.col(), "expected '(body ...)' in when hook"),
                }
            }
            Ok(Item {
                name: format!("when_{}", line),
                kind: ItemKind::When(cond),
                params: Vec::new(),
                body,
                line,
                origin: String::new(),
            })
        }
        "every" => {
            if children.len() < 3 {
                return err(line, 1, "'(every <period> (body ...))' requires a period and a body");
            }
            let period = match &children[1] {
                SExpr::Atom(s, l, c) => s.parse::<f64>().map_err(|_| ParseError {
                    line: *l,
                    col: *c,
                    msg: "period in 'every' must be a numeric literal".into(),
                    code: "E_KAZE_PARSE",
                })?,
                SExpr::String(s, l, c) => s.parse::<f64>().map_err(|_| ParseError {
                    line: *l,
                    col: *c,
                    msg: "period in 'every' must be a numeric literal".into(),
                    code: "E_KAZE_PARSE",
                })?,
                _ => return err(children[1].line(), children[1].col(), "period in 'every' must be a number"),
            };
            let mut body = Vec::new();
            for child in &children[2..] {
                match child {
                    SExpr::List(form, fl, fc) if !form.is_empty() => {
                        let subtag = match &form[0] {
                            SExpr::Atom(t, _, _) => t.as_str(),
                            _ => return err(*fl, *fc, "expected tag symbol"),
                        };
                        if subtag == "body" {
                            for stmt_sexpr in &form[1..] {
                                body.push(lower_stmt(stmt_sexpr)?);
                            }
                        } else {
                            return err(*fl, *fc, format!("unexpected form '{}' in every hook", subtag));
                        }
                    }
                    _ => return err(child.line(), child.col(), "expected '(body ...)' in every hook"),
                }
            }
            Ok(Item {
                name: format!("every_{}", line),
                kind: ItemKind::Every(period),
                params: Vec::new(),
                body,
                line,
                origin: String::new(),
            })
        }
        unknown => err(line, 1, format!("unknown item kind '{}'", unknown)),
    }
}

fn lower_stmt(sexpr: &SExpr) -> Result<Stmt, ParseError> {
    let line = sexpr.line();
    let elements = match sexpr {
        SExpr::List(e, _, _) if !e.is_empty() => e,
        _ => return err(line, sexpr.col(), "statements must be non-empty lists"),
    };

    let tag = match &elements[0] {
        SExpr::Atom(t, _, _) => t.as_str(),
        _ => return err(line, sexpr.col(), "statement head must be an operator tag"),
    };

    match tag {
        "let" => {
            if elements.len() != 3 {
                return err(line, sexpr.col(), "(let \"<var>\" <expr>) takes exactly two arguments");
            }
            let var = match &elements[1] {
                SExpr::Atom(s, _, _) | SExpr::String(s, _, _) => s.clone(),
                _ => return err(line, elements[1].col(), "variable name in 'let' must be an identifier"),
            };
            let expr = lower_expr(&elements[2])?;
            Ok(Stmt::Let(var, expr, line))
        }
        "set" => {
            if elements.len() != 3 {
                return err(line, sexpr.col(), "(set \"<var>\" <expr>) takes exactly two arguments");
            }
            let var = match &elements[1] {
                SExpr::Atom(s, _, _) | SExpr::String(s, _, _) => s.clone(),
                _ => return err(line, elements[1].col(), "variable name in 'set' must be an identifier"),
            };
            let expr = lower_expr(&elements[2])?;
            Ok(Stmt::Set(var, expr, line))
        }
        "eval" => {
            if elements.len() != 2 {
                return err(line, sexpr.col(), "(eval <expr>) takes exactly one expression");
            }
            let expr = lower_expr(&elements[1])?;
            Ok(Stmt::Eval(expr, line))
        }
        "return" => {
            if elements.len() == 1 {
                Ok(Stmt::Return(None, line))
            } else if elements.len() == 2 {
                let expr = lower_expr(&elements[1])?;
                Ok(Stmt::Return(Some(expr), line))
            } else {
                err(line, sexpr.col(), "(return [<expr>]) takes at most one argument")
            }
        }
        "if" => {
            if elements.len() < 3 {
                return err(line, sexpr.col(), "(if <cond> (then ...) [(otherwise ...)]) requires condition and then block");
            }
            let cond = lower_expr(&elements[1])?;
            let mut then_stmts = Vec::new();
            let mut otherwise_stmts = Vec::new();

            for block in &elements[2..] {
                match block {
                    SExpr::List(b, bl, bc) if !b.is_empty() => {
                        let btag = match &b[0] {
                            SExpr::Atom(t, _, _) => t.as_str(),
                            _ => return err(*bl, *bc, "expected block tag 'then' or 'otherwise'"),
                        };
                        match btag {
                            "then" => {
                                for s in &b[1..] {
                                    then_stmts.push(lower_stmt(s)?);
                                }
                            }
                            "otherwise" | "else" => {
                                for s in &b[1..] {
                                    otherwise_stmts.push(lower_stmt(s)?);
                                }
                            }
                            unknown => return err(*bl, *bc, format!("unknown block in 'if': '{}'", unknown)),
                        }
                    }
                    _ => return err(block.line(), block.col(), "expected '(then ...)' or '(otherwise ...)' in 'if'"),
                }
            }
            Ok(Stmt::If {
                cond,
                then: then_stmts,
                otherwise: otherwise_stmts,
                line,
            })
        }
        "while" => {
            if elements.len() < 3 {
                return err(line, sexpr.col(), "(while <cond> (body ...)) requires condition and body");
            }
            let cond = lower_expr(&elements[1])?;
            let mut body = Vec::new();
            for block in &elements[2..] {
                match block {
                    SExpr::List(b, bl, bc) if !b.is_empty() => {
                        let btag = match &b[0] {
                            SExpr::Atom(t, _, _) => t.as_str(),
                            _ => return err(*bl, *bc, "expected 'body' tag"),
                        };
                        if btag == "body" {
                            for s in &b[1..] {
                                body.push(lower_stmt(s)?);
                            }
                        } else {
                            return err(*bl, *bc, format!("unexpected block in while: '{}'", btag));
                        }
                    }
                    _ => return err(block.line(), block.col(), "expected '(body ...)' in 'while'"),
                }
            }
            Ok(Stmt::While { cond, body, line })
        }
        "for-each" | "for_each" | "for" => {
            if elements.len() < 4 {
                return err(line, sexpr.col(), "(for-each \"<var>\" <list> (body ...)) requires var, list, and body");
            }
            let var = match &elements[1] {
                SExpr::Atom(s, _, _) | SExpr::String(s, _, _) => s.clone(),
                _ => return err(line, elements[1].col(), "variable in 'for-each' must be an identifier"),
            };
            let list = lower_expr(&elements[2])?;
            let mut body = Vec::new();
            for block in &elements[3..] {
                match block {
                    SExpr::List(b, bl, bc) if !b.is_empty() => {
                        let btag = match &b[0] {
                            SExpr::Atom(t, _, _) => t.as_str(),
                            _ => return err(*bl, *bc, "expected 'body' tag"),
                        };
                        if btag == "body" {
                            for s in &b[1..] {
                                body.push(lower_stmt(s)?);
                            }
                        } else {
                            return err(*bl, *bc, format!("unexpected block in for-each: '{}'", btag));
                        }
                    }
                    _ => return err(block.line(), block.col(), "expected '(body ...)' in 'for-each'"),
                }
            }
            Ok(Stmt::ForEach {
                var,
                list,
                body,
                line,
            })
        }
        "try" => {
            let mut body = Vec::new();
            let mut otherwise = Vec::new();
            let mut error_name = None;

            for block in &elements[1..] {
                match block {
                    SExpr::List(b, bl, bc) if !b.is_empty() => {
                        let btag = match &b[0] {
                            SExpr::Atom(t, _, _) => t.as_str(),
                            _ => return err(*bl, *bc, "expected 'body' or 'otherwise' in 'try'"),
                        };
                        match btag {
                            "body" => {
                                for s in &b[1..] {
                                    body.push(lower_stmt(s)?);
                                }
                            }
                            "otherwise" => {
                                let mut start_idx = 1;
                                if b.len() > 1 {
                                    if let SExpr::List(with_form, _, _) = &b[1] {
                                        if !with_form.is_empty() {
                                            if let SExpr::Atom(w, _, _) = &with_form[0] {
                                                if w == "with" && with_form.len() == 2 {
                                                    match &with_form[1] {
                                                        SExpr::Atom(n, _, _) | SExpr::String(n, _, _) => {
                                                            error_name = Some(n.clone());
                                                            start_idx = 2;
                                                        }
                                                        _ => {}
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                                for s in &b[start_idx..] {
                                    otherwise.push(lower_stmt(s)?);
                                }
                            }
                            unknown => return err(*bl, *bc, format!("unexpected block in try: '{}'", unknown)),
                        }
                    }
                    _ => return err(block.line(), block.col(), "expected '(body ...)' or '(otherwise ...)' in 'try'"),
                }
            }

            Ok(Stmt::Try {
                body,
                otherwise,
                error_name,
                line,
            })
        }
        _ => {
            // General expression evaluated for effect in statement position
            let expr = lower_expr(sexpr)?;
            Ok(Stmt::Eval(expr, line))
        }
    }
}

fn parse_unit(u: &str) -> Unit {
    match u {
        "meters" | "meter" | "m" => Unit::Meter,
        "m/s" | "mps" => Unit::Mps,
        "degrees" | "degree" | "deg" => Unit::Deg,
        "seconds" | "second" | "s" | "ms" => Unit::Sec,
        "%" | "pct" | "percent" => Unit::Pct,
        _ => Unit::None,
    }
}

fn lower_expr(sexpr: &SExpr) -> Result<Expr, ParseError> {
    let line = sexpr.line();
    let col = sexpr.col();

    match sexpr {
        SExpr::String(s, _, _) => Ok(Expr::Str(s.clone())),
        SExpr::Atom(a, _, _) => {
            if a == "true" {
                return Ok(Expr::Bool(true));
            }
            if a == "false" {
                return Ok(Expr::Bool(false));
            }
            if let Ok(n) = a.parse::<f64>() {
                return Ok(Expr::Num(n, Unit::None));
            }
            Ok(Expr::Name(a.clone()))
        }
        SExpr::List(elements, _, _) => {
            if elements.is_empty() {
                return err(line, col, "empty expression list '()'");
            }
            let tag = match &elements[0] {
                SExpr::Atom(t, _, _) => t.as_str(),
                _ => return err(line, col, "head of expression must be a tag identifier"),
            };

            match tag {
                "number" | "num" => {
                    if elements.len() < 2 || elements.len() > 3 {
                        return err(line, col, "(number \"<value>\" [\"<unit>\"]) takes 1 or 2 arguments");
                    }
                    let val_str = match &elements[1] {
                        SExpr::Atom(s, _, _) | SExpr::String(s, _, _) => s.as_str(),
                        _ => return err(line, col, "numeric value must be a string or atom"),
                    };
                    let val = val_str.parse::<f64>().map_err(|_| ParseError {
                        line,
                        col,
                        msg: format!("invalid number '{}'", val_str),
                        code: "E_KAZE_PARSE",
                    })?;
                    let unit = if elements.len() == 3 {
                        let u_str = match &elements[2] {
                            SExpr::Atom(s, _, _) | SExpr::String(s, _, _) => s.as_str(),
                            _ => return err(line, col, "unit must be a string or atom"),
                        };
                        parse_unit(u_str)
                    } else {
                        Unit::None
                    };
                    Ok(Expr::Num(val, unit))
                }
                "string" | "str" => {
                    if elements.len() != 2 {
                        return err(line, col, "(string \"<value>\") takes exactly 1 argument");
                    }
                    let s = match &elements[1] {
                        SExpr::String(s, _, _) | SExpr::Atom(s, _, _) => s.clone(),
                        _ => return err(line, col, "string value must be a literal"),
                    };
                    Ok(Expr::Str(s))
                }
                "bool" => {
                    if elements.len() != 2 {
                        return err(line, col, "(bool true|false) takes exactly 1 argument");
                    }
                    let b = match &elements[1] {
                        SExpr::Atom(s, _, _) => match s.as_str() {
                            "true" => true,
                            "false" => false,
                            _ => return err(line, col, "boolean argument must be 'true' or 'false'"),
                        },
                        _ => return err(line, col, "boolean argument must be 'true' or 'false'"),
                    };
                    Ok(Expr::Bool(b))
                }
                "ref" | "var" => {
                    if elements.len() != 2 {
                        return err(line, col, "(ref \"<name>\") takes exactly 1 argument");
                    }
                    let name = match &elements[1] {
                        SExpr::Atom(s, _, _) | SExpr::String(s, _, _) => s.clone(),
                        _ => return err(line, col, "identifier in 'ref' must be an atom or string"),
                    };
                    Ok(Expr::Name(name))
                }
                "ambient" => {
                    if elements.len() != 3 {
                        return err(line, col, "(ambient \"<field>\" \"<family>\") takes 2 arguments");
                    }
                    let field = match &elements[1] {
                        SExpr::Atom(s, _, _) | SExpr::String(s, _, _) => s.clone(),
                        _ => return err(line, col, "ambient field must be a string"),
                    };
                    let family_str = match &elements[2] {
                        SExpr::Atom(s, _, _) | SExpr::String(s, _, _) => s.as_str(),
                        _ => return err(line, col, "ambient family must be a string"),
                    };
                    let family: &'static str = match family_str {
                        "drone" => "drone",
                        "rover" => "rover",
                        "sensor" => "sensor",
                        "device" => "device",
                        _ => "drone",
                    };
                    Ok(Expr::Ambient { field, family })
                }
                "call" => {
                    if elements.len() < 3 {
                        return err(line, col, "(call \"<recv>\" \"<op>\" <args>...) takes at least 2 arguments");
                    }
                    let recv = match &elements[1] {
                        SExpr::Atom(s, _, _) | SExpr::String(s, _, _) => s.clone(),
                        _ => return err(line, col, "call receiver must be an atom or string"),
                    };
                    let op = match &elements[2] {
                        SExpr::Atom(s, _, _) | SExpr::String(s, _, _) => s.clone(),
                        _ => return err(line, col, "call operation must be an atom or string"),
                    };
                    let mut args = Vec::new();
                    for arg in &elements[3..] {
                        args.push(lower_expr(arg)?);
                    }
                    Ok(Expr::Call { recv, op, args, line })
                }
                "field" => {
                    if elements.len() != 3 {
                        return err(line, col, "(field \"<base>\" \"<field>\") takes 2 arguments");
                    }
                    let base = match &elements[1] {
                        SExpr::Atom(s, _, _) | SExpr::String(s, _, _) => s.clone(),
                        _ => return err(line, col, "field base must be an identifier"),
                    };
                    let f = match &elements[2] {
                        SExpr::Atom(s, _, _) | SExpr::String(s, _, _) => s.clone(),
                        _ => return err(line, col, "field name must be an identifier"),
                    };
                    Ok(Expr::Field { base, field: f, line })
                }
                "allow" => {
                    let mut args = Vec::new();
                    for a in &elements[1..] {
                        args.push(lower_expr(a)?);
                    }
                    Ok(Expr::Call { recv: "policy".into(), op: "allow".into(), args, line })
                }
                "reject" => {
                    if elements.len() < 2 {
                        return err(line, col, "(reject <reason>) requires a reason argument");
                    }
                    let reason = lower_expr(&elements[1])?;
                    Ok(Expr::Call { recv: "policy".into(), op: "reject".into(), args: vec![reason], line })
                }
                // Binary operations
                "add" | "+" => bin_op(BinOp::Add, elements, line),
                "sub" | "-" => bin_op(BinOp::Sub, elements, line),
                "mul" | "*" => bin_op(BinOp::Mul, elements, line),
                "div" | "/" => bin_op(BinOp::Div, elements, line),
                "mod" | "%" => bin_op(BinOp::Mod, elements, line),
                "lt" | "<" => bin_op(BinOp::Lt, elements, line),
                "le" | "<=" => bin_op(BinOp::Le, elements, line),
                "gt" | ">" => bin_op(BinOp::Gt, elements, line),
                "ge" | ">=" => bin_op(BinOp::Ge, elements, line),
                "eq" | "==" => bin_op(BinOp::Eq, elements, line),
                "ne" | "!=" => bin_op(BinOp::Ne, elements, line),
                "and" => bin_op(BinOp::And, elements, line),
                "or" => bin_op(BinOp::Or, elements, line),
                "bin" => {
                    if elements.len() != 4 {
                        return err(line, col, "(bin \"<op>\" <left> <right>) takes 3 arguments");
                    }
                    let op_str = match &elements[1] {
                        SExpr::Atom(s, _, _) | SExpr::String(s, _, _) => s.as_str(),
                        _ => return err(line, col, "operator in 'bin' must be an atom or string"),
                    };
                    let op = match op_str {
                        "+" | "add" => BinOp::Add,
                        "-" | "sub" => BinOp::Sub,
                        "*" | "mul" => BinOp::Mul,
                        "/" | "div" => BinOp::Div,
                        "%" | "mod" => BinOp::Mod,
                        "<" | "lt" => BinOp::Lt,
                        "<=" | "le" => BinOp::Le,
                        ">" | "gt" => BinOp::Gt,
                        ">=" | "ge" => BinOp::Ge,
                        "==" | "eq" => BinOp::Eq,
                        "!=" | "ne" => BinOp::Ne,
                        "and" => BinOp::And,
                        "or" => BinOp::Or,
                        unknown => return err(line, col, format!("unsupported binary operator '{}'", unknown)),
                    };
                    let left = lower_expr(&elements[2])?;
                    let right = lower_expr(&elements[3])?;
                    Ok(Expr::Bin(op, Box::new(left), Box::new(right)))
                }
                "not" => {
                    if elements.len() != 2 {
                        return err(line, col, "(not <expr>) takes exactly 1 argument");
                    }
                    let inner = lower_expr(&elements[1])?;
                    Ok(Expr::Not(Box::new(inner)))
                }
                "len" | "length" => {
                    if elements.len() != 2 {
                        return err(line, col, "(len <expr>) takes exactly 1 argument");
                    }
                    let inner = lower_expr(&elements[1])?;
                    Ok(Expr::Len(Box::new(inner)))
                }
                "index" => {
                    if elements.len() != 3 {
                        return err(line, col, "(index <list> <idx>) takes 2 arguments");
                    }
                    let base = lower_expr(&elements[1])?;
                    let idx = lower_expr(&elements[2])?;
                    Ok(Expr::Index(Box::new(base), Box::new(idx)))
                }
                "list" => {
                    let mut items = Vec::new();
                    for item in &elements[1..] {
                        items.push(lower_expr(item)?);
                    }
                    Ok(Expr::List(items))
                }
                // Spatial & Kinematics Math
                "vector" => {
                    let mut args = Vec::new();
                    for arg in &elements[1..] {
                        args.push(lower_expr(arg)?);
                    }
                    Ok(Expr::Math { kind: MathKind::Vector, args, line })
                }
                "distance" => {
                    if elements.len() != 3 {
                        return err(line, col, "(distance <a> <b>) takes 2 arguments");
                    }
                    let a = lower_expr(&elements[1])?;
                    let b = lower_expr(&elements[2])?;
                    Ok(Expr::Math { kind: MathKind::Distance, args: vec![a, b], line })
                }
                "slant-distance" | "slant_distance" => {
                    if elements.len() != 3 {
                        return err(line, col, "(slant-distance <a> <b>) takes 2 arguments");
                    }
                    let a = lower_expr(&elements[1])?;
                    let b = lower_expr(&elements[2])?;
                    Ok(Expr::Math { kind: MathKind::SlantDistance, args: vec![a, b], line })
                }
                "bearing-of" | "bearing_of" => {
                    if elements.len() != 2 {
                        return err(line, col, "(bearing-of <v>) takes 1 argument");
                    }
                    let v = lower_expr(&elements[1])?;
                    Ok(Expr::Math { kind: MathKind::BearingOf, args: vec![v], line })
                }
                "bearing-from" | "bearing_from" => {
                    if elements.len() != 3 {
                        return err(line, col, "(bearing-from <a> <b>) takes 2 arguments");
                    }
                    let a = lower_expr(&elements[1])?;
                    let b = lower_expr(&elements[2])?;
                    Ok(Expr::Math { kind: MathKind::BearingFrom, args: vec![a, b], line })
                }
                "sin" => {
                    if elements.len() != 2 {
                        return err(line, col, "(sin <x>) takes 1 argument");
                    }
                    let x = lower_expr(&elements[1])?;
                    Ok(Expr::Math { kind: MathKind::Sin, args: vec![x], line })
                }
                "cos" => {
                    if elements.len() != 2 {
                        return err(line, col, "(cos <x>) takes 1 argument");
                    }
                    let x = lower_expr(&elements[1])?;
                    Ok(Expr::Math { kind: MathKind::Cos, args: vec![x], line })
                }
                "atan2" => {
                    if elements.len() != 3 {
                        return err(line, col, "(atan2 <y> <x>) takes 2 arguments");
                    }
                    let y = lower_expr(&elements[1])?;
                    let x = lower_expr(&elements[2])?;
                    Ok(Expr::Math { kind: MathKind::Atan2, args: vec![y, x], line })
                }
                "sqrt" => {
                    if elements.len() != 2 {
                        return err(line, col, "(sqrt <x>) takes 1 argument");
                    }
                    let x = lower_expr(&elements[1])?;
                    Ok(Expr::Math { kind: MathKind::Sqrt, args: vec![x], line })
                }
                "to-radians" | "to_radians" => {
                    if elements.len() != 2 {
                        return err(line, col, "(to-radians <x>) takes 1 argument");
                    }
                    let x = lower_expr(&elements[1])?;
                    Ok(Expr::Math { kind: MathKind::ToRadians, args: vec![x], line })
                }
                "to-degrees" | "to_degrees" => {
                    if elements.len() != 2 {
                        return err(line, col, "(to-degrees <x>) takes 1 argument");
                    }
                    let x = lower_expr(&elements[1])?;
                    Ok(Expr::Math { kind: MathKind::ToDegrees, args: vec![x], line })
                }
                "link-lost" | "link_lost" => {
                    if elements.len() != 2 {
                        return err(line, col, "(link-lost <ms>) takes 1 argument");
                    }
                    let ms = match &elements[1] {
                        SExpr::Atom(s, _, _) | SExpr::String(s, _, _) => s.parse::<f64>().unwrap_or(0.0),
                        SExpr::List(l, _, _) if l.len() >= 2 => {
                            match &l[1] {
                                SExpr::Atom(s, _, _) | SExpr::String(s, _, _) => {
                                    let n = s.parse::<f64>().unwrap_or(0.0);
                                    if l.len() >= 3 {
                                        let u = match &l[2] {
                                            SExpr::Atom(u, _, _) | SExpr::String(u, _, _) => u.as_str(),
                                            _ => "",
                                        };
                                        if u == "seconds" || u == "second" || u == "s" {
                                            n * 1000.0
                                        } else {
                                            n
                                        }
                                    } else {
                                        n
                                    }
                                }
                                _ => 0.0,
                            }
                        }
                        _ => 0.0,
                    };
                    Ok(Expr::LinkLost { after_ms: ms, line })
                }
                unknown => err(line, col, format!("unknown expression operator or tag '{}'", unknown)),
            }
        }
    }
}

fn bin_op(op: BinOp, elements: &[SExpr], line: u32) -> Result<Expr, ParseError> {
    if elements.len() != 3 {
        return err(line, elements[0].col(), format!("binary operator '({:?})' takes 2 arguments", op));
    }
    let left = lower_expr(&elements[1])?;
    let right = lower_expr(&elements[2])?;
    Ok(Expr::Bin(op, Box::new(left), Box::new(right)))
}

// ============================================================================
// Emitter: AST Program -> Machine Syntax (Prefix-tree text)
// ============================================================================

fn push_indent(out: &mut String, depth: usize) {
    for _ in 0..depth {
        out.push_str("  ");
    }
}

fn push_indent4(out: &mut String, depth: usize) {
    for _ in 0..depth {
        out.push_str("    ");
    }
}

fn push_num(out: &mut String, n: f64) {
    use std::fmt::Write;
    let _ = write!(out, "{}", n);
}

fn escape_str_to(s: &str, out: &mut String) {
    for ch in s.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            '\r' => out.push_str("\\r"),
            c => out.push(c),
        }
    }
}

#[allow(dead_code)]
fn escape_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    escape_str_to(s, &mut out);
    out
}

/// Pretty-print an AST Program into canonical Machine Syntax prefix-tree text.
pub fn emit_machine(prog: &Program) -> String {
    let mut out = String::with_capacity(512);
    out.push_str("(kaze\n  (format 1)\n");

    if !prog.needs.is_empty() {
        out.push_str("  (needs");
        for cap in &prog.needs {
            out.push_str(" \"");
            escape_str_to(cap, &mut out);
            out.push('"');
        }
        out.push_str(")\n");
    }

    for u in &prog.uses {
        out.push_str("  (use \"");
        escape_str_to(&u.path, &mut out);
        out.push_str("\")\n");
    }

    for item in &prog.items {
        emit_item_machine(item, &mut out, 1);
    }

    out.push_str(")\n");
    out
}

fn emit_item_machine(item: &Item, out: &mut String, depth: usize) {
    push_indent(out, depth);
    match &item.kind {
        ItemKind::When(cond) => {
            out.push_str("(when ");
            emit_expr_machine(cond, out);
            out.push('\n');
            push_indent(out, depth + 1);
            out.push_str("(body\n");
            for stmt in &item.body {
                emit_stmt_machine(stmt, out, depth + 2);
            }
            push_indent(out, depth + 1);
            out.push_str("))\n");
            return;
        }
        ItemKind::Every(secs) => {
            out.push_str("(every ");
            push_num(out, *secs);
            out.push('\n');
            push_indent(out, depth + 1);
            out.push_str("(body\n");
            for stmt in &item.body {
                emit_stmt_machine(stmt, out, depth + 2);
            }
            push_indent(out, depth + 1);
            out.push_str("))\n");
            return;
        }
        ItemKind::Action => out.push_str("(action \""),
        ItemKind::Check => out.push_str("(check \""),
        ItemKind::Filter => out.push_str("(filter \""),
    }

    escape_str_to(&item.name, out);
    out.push_str("\"\n");

    if !item.params.is_empty() {
        push_indent(out, depth + 1);
        out.push_str("(params");
        for p in &item.params {
            out.push_str(" \"");
            escape_str_to(p, out);
            out.push('"');
        }
        out.push_str(")\n");
    }

    push_indent(out, depth + 1);
    out.push_str("(body\n");
    for stmt in &item.body {
        emit_stmt_machine(stmt, out, depth + 2);
    }
    push_indent(out, depth + 1);
    out.push_str("))\n");
}

fn emit_stmt_machine(stmt: &Stmt, out: &mut String, depth: usize) {
    push_indent(out, depth);
    match stmt {
        Stmt::Let(var, expr, _) => {
            out.push_str("(let \"");
            escape_str_to(var, out);
            out.push_str("\" ");
            emit_expr_machine(expr, out);
            out.push_str(")\n");
        }
        Stmt::Set(var, expr, _) => {
            out.push_str("(set \"");
            escape_str_to(var, out);
            out.push_str("\" ");
            emit_expr_machine(expr, out);
            out.push_str(")\n");
        }
        Stmt::Eval(expr, _) => {
            out.push_str("(eval ");
            emit_expr_machine(expr, out);
            out.push_str(")\n");
        }
        Stmt::Return(None, _) => {
            out.push_str("(return)\n");
        }
        Stmt::Return(Some(expr), _) => {
            out.push_str("(return ");
            emit_expr_machine(expr, out);
            out.push_str(")\n");
        }
        Stmt::If { cond, then, otherwise, .. } => {
            out.push_str("(if ");
            emit_expr_machine(cond, out);
            out.push('\n');
            push_indent(out, depth + 1);
            out.push_str("(then\n");
            for s in then {
                emit_stmt_machine(s, out, depth + 2);
            }
            push_indent(out, depth + 1);
            out.push(')');
            if !otherwise.is_empty() {
                out.push('\n');
                push_indent(out, depth + 1);
                out.push_str("(otherwise\n");
                for s in otherwise {
                    emit_stmt_machine(s, out, depth + 2);
                }
                push_indent(out, depth + 1);
                out.push(')');
            }
            out.push_str(")\n");
        }
        Stmt::While { cond, body, .. } => {
            out.push_str("(while ");
            emit_expr_machine(cond, out);
            out.push('\n');
            push_indent(out, depth + 1);
            out.push_str("(body\n");
            for s in body {
                emit_stmt_machine(s, out, depth + 2);
            }
            push_indent(out, depth + 1);
            out.push_str("))\n");
        }
        Stmt::ForEach { var, list, body, .. } => {
            out.push_str("(for-each \"");
            escape_str_to(var, out);
            out.push_str("\" ");
            emit_expr_machine(list, out);
            out.push('\n');
            push_indent(out, depth + 1);
            out.push_str("(body\n");
            for s in body {
                emit_stmt_machine(s, out, depth + 2);
            }
            push_indent(out, depth + 1);
            out.push_str("))\n");
        }
        Stmt::Try { body, otherwise, error_name, .. } => {
            out.push_str("(try\n");
            push_indent(out, depth + 1);
            out.push_str("(body\n");
            for s in body {
                emit_stmt_machine(s, out, depth + 2);
            }
            push_indent(out, depth + 1);
            out.push_str(")\n");
            push_indent(out, depth + 1);
            if let Some(err_var) = error_name {
                out.push_str("(otherwise (with \"");
                escape_str_to(err_var, out);
                out.push_str("\")\n");
            } else {
                out.push_str("(otherwise\n");
            }
            for s in otherwise {
                emit_stmt_machine(s, out, depth + 2);
            }
            push_indent(out, depth + 1);
            out.push_str("))\n");
        }
    }
}

fn emit_expr_machine(expr: &Expr, out: &mut String) {
    match expr {
        Expr::Num(n, unit) => {
            out.push_str("(number \"");
            push_num(out, *n);
            if *unit != Unit::None {
                out.push_str("\" \"");
                out.push_str(unit.label());
            }
            out.push_str("\")");
        }
        Expr::Str(s) => {
            out.push_str("(string \"");
            escape_str_to(s, out);
            out.push_str("\")");
        }
        Expr::Bool(b) => {
            out.push_str(if *b { "(bool true)" } else { "(bool false)" });
        }
        Expr::Name(name) => {
            out.push_str("(ref \"");
            escape_str_to(name, out);
            out.push_str("\")");
        }
        Expr::Ambient { field, family } => {
            out.push_str("(ambient \"");
            escape_str_to(field, out);
            out.push_str("\" \"");
            escape_str_to(family, out);
            out.push_str("\")");
        }
        Expr::Call { recv, op, args, .. } => {
            out.push_str("(call \"");
            escape_str_to(recv, out);
            out.push_str("\" \"");
            escape_str_to(op, out);
            out.push('"');
            for a in args {
                out.push(' ');
                emit_expr_machine(a, out);
            }
            out.push(')');
        }
        Expr::Field { base, field, .. } => {
            out.push_str("(field \"");
            escape_str_to(base, out);
            out.push_str("\" \"");
            escape_str_to(field, out);
            out.push_str("\")");
        }
        Expr::Bin(op, left, right) => {
            out.push('(');
            let op_str = match op {
                BinOp::Add => "add",
                BinOp::Sub => "sub",
                BinOp::Mul => "mul",
                BinOp::Div => "div",
                BinOp::Mod => "mod",
                BinOp::Lt => "lt",
                BinOp::Le => "le",
                BinOp::Gt => "gt",
                BinOp::Ge => "ge",
                BinOp::Eq => "eq",
                BinOp::Ne => "ne",
                BinOp::And => "and",
                BinOp::Or => "or",
            };
            out.push_str(op_str);
            out.push(' ');
            emit_expr_machine(left, out);
            out.push(' ');
            emit_expr_machine(right, out);
            out.push(')');
        }
        Expr::Not(inner) => {
            out.push_str("(not ");
            emit_expr_machine(inner, out);
            out.push(')');
        }
        Expr::Len(inner) => {
            out.push_str("(len ");
            emit_expr_machine(inner, out);
            out.push(')');
        }
        Expr::Index(base, idx) => {
            out.push_str("(index ");
            emit_expr_machine(base, out);
            out.push(' ');
            emit_expr_machine(idx, out);
            out.push(')');
        }
        Expr::List(items) => {
            out.push_str("(list");
            for item in items {
                out.push(' ');
                emit_expr_machine(item, out);
            }
            out.push(')');
        }
        Expr::Math { kind, args, .. } => {
            out.push('(');
            let k_str = match kind {
                MathKind::Vector => "vector",
                MathKind::Distance => "distance",
                MathKind::BearingOf => "bearing-of",
                MathKind::BearingFrom => "bearing-from",
                MathKind::SlantDistance => "slant-distance",
                MathKind::Sin => "sin",
                MathKind::Cos => "cos",
                MathKind::Atan2 => "atan2",
                MathKind::Sqrt => "sqrt",
                MathKind::ToRadians => "to-radians",
                MathKind::ToDegrees => "to-degrees",
            };
            out.push_str(k_str);
            for a in args {
                out.push(' ');
                emit_expr_machine(a, out);
            }
            out.push(')');
        }
        Expr::LinkLost { after_ms, .. } => {
            out.push_str("(link-lost ");
            push_num(out, *after_ms);
            out.push(')');
        }
    }
}

// ============================================================================
// Emitter: AST Program -> Canonical Human Syntax (Indentation-based)
// ============================================================================

/// Pretty-print an AST Program into canonical Human Syntax text.
pub fn emit_human(prog: &Program) -> String {
    let mut out = String::with_capacity(512);

    if !prog.needs.is_empty() {
        out.push_str("needs: ");
        for (i, cap) in prog.needs.iter().enumerate() {
            if i > 0 {
                out.push_str(", ");
            }
            out.push_str(cap);
        }
        out.push_str("\n\n");
    }

    for u in &prog.uses {
        out.push_str("use \"");
        escape_str_to(&u.path, &mut out);
        out.push_str("\"\n");
    }
    if !prog.uses.is_empty() {
        out.push('\n');
    }

    for (idx, item) in prog.items.iter().enumerate() {
        if idx > 0 {
            out.push('\n');
        }
        emit_item_human(item, &mut out);
    }

    out
}

fn emit_item_human(item: &Item, out: &mut String) {
    match &item.kind {
        ItemKind::Action => {
            out.push_str("action ");
            out.push_str(&item.name);
            out.push('(');
            for (i, p) in item.params.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(p);
            }
            out.push_str("):\n");
        }
        ItemKind::Check => {
            out.push_str("check ");
            out.push_str(&item.name);
            out.push('(');
            for (i, p) in item.params.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(p);
            }
            out.push_str("):\n");
        }
        ItemKind::Filter => {
            out.push_str("filter ");
            out.push_str(&item.name);
            out.push('(');
            for (i, p) in item.params.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(p);
            }
            out.push_str("):\n");
        }
        ItemKind::When(cond) => {
            out.push_str("when ");
            emit_expr_human(cond, out);
            out.push_str(":\n");
        }
        ItemKind::Every(secs) => {
            out.push_str("every ");
            push_num(out, *secs);
            out.push_str(" seconds:\n");
        }
    }

    if item.body.is_empty() {
        out.push_str("    return\n");
    } else {
        for stmt in &item.body {
            emit_stmt_human(stmt, out, 1);
        }
    }
}

fn emit_stmt_human(stmt: &Stmt, out: &mut String, depth: usize) {
    push_indent4(out, depth);
    match stmt {
        Stmt::Let(var, expr, _) => {
            out.push_str("let ");
            out.push_str(var);
            out.push_str(" = ");
            emit_expr_human(expr, out);
            out.push('\n');
        }
        Stmt::Set(var, expr, _) => {
            out.push_str("set ");
            out.push_str(var);
            out.push_str(" = ");
            emit_expr_human(expr, out);
            out.push('\n');
        }
        Stmt::Eval(expr, _) => {
            emit_expr_human(expr, out);
            out.push('\n');
        }
        Stmt::Return(None, _) => {
            out.push_str("return\n");
        }
        Stmt::Return(Some(expr), _) => {
            out.push_str("return ");
            emit_expr_human(expr, out);
            out.push('\n');
        }
        Stmt::If { cond, then, otherwise, .. } => {
            out.push_str("if ");
            emit_expr_human(cond, out);
            out.push_str(":\n");
            for s in then {
                emit_stmt_human(s, out, depth + 1);
            }
            if !otherwise.is_empty() {
                push_indent4(out, depth);
                out.push_str("otherwise:\n");
                for s in otherwise {
                    emit_stmt_human(s, out, depth + 1);
                }
            }
        }
        Stmt::While { cond, body, .. } => {
            out.push_str("while ");
            emit_expr_human(cond, out);
            out.push_str(":\n");
            for s in body {
                emit_stmt_human(s, out, depth + 1);
            }
        }
        Stmt::ForEach { var, list, body, .. } => {
            out.push_str("for ");
            out.push_str(var);
            out.push_str(" in ");
            emit_expr_human(list, out);
            out.push_str(":\n");
            for s in body {
                emit_stmt_human(s, out, depth + 1);
            }
        }
        Stmt::Try { body, otherwise, error_name, .. } => {
            out.push_str("try:\n");
            for s in body {
                emit_stmt_human(s, out, depth + 1);
            }
            push_indent4(out, depth);
            if let Some(err_name) = error_name {
                out.push_str("otherwise with ");
                out.push_str(err_name);
                out.push_str(":\n");
            } else {
                out.push_str("otherwise:\n");
            }
            for s in otherwise {
                emit_stmt_human(s, out, depth + 1);
            }
        }
    }
}

fn emit_expr_human(expr: &Expr, out: &mut String) {
    match expr {
        Expr::Num(n, unit) => {
            push_num(out, *n);
            match unit {
                Unit::None => {}
                Unit::Meter => out.push_str(" meters"),
                Unit::Mps => out.push_str(" m/s"),
                Unit::Deg => out.push_str(" degrees"),
                Unit::Sec => out.push_str(" seconds"),
                Unit::Pct => out.push('%'),
            }
        }
        Expr::Str(s) => {
            out.push('"');
            escape_str_to(s, out);
            out.push('"');
        }
        Expr::Bool(b) => {
            out.push_str(if *b { "true" } else { "false" });
        }
        Expr::Name(name) => out.push_str(name),
        Expr::Ambient { field, .. } => out.push_str(field),
        Expr::Call { recv, op, args, .. } => {
            if recv == "policy" && op == "allow" {
                if args.is_empty() {
                    out.push_str("allow");
                } else {
                    out.push_str("allow ");
                    emit_expr_human(&args[0], out);
                }
                return;
            }
            if recv == "policy" && op == "reject" && !args.is_empty() {
                out.push_str("reject with ");
                emit_expr_human(&args[0], out);
                return;
            }
            out.push_str(recv);
            out.push('.');
            out.push_str(op);
            out.push('(');
            for (i, a) in args.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                emit_expr_human(a, out);
            }
            out.push(')');
        }
        Expr::Field { base, field, .. } => {
            out.push_str(base);
            out.push('.');
            out.push_str(field);
        }
        Expr::Bin(op, left, right) => {
            out.push('(');
            emit_expr_human(left, out);
            out.push(' ');
            out.push_str(op.symbol());
            out.push(' ');
            emit_expr_human(right, out);
            out.push(')');
        }
        Expr::Not(inner) => {
            out.push_str("not (");
            emit_expr_human(inner, out);
            out.push(')');
        }
        Expr::Len(inner) => {
            out.push_str("length of ");
            emit_expr_human(inner, out);
        }
        Expr::Index(base, idx) => {
            emit_expr_human(base, out);
            out.push('[');
            emit_expr_human(idx, out);
            out.push(']');
        }
        Expr::List(items) => {
            out.push('[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                emit_expr_human(item, out);
            }
            out.push(']');
        }
        Expr::Math { kind, args, .. } => {
            match kind {
                MathKind::Vector => {
                    out.push_str("vector(");
                    for (i, a) in args.iter().enumerate() {
                        if i > 0 {
                            out.push_str(", ");
                        }
                        emit_expr_human(a, out);
                    }
                    out.push(')');
                }
                MathKind::Distance => {
                    out.push_str("distance between ");
                    emit_expr_human(&args[0], out);
                    out.push_str(" and ");
                    emit_expr_human(&args[1], out);
                }
                MathKind::SlantDistance => {
                    out.push_str("slant distance between ");
                    emit_expr_human(&args[0], out);
                    out.push_str(" and ");
                    emit_expr_human(&args[1], out);
                }
                MathKind::BearingOf => {
                    out.push_str("bearing of ");
                    emit_expr_human(&args[0], out);
                }
                MathKind::BearingFrom => {
                    out.push_str("bearing from ");
                    emit_expr_human(&args[0], out);
                    out.push_str(" to ");
                    emit_expr_human(&args[1], out);
                }
                MathKind::Sin => {
                    out.push_str("sin(");
                    emit_expr_human(&args[0], out);
                    out.push(')');
                }
                MathKind::Cos => {
                    out.push_str("cos(");
                    emit_expr_human(&args[0], out);
                    out.push(')');
                }
                MathKind::Atan2 => {
                    out.push_str("atan2(");
                    emit_expr_human(&args[0], out);
                    out.push_str(", ");
                    emit_expr_human(&args[1], out);
                    out.push(')');
                }
                MathKind::Sqrt => {
                    out.push_str("sqrt(");
                    emit_expr_human(&args[0], out);
                    out.push(')');
                }
                MathKind::ToRadians => {
                    out.push('(');
                    emit_expr_human(&args[0], out);
                    out.push_str(" in radians)");
                }
                MathKind::ToDegrees => {
                    out.push('(');
                    emit_expr_human(&args[0], out);
                    out.push_str(" in degrees)");
                }
            }
        }
        Expr::LinkLost { after_ms, .. } => {
            out.push_str("link lost for ");
            push_num(out, *after_ms);
            out.push_str(" ms");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_machine_syntax() {
        assert!(is_machine_syntax("(kaze (format 1))"));
        assert!(is_machine_syntax("  ; comment\n  (kaze\n    (format 1))"));
        assert!(is_machine_syntax("# comment\n(kaze (format 1))"));
        assert!(!is_machine_syntax("action main():\n    return 1"));
        assert!(!is_machine_syntax("needs: drone, log\n\naction main():\n    return 42"));
    }

    #[test]
    fn parses_simple_machine_program() {
        let src = r#"
        (kaze
          (format 1)
          (needs "drone" "log")
          (action "main"
            (params "amount")
            (body
              (return (add (ref "amount") (number "1"))))))
        "#;
        let prog = parse_machine(src).expect("parse machine syntax");
        assert_eq!(prog.needs, vec!["drone", "log"]);
        assert_eq!(prog.items.len(), 1);
        assert_eq!(prog.items[0].name, "main");
        assert_eq!(prog.items[0].params, vec!["amount"]);
    }

    #[test]
    fn bidirectional_roundtrip() {
        let machine_src = r#"(kaze
  (format 1)
  (needs "drone" "log")
  (action "patrol"
    (params "target_alt")
    (body
      (eval (call "drone" "arm"))
      (eval (call "drone" "takeoff" (number "10" "meters")))
      (let "loc" (vector (number "47.37") (number "8.54") (ref "target_alt")))
      (let "d" (distance (ref "loc") (vector (number "47.38") (number "8.55") (number "100"))))
      (if (gt (ref "d") (number "50"))
        (then
          (eval (call "log" "info" (string "target reached")))
        )
        (otherwise
          (return (ref "d"))
        ))
      (try
        (body
          (eval (call "drone" "land"))
        )
        (otherwise (with "err")
          (eval (call "log" "info" (field "err" "message")))
        ))
      (return (ref "loc"))
    ))
)
"#;
        let prog = parse_machine(machine_src).expect("parse machine");
        let human = emit_human(&prog);
        assert!(human.contains("action patrol(target_alt):"));
        assert!(human.contains("drone.arm()"));
        assert!(human.contains("vector(47.37, 8.54, target_alt)"));
        assert!(human.contains("try:"));
        assert!(human.contains("otherwise with err:"));

        let emitted_machine = emit_machine(&prog);
        assert!(emitted_machine.contains("(action \"patrol\""));
        assert!(emitted_machine.contains("(call \"drone\" \"arm\")"));
        assert!(emitted_machine.contains("(vector"));
    }

    #[test]
    fn rejects_unsupported_format_version() {
        let src = "(kaze (format 99) (action \"a\" (body (return))))";
        let err = parse_machine(src).unwrap_err();
        assert!(err.msg.contains("unsupported machine syntax format version"));
    }

    #[test]
    fn rejects_excessive_nesting() {
        let mut deep = "(kaze (format 1) (action \"a\" (body ".to_string();
        for _ in 0..300 {
            deep.push_str("(add (number \"1\") ");
        }
        deep.push_str("(number \"1\")");
        for _ in 0..300 {
            deep.push(')');
        }
        deep.push_str(")))");
        let err = parse_machine(&deep).unwrap_err();
        assert!(err.msg.contains("nesting depth"));
    }
}
