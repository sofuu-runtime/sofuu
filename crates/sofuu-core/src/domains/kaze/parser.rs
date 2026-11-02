// sfs/parser.rs — recursive-descent parser for the natural-phrasing DSL.
//
// PLAN-KAZE-LANGUAGE §3.2 / §3.3 / §5 (M-K1). The grammar is deliberately
// sentence-shaped (`takeoff to 15 meters`, `turn on fan`) while staying
// strictly unambiguous: every verb has one shape, and a shape that does not
// match produces a message naming the line and the two things it could have
// been. Natural phrasing is only worth it if the failure mode is legible.
//
// Scope resolution happens here (not in the checker) because the surface has
// three kinds of name that look identical: a binding (`let`, a parameter, a
// `for each` variable), a telemetry field (`battery`), and a capability
// receiver (`lidar.front_distance()`). Reading them at parse time keeps the
// AST explicit, so a later pass never has to guess.

use crate::sfs::ast::*;
use crate::sfs::caps;
use crate::sfs::lexer::{tokenize, LexError, Tok, Token};

#[derive(Debug, Clone)]
pub struct ParseError {
    pub line: u32,
    pub col: u32,
    pub msg: String,
}

impl ParseError {
    pub fn render(&self, source_name: &str) -> String {
        format!("{}:{}:{}: {}", source_name, self.line, self.col, self.msg)
    }

    fn from_lex(e: LexError) -> Self {
        ParseError { line: e.line, col: e.col, msg: e.msg }
    }
}

fn err<T>(line: u32, col: u32, msg: impl Into<String>) -> Result<T, ParseError> {
    Err(ParseError { line, col, msg: msg.into() })
}

/// Parse a whole script.
pub fn parse(src: &str) -> Result<Program, ParseError> {
    let toks = tokenize(src).map_err(ParseError::from_lex)?;
    let mut p = Parser {
        toks,
        pos: 0,
        scopes: Vec::new(),
        needs: Vec::new(),
        ambient: None,
        hook_seq: 0,
    };
    p.program()
}

struct Parser {
    toks: Vec<Token>,
    pos: usize,
    scopes: Vec<Vec<String>>,
    needs: Vec<String>,
    /// The script's single hardware family, when it declared exactly one.
    ambient: Option<&'static str>,
    hook_seq: u32,
}

impl Parser {
    // ── token plumbing ───────────────────────────────────────────────
    fn peek(&self) -> &Tok {
        &self.toks.get(self.pos).map(|t| &t.tok).unwrap_or(&Tok::Eof)
    }

    fn peek_at(&self, n: usize) -> &Tok {
        &self.toks.get(self.pos + n).map(|t| &t.tok).unwrap_or(&Tok::Eof)
    }

    fn here(&self) -> (u32, u32) {
        self.toks
            .get(self.pos)
            .map(|t| (t.line, t.col))
            .unwrap_or((0, 0))
    }

    fn bump(&mut self) -> Tok {
        let t = self.toks.get(self.pos).map(|t| t.tok.clone()).unwrap_or(Tok::Eof);
        if self.pos < self.toks.len() {
            self.pos += 1;
        }
        t
    }

    fn at_ident(&self, word: &str) -> bool {
        matches!(self.peek(), Tok::Ident(w) if w == word)
    }

    fn eat_word(&mut self, word: &str) -> bool {
        if self.at_ident(word) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn expect_word(&mut self, word: &str) -> Result<(), ParseError> {
        if self.eat_word(word) {
            Ok(())
        } else {
            let (l, c) = self.here();
            err(l, c, format!("expected '{}' here", word))
        }
    }

    fn eat_punct(&mut self, ch: char) -> bool {
        if matches!(self.peek(), Tok::Punct(c) if *c == ch) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn expect_punct(&mut self, ch: char) -> Result<(), ParseError> {
        if self.eat_punct(ch) {
            Ok(())
        } else {
            let (l, c) = self.here();
            err(l, c, format!("expected '{}' here", ch))
        }
    }

    fn expect_ident(&mut self) -> Result<String, ParseError> {
        match self.peek().clone() {
            Tok::Ident(w) => {
                self.bump();
                Ok(w)
            }
            other => {
                let (l, c) = self.here();
                err(l, c, format!("expected a name here, found {}", describe(&other)))
            }
        }
    }

    fn skip_newlines(&mut self) {
        while matches!(self.peek(), Tok::Newline) {
            self.bump();
        }
    }

    // ── scopes ───────────────────────────────────────────────────────
    fn push_scope(&mut self) {
        self.scopes.push(Vec::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn bind(&mut self, name: &str) {
        if let Some(top) = self.scopes.last_mut() {
            top.push(name.to_string());
        }
    }

    fn is_bound(&self, name: &str) -> bool {
        self.scopes.iter().any(|s| s.iter().any(|n| n == name))
    }

    // ── program ──────────────────────────────────────────────────────
    fn program(&mut self) -> Result<Program, ParseError> {
        self.skip_newlines();
        if self.at_ident("needs") {
            self.bump();
            self.expect_punct(':')?;
            loop {
                let name = self.expect_ident()?;
                if self.needs.iter().any(|n| *n == name) {
                    let (l, c) = self.here();
                    return err(l, c, format!("'{}' is declared twice in needs:", name));
                }
                self.needs.push(name);
                if !self.eat_punct(',') {
                    break;
                }
            }
            if !matches!(self.peek(), Tok::Newline | Tok::Eof) {
                let (l, c) = self.here();
                return err(l, c, "expected the needs line to end after the capability list");
            }
            self.skip_newlines();
        }

        // Exactly one hardware family means a bare `battery` is unambiguous.
        let hw: Vec<&'static str> = self
            .needs
            .iter()
            .map(|n| caps::family_of(n))
            .filter(|f| caps::hardware_families().contains(f))
            .collect();
        if hw.len() == 1 {
            self.ambient = Some(hw[0]);
        }

        let mut items: Vec<Item> = Vec::new();
        while !matches!(self.peek(), Tok::Eof) {
            let item = self.item()?;
            if items.iter().any(|i| i.name == item.name) {
                let (l, c) = self.here();
                return err(l, c, format!("'{}' is defined twice", item.name));
            }
            items.push(item);
            self.skip_newlines();
        }

        if items.is_empty() {
            return err(1, 1, "this script is empty: add an action, check, filter, when or every block");
        }
        Ok(Program { needs: self.needs.clone(), items })
    }

    fn item(&mut self) -> Result<Item, ParseError> {
        let (line, col) = self.here();
        if self.at_ident("when") {
            self.bump();
            let cond = self.expr()?;
            self.expect_punct(':')?;
            self.hook_seq += 1;
            let name = format!("when#{}", self.hook_seq);
            self.push_scope();
            let body = self.block()?;
            self.pop_scope();
            return Ok(Item { name, kind: ItemKind::When(cond), params: Vec::new(), body, line });
        }
        if self.at_ident("every") {
            self.bump();
            let (num, unit) = self.number_with_unit()?;
            let ms = match unit {
                Unit::Sec => num * 1000.0,
                _ => {
                    return err(
                        line,
                        col,
                        "every needs a time period, e.g. 'every 100ms' or 'every 2 seconds'",
                    )
                }
            };
            self.expect_punct(':')?;
            self.hook_seq += 1;
            let name = format!("every#{}", self.hook_seq);
            self.push_scope();
            let body = self.block()?;
            self.pop_scope();
            return Ok(Item { name, kind: ItemKind::Every(ms), params: Vec::new(), body, line });
        }

        let kind = match self.peek().clone() {
            Tok::Ident(w) if w == "action" => ItemKind::Action,
            Tok::Ident(w) if w == "check" => ItemKind::Check,
            Tok::Ident(w) if w == "filter" => ItemKind::Filter,
            other => {
                return err(
                    line,
                    col,
                    format!(
                        "expected 'action', 'check', 'filter', 'when' or 'every' here, found {}",
                        describe(&other)
                    ),
                )
            }
        };
        self.bump();
        let name = self.expect_ident()?;
        self.expect_punct('(')?;
        let mut params: Vec<String> = Vec::new();
        if !self.eat_punct(')') {
            loop {
                let p = self.expect_ident()?;
                if params.iter().any(|x| *x == p) {
                    let (l, c) = self.here();
                    return err(l, c, format!("parameter '{}' is repeated", p));
                }
                params.push(p);
                if self.eat_punct(',') {
                    continue;
                }
                self.expect_punct(')')?;
                break;
            }
        }
        self.expect_punct(':')?;
        self.push_scope();
        for p in &params {
            self.bind(p);
        }
        let body = self.block()?;
        self.pop_scope();
        if body.is_empty() {
            return err(line, col, format!("'{}' has an empty body", name));
        }
        Ok(Item { name, kind, params, body, line })
    }

    fn block(&mut self) -> Result<Vec<Stmt>, ParseError> {
        if !matches!(self.peek(), Tok::Newline) {
            let (l, c) = self.here();
            return err(l, c, "put the block on the next line, indented by four spaces");
        }
        self.bump();
        if !matches!(self.peek(), Tok::Indent) {
            let (l, c) = self.here();
            return err(l, c, "this block needs one indented line at least");
        }
        self.bump();
        self.push_scope();
        let mut out: Vec<Stmt> = Vec::new();
        loop {
            match self.peek() {
                Tok::Dedent => {
                    self.bump();
                    break;
                }
                Tok::Eof => {
                    let (l, c) = self.here();
                    self.pop_scope();
                    return err(l, c, "this block is never closed");
                }
                Tok::Newline => {
                    self.bump();
                    continue;
                }
                _ => {}
            }
            let stmt = self.stmt()?;
            out.push(stmt);
            match self.peek() {
                Tok::Newline => {
                    self.bump();
                }
                Tok::Dedent | Tok::Eof => {}
                _ => {}
            }
            if matches!(self.peek(), Tok::Dedent) {
                self.bump();
                break;
            }
        }
        self.pop_scope();
        Ok(out)
    }

    // ── statements ───────────────────────────────────────────────────
    fn stmt(&mut self) -> Result<Stmt, ParseError> {
        let (line, col) = self.here();
        if self.at_ident("let") {
            self.bump();
            let name = self.expect_ident()?;
            self.expect_punct('=')?;
            let value = self.expr()?;
            self.bind(&name);
            return Ok(Stmt::Let(name, value, line));
        }
        if self.at_ident("set") {
            self.bump();
            let name = self.expect_ident()?;
            if !self.eat_punct('=') {
                self.expect_word("to")?;
            }
            let value = self.expr()?;
            if !self.is_bound(&name) {
                return err(line, col, format!("'{}' is not defined yet; use 'let {} = ...' first", name, name));
            }
            return Ok(Stmt::Set(name, value, line));
        }
        if self.at_ident("if") {
            self.bump();
            let cond = self.expr()?;
            self.expect_punct(':')?;
            let then = self.block()?;
            let mut otherwise: Vec<Stmt> = Vec::new();
            if self.at_ident("otherwise") {
                self.bump();
                self.expect_punct(':')?;
                otherwise = self.block()?;
            }
            return Ok(Stmt::If { cond, then, otherwise, line });
        }
        if self.at_ident("for") {
            self.bump();
            self.expect_word("each")?;
            let var = self.expect_ident()?;
            self.expect_word("in")?;
            let list = self.expr()?;
            self.expect_punct(':')?;
            self.push_scope();
            self.bind(&var);
            let body = self.block()?;
            self.pop_scope();
            return Ok(Stmt::ForEach { var, list, body, line });
        }
        if self.at_ident("while") {
            self.bump();
            let cond = self.expr()?;
            self.expect_punct(':')?;
            let body = self.block()?;
            return Ok(Stmt::While { cond, body, line });
        }
        if self.at_ident("return") {
            // `return to launch` is the drone verb; `return` alone ends the action.
            if matches!(self.peek_at(1), Tok::Ident(w) if w == "to") {
                self.bump();
                self.bump();
                self.expect_word("launch")?;
                return Ok(Stmt::Eval(
                    Expr::Call { recv: "drone".into(), op: "rtl".into(), args: Vec::new(), line },
                    line,
                ));
            }
            self.bump();
            if matches!(self.peek(), Tok::Newline | Tok::Dedent | Tok::Eof) {
                return Ok(Stmt::Return(None, line));
            }
            let value = self.expr()?;
            return Ok(Stmt::Return(Some(value), line));
        }
        let expr = self.verb_or_expr()?;
        Ok(Stmt::Eval(expr, line))
    }

    /// A sentence-verb statement, or a plain expression statement.
    fn verb_or_expr(&mut self) -> Result<Expr, ParseError> {
        let (line, _col) = self.here();
        let word = match self.peek().clone() {
            Tok::Ident(w) => w,
            _ => return self.expr(),
        };
        match word.as_str() {
            "log" => {
                self.bump();
                let level = match self.peek().clone() {
                    Tok::Ident(w) if ["info", "warn", "warning", "error", "debug"].contains(&w.as_str()) => {
                        self.bump();
                        if w == "warning" { "warn".to_string() } else { w }
                    }
                    _ => "info".to_string(),
                };
                let msg = self.expr()?;
                Ok(Expr::Call { recv: "log".into(), op: level, args: vec![msg], line })
            }
            "send" => {
                self.bump();
                self.expect_word("alert")?;
                let msg = self.expr()?;
                Ok(Expr::Call { recv: "alert".into(), op: "send".into(), args: vec![msg], line })
            }
            "remember" => {
                self.bump();
                let value = self.expr()?;
                self.expect_word("as")?;
                let label = self.expr()?;
                Ok(Expr::Call {
                    recv: "memory".into(),
                    op: "save".into(),
                    args: vec![label, value],
                    line,
                })
            }
            "reject" => {
                self.bump();
                self.expect_word("with")?;
                let reason = self.expr()?;
                Ok(Expr::Call { recv: "policy".into(), op: "reject".into(), args: vec![reason], line })
            }
            "allow" => {
                self.bump();
                let mut args = Vec::new();
                if !matches!(self.peek(), Tok::Newline | Tok::Dedent | Tok::Eof) {
                    args.push(self.expr()?);
                }
                Ok(Expr::Call { recv: "policy".into(), op: "allow".into(), args, line })
            }
            "pulse" => {
                self.bump();
                let device = self.expect_ident()?;
                self.expect_word("for")?;
                let secs = self.expr()?;
                let name = self.literal_text(&device);
                Ok(Expr::Call {
                    recv: "device".into(),
                    op: "pulse".into(),
                    args: vec![name, secs],
                    line,
                })
            }
            "turn" => {
                self.bump();
                match self.peek().clone() {
                    Tok::Ident(w) if w == "on" || w == "off" => {
                        self.bump();
                        let device = self.expect_ident()?;
                        let name = self.literal_text(&device);
                        Ok(Expr::Call {
                            recv: "device".into(),
                            op: "set_relay".into(),
                            args: vec![name, Expr::Bool(w == "on")],
                            line,
                        })
                    }
                    Tok::Ident(w) if w == "left" || w == "right" => {
                        self.bump();
                        self.expect_word("by")?;
                        let deg = self.expr()?;
                        let op = if w == "left" { "turn_left" } else { "turn_right" };
                        Ok(Expr::Call { recv: "rover".into(), op: op.into(), args: vec![deg], line })
                    }
                    _ => err(line, line_col(self).1, "expected 'on', 'off', 'left' or 'right' after 'turn'"),
                }
            }
            "arm" => {
                self.bump();
                let noun = self.expect_ident()?;
                let family = caps::family_of(&noun);
                Ok(Expr::Call { recv: family.into(), op: "arm".into(), args: Vec::new(), line })
            }
            "takeoff" => {
                self.bump();
                self.expect_word("to")?;
                let alt = self.expr()?;
                Ok(Expr::Call { recv: "drone".into(), op: "takeoff".into(), args: vec![alt], line })
            }
            "fly" => {
                self.bump();
                self.expect_word("to")?;
                let point = self.expr()?;
                let mut args = vec![point];
                if self.eat_word("at") {
                    args.push(self.expr()?);
                }
                Ok(Expr::Call { recv: "drone".into(), op: "goto".into(), args, line })
            }
            "land" => {
                self.bump();
                let family = match self.peek().clone() {
                    Tok::Ident(w) if !matches!(w.as_str(), "to" | "on") => {
                        self.bump();
                        caps::family_of(&w)
                    }
                    _ => "drone",
                };
                Ok(Expr::Call { recv: family.into(), op: "land".into(), args: Vec::new(), line })
            }
            "hold" => {
                self.bump();
                self.eat_word("position");
                Ok(Expr::Call { recv: "drone".into(), op: "hold".into(), args: Vec::new(), line })
            }
            "stop" => {
                self.bump();
                let noun = self.expect_ident()?;
                let family = caps::family_of(&noun);
                Ok(Expr::Call { recv: family.into(), op: "stop".into(), args: Vec::new(), line })
            }
            "drive" => {
                self.bump();
                let op = match self.peek().clone() {
                    Tok::Ident(w) if w == "forward" || w == "backward" => {
                        self.bump();
                        w
                    }
                    _ => "forward".to_string(),
                };
                let mut args = Vec::new();
                if self.eat_word("at") {
                    args.push(self.expr()?);
                }
                Ok(Expr::Call { recv: "rover".into(), op, args, line })
            }
            "recall" => {
                self.bump();
                self.expect_word("memories")?;
                self.expect_word("matching")?;
                let query = self.expr()?;
                Ok(Expr::Call { recv: "memory".into(), op: "recall".into(), args: vec![query], line })
            }
            _ => self.expr(),
        }
    }

    /// A bare name in a verb slot becomes text (`turn on fan` → "fan").
    fn literal_text(&self, name: &str) -> Expr {
        Expr::Str(name.to_string())
    }

    // ── expressions ──────────────────────────────────────────────────
    fn expr(&mut self) -> Result<Expr, ParseError> {
        self.or_expr()
    }

    fn or_expr(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.and_expr()?;
        while self.eat_word("or") {
            let right = self.and_expr()?;
            left = Expr::Bin(BinOp::Or, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn and_expr(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.not_expr()?;
        while self.eat_word("and") {
            let right = self.not_expr()?;
            left = Expr::Bin(BinOp::And, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn not_expr(&mut self) -> Result<Expr, ParseError> {
        if self.eat_word("not") {
            let inner = self.not_expr()?;
            return Ok(Expr::Not(Box::new(inner)));
        }
        self.cmp_expr()
    }

    fn cmp_expr(&mut self) -> Result<Expr, ParseError> {
        let left = self.add_expr()?;
        let op = match self.peek().clone() {
            Tok::Punct('<') => Some(BinOp::Lt),
            Tok::Le => Some(BinOp::Le),
            Tok::Punct('>') => Some(BinOp::Gt),
            Tok::Ge => Some(BinOp::Ge),
            Tok::EqEq => Some(BinOp::Eq),
            Tok::Ne => Some(BinOp::Ne),
            Tok::Punct('=') => {
                let (l, c) = self.here();
                return err(l, c, "use '==' to compare; '=' assigns");
            }
            _ => None,
        };
        if let Some(op) = op {
            self.bump();
            let right = self.add_expr()?;
            return Ok(Expr::Bin(op, Box::new(left), Box::new(right)));
        }
        Ok(left)
    }

    fn add_expr(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.mul_expr()?;
        loop {
            let op = match self.peek() {
                Tok::Punct('+') => BinOp::Add,
                Tok::Punct('-') => BinOp::Sub,
                _ => break,
            };
            self.bump();
            let right = self.mul_expr()?;
            left = Expr::Bin(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn mul_expr(&mut self) -> Result<Expr, ParseError> {
        let mut left = self.unary_expr()?;
        loop {
            let op = match self.peek() {
                Tok::Punct('*') => BinOp::Mul,
                Tok::Punct('/') => BinOp::Div,
                Tok::Punct('%') => BinOp::Mod,
                _ => break,
            };
            // `1 m/s` was already consumed as a unit by number_with_unit, so a
            // '/' here is always real division.
            self.bump();
            let right = self.unary_expr()?;
            left = Expr::Bin(op, Box::new(left), Box::new(right));
        }
        Ok(left)
    }

    fn unary_expr(&mut self) -> Result<Expr, ParseError> {
        if self.eat_punct('-') {
            let inner = self.unary_expr()?;
            return Ok(Expr::Bin(BinOp::Sub, Box::new(Expr::Num(0.0, Unit::None)), Box::new(inner)));
        }
        self.postfix_expr()
    }

    fn postfix_expr(&mut self) -> Result<Expr, ParseError> {
        let mut value = self.primary()?;
        loop {
            if self.eat_punct('[') {
                let idx = self.expr()?;
                self.expect_punct(']')?;
                value = Expr::Index(Box::new(value), Box::new(idx));
                continue;
            }
            break;
        }
        Ok(value)
    }

    fn primary(&mut self) -> Result<Expr, ParseError> {
        let (line, col) = self.here();
        match self.peek().clone() {
            Tok::Num(n) => {
                self.bump();
                let unit = self.take_unit()?;
                let (value, unit) = normalize_unit(n, unit);
                Ok(Expr::Num(value, unit))
            }
            Tok::Str(s) => {
                self.bump();
                Ok(Expr::Str(s))
            }
            Tok::Punct('(') => {
                self.bump();
                let inner = self.expr()?;
                self.expect_punct(')')?;
                Ok(inner)
            }
            Tok::Punct('[') => {
                self.bump();
                let mut items = Vec::new();
                if !self.eat_punct(']') {
                    loop {
                        items.push(self.expr()?);
                        if self.eat_punct(',') {
                            continue;
                        }
                        self.expect_punct(']')?;
                        break;
                    }
                }
                Ok(Expr::List(items))
            }
            Tok::Ident(word) => {
                if word == "true" || word == "false" {
                    self.bump();
                    return Ok(Expr::Bool(word == "true"));
                }
                if word == "length" && matches!(self.peek_at(1), Tok::Ident(w) if w == "of") {
                    self.bump();
                    self.bump();
                    let inner = self.add_expr()?;
                    return Ok(Expr::Len(Box::new(inner)));
                }
                if word == "policy" && matches!(self.peek_at(1), Tok::Ident(w) if w == "detects" || w == "detect") {
                    self.bump();
                    self.bump();
                    let kind = self.expect_ident()?;
                    self.expect_word("in")?;
                    let subject = self.add_expr()?;
                    return Ok(Expr::Call {
                        recv: "policy".into(),
                        op: "detects".into(),
                        args: vec![Expr::Str(kind), subject],
                        line,
                    });
                }
                if word == "recall" && matches!(self.peek_at(1), Tok::Ident(w) if w == "memories") {
                    self.bump();
                    self.expect_word("memories")?;
                    self.expect_word("matching")?;
                    let query = self.expr()?;
                    return Ok(Expr::Call {
                        recv: "memory".into(),
                        op: "recall".into(),
                        args: vec![query],
                        line,
                    });
                }
                self.bump();
                // `recv.op(args)`, or `record.field` when no call follows.
                if self.eat_punct('.') {
                    let op = self.expect_ident()?;
                    if !self.eat_punct('(') {
                        return Ok(Expr::Field { base: word, field: op, line });
                    }
                    let mut args = Vec::new();
                    if !self.eat_punct(')') {
                        loop {
                            args.push(self.expr()?);
                            if self.eat_punct(',') {
                                continue;
                            }
                            self.expect_punct(')')?;
                            break;
                        }
                    }
                    return Ok(Expr::Call { recv: word, op, args, line });
                }
                if self.is_bound(&word) {
                    return Ok(Expr::Name(word));
                }
                if caps::is_telemetry(&word) {
                    if let Some(family) = self.ambient {
                        return Ok(Expr::Ambient { field: word, family });
                    }
                    return err(
                        line,
                        col,
                        format!(
                            "'{}' is read from a machine, but this script declares no single \
                             hardware capability to read it from; write e.g. drone.telemetry(\"{}\") \
                             or pass it as a parameter",
                            word, word
                        ),
                    );
                }
                Ok(Expr::Name(word))
            }
            other => err(line, col, format!("expected a value here, found {}", describe(&other))),
        }
    }

    /// Consume a unit suffix if one is written.
    fn take_unit(&mut self) -> Result<Option<(&'static str, Unit)>, ParseError> {
        let word = match self.peek().clone() {
            Tok::Ident(w) => w,
            Tok::Punct('%') => {
                self.bump();
                return Ok(Some(("%", Unit::Pct)));
            }
            _ => return Ok(None),
        };
        let unit = match word.as_str() {
            "meters" | "meter" | "m" => {
                // `m/s` is two tokens; only in unit position do they join.
                if matches!(self.peek_at(1), Tok::Punct('/')) && matches!(self.peek_at(2), Tok::Ident(s) if s == "s") {
                    self.bump();
                    self.bump();
                    self.bump();
                    return Ok(Some(("m/s", Unit::Mps)));
                }
                self.bump();
                ("meters", Unit::Meter)
            }
            "degrees" | "degree" | "deg" => {
                self.bump();
                ("degrees", Unit::Deg)
            }
            "seconds" | "second" | "sec" | "s" => {
                self.bump();
                ("seconds", Unit::Sec)
            }
            "ms" | "milliseconds" | "millisecond" => {
                self.bump();
                ("ms", Unit::Sec)
            }
            "percent" => {
                self.bump();
                ("%", Unit::Pct)
            }
            _ => return Ok(None),
        };
        Ok(Some(unit))
    }

    /// A number with an optional unit — used where a period is required.
    fn number_with_unit(&mut self) -> Result<(f64, Unit), ParseError> {
        let (line, col) = self.here();
        let n = match self.peek().clone() {
            Tok::Num(n) => {
                self.bump();
                n
            }
            other => return err(line, col, format!("expected a number here, found {}", describe(&other))),
        };
        let unit = self.take_unit()?;
        let (value, unit) = normalize_unit(n, unit);
        Ok((value, unit))
    }
}

fn line_col(p: &Parser) -> (u32, u32) {
    p.here()
}

/// `500 ms` is half a second: time units normalize to seconds, lengths to
/// meters, angles to degrees, and percent stays a whole number.
fn normalize_unit(n: f64, unit: Option<(&'static str, Unit)>) -> (f64, Unit) {
    match unit {
        None => (n, Unit::None),
        Some((label, u)) => match (label, u) {
            ("ms", Unit::Sec) => (n / 1000.0, Unit::Sec),
            _ => (n, u),
        },
    }
}

fn describe(tok: &Tok) -> String {
    match tok {
        Tok::Ident(w) => format!("'{}'", w),
        Tok::Num(n) => format!("the number {}", crate::sfs::arena::fmt_num(*n)),
        Tok::Str(_) => "text".to_string(),
        Tok::Punct(c) => format!("'{}'", c),
        Tok::Le => "'<='".to_string(),
        Tok::Ge => "'>='".to_string(),
        Tok::EqEq => "'=='".to_string(),
        Tok::Ne => "'!='".to_string(),
        Tok::Newline => "the end of the line".to_string(),
        Tok::Indent => "an indent".to_string(),
        Tok::Dedent => "the end of the block".to_string(),
        Tok::Eof => "the end of the script".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(src: &str) -> Program {
        parse(src).unwrap_or_else(|e| panic!("parse failed: {}", e.msg))
    }

    fn fails(src: &str) -> String {
        match parse(src) {
            Ok(_) => panic!("expected a parse error"),
            Err(e) => e.msg,
        }
    }

    #[test]
    fn parses_the_patrol_example_from_the_plan() {
        let src = "needs: drone, log\n\naction patrol(points):\n    log \"Starting automated perimeter patrol\"\n    arm drone\n    takeoff to 15 meters\n    for each point in points:\n        fly to point at 15 meters\n    return to launch\n\nwhen battery < 20%:\n    log warning \"Battery critical!\"\n    return to launch\n";
        let p = ok(src);
        assert_eq!(p.needs, vec!["drone".to_string(), "log".to_string()]);
        assert_eq!(p.items.len(), 2);
        let patrol = p.item("patrol").expect("patrol");
        assert!(matches!(patrol.kind, ItemKind::Action));
        assert_eq!(patrol.params, vec!["points".to_string()]);
        assert!(matches!(p.items[1].kind, ItemKind::When(_)));
    }

    #[test]
    fn parses_the_rover_example() {
        let src = "needs: rover, lidar, log\n\nevery 100ms:\n    let front = lidar.front_distance()\n    if front < 0.5 meters:\n        log warning \"Obstacle detected ahead!\"\n        stop rover\n        turn right by 90 degrees\n    otherwise:\n        drive forward at 1.2 m/s\n";
        let p = ok(src);
        match &p.items[0].kind {
            ItemKind::Every(ms) => assert_eq!(*ms, 100.0),
            other => panic!("expected an every hook, got {:?}", other),
        }
        let body = &p.items[0].body;
        assert!(matches!(body[0], Stmt::Let(_, _, _)));
        match &body[1] {
            Stmt::If { then, otherwise, .. } => {
                assert_eq!(then.len(), 3);
                assert_eq!(otherwise.len(), 1);
            }
            other => panic!("expected an if, got {:?}", other),
        }
    }

    #[test]
    fn parses_the_greenhouse_and_guardrail_examples() {
        let greenhouse = "needs: sensor, fan, mister, alert\n\ncheck climate(target_temp, target_humidity):\n    if sensor.temperature() > target_temp + 2:\n        turn on fan\n        send alert \"Cooling fan activated\"\n    otherwise:\n        turn off fan\n    if sensor.humidity() < target_humidity:\n        pulse mister for 5 seconds\n";
        let p = ok(greenhouse);
        assert_eq!(p.items[0].params.len(), 2);
        assert_eq!(p.items[0].body.len(), 2);

        let guardrail = "needs: memory, policy\n\nfilter user_prompt(prompt):\n    if length of prompt > 2000:\n        reject with \"Prompt exceeds maximum allowed length\"\n    if policy detects secrets in prompt:\n        reject with \"Prompt contains private API keys\"\n    remember prompt as \"last_valid_query\"\n    allow prompt\n";
        let p = ok(guardrail);
        assert!(matches!(p.items[0].kind, ItemKind::Filter));
        assert_eq!(p.items[0].body.len(), 4);
    }

    #[test]
    fn units_normalize_but_keep_their_identity() {
        let p = ok("needs: log\naction a():\n    let x = 500 ms\n    let y = 1.5 m/s\n    let z = 20%\n");
        let Stmt::Let(_, Expr::Num(v, u), _) = &p.items[0].body[0] else { panic!() };
        assert_eq!(*v, 0.5);
        assert_eq!(*u, Unit::Sec);
        let Stmt::Let(_, Expr::Num(v, u), _) = &p.items[0].body[1] else { panic!() };
        assert_eq!(*v, 1.5);
        assert_eq!(*u, Unit::Mps);
        let Stmt::Let(_, Expr::Num(v, u), _) = &p.items[0].body[2] else { panic!() };
        assert_eq!(*v, 20.0);
        assert_eq!(*u, Unit::Pct);
    }

    #[test]
    fn bare_telemetry_needs_exactly_one_hardware_family() {
        let msg = fails("needs: drone, rover\n\nwhen battery < 20%:\n    stop rover\n");
        assert!(msg.contains("drone.telemetry"), "{}", msg);
        let p = ok("needs: drone\n\nwhen battery < 20%:\n    stop drone\n");
        match &p.items[0].kind {
            ItemKind::When(Expr::Bin(_, left, _)) => match &**left {
                Expr::Ambient { field, family } => {
                    assert_eq!(field, "battery");
                    assert_eq!(*family, "drone");
                }
                other => panic!("expected ambient, got {:?}", other),
            },
            other => panic!("unexpected shape {:?}", other),
        }
    }

    #[test]
    fn unknown_names_survive_to_the_checker_as_names() {
        let p = ok("needs: log\naction a():\n    log mystery\n");
        let Stmt::Eval(Expr::Call { args, .. }, _) = &p.items[0].body[0] else { panic!() };
        assert!(matches!(args[0], Expr::Name(ref n) if n == "mystery"));
    }

    #[test]
    fn a_single_line_block_is_refused_with_advice() {
        let msg = fails("needs: log\naction a(): log \"x\"\n");
        assert!(msg.contains("next line"), "{}", msg);
    }

    #[test]
    fn duplicates_are_refused() {
        assert!(fails("needs: log\naction a():\n    log \"1\"\naction a():\n    log \"2\"\n")
            .contains("defined twice"));
        assert!(fails("needs: log, log\naction a():\n    log \"1\"\n").contains("twice"));
    }

    #[test]
    fn missing_needs_is_fine_until_a_capability_is_used() {
        let p = ok("action a():\n    let x = 1 + 1\n");
        assert!(p.needs.is_empty());
    }
}
