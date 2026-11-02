// sfs/lexer.rs — hand-rolled zero-dependency tokenizer.
//
// PLAN-KAZE-LANGUAGE §5 (M-K1) and §3.1: the surface reads like English, so the
// tokenizer's job is to make the *shape* of a line unambiguous without a
// grammar generator: words stay words, and indentation becomes explicit
// INDENT/DEDENT tokens the parser can trust.
//
// Rules that matter for determinism and for kind error messages:
//   * Indentation is spaces only; a tab in the indent is refused by column.
//   * A dedent that does not land on an existing level is refused (no
//     guesswork about what the author meant).
//   * Newlines inside (`(`, `[`) are continuation, not statement ends.
//   * `#` starts a comment to end of line; blank lines emit nothing.

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    Ident(String),
    Num(f64),
    Str(String),
    /// `: ( ) , . % + - * / < > = [ ]`
    Punct(char),
    Le,
    Ge,
    EqEq,
    Ne,
    Newline,
    Indent,
    Dedent,
    Eof,
}

#[derive(Clone, Debug)]
pub struct Token {
    pub tok: Tok,
    /// 1-based line, 1-based column — what a human reads in an editor.
    pub line: u32,
    pub col: u32,
}

#[derive(Debug, Clone)]
pub struct LexError {
    pub line: u32,
    pub col: u32,
    pub msg: String,
}

impl LexError {
    pub fn render(&self, source_name: &str) -> String {
        format!("{}:{}:{}: {}", source_name, self.line, self.col, self.msg)
    }
}

fn err(line: u32, col: u32, msg: impl Into<String>) -> LexError {
    LexError { line, col, msg: msg.into() }
}

/// Tokenize a whole script. The result always ends with `Tok::Eof`.
pub fn tokenize(src: &str) -> Result<Vec<Token>, LexError> {
    let bytes = src.as_bytes();
    let mut out: Vec<Token> = Vec::new();
    let mut i = 0usize;
    let mut line: u32 = 1;
    let mut col: u32 = 1;
    let mut indents: Vec<u32> = vec![0];
    let mut depth: u32 = 0; // () and [] nesting disables indent handling
    let mut line_has_token = false;

    while i < bytes.len() {
        let b = bytes[i];
        match b {
            b' ' | b'\r' => {
                i += 1;
                col += 1;
            }
            b'\t' => {
                return Err(err(line, col, "tabs are not allowed; indent with spaces"));
            }
            b'\n' => {
                if depth == 0 && line_has_token {
                    out.push(Token { tok: Tok::Newline, line, col });
                }
                i += 1;
                line += 1;
                col = 1;
                line_has_token = false;
                // measure the next line's indentation up front
                if depth == 0 {
                    let mut j = i;
                    let mut width: u32 = 0;
                    while j < bytes.len() && (bytes[j] == b' ' || bytes[j] == b'\r') {
                        if bytes[j] == b' ' {
                            width += 1;
                        }
                        j += 1;
                    }
                    // blank line or comment-only line: no INDENT/DEDENT
                    let blank = j >= bytes.len() || bytes[j] == b'\n' || bytes[j] == b'#';
                    if !blank {
                        let top = *indents.last().unwrap_or(&0);
                        if width > top {
                            indents.push(width);
                            out.push(Token { tok: Tok::Indent, line, col });
                        } else if width < top {
                            while *indents.last().unwrap_or(&0) > width {
                                indents.pop();
                                out.push(Token { tok: Tok::Dedent, line, col });
                            }
                            if *indents.last().unwrap_or(&0) != width {
                                return Err(err(
                                    line,
                                    1,
                                    "inconsistent indentation: this line does not line up \
                                     with any enclosing block",
                                ));
                            }
                        }
                    }
                }
            }
            b'#' => {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                    col += 1;
                }
            }
            b'"' => {
                let start_line = line;
                let start_col = col;
                i += 1;
                col += 1;
                let mut s = String::new();
                loop {
                    if i >= bytes.len() {
                        return Err(err(
                            start_line,
                            start_col,
                            "this text is missing its closing quote",
                        ));
                    }
                    let c = bytes[i];
                    if c == b'"' {
                        i += 1;
                        col += 1;
                        break;
                    }
                    if c == b'\n' {
                        return Err(err(
                            start_line,
                            start_col,
                            "text cannot span lines; close the quote or drop the newline",
                        ));
                    }
                    if c == b'\\' {
                        i += 1;
                        col += 1;
                        if i >= bytes.len() {
                            return Err(err(start_line, start_col, "text ends in a lone backslash"));
                        }
                        let e = bytes[i];
                        let ch = match e {
                            b'n' => '\n',
                            b't' => '\t',
                            b'"' => '"',
                            b'\\' => '\\',
                            other => {
                                return Err(err(
                                    line,
                                    col,
                                    format!(
                                        "unsupported escape \\{} (use \\n, \\t or \\\\)",
                                        other as char
                                    ),
                                ))
                            }
                        };
                        s.push(ch);
                        i += 1;
                        col += 1;
                        continue;
                    }
                    // copy one UTF-8 scalar
                    let ch_len = utf8_len(c);
                    let end = (i + ch_len).min(bytes.len());
                    if let Ok(part) = std::str::from_utf8(&bytes[i..end]) {
                        s.push_str(part);
                    }
                    i = end;
                    col += 1;
                }
                out.push(Token { tok: Tok::Str(s), line: start_line, col: start_col });
                line_has_token = true;
            }
            b'0'..=b'9' => {
                let start = i;
                let start_col = col;
                while i < bytes.len() && bytes[i].is_ascii_digit() {
                    i += 1;
                    col += 1;
                }
                if i < bytes.len() && bytes[i] == b'.' && i + 1 < bytes.len() && bytes[i + 1].is_ascii_digit() {
                    i += 1;
                    col += 1;
                    while i < bytes.len() && bytes[i].is_ascii_digit() {
                        i += 1;
                        col += 1;
                    }
                }
                let text = std::str::from_utf8(&bytes[start..i]).unwrap_or("0");
                let n: f64 = text
                    .parse()
                    .map_err(|_| err(line, start_col, format!("'{}' is not a number", text)))?;
                out.push(Token { tok: Tok::Num(n), line, col: start_col });
                line_has_token = true;
            }
            c if c.is_ascii_alphabetic() || c == b'_' => {
                let start = i;
                let start_col = col;
                while i < bytes.len()
                    && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_')
                {
                    i += 1;
                    col += 1;
                }
                let word = String::from_utf8_lossy(&bytes[start..i]).into_owned();
                out.push(Token { tok: Tok::Ident(word), line, col: start_col });
                line_has_token = true;
            }
            _ => {
                let start_col = col;
                let two = if i + 1 < bytes.len() { bytes[i + 1] } else { 0 };
                let tok = match (b, two) {
                    (b'<', b'=') => Some(Tok::Le),
                    (b'>', b'=') => Some(Tok::Ge),
                    (b'=', b'=') => Some(Tok::EqEq),
                    (b'!', b'=') => Some(Tok::Ne),
                    _ => None,
                };
                if let Some(t) = tok {
                    i += 2;
                    col += 2;
                    out.push(Token { tok: t, line, col: start_col });
                    line_has_token = true;
                    continue;
                }
                let ch = match b {
                    b':' | b'(' | b')' | b',' | b'.' | b'%' | b'+' | b'-' | b'*' | b'/'
                    | b'<' | b'>' | b'=' | b'[' | b']' => b as char,
                    other => {
                        return Err(err(
                            line,
                            col,
                            format!("unexpected character '{}'", other as char),
                        ))
                    }
                };
                if ch == '(' || ch == '[' {
                    depth += 1;
                } else if ch == ')' || ch == ']' {
                    depth = depth.saturating_sub(1);
                }
                i += 1;
                col += 1;
                out.push(Token { tok: Tok::Punct(ch), line, col: start_col });
                line_has_token = true;
            }
        }
    }

    if line_has_token {
        out.push(Token { tok: Tok::Newline, line, col });
    }
    while indents.len() > 1 {
        indents.pop();
        out.push(Token { tok: Tok::Dedent, line, col });
    }
    out.push(Token { tok: Tok::Eof, line, col });
    Ok(out)
}

fn utf8_len(b: u8) -> usize {
    if b < 0x80 {
        1
    } else if b >> 5 == 0b110 {
        2
    } else if b >> 4 == 0b1110 {
        3
    } else {
        4
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(src: &str) -> Vec<Tok> {
        tokenize(src).unwrap().into_iter().map(|t| t.tok).collect()
    }

    #[test]
    fn indentation_becomes_indent_and_dedent() {
        let t = kinds("action a():\n    let x = 1\n");
        assert!(t.contains(&Tok::Indent));
        assert!(t.contains(&Tok::Dedent));
    }

    #[test]
    fn comment_only_lines_do_not_move_indentation() {
        let t = kinds("action a():\n    let x = 1\n    # note\n    let y = 2\n");
        assert_eq!(t.iter().filter(|k| **k == Tok::Indent).count(), 1);
        assert_eq!(t.iter().filter(|k| **k == Tok::Dedent).count(), 1);
    }

    #[test]
    fn tabs_are_refused_by_column() {
        let e = tokenize("action a():\n\tlet x = 1\n").unwrap_err();
        assert!(e.msg.contains("tabs"), "{}", e.msg);
        assert_eq!(e.line, 2);
    }

    #[test]
    fn breaking_indentation_is_refused() {
        let e = tokenize("action a():\n    if true:\n        let x = 1\n      let y = 2\n")
            .unwrap_err();
        assert!(e.msg.contains("inconsistent indentation"), "{}", e.msg);
    }

    #[test]
    fn numbers_units_and_percent_tokenize_adjacently() {
        let t = kinds("every 100ms\nif battery < 20%:\n");
        assert!(t.contains(&Tok::Num(100.0)));
        assert!(t.contains(&Tok::Ident("ms".to_string())));
        assert!(t.contains(&Tok::Num(20.0)));
        assert!(t.contains(&Tok::Punct('%')));
    }

    #[test]
    fn strings_carry_escapes() {
        let t = kinds("log \"a\\nb\"\n");
        assert!(t.contains(&Tok::Str("a\nb".to_string())));
    }

    #[test]
    fn unterminated_string_names_the_line() {
        let e = tokenize("action a():\n    log \"oops\n").unwrap_err();
        assert_eq!(e.line, 2);
        assert!(e.msg.contains("quote"), "{}", e.msg);
    }

    #[test]
    fn brackets_suppress_newlines() {
        let t = kinds("let x = [1,\n2]\n");
        assert_eq!(t.iter().filter(|k| **k == Tok::Newline).count(), 1);
    }
}
