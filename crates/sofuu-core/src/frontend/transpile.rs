// crates/sofuu-core/src/frontend/transpile.rs
//
// Fast, streaming, zero-AST JSX and TypeScript transpiler in pure Rust.
// Converts JSX syntax:
//   `<div className="title">Hello {name}</div>`
// into standard function calls:
//   `h("div", {className: "title"}, "Hello ", (name))`
//
// Handles:
//   - Self-closing tags: `<input type="text" />` -> `h("input", {type: "text"})`
//   - Fragment tags: `<>...</>` -> `h(Fragment, null, ...)`
//   - Attribute expressions: `<button onClick={onClick} />` -> `h("button", {onClick: (onClick)})`
//   - Boolean attributes: `<button disabled />` -> `h("button", {disabled: true})`
//   - Spread attributes: `<div {...props} />` -> `h("div", {...props})`
//   - Bare import rewrites (`sofuu/ui`, `react`, `preact` -> `/_sofuu/ui.js`)
//   - TypeScript type stripping via `crate::ts::strip`

use crate::ts;

/// Transpile a JSX or TSX source string into standard ES Module JavaScript.
pub fn transpile(src: &str, is_ts: bool) -> Result<String, String> {
    // Step 1: Rewrite bare imports (sofuu/ui, react, preact -> /_sofuu/ui.js)
    let rewritten_imports = rewrite_imports(src);

    // Step 2: Transpile JSX tags to h() calls FIRST
    // This prevents TS type strippers from confusing `<Tag>` with generics `<T>`
    let mut transpiler = JsxTranspiler::new(&rewritten_imports);
    let jsx_transpiled = transpiler.transpile()?;

    // Step 3: Strip TypeScript types if requested (preserves line numbers)
    let mut result = if is_ts {
        ts::strip(&jsx_transpiled)
    } else {
        jsx_transpiled
    };

    // Step 4: Inject `import { h, Fragment } from "/_sofuu/ui.js";` if JSX was used
    // and not already imported
    if transpiler.has_jsx && !result.contains("from \"/_sofuu/ui.js\"") && !result.contains("from '/_sofuu/ui.js'") {
        result = format!("import {{ h, Fragment }} from \"/_sofuu/ui.js\";\n{}", result);
    }

    Ok(result)
}

/// JSX → `h()` calls only: no bare-import rewriting and no `/_sofuu/ui.js`
/// import injection. The sandbox has no module system, so it installs its own
/// `h`/`Fragment` in the realm and evaluates the result as a script.
pub fn transpile_jsx_only(src: &str) -> Result<String, String> {
    let mut transpiler = JsxTranspiler::new(src);
    transpiler.transpile()
}

fn rewrite_imports(src: &str) -> String {
    let targets = [
        ("\"sofuu/ui\"", "\"/_sofuu/ui.js\""),
        ("'sofuu/ui'", "'/_sofuu/ui.js'"),
        ("\"sofuu/ai\"", "\"/_sofuu/ui.js\""),
        ("'sofuu/ai'", "'/_sofuu/ui.js'"),
        ("\"sofuu/router\"", "\"/_sofuu/ui.js\""),
        ("'sofuu/router'", "'/_sofuu/ui.js'"),
        ("\"sofuu/store\"", "\"/_sofuu/ui.js\""),
        ("'sofuu/store'", "'/_sofuu/ui.js'"),
        ("\"react\"", "\"/_sofuu/ui.js\""),
        ("'react'", "'/_sofuu/ui.js'"),
        ("\"react-dom\"", "\"/_sofuu/ui.js\""),
        ("'react-dom'", "'/_sofuu/ui.js'"),
        ("\"react-dom/client\"", "\"/_sofuu/ui.js\""),
        ("'react-dom/client'", "'/_sofuu/ui.js'"),
        ("\"react/jsx-runtime\"", "\"/_sofuu/ui.js\""),
        ("'react/jsx-runtime'", "'/_sofuu/ui.js'"),
        ("\"react/jsx-dev-runtime\"", "\"/_sofuu/ui.js\""),
        ("'react/jsx-dev-runtime'", "'/_sofuu/ui.js'"),
        ("\"preact\"", "\"/_sofuu/ui.js\""),
        ("'preact'", "'/_sofuu/ui.js'"),
        ("\"preact/hooks\"", "\"/_sofuu/ui.js\""),
        ("'preact/hooks'", "'/_sofuu/ui.js'"),
        ("\"swr\"", "\"/_sofuu/ui.js\""),
        ("'swr'", "'/_sofuu/ui.js'"),
    ];

    let mut result = src.to_string();
    for (target, replacement) in &targets {
        result = result.replace(target, replacement);
    }
    result
}

fn transpile_jsx(src: &str) -> Result<String, String> {
    let mut transpiler = JsxTranspiler::new(src);
    transpiler.transpile()
}

struct JsxTranspiler<'a> {
    src: &'a str,
    bytes: &'a [u8],
    len: usize,
    pos: usize,
    has_jsx: bool,
}

impl<'a> JsxTranspiler<'a> {
    fn new(src: &'a str) -> Self {
        let bytes = src.as_bytes();
        let len = bytes.len();
        Self {
            src,
            bytes,
            len,
            pos: 0,
            has_jsx: false,
        }
    }

    fn transpile(&mut self) -> Result<String, String> {
        let mut out = String::with_capacity(self.len + 128);
        let mut last_sig = false;

        while self.pos < self.len {
            let c = self.bytes[self.pos];

            // ── Comments ──
            if c == b'/' && self.pos + 1 < self.len && self.bytes[self.pos + 1] == b'/' {
                let start = self.pos;
                self.pos += 2;
                while self.pos < self.len && self.bytes[self.pos] != b'\n' {
                    self.pos += 1;
                }
                out.push_str(self.slice_str(start, self.pos));
                continue;
            }
            if c == b'/' && self.pos + 1 < self.len && self.bytes[self.pos + 1] == b'*' {
                let start = self.pos;
                self.pos += 2;
                while self.pos + 1 < self.len && !(self.bytes[self.pos] == b'*' && self.bytes[self.pos + 1] == b'/') {
                    self.pos += 1;
                }
                if self.pos + 1 < self.len {
                    self.pos += 2;
                }
                out.push_str(self.slice_str(start, self.pos));
                continue;
            }

            // ── Strings & Templates ──
            if c == b'\'' || c == b'"' {
                let quote = c;
                let start = self.pos;
                self.pos += 1;
                while self.pos < self.len && self.bytes[self.pos] != quote {
                    if self.bytes[self.pos] == b'\\' {
                        self.pos += 2;
                        continue;
                    }
                    self.pos += 1;
                }
                if self.pos < self.len {
                    self.pos += 1;
                }
                out.push_str(self.slice_str(start, self.pos));
                last_sig = true;
                continue;
            }

            if c == b'`' {
                let start = self.pos;
                self.pos += 1;
                while self.pos < self.len && self.bytes[self.pos] != b'`' {
                    if self.bytes[self.pos] == b'\\' {
                        self.pos += 2;
                        continue;
                    }
                    // Handle template expression ${ ... }
                    if self.bytes[self.pos] == b'$' && self.pos + 1 < self.len && self.bytes[self.pos + 1] == b'{' {
                        self.pos += 2;
                        let mut braces = 1;
                        while self.pos < self.len && braces > 0 {
                            if self.bytes[self.pos] == b'{' {
                                braces += 1;
                            } else if self.bytes[self.pos] == b'}' {
                                braces -= 1;
                            }
                            self.pos += 1;
                        }
                        continue;
                    }
                    self.pos += 1;
                }
                if self.pos < self.len {
                    self.pos += 1;
                }
                out.push_str(self.slice_str(start, self.pos));
                last_sig = true;
                continue;
            }

            // ── Potential JSX Element or Fragment: `<` ──
            // In JS, a JSX tag begins with `<` when not preceded by an operand (identifier, number, closing paren, etc.)
            // EXCEPT after `return`, `yield`, `=>`, `=`, `(`, `[`, `{`, `:`, `,`, `?`, `&&`, `||`, `;`, etc.
            if c == b'<' && (!last_sig || self.is_jsx_context(&out)) {
                if let Some(jsx_out) = self.try_parse_element()? {
                    out.push_str(&jsx_out);
                    self.has_jsx = true;
                    last_sig = true;
                    continue;
                }
            }

            // Track whether previous non-whitespace token was an operand
            if c == b' ' || c == b'\t' || c == b'\r' || c == b'\n' {
                out.push(c as char);
                self.pos += 1;
                continue;
            }

            if c == b'(' || c == b'[' || c == b'{' || c == b'=' || c == b':' || c == b',' || c == b';' || c == b'?' || c == b'&' || c == b'|' {
                last_sig = false;
            } else {
                last_sig = true;
            }

            if c < 128 {
                out.push(c as char);
                self.pos += 1;
            } else {
                let ch = self.src[self.pos..].chars().next().unwrap_or(' ');
                out.push(ch);
                self.pos += ch.len_utf8();
            }
        }

        Ok(out)
    }

    fn is_jsx_context(&self, current_out: &str) -> bool {
        let trimmed = current_out.trim_end();
        if trimmed.is_empty() {
            return true;
        }
        let last_word = trimmed.split(|c: char| !c.is_alphanumeric() && c != '_').filter(|s| !s.is_empty()).last().unwrap_or("");
        matches!(last_word, "return" | "yield" | "default" | "case")
            || trimmed.ends_with('(')
            || trimmed.ends_with('[')
            || trimmed.ends_with('{')
            || trimmed.ends_with('=')
            || trimmed.ends_with('>') // e.g. `=>`
            || trimmed.ends_with(':')
            || trimmed.ends_with('?')
            || trimmed.ends_with('&')
            || trimmed.ends_with('|')
            || trimmed.ends_with(',')
            || trimmed.ends_with(';')
    }

    fn try_parse_element(&mut self) -> Result<Option<String>, String> {
        if self.pos >= self.len || self.bytes[self.pos] != b'<' {
            return Ok(None);
        }

        // Must not be `<<` or `<=` or `<!--`
        if self.pos + 1 < self.len {
            let next = self.bytes[self.pos + 1];
            if next == b'<' || next == b'=' || next == b'!' {
                return Ok(None);
            }
        }

        let saved = self.pos;
        match self.parse_element() {
            Ok(res) => Ok(Some(res)),
            Err(_) => {
                // Not valid JSX, rewind and treat as regular `<`
                self.pos = saved;
                Ok(None)
            }
        }
    }

    fn parse_element(&mut self) -> Result<String, String> {
        self.expect(b'<')?;
        self.skip_whitespace();

        // ── 1. Tag Name or Fragment ──
        let (tag_repr, is_fragment) = if self.pos < self.len && self.bytes[self.pos] == b'>' {
            self.pos += 1;
            ("Fragment".to_string(), true)
        } else {
            let tag_name = self.parse_tag_name()?;
            let repr = if tag_name.starts_with(|c: char| c.is_ascii_uppercase()) || tag_name.contains('.') {
                tag_name
            } else {
                format!("\"{}\"", tag_name)
            };
            (repr, false)
        };

        // ── 2. Attributes (if not fragment) ──
        let mut is_self_closing = false;
        let attrs_obj = if !is_fragment {
            self.skip_whitespace();
            let mut attrs = Vec::new();

            while self.pos < self.len {
                if self.bytes[self.pos] == b'/' && self.pos + 1 < self.len && self.bytes[self.pos + 1] == b'>' {
                    self.pos += 2;
                    is_self_closing = true;
                    break;
                }
                if self.bytes[self.pos] == b'>' {
                    self.pos += 1;
                    break;
                }

                // Spread attribute: `{...props}`
                if self.bytes[self.pos] == b'{' {
                    self.pos += 1;
                    self.skip_whitespace();
                    if self.pos + 3 <= self.len && &self.bytes[self.pos..self.pos + 3] == b"..." {
                        self.pos += 3;
                        let expr = self.parse_balanced(b'{', b'}')?;
                        attrs.push(format!("...{}", expr.trim()));
                        self.skip_whitespace();
                        continue;
                    }
                }

                // Regular attribute: `name="val"` or `name={expr}` or boolean `name`
                let attr_name = self.parse_attr_name()?;
                self.skip_whitespace();
                let key_repr = format_attr_key(&attr_name);

                if self.pos < self.len && self.bytes[self.pos] == b'=' {
                    self.pos += 1;
                    self.skip_whitespace();
                    if self.pos < self.len && (self.bytes[self.pos] == b'"' || self.bytes[self.pos] == b'\'') {
                        let quote = self.bytes[self.pos];
                        self.pos += 1;
                        let val_start = self.pos;
                        while self.pos < self.len && self.bytes[self.pos] != quote {
                            if self.bytes[self.pos] == b'\\' {
                                self.pos += 2;
                                continue;
                            }
                            self.pos += 1;
                        }
                        let val = self.slice_str(val_start, self.pos);
                        if self.pos < self.len {
                            self.pos += 1;
                        }
                        attrs.push(format!("{}: \"{}\"", key_repr, escape_str(val)));
                    } else if self.pos < self.len && self.bytes[self.pos] == b'{' {
                        self.pos += 1;
                        let expr = self.parse_balanced(b'{', b'}')?;
                        let transpiled = transpile_jsx(expr.trim())?;
                        attrs.push(format!("{}: ({})", key_repr, transpiled));
                    } else {
                        return Err(format!("Unexpected character in attribute value: {}", self.bytes[self.pos] as char));
                    }
                } else {
                    // Boolean attribute: `<button disabled />`
                    attrs.push(format!("{}: true", key_repr));
                }

                self.skip_whitespace();
            }

            if attrs.is_empty() {
                "null".to_string()
            } else {
                format!("{{{}}}", attrs.join(", "))
            }
        } else {
            "null".to_string()
        };

        if is_self_closing {
            return Ok(format!("h({}, {})", tag_repr, attrs_obj));
        }

        // ── 3. Children ──
        let mut children = Vec::new();
        let close_tag = if is_fragment {
            "</>".to_string()
        } else {
            // Raw tag name without quotes
            let clean_name = tag_repr.trim_matches('"');
            format!("</{}>", clean_name)
        };
        let close_bytes = close_tag.as_bytes();

        while self.pos < self.len {
            // Check for closing tag `</...>` (exact match or space-tolerant match)
            if self.pos + 2 <= self.len && &self.bytes[self.pos..self.pos + 2] == b"</" {
                let saved = self.pos;
                self.pos += 2;
                self.skip_whitespace();
                let close_name = if self.pos < self.len && self.bytes[self.pos] == b'>' {
                    self.pos += 1;
                    "".to_string()
                } else {
                    let name = self.parse_tag_name().unwrap_or_default();
                    self.skip_whitespace();
                    if self.pos < self.len && self.bytes[self.pos] == b'>' {
                        self.pos += 1;
                    }
                    name
                };

                let clean_name = tag_repr.trim_matches('"');
                if is_fragment && close_name.is_empty() {
                    break;
                } else if !is_fragment && (close_name == clean_name || close_name.is_empty()) {
                    break;
                } else {
                    // Belongs to parent element, rewind so parent closes cleanly
                    self.pos = saved;
                    break;
                }
            }

            // Child JSX element
            if self.bytes[self.pos] == b'<' {
                let child = self.parse_element()?;
                children.push(child);
                continue;
            }

            // Child Expression container: `{expr}`
            if self.bytes[self.pos] == b'{' {
                self.pos += 1;
                let expr = self.parse_balanced(b'{', b'}')?;
                let trimmed = expr.trim();
                if !trimmed.is_empty() {
                    // Ignore JSX comments like `{/* ... */}` or `{// ...}`
                    if (trimmed.starts_with("/*") && trimmed.ends_with("*/"))
                        || (trimmed.starts_with("//") && !trimmed.contains('\n'))
                    {
                        continue;
                    }
                    let transpiled = transpile_jsx(trimmed)?;
                    let clean = transpiled.trim();
                    if !clean.is_empty() && clean != "()" {
                        children.push(format!("({})", clean));
                    }
                }
                continue;
            }

            // Child Text
            let text_start = self.pos;
            while self.pos < self.len && self.bytes[self.pos] != b'<' && self.bytes[self.pos] != b'{' {
                self.pos += 1;
            }
            let raw_text = self.slice_str(text_start, self.pos);
            let trimmed = raw_text.trim();
            if !trimmed.is_empty() {
                children.push(format!("\"{}\"", escape_str(trimmed)));
            }
        }

        if children.is_empty() {
            Ok(format!("h({}, {})", tag_repr, attrs_obj))
        } else {
            Ok(format!("h({}, {}, {})", tag_repr, attrs_obj, children.join(", ")))
        }
    }

    fn slice_str(&self, start: usize, end: usize) -> &'a str {
        if start <= end && end <= self.len {
            std::str::from_utf8(&self.bytes[start..end]).unwrap_or("")
        } else {
            ""
        }
    }

    fn parse_tag_name(&mut self) -> Result<String, String> {
        let start = self.pos;
        while self.pos < self.len {
            let c = self.bytes[self.pos];
            if c.is_ascii_alphanumeric() || c == b'_' || c == b'$' || c == b'.' || c == b'-' {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos == start {
            return Err("Expected tag name".to_string());
        }
        Ok(self.slice_str(start, self.pos).to_string())
    }

    fn parse_attr_name(&mut self) -> Result<String, String> {
        let start = self.pos;
        while self.pos < self.len {
            let c = self.bytes[self.pos];
            if c.is_ascii_alphanumeric() || c == b'_' || c == b'$' || c == b'-' || c == b':' {
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos == start {
            return Err("Expected attribute name".to_string());
        }
        Ok(self.slice_str(start, self.pos).to_string())
    }

    fn parse_balanced(&mut self, open: u8, close: u8) -> Result<String, String> {
        let start = self.pos;
        let mut depth = 1;

        while self.pos < self.len && depth > 0 {
            let c = self.bytes[self.pos];
            if c == open {
                depth += 1;
            } else if c == close {
                depth -= 1;
                if depth == 0 {
                    let res = self.slice_str(start, self.pos).to_string();
                    self.pos += 1;
                    return Ok(res);
                }
            } else if c == b'\'' || c == b'"' {
                let quote = c;
                self.pos += 1;
                while self.pos < self.len && self.bytes[self.pos] != quote {
                    if self.bytes[self.pos] == b'\\' {
                        self.pos += 2;
                        continue;
                    }
                    self.pos += 1;
                }
            } else if c == b'`' {
                self.pos += 1;
                while self.pos < self.len && self.bytes[self.pos] != b'`' {
                    if self.bytes[self.pos] == b'\\' {
                        self.pos += 2;
                        continue;
                    }
                    self.pos += 1;
                }
            }
            self.pos += 1;
        }

        if depth != 0 {
            Err("Unmatched closing delimiter".to_string())
        } else {
            let res = self.slice_str(start, self.pos - 1).to_string();
            Ok(res)
        }
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.len {
            let c = self.bytes[self.pos];
            if c == b' ' || c == b'\t' || c == b'\r' || c == b'\n' {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn expect(&mut self, expected: u8) -> Result<(), String> {
        if self.pos < self.len && self.bytes[self.pos] == expected {
            self.pos += 1;
            Ok(())
        } else {
            Err(format!("Expected '{}'", expected as char))
        }
    }
}

fn escape_str(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

fn format_attr_key(name: &str) -> String {
    if name.contains('-') || name.contains(':') || name.contains('.') {
        format!("\"{}\"", name)
    } else {
        name.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpile_simple_jsx() {
        let input = r#"
            export default function App() {
                return <div className="card"><h1>Hello world</h1></div>;
            }
        "#;
        let output = transpile(input, false).unwrap();
        assert!(output.contains(r#"h("div", {className: "card"}, h("h1", null, "Hello world"))"#));
        assert!(output.contains(r#"import { h, Fragment } from "/_sofuu/ui.js";"#));
    }

    #[test]
    fn test_transpile_multibyte_utf8_emoji() {
        let input = r#"
            export default function App() {
                return <h1>⚡ Sofuu In-Process AI Chat 🚀</h1>;
            }
        "#;
        let output = transpile(input, false).unwrap();
        assert!(output.contains(r#"h("h1", null, "⚡ Sofuu In-Process AI Chat 🚀")"#));
    }

    #[test]
    fn test_transpile_self_closing_and_expressions() {
        let input = r#"
            const el = <input type="text" value={name} disabled />;
        "#;
        let output = transpile(input, false).unwrap();
        assert!(output.contains(r#"h("input", {type: "text", value: (name), disabled: true})"#));
    }

    #[test]
    fn test_transpile_component_and_fragment() {
        let input = r#"
            return (
                <>
                    <Header title="AI Chat" />
                    <span>Welcome</span>
                </>
            );
        "#;
        let output = transpile(input, false).unwrap();
        assert!(output.contains("h(Fragment, null"));
        assert!(output.contains(r#"h(Header, {title: "AI Chat"})"#));
        assert!(output.contains(r#"h("span", null, "Welcome")"#));
    }

    #[test]
    fn test_transpile_typescript_stripping() {
        let input = r#"
            interface UserProps {
                name: string;
                age: number;
            }
            export function UserCard({ name, age }: UserProps) {
                return <div>{name}: {age}</div>;
            }
        "#;
        let output = transpile(input, true).unwrap();
        assert!(!output.contains("interface UserProps"));
        assert!(!output.contains(": UserProps"));
        assert!(output.contains(r#"h("div", null, (name), ":", (age))"#));
    }

    #[test]
    fn test_transpile_nested_jsx_in_expressions() {
        let input = r#"
            return (
                <main>
                    {items.length === 0 && (
                        <div className="empty">No items</div>
                    )}
                    {items.map(item => (
                        <div key={item.id}>
                            <strong>{item.title}</strong>
                        </div>
                    ))}
                </main>
            );
        "#;
        let output = transpile(input, false).unwrap();
        assert!(!output.contains("<div"));
        assert!(!output.contains("<strong"));
        assert!(output.contains(r#"h("main", null"#));
        assert!(output.contains(r#"h("div", {className: "empty"}, "No items")"#));
        assert!(output.contains(r#"h("strong", null, (item.title))"#));
    }

    #[test]
    fn test_rewrite_bare_imports() {
        let input = r#"
            import React, { useState, useQuery } from "react";
            import ReactDOM from "react-dom";
            import { createRoot } from "react-dom/client";
            import { useChat, ThinkingBlock } from "sofuu/ai";
            import { Link, useRouter } from "sofuu/router";
            import { createStore } from "sofuu/store";
            export default function Chat() { return <Link href="/"><div /></Link>; }
        "#;
        let output = transpile(input, false).unwrap();
        assert!(output.contains(r#"from "/_sofuu/ui.js""#));
        assert!(!output.contains(r#"from "react""#));
        assert!(!output.contains(r#"from "react-dom""#));
        assert!(!output.contains(r#"from "react-dom/client""#));
        assert!(!output.contains(r#"from "sofuu/ui""#));
        assert!(!output.contains(r#"from "sofuu/ai""#));
        assert!(!output.contains(r#"from "sofuu/router""#));
        assert!(!output.contains(r#"from "sofuu/store""#));
    }

    #[test]
    fn test_rewrite_multiline_imports() {
        let input = r#"
            import {
                useState,
                useEffect,
                useRef
            } from "sofuu/ui";
            export default function App() { return <div />; }
        "#;
        let output = transpile(input, false).unwrap();
        assert!(output.contains(r#"from "/_sofuu/ui.js""#));
        assert!(!output.contains(r#"from "sofuu/ui""#));
    }

    #[test]
    fn test_transpile_jsx_comment() {
        let input = r#"
            return (
                <div>
                    {/* This is a comment */}
                    <span>Hello</span>
                </div>
            );
        "#;
        let output = transpile(input, false).unwrap();
        assert!(!output.contains("()"));
        assert!(output.contains(r#"h("div", null, h("span", null, "Hello"))"#));
    }

    #[test]
    fn test_transpile_hyphenated_and_namespaced_attributes() {
        let input = r#"
            const el = <button data-testid="chat-btn" aria-label="Send" stroke-width="2" disabled />;
        "#;
        let output = transpile(input, false).unwrap();
        assert!(output.contains(r#""data-testid": "chat-btn""#));
        assert!(output.contains(r#""aria-label": "Send""#));
        assert!(output.contains(r#""stroke-width": "2""#));
        assert!(output.contains("disabled: true"));
    }
}
