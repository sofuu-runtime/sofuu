//! Purpose-built, GPU-native, automatically adaptable UI components for `sofuu-ui`.
//!
//! Zero generic wrappers: every component emits precision `QuadInstance`, `StrokeInstance`,
//! and `GlyphInstance` buffers tailored to its bounds, data cardinality, scale, and input state.

pub mod code;
pub mod controls;
pub mod feed;
pub mod graph;
pub mod list;
pub mod media;
pub mod nav;
pub mod plot;
pub mod stat;
pub mod table;

use crate::text::GlyphAtlas;

/// Measure and truncate `text` with `".."` if wider than `max_w` logical pixels.
pub fn fit_text(atlas: &GlyphAtlas, text: &str, px_size: f32, mono: bool, max_w: f32) -> String {
    if max_w <= px_size * 0.6 {
        return String::new();
    }
    let (w, _) = atlas.measure(text, px_size, mono);
    if w <= max_w {
        return text.to_string();
    }
    let (dots_w, _) = atlas.measure("..", px_size, mono);
    let budget = (max_w - dots_w).max(0.0);
    let mut acc = 0.0f32;
    let mut out = String::new();
    for ch in text.chars() {
        let cw = atlas.char_advance_em(ch, mono) * px_size;
        if acc + cw > budget {
            break;
        }
        acc += cw;
        out.push(ch);
    }
    if out.is_empty() {
        return String::new();
    }
    out.push_str("..");
    out
}

/// Format numeric scalar adaptively with clean grouping/compact magnitude suffixes.
pub fn format_number(val: f64, precision: Option<usize>) -> String {
    if !val.is_finite() {
        return "--".to_string();
    }
    let abs = val.abs();
    if let Some(p) = precision {
        return format_fixed_grouped(val, p);
    }
    if abs >= 1_000_000_000.0 {
        format!("{:.2}B", val / 1_000_000_000.0)
    } else if abs >= 1_000_000.0 {
        format!("{:.2}M", val / 1_000_000.0)
    } else if abs >= 100_000.0 {
        format!("{:.1}K", val / 1_000.0)
    } else if abs >= 1_000.0 {
        format_fixed_grouped(val, 2)
    } else if abs >= 1.0 || abs == 0.0 {
        format!("{val:.2}")
    } else if abs >= 0.01 {
        format!("{val:.4}")
    } else {
        format!("{val:.6}")
    }
}

fn format_fixed_grouped(val: f64, decimals: usize) -> String {
    let raw = format!("{val:.decimals$}");
    let (neg, body) = if let Some(rest) = raw.strip_prefix('-') {
        (true, rest)
    } else {
        (false, raw.as_str())
    };
    let (int_part, frac_part) = match body.split_once('.') {
        Some((i, f)) => (i, Some(f)),
        None => (body, None),
    };
    let mut grouped = String::with_capacity(int_part.len() + 6);
    if neg {
        grouped.push('-');
    }
    for (idx, ch) in int_part.chars().enumerate() {
        let rem = int_part.len() - idx;
        if idx > 0 && rem % 3 == 0 {
            grouped.push(',');
        }
        grouped.push(ch);
    }
    if let Some(f) = frac_part {
        grouped.push('.');
        grouped.push_str(f);
    }
    grouped
}
