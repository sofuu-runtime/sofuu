//! Virtualized list math: total extent, scroll clamping, and the visible window.
//!
//! Layout places only the rows intersecting the window (plus `overscan` screens),
//! so a 50,000-row list compiles exactly like a short one. These helpers are kept
//! independent of layout so compiler, host scrolling, and tests share one policy.

use crate::ir::ListViewData;

/// Total scrollable content height using explicit row heights and estimates.
pub fn total_height(data: &ListViewData) -> f32 {
    let spacing = data.spacing.max(0.0);
    let mut total = 0.0f32;
    for (i, item) in data.items.iter().enumerate() {
        total += item.estimated_or(data.estimated_height);
        if i + 1 < data.items.len() {
            total += spacing;
        }
    }
    total
}

/// Maximum scroll offset for a viewport of `viewport_h`.
#[inline]
pub fn max_scroll(data: &ListViewData, viewport_h: f32) -> f32 {
    (total_height(data) - viewport_h).max(0.0)
}

/// Vertical window `[top, bottom]` (in content coordinates) including overscan.
#[inline]
pub fn visible_window(scroll: f32, viewport_h: f32, overscan: f32) -> (f32, f32) {
    let pad = overscan.max(0.0) * viewport_h.max(0.0);
    (scroll - pad, scroll + viewport_h.max(0.0) + pad)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{ListItem, Node, NodeKind};

    fn item(id: &str, height: Option<f32>) -> ListItem {
        ListItem {
            id: id.into(),
            height,
            child: Node::new(id, NodeKind::Spacer),
        }
    }

    fn data(items: Vec<ListItem>) -> ListViewData {
        ListViewData {
            items,
            estimated_height: 50.0,
            spacing: 2.0,
            overscan: 1.0,
            style: None,
        }
    }

    #[test]
    fn total_height_mixes_explicit_and_estimated_rows() {
        let d = data(vec![item("a", Some(10.0)), item("b", None), item("c", Some(30.0))]);
        assert!((total_height(&d) - (10.0 + 2.0 + 50.0 + 2.0 + 30.0)).abs() < 0.001);
        assert!((max_scroll(&d, 40.0) - 54.0).abs() < 0.001);
        assert_eq!(max_scroll(&d, 200.0), 0.0);
    }

    #[test]
    fn overscan_expands_the_window_on_both_sides() {
        let (top, bottom) = visible_window(100.0, 300.0, 1.0);
        assert_eq!(top, -200.0);
        assert_eq!(bottom, 700.0);
    }
}
