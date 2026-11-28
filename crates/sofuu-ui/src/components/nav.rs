//! Navigation stack transitions and route-stack helpers.
//!
//! `NavStack` rendering lives in the compiler (it needs the layout rects and the
//! animation progress); this module owns the pure state transitions so hosts, the
//! JS kit, and patches all agree on `push`/`pop`/`replace` semantics.

use crate::ir::NavRoute;

/// Push a route and return its new active index.
pub fn push(routes: &mut Vec<NavRoute>, route: NavRoute) -> usize {
    routes.push(route);
    routes.len().saturating_sub(1)
}

/// Pop the active route. Returns `false` at the root.
pub fn pop(routes: &mut Vec<NavRoute>, active: &mut usize) -> bool {
    if *active == 0 || routes.is_empty() {
        return false;
    }
    routes.remove(*active);
    *active = active.saturating_sub(1);
    true
}

/// Replace the active route in place, keeping the stack depth.
pub fn replace(routes: &mut [NavRoute], active: usize, route: NavRoute) -> bool {
    if let Some(slot) = routes.get_mut(active) {
        *slot = route;
        true
    } else {
        false
    }
}

/// Drop every route above the root and settle on index `0`.
pub fn pop_to_root(routes: &mut Vec<NavRoute>, active: &mut usize) -> bool {
    if *active == 0 {
        return false;
    }
    routes.truncate(1);
    *active = 0;
    true
}

/// Clamp `active` after a scene rebuild.
#[inline]
pub fn clamp_active(len: usize, active: usize) -> usize {
    active.min(len.saturating_sub(1))
}

/// Edge-swipe release policy: commit a back navigation once the user has
/// dragged more than `threshold` of the screen width.
#[inline]
pub fn should_commit_back(progress: f32, threshold: f32) -> bool {
    progress.clamp(0.0, 1.0) >= threshold.clamp(0.05, 0.95)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ir::{Node, NodeKind};

    fn route(id: &str) -> NavRoute {
        NavRoute {
            id: id.into(),
            title: id.into(),
            child: Node::new(id, NodeKind::Spacer),
        }
    }

    #[test]
    fn push_pop_replace_and_pop_to_root_keep_active_in_sync() {
        let mut routes = vec![route("root")];
        let mut active = push(&mut routes, route("a"));
        assert_eq!(active, 1);
        assert_eq!(routes.len(), 2);
        assert!(replace(&mut routes, active, route("b")));
        assert_eq!(routes[1].id, "b");
        active = push(&mut routes, route("c"));
        assert!(pop_to_root(&mut routes, &mut active));
        assert_eq!((routes.len(), active), (1, 0));
        assert!(!pop(&mut routes, &mut active));
        assert!(!pop_to_root(&mut routes, &mut active));
    }

    #[test]
    fn edge_swipe_commits_past_the_threshold() {
        assert!(!should_commit_back(0.2, 0.4));
        assert!(should_commit_back(0.5, 0.4));
        assert!(should_commit_back(1.0, 0.4));
    }
}
