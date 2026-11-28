//! Backend-free scenario fixtures.
//!
//! A UI module may export:
//!
//! ```js
//! export function renderApp(props) { ... }
//! export const preview = {
//!   title: 'Wallet Checkout Flow',
//!   scenarios: {
//!     Loaded:    { balance: 1240.50, items: ['Node Pack'] },
//!     EmptyCart: { balance: 1240.50, items: [] },
//!     HitlGate:  { pendingApproval: { tool: 'wire_transfer' } },
//!   },
//! };
//! ```
//!
//! The emulator treats the scenario payload as an opaque JSON object. This
//! module only parses/merges it; no app-specific keys are ever interpreted.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Scenario {
    pub name: String,
    #[serde(default)]
    pub props: Value,
}

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct ScenarioSet {
    #[serde(default)]
    pub title: Option<String>,
    /// Ordered names for stable devtools panels.
    #[serde(default)]
    pub order: Vec<String>,
    #[serde(default)]
    pub scenarios: BTreeMap<String, Value>,
}

impl ScenarioSet {
    /// Parse either `{ title, scenarios: {...} }` or a bare `{ name: props }`
    /// map. Non-object payloads are rejected by the caller with a clear error.
    pub fn from_json(value: &Value) -> Self {
        let mut out = ScenarioSet::default();
        let map = match value {
            Value::Object(m) => m,
            _ => return empty_set(),
        };
        if let Some(Value::Object(sc)) = map.get("scenarios") {
            if let Some(Value::String(t)) = map.get("title") {
                out.title = Some(t.clone());
            }
            for (k, v) in sc {
                out.order.push(k.clone());
                out.scenarios.insert(k.clone(), v.clone());
            }
            return out;
        }
        if map.contains_key("scenarios") {
            return empty_set();
        }
        for (k, v) in map {
            out.order.push(k.clone());
            out.scenarios.insert(k.clone(), v.clone());
        }
        out
    }

    /// Generic states every UI can exercise without a backend.
    pub fn builtin() -> Self {
        let mut out = ScenarioSet::default();
        out.title = Some("Preview".to_string());
        for (name, props) in [
            ("Loaded", serde_json::json!({ "status": "ready" })),
            ("Empty", serde_json::json!({ "status": "empty", "items": [] })),
            ("Loading", serde_json::json!({ "status": "loading", "items": [] })),
            ("Overflow", serde_json::json!({ "status": "ready", "overflow": true })),
            ("Error", serde_json::json!({ "status": "error" })),
            (
                "ApprovalPending",
                serde_json::json!({
                    "status": "awaiting_approval",
                    "pendingApproval": { "tool": "example_tool" }
                }),
            ),
        ] {
            out.order.push(name.to_string());
            out.scenarios.insert(name.to_string(), props);
        }
        out
    }

    pub fn names(&self) -> Vec<String> {
        if self.order.is_empty() {
            self.scenarios.keys().cloned().collect()
        } else {
            self.order.clone()
        }
    }

    pub fn resolve(&self, name: &str) -> Result<Value, String> {
        self.scenarios
            .get(name)
            .cloned()
            .ok_or_else(|| {
                format!(
                    "unknown scenario `{name}` (available: {})",
                    self.names().join(", ")
                )
            })
    }

    /// Deep-merge scenario props over a base object. Objects merge
    /// recursively; arrays/scalars in the scenario replace the base value.
    pub fn apply_over(&self, name: &str, base: &Value) -> Result<Value, String> {
        let props = self.resolve(name)?;
        Ok(deep_merge(base.clone(), props))
    }
}

fn empty_set() -> ScenarioSet {
    ScenarioSet::builtin()
}

pub fn deep_merge(mut base: Value, overlay: Value) -> Value {
    match (&mut base, overlay) {
        (Value::Object(a), Value::Object(b)) => {
            for (k, v) in b {
                let next = match a.remove(&k) {
                    Some(existing) => deep_merge(existing, v),
                    None => v,
                };
                a.insert(k, next);
            }
            Value::Object(a.clone())
        }
        (_, overlay) => overlay,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_preview_contract_and_bare_maps() {
        let preview = serde_json::json!({
            "title": "Wallet",
            "scenarios": {
                "Loaded": { "balance": 1240.5, "items": ["Node Pack"] },
                "EmptyCart": { "balance": 1240.5, "items": [] }
            }
        });
        let set = ScenarioSet::from_json(&preview);
        assert_eq!(set.title.as_deref(), Some("Wallet"));
        // `serde_json::Map` iterates in sorted key order by default, which is
        // the stable ordering exposed to the devtools switcher.
        assert_eq!(set.names(), vec!["EmptyCart", "Loaded"]);
        assert_eq!(set.resolve("Loaded").unwrap()["balance"], 1240.5);

        let bare = serde_json::json!({ "A": { "x": 1 }, "B": { "x": 2 } });
        let set2 = ScenarioSet::from_json(&bare);
        assert_eq!(set2.names(), vec!["A", "B"]);
    }

    #[test]
    fn unknown_scenario_lists_available_names() {
        let err = ScenarioSet::builtin().resolve("Nope").unwrap_err();
        assert!(err.contains("Loaded"));
        assert!(err.contains("ApprovalPending"));
    }

    #[test]
    fn deep_merge_keeps_base_keys() {
        let base = serde_json::json!({ "a": 1, "nested": { "x": 1, "y": 2 } });
        let over = serde_json::json!({ "nested": { "y": 9 }, "b": true });
        let out = deep_merge(base, over);
        assert_eq!(out["a"], 1);
        assert_eq!(out["nested"]["x"], 1);
        assert_eq!(out["nested"]["y"], 9);
        assert_eq!(out["b"], true);
    }

    #[test]
    fn builtin_covers_documented_states() {
        let names = ScenarioSet::builtin().names();
        for n in ["Loaded", "Empty", "Loading", "Overflow", "Error", "ApprovalPending"] {
            assert!(names.contains(&n.to_string()), "missing {n}");
        }
    }
}
