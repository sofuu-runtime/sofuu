# Wire fixtures

These are the exact JSON envelopes `src/js/ui.js` emits, kept as files so the
`sofuu-ui` crate's tests are self-contained when the crate is checked out as
its own repository (PLAN-SPLIT-UI-EFFECTS-EXTERNAL-REPO §6, Phase 1.2).

* `wire_showcase.json` — `replace_scene` from the component showcase.
* `wire_v2_handles.json` — `replace_scene` from the V2 handle builders.

They are parsed by `tests/js_wire_parity.rs`; update the JS first, then the
fixture, so the test keeps describing what the driver actually writes.
