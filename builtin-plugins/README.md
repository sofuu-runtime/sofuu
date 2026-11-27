# builtin-plugins

First-party plugins that are **eligible** to ship inside the Sofuu binary.

Nothing in this directory is bundled today. The build (`crates/sofuu-core/src/shipped.rs`)
does not reference it, no `include_str!` points at it, and no `sofuu.json` in the
repository enables it. That is deliberate: the developer decides which plugins
are compiled into which binary profile, and that decision has not been made.

## Layout

```
builtin-plugins/
  collab/
    collab.ts          single-file plugin (tools + service)
    README.md          contract, configuration, guarantees and limits
    test/
      collab_test.js   contract suite (drives the real hooks)
      collab_race_worker.js  one process in the cross-process race
      run.sh           sets up a throwaway workspace and runs both suites
```

A built-in plugin is a normal plugin. It does not get a private API, a private
storage format, or a bypass of the 5 shields. The only difference is where the
source comes from at boot.

## Running one without bundling

Point the project manifest at the folder (discovery loads top-level `.js`,
`.ts` and `.wasm` entries and ignores everything else):

```json
{
  "plugins": [{ "source": "./builtin-plugins/collab" }]
}
```

or copy the entry file to `./plugins/` for a local experiment.

## Bundling a plugin later

When a plugin is approved for a binary profile, the embedding step is:

1. Add `pub const <NAME>_JS: &str = include_str!("../../../builtin-plugins/<name>/<entry>.ts");`
   to `crates/sofuu-core/src/shipped.rs` (release builds additionally need a
   `.bc.gz` snapshot, exactly like the existing drivers).
2. Evaluate it through the existing `eval_shipped_with_bytecode` seam at boot,
   behind the profile gate that selects which built-ins ship.
3. Add the plugin name to that profile's manifest.

No plugin does step 1-3 until the developer names it. In particular, `collab`
is not embedded, not in a profile list, and not auto-loaded.

## Acceptance

```sh
builtin-plugins/collab/test/run.sh
```

The suite runs in a temp workspace, uses the real `sofuu.fs`, `sofuu.policy`
and service dispatcher, and never touches a model or the network.
