# Sofuu plugins - four worked examples

Every plugin in this directory is a plain `.ts` file that the runtime loads,
type-strips and evaluates at discovery time. There is no build step, no bundler
and no `node_modules` requirement, and the files are checked in as source you
can copy into your own project.

| File | Category | What it demonstrates |
| --- | --- | --- |
| `fetch_web.ts` | tool (`defineTool`) | A read-only HTTP fetch with a strict schema, a hard timeout, byte-cap truncation, HTML-to-text, and a deliberately absent `permissions.net` (so Shield 2 keeps the private-IP guard on). |
| `sqlite_query.ts` | tool (`defineTool`) | A read-only query surface over the native QTSQ structured store, including a refusal path that rejects writes before anything is opened. |
| `ollama_provider.ts` | provider (`defineProvider`) | A streaming NDJSON `/api/chat` adapter with tool-call passthrough and a `/api/tags` health probe. |
| `token_logger.ts` | service (`defineService`) | A per-turn hook that appends one JSONL line, keeps cumulative totals per agent, flushes on stop, and never throws into a turn. |

## Typing

`sofuu.d.ts` at the repo root is the hand-written type surface for the plugin
SDK. Point TypeScript at it once, for example in `tsconfig.json`:

```json
{
  "compilerOptions": {
    "strict": true,
    "target": "ES2023",
    "module": "ESNext",
    "moduleResolution": "Bundler",
    "paths": { "sofuu": ["./sofuu.d.ts"] }
  }
}
```

Then `import { defineTool, z } from 'sofuu'` resolves to real declarations with
generics: `z.object({...})` infers the argument type of `run`, so `args.url` is
a `string`, and `z.enum(['text', 'markdown']).optional()` makes the field
optional. The globals `sofuu`, `z`, `defineTool`, `defineProvider`,
`defineStorage`, `defineVectorStore` and `defineService` are declared too, so a
plugin that skips the import still typechecks.

The npm package ships the same file (`"types": "sofuu.d.ts"`), so an installed
`sofuu` needs no extra configuration.

## Running them

```
sofuu plugin list                        # discovered tools, providers, stores, services
sofuu plugin validate plugins/fetch_web.ts
sofuu plugin run fetch_web '{"url":"https://example.com"}'
sofuu plugin test fetch_web '{"url":"https://example.com"}'   # alias of run
```

`plugin test` prefers the isolated driver entry point (`cliTest`) and falls back
to `cliRun` when the shipped bytecode predates it, so the alias works on older
builds.

Files dropped into `./plugins` (and `~/.sofuu/plugins`) are discovered
automatically. A `sofuu.json` at the project root adds more sources:

```json
{ "plugins": ["./plugins", { "source": "./vendor/my-plugin.ts", "enabled": true }] }
```

## The 5-Shield safety boundary

Tool calls run behind five independent shields, each with its own error code
from the Sofuu taxonomy (`E_SCHEMA | E_TIMEOUT | E_DENIED | E_CIRCUIT_OPEN |
E_TOOL | E_CAPABILITY`):

| Shield | Guarantee | Code |
| --- | --- | --- |
| 1 | Input/output schema validation (micro-Zod or JSON Schema) | `E_SCHEMA` |
| 2 | Security and network jailing: path traversal, sensitive files, private-IP / SSRF egress guard | `E_DENIED` |
| 3 | Automated circuit breaker (CLOSED / OPEN / HALF_OPEN) | `E_CIRCUIT_OPEN` |
| 4 | Hardware timer watchdog with signal abort (`timeoutMs`) | `E_TIMEOUT` |
| 5 | Isolated execution with a panic guard, so the host never crashes | `E_TOOL` |

Two consequences worth knowing before you copy these files:

* Declaring `permissions.net: ['host']` makes the allowlist authoritative and
  *replaces* the private-IP guard for that tool. `fetch_web.ts` omits it on
  purpose, so the SSRF protection stays on.
* A missing runtime capability (for example `sofuu.storage` in a build without
  the store) is reported as `E_CAPABILITY`, not as a crash. `sqlite_query.ts`
  checks for it up front.

## Why there is no `sqlite_query` that speaks SQLite

This runtime does not link SQLite. The only mention of it in the source is a
note recording that it was replaced by QTSQ, the native columnar store exposed
as `sofuu.storage`. `sqlite_query.ts` keeps the familiar name and the SELECT
shape, then maps the query onto that store, so an agent can inspect sessions,
traces and KV rows without a new dependency or a subprocess. Anything outside
the small documented subset is rejected with `E_SCHEMA`; every write and DDL
verb is rejected with `E_DENIED` before the store is even opened.

## How these files are loaded (and what that means for your code)

Discovery does not run `tsc`. It applies a fixed textual transformation, then
evaluates the result:

1. `import ... from '...'` lines are removed.
2. `export default ` becomes an assignment to the module slot;
   `export const|let|var|function|class` loses its `export` keyword.
3. A small set of regexes removes `interface`, `type` aliases, `as T`, return
   type annotations, parameter annotations and `var x: T` annotations.

That keeps the files readable and dependency-free, but it means a construct the
regexes cannot express is a runtime syntax error rather than a type error.
These files obey the contract by construction:

* No decorators, namespaces, enums, overloads or `implements` clauses.
* One-line flat interfaces (no nested object types inside a body).
* Type aliases contain no internal `;` (use commas).
* Every annotation is a single identifier, a single-level generic, or a name
  that strips cleanly. Unions and function types get a named `type` alias
  first, which also makes the code easier to read.
* No type arguments on a call expression (`new Promise<never>(...)`,
  `foo<T>(x)`). Nothing strips them, and plain JS reads `new Promise<never>(`
  as two comparisons. Annotate the variable instead.
* No `key: identifier` entries in object literals where the two names differ.
  The parameter rule rewrites `, key: value` into `, key`, which parses as a
  shorthand property and then throws at runtime. `{ text: text, truncated:
  truncated }` is safe; `{ text: body, truncated: true }` is not. When a
  literal is unavoidable, assign the field after construction.

All four files are verified two ways: `tsc --strict --noEmit` against
`sofuu.d.ts`, and a harness that replays the exact discovery pipeline and then
evaluates the stripped output.
