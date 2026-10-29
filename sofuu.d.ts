/**
 * sofuu.d.ts - TypeScript declarations for the Sofuu plugin SDK.
 *
 * Provenance: every member below is read off the shipped runtime, not invented.
 *   src/js/plugins.js                     defineTool / defineProvider / defineStorage /
 *                                         defineVectorStore / defineService / useOpenAPI /
 *                                         micro-Zod / the 5-Shield ToolContext
 *   src/js/storage.js                     sofuu.storage (QTSQ structured store)
 *   src/js/memory_api.js                  sofuu.memory
 *   src/js/agent.js                       sofuu.ai / sofuu.tools / sofuu.policy / sofuu.mcp
 *   src/js/observability.js               sofuu.observability.cli / renderRun
 *   crates/sofuu-core/src/rt/engine.rs    sofuu.version, sofuu.fs
 *
 * Two rules this file obeys:
 *   1. No method is declared unless the runtime can answer it today. Where a
 *      declaration is advisory (a shape the runtime registers but does not yet
 *      call, or a driver hook that lands in a later bytecode refresh), the doc
 *      comment says so in as many words.
 *   2. Everything is structural. Nothing here needs a build step, so these
 *      declarations double as the reference for what a plugin may touch.
 *
 * The declarations reference the host fetch/AbortController globals (Response,
 * RequestInit, AbortSignal). Sofuu provides all three at runtime, so include a
 * DOM or Node type library in the consuming project.
 *
 * Usage:
 *   import { defineTool, z } from 'sofuu';
 *
 *   export default defineTool({
 *     name: 'echo',
 *     description: 'Echo the input back',
 *     schema: z.object({ text: z.string() }),
 *     async run(args, ctx) {
 *       ctx.log('echoing', args.text.length, 'characters');
 *       return { text: args.text };
 *     }
 *   });
 */

/* ================================================================== 
 * Error taxonomy
 * ================================================================== */

/**
 * The normative Sofuu error codes. plugins.js defaults an unclassified throw
 * inside a tool to E_TOOL; the other codes are produced by the shield that
 * rejected the call:
 *   E_SCHEMA       Shield 1 - input/output validation
 *   E_DENIED       Shield 2 - policy, SSRF or filesystem jail
 *   E_CIRCUIT_OPEN Shield 3 - circuit breaker tripped
 *   E_TIMEOUT      Shield 4 - deadline watchdog fired
 *   E_TOOL         Shield 5 - the tool itself threw
 *   E_CAPABILITY   the runtime cannot answer (missing driver or feature)
 *   E_BACKPRESSURE a queue or budget refused the work
 *   E_RATE_LIMIT   a provider or queue throttled the caller
 *   E_INTERNAL     an engine bug, never a user or provider mistake
 *
 * The last two complete the normative taxonomy in OBSERVABILITY-SPEC §0: a
 * retryable throttle and the last-resort internal code. Codes the spec
 * forbids (E_NETWORK_TIMEOUT, E_SERVER_ERROR, E_UNCAUGHT_THROW) are
 * deliberately absent - network and timeout faults map to E_TIMEOUT, tool
 * faults to E_TOOL, policy to E_DENIED, validation to E_SCHEMA.
 */
export type SofuuErrorCode =
  | 'E_SCHEMA'
  | 'E_TIMEOUT'
  | 'E_DENIED'
  | 'E_CIRCUIT_OPEN'
  | 'E_TOOL'
  | 'E_CAPABILITY'
  | 'E_BACKPRESSURE'
  | 'E_RATE_LIMIT'
  | 'E_INTERNAL';

/**
 * A plugin-raised error. code is what the shield loop reads; a plain Error
 * (no code) is treated as E_TOOL.
 */
export interface SofuuError extends Error {
  code: SofuuErrorCode;
  details?: unknown;
}

/**
 * An error code from the taxonomy, or any string a plugin chose to throw.
 * Keeps editor completion for the seven normative codes without rejecting a
 * plugin that invents its own.
 */
export type ErrorCodeLike = SofuuErrorCode | (string & {});

/** The { ok: false, error } payload plugins.js builds for a failed call. */
export interface ToolErrorInfo {
  code: ErrorCodeLike;
  message: string;
  /** Dotted path of the offending argument, when Shield 1 rejected input. */
  path?: string;
  /**
   * JSON-RPC code beside the engine code, for JSON-RPC / MCP clients:
   * -32602 invalid params, -32601 unknown tool, -32603 internal error,
   * -32000 timeout, -32001 denied, -32002 circuit open, -32003 backpressure.
   */
  jsonrpc?: number;
  /** Structured JSON-RPC error data (tool name, code, offending path). */
  data?: { tool?: string; code?: string; path?: string } & Record<string, unknown>;
}

/* ================================================================== 
 * JSON Schema (what defineTool stores after zodToJSONSchema)
 * ================================================================== */

/**
 * The JSON Schema subset the runtime validator understands
 * (validateJSONSchema in plugins.js checks exactly these keywords).
 */
export interface JSONSchema {
  type?: 'object' | 'array' | 'string' | 'number' | 'integer' | 'boolean' | 'any';
  description?: string;
  properties?: Record<string, JSONSchema>;
  required?: string[];
  items?: JSONSchema;
  enum?: Array<string | number | boolean>;
  minimum?: number;
  [extension: string]: unknown;
}

/* ================================================================== 
 * Micro-Zod - mirrors the z object plugins.js installs
 * ================================================================== */

/** Validation outcome returned by every schema safeParse. */
export interface ZodSafeParseResult<T> {
  success: boolean;
  data?: T;
  error?: { message: string; path?: Array<string | number> };
}

/**
 * The brand every micro-Zod schema carries. plugins.js duck-types a schema
 * with isZodSchema() (parse / safeParse / _def), so a real zod instance is
 * accepted too.
 */
export interface ZodTypeDef {
  typeName?: string;
  type?: ZodTypeAny;
  shape?: ZodRawShape | (() => ZodRawShape);
  [extension: string]: unknown;
}

/**
 * Base schema contract.
 *
 * Note on optional(): the runtime mutates the SAME schema instance
 * (isOptional = true) and returns it. The ZodOptional wrapper below exists
 * only so z.infer can see '| undefined'; it is not a distinct object at
 * runtime. Likewise describe() returns the same instance.
 */
export interface ZodType<Output = unknown, Def extends ZodTypeDef = ZodTypeDef, Input = Output> {
  readonly _def: Def;
  readonly _output: Output;
  readonly _input: Input;
  /** True on every micro-Zod schema; absent on a real zod instance. */
  readonly _isJSONSchema?: boolean;
  description?: string;
  isOptional?: boolean;
  /** The value used when the key is absent (`.default(v)`). */
  _default?: Output;
  safeParse(value: unknown): ZodSafeParseResult<Output>;
  describe(text: string): this;
  optional(): ZodOptional<this>;
  /**
   * Supplies the value used when the key is absent, and marks the key
   * optional. The runtime applies defaults on a shallow copy, so the caller's
   * argument object is never mutated.
   */
  default(value: Output): this;
}

/** Anything schema-shaped: a micro-Zod schema or a real zod schema. */
export type ZodTypeAny = ZodType<unknown, ZodTypeDef, unknown>;

/**
 * A schema marked optional; optional() tags the runtime instance.
 *
 * The _optional brand keeps ZodOptional structurally distinct from the scalar
 * builders, so InferObject can tell an optional key from a required one even
 * though every schema is structurally compatible with a wider output type.
 */
export interface ZodOptional<T extends ZodTypeAny> extends ZodType<Infer<T> | undefined, ZodTypeDef, Infer<T> | undefined> {
  readonly _optional: true;
}

export interface ZodString extends ZodType<string, ZodTypeDef, string> {}

export interface ZodNumber extends ZodType<number, ZodTypeDef, number> {
  /** Switches the JSON Schema type to integer and enforces it on parse. */
  int(): ZodNumber;
  /** Sets minimum = 1 and rejects values <= 0. */
  positive(): ZodNumber;
}

export interface ZodBoolean extends ZodType<boolean, ZodTypeDef, boolean> {}

export interface ZodArray<T extends ZodTypeAny> extends ZodType<Array<Infer<T>>, ZodTypeDef, Array<Infer<T>>> {}

export interface ZodEnum<Values extends string = string> extends ZodType<Values, ZodTypeDef, Values> {
  /** The accepted string values, exactly as passed to z.enum(). */
  readonly enum: string[];
}

export interface ZodLiteral<T extends string | number | boolean> extends ZodType<T, ZodTypeDef, T> {}

export interface ZodAny extends ZodType<unknown, ZodTypeDef, unknown> {}

export interface ZodRecord extends ZodType<Record<string, unknown>, ZodTypeDef, Record<string, unknown>> {}

/** The shape argument of z.object({...}). */
export type ZodRawShape = Record<string, ZodTypeAny>;

/**
 * An object schema. Unlike the scalar builders it also carries parse()
 * (which throws E_SCHEMA) and the materialised JSON Schema keywords.
 */
export interface ZodObject<Shape extends ZodRawShape = ZodRawShape>
  extends ZodType<InferObject<Shape>, ZodTypeDef, InferObject<Shape>> {
  readonly shape: Shape;
  readonly properties: Record<string, JSONSchema>;
  readonly required: string[];
  parse(value: unknown): InferObject<Shape>;
}

/** Keys whose schema is an optional(): omitted from the required list. */
export type OptionalKeys<Shape extends ZodRawShape> = {
  [K in keyof Shape]: Shape[K] extends { readonly _optional: true } ? K : never;
}[keyof Shape];

/** Keys whose schema is required. */
export type RequiredKeys<Shape extends ZodRawShape> = Exclude<keyof Shape, OptionalKeys<Shape>>;

/** The object type z.infer produces for a shape. */
export type InferObject<Shape extends ZodRawShape> = {
  [K in RequiredKeys<Shape>]: Infer<Shape[K]>;
} & {
  [K in OptionalKeys<Shape>]?: Infer<Shape[K]>;
};

/** The value type a schema validates to. Equivalent to zod z.infer. */
export type Infer<T extends ZodTypeAny> = T extends ZodType<infer Output, ZodTypeDef, unknown> ? Output : never;

/**
 * The z value: the micro-Zod builders. Declared as a namespace so both forms
 * work - z.string() (value) and z.infer<typeof schema> (type), matching how
 * zod itself is consumed.
 */
export interface MicroZod {
  string(): ZodString;
  number(): ZodNumber;
  boolean(): ZodBoolean;
  array<T extends ZodTypeAny>(item: T): ZodArray<T>;
  enum<Values extends string>(values: ReadonlyArray<Values>): ZodEnum<Values>;
  literal<T extends string | number | boolean>(expected: T): ZodLiteral<T>;
  any(): ZodAny;
  record(): ZodRecord;
  object<Shape extends ZodRawShape>(shape: Shape): ZodObject<Shape>;
}

/**
 * The `z` value: the micro-Zod builders.
 *
 * Declared as a const (the value) merged with a namespace (the types), so
 * both zod idioms work: `z.string()` and `z.infer<typeof schema>`.
 */
export declare const z: MicroZod;

export declare namespace z {
  /** The validated type of a schema. */
  type infer<T extends ZodTypeAny> = Infer<T>;
  /** The validated type of an object shape (what z.object returns). */
  type inferObject<Shape extends ZodRawShape> = InferObject<Shape>;
  /** Any schema-shaped value (alias of ZodTypeAny). */
  type Type = ZodTypeAny;
}

/* ================================================================== 
 * Tool plugins (Category 1)
 * ================================================================== */

/**
 * The jail ctx.fs exposes. Every path goes through the plugins.js resolvePath:
 * dot-dot and dot-slash segments, .env/.ssh/.aws/.git and anything outside cwd
 * throw E_DENIED. The handle is null when sofuu.fs is not registered.
 */
export interface JailedFs {
  readFile(path: string, options?: { encoding?: string }): Promise<string>;
  writeFile(path: string, data: string | Uint8Array, options?: { encoding?: string }): Promise<void>;
  exists(path: string): Promise<boolean>;
  readdir(path: string): Promise<string[]>;
  listDir(path: string): Promise<string[]>;
}

/** Network and filesystem intent a tool declares up front. */
export interface ToolPermissions {
  /**
   * Allowed hosts. When present the sandboxed fetch permits ONLY these
   * (exact host or a dot-suffix match) and skips the private-IP guard, so a
   * loopback tool must name its host here. When absent, every RFC1918,
   * link-local, loopback and metadata host is rejected with E_DENIED.
   */
  net?: string[];
  /**
   * Filesystem intent. workspace keeps the cwd jail; off declares that the
   * tool must not touch disk. Today the jail is always applied to ctx.fs;
   * this field documents intent for policy evaluators.
   */
  fs?: 'workspace' | 'off';
  [extension: string]: unknown;
}

/** Shield 3 configuration. Defaults: 5 failures, 30 s cool-down. */
export interface CircuitBreakerOptions {
  failureThreshold?: number;
  resetTimeoutMs?: number;
}

/** Circuit breaker states plugins.js transitions through. */
export type CircuitState = 'CLOSED' | 'OPEN' | 'HALF_OPEN';

/** Per-invocation context handed to ToolSpec.run as its second argument. */
export interface ToolContext {
  /** The tool registered name. */
  readonly name: string;
  /**
   * Sandboxed fetch (Shield 2). Applies permissions.net or the private-IP
   * guard, and inherits the watchdog AbortSignal unless the call passes its
   * own. Falls back to the host fetch only when the build has no sandboxed
   * wrapper.
   */
  fetch(input: string, init?: RequestInit): Promise<Response>;
  /** Jailed filesystem view rooted at cwd; null when sofuu.fs is absent. */
  readonly fs: JailedFs | null;
  /** Aborted when the watchdog fires (timeoutMs elapsed) or the host cancels. */
  readonly signal?: AbortSignal;
  /** Alias of signal, kept because the runtime sets both names. */
  readonly abortSignal?: AbortSignal;
  /** process.env, read-only by convention. */
  readonly secrets: Record<string, string | undefined>;
  /** process.env, same object as secrets. */
  readonly env: Record<string, string | undefined>;
  /** process.cwd() at call time; the jail root. */
  readonly cwd: string;
  /** Model name the host supplied for this run (empty when unknown). */
  readonly model: string;
  /** Mutable per-run scratch space shared by the host ({} when absent). */
  readonly state: Record<string, unknown>;
  /** Prefixed with the tool name and routed to the host logger. */
  log(...parts: unknown[]): void;
  /** Prefixed with the tool name and written to stderr. */
  error(...parts: unknown[]): void;
}

/** The ctx overrides a host may pass to Tool.run/execute. */
export interface ToolContextOverrides {
  model?: string;
  state?: Record<string, unknown>;
  log?(message: string): void;
}

/**
 * What Tool.run/execute resolves to. data carries whatever the plugin
 * returned; the runtime also passes a plugin-returned object through
 * untouched when it already has a boolean ok member.
 */
export type ToolResult<TOutput = unknown> =
  | { ok: true; source: string; data: TOutput; text: string; duration_ms: number }
  | { ok: false; source: string; error: ToolErrorInfo; text: string; duration_ms?: number };

/**
 * The spec handed to defineTool.
 *
 * TSchema drives argument validation (Shield 1) and the JSON Schema the model
 * sees. TOutput is a typing/documentation aid only: Shield 1 validates
 * inputs, and the runtime never inspects the return value.
 */
export interface ToolSpec<TSchema extends ZodTypeAny = ZodTypeAny, TOutput = unknown> {
  /** Registered tool name; also the circuit breaker key. Required, non-empty. */
  name: string;
  /** Shown to the model. The runtime also accepts the summary alias. */
  description?: string;
  /** micro-Zod object schema (or any schema-shaped value). */
  schema: TSchema;
  /**
   * Advisory: the declared result shape. The runtime does not validate
   * output yet, so this is documentation for readers and future shields.
   */
  output?: ZodTypeAny;
  /** Deadline enforced by Shield 4. Default 30000. */
  timeoutMs?: number;
  /** Forwarded to the host so it can gate the call behind approval. */
  requiresApproval?: boolean;
  /** Network/fs intent, enforced by Shield 2. */
  permissions?: ToolPermissions;
  /** Shield 3 tuning. */
  circuitBreaker?: CircuitBreakerOptions;
  /** Executes the tool. The spec object is bound as this. */
  run(args: Infer<TSchema>, ctx: ToolContext): TOutput | Promise<TOutput>;
  /** Runtime-accepted alias of run. */
  execute?(args: Infer<TSchema>, ctx: ToolContext): TOutput | Promise<TOutput>;
  /** Runtime-accepted alias of description. */
  summary?: string;
}

/**
 * The object defineTool returns (and registers in sofuu.plugins.TOOLS and
 * sofuu.tools.TOOLS).
 */
export interface Tool<TSchema extends ZodTypeAny = ZodTypeAny, TOutput = unknown> {
  readonly name: string;
  readonly description: string;
  /** Materialised JSON Schema (zodToJSONSchema of schema). */
  readonly schema: JSONSchema;
  /** Same object as schema; the name the model-side tool loop reads. */
  readonly parameters: JSONSchema;
  readonly timeoutMs: number;
  readonly requiresApproval?: boolean;
  readonly permissions: ToolPermissions;
  readonly category: 'tool';
  /** Runs the 5 shields, then the plugin body. */
  run(args: Infer<TSchema>, ctx?: ToolContextOverrides): Promise<ToolResult<TOutput>>;
  /** Same function as run. */
  execute(args: Infer<TSchema>, ctx?: ToolContextOverrides): Promise<ToolResult<TOutput>>;
}

/* ================================================================== 
 * Inference providers (Category 2)
 * ================================================================== */

/** One chat turn, in the shape every provider adapter receives. */
export interface ProviderMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string;
  /** Tool name, for role tool. */
  name?: string;
  /** Correlates a tool result with the call that produced it. */
  toolCallId?: string;
  /** Assistant turns may carry the calls the model requested. */
  toolCalls?: ProviderToolCall[];
  [extension: string]: unknown;
}

/** A model-requested tool invocation. */
export interface ProviderToolCall {
  id?: string;
  name: string;
  /** JSON-encoded arguments as the model emitted them. */
  arguments: string | Record<string, unknown>;
}

/** A tool the model may call, in JSON Schema form. */
export interface ProviderToolDef {
  name: string;
  description?: string;
  parameters?: JSONSchema;
}

/** Token accounting for one completion. */
export interface ProviderUsage {
  promptTokens: number;
  completionTokens: number;
  totalTokens?: number;
}

/** One request as the host hands it to ProviderSpec.complete/stream. */
export interface ProviderRequest {
  messages: ProviderMessage[];
  model?: string;
  /** True when the host expects incremental events. */
  stream?: boolean;
  tools?: ProviderToolDef[];
  temperature?: number;
  maxTokens?: number;
  /** Convenience: the system prompt, when the host split it out. */
  system?: string;
  /** Aborted when the turn is cancelled. */
  signal?: AbortSignal;
  [extension: string]: unknown;
}

/** What a completion resolves to. */
export interface ProviderResult {
  text: string;
  toolCalls?: ProviderToolCall[];
  usage?: ProviderUsage;
  model?: string;
  finishReason?: string;
  [extension: string]: unknown;
}

/**
 * One streamed increment. type is open: adapters emit their own kinds and the
 * host forwards unknown ones to listeners.
 */
export interface ProviderEvent {
  type: 'text' | 'thinking' | 'tool_call' | 'usage' | 'done' | 'error' | (string & {});
  text?: string;
  toolCall?: ProviderToolCall;
  usage?: ProviderUsage;
  error?: ToolErrorInfo;
  [extension: string]: unknown;
}

/** Callback a streaming provider pushes increments into. */
export type ProviderEventSink = (event: ProviderEvent) => void;

/** Context handed to a provider adapter (same env and logs as a tool). */
export interface ProviderContext {
  readonly name: string;
  fetch(input: string, init?: RequestInit): Promise<Response>;
  readonly env: Record<string, string | undefined>;
  readonly cwd: string;
  readonly signal?: AbortSignal;
  log(...parts: unknown[]): void;
  error(...parts: unknown[]): void;
}

/** Result of an adapter-defined health probe; the runtime never calls it. */
export interface ProviderHealth {
  ok: boolean;
  models?: string[];
  version?: string;
  latencyMs?: number;
  error?: string;
}

/**
 * The spec handed to defineProvider. defineProvider registers it in
 * sofuu.plugins.PROVIDERS and returns the same object tagged category
 * provider.
 *
 * Wiring an adapter into the CLI is separate: sofuu -p NAME only accepts
 * known wire names or names saved with /provider, and a custom endpoint is
 * reached with --base-url.
 */
export interface ProviderSpec {
  name: string;
  /** Which wire dialect the adapter speaks. */
  kind: 'ollama' | 'llamacpp' | 'openai' | 'custom';
  /** Endpoint root, e.g. http://127.0.0.1:11434. */
  baseUrl?: string;
  /** Default model for this adapter. */
  model?: string;
  description?: string;
  /** Non-streaming completion; a bare string is also accepted. */
  complete(req: ProviderRequest, ctx: ProviderContext): Promise<ProviderResult> | Promise<string>;
  /** Streaming completion; push increments into onEvent, then resolve. */
  stream?(req: ProviderRequest, ctx: ProviderContext, onEvent: ProviderEventSink): Promise<ProviderResult>;
  /**
   * Adapter-defined probe (e.g. GET /api/tags). The runtime does not call
   * it; it rides along on the registered object for CLI use and scripts.
   */
  health?(): Promise<ProviderHealth>;
  [extension: string]: unknown;
}

/** The registered provider object. */
export interface Provider extends ProviderSpec {
  readonly category: 'provider';
}

/* ================================================================== 
 * Storage and vector stores (Category 3)
 * ================================================================== */

/**
 * Configuration handed to StorageSpec.init. Nothing in the shipped drivers
 * constructs one yet; adapters should treat every field as optional.
 */
export interface StorageInit {
  path?: string;
  password?: string;
  [extension: string]: unknown;
}

/**
 * A key-value storage adapter.
 *
 * Honest status: defineStorage registers the adapter in sofuu.plugins.STORES
 * and 'sofuu plugin list' reports it, but no shipped driver calls these
 * methods yet - the local QTSQ store (sofuu.storage) backs sessions and
 * memory today. Implement the contract so a future mirror/dual-write path
 * can adopt it without a rewrite.
 */
export interface StorageSpec {
  name: string;
  description?: string;
  init?(options?: StorageInit): Promise<void> | void;
  get?(key: string): Promise<unknown> | unknown;
  set?(key: string, value: unknown): Promise<void> | void;
  delete?(key: string): Promise<boolean> | boolean;
  keys?(): Promise<string[]> | string[];
  [extension: string]: unknown;
}

/** The registered storage object. */
export interface Storage extends StorageSpec {
  readonly category: 'store';
}

/** One record in a vector store. */
export interface VectorRecord {
  id: string;
  vector: number[];
  text?: string;
  metadata?: Record<string, unknown>;
}

/** A similarity query. */
export interface VectorQuery {
  vector: number[];
  /** Maximum hits to return. */
  limit?: number;
  /** Metadata equality filter, applied by the adapter. */
  filter?: Record<string, unknown>;
  [extension: string]: unknown;
}

/** One similarity hit. */
export interface VectorQueryResult {
  id: string;
  /** Higher is closer; adapters normalise to 0..1 where they can. */
  score: number;
  text?: string;
  metadata?: Record<string, unknown>;
}

/**
 * A semantic vector store adapter. upsert/query are the members the discovery
 * duck-type looks for (plugins.js checks upsert/query).
 */
export interface VectorStoreSpec {
  name: string;
  description?: string;
  init?(options?: StorageInit): Promise<void> | void;
  upsert(record: VectorRecord | VectorRecord[]): Promise<void> | void;
  query(query: VectorQuery): Promise<VectorQueryResult[]>;
  delete?(id: string): Promise<boolean> | boolean;
  count?(): Promise<number> | number;
  [extension: string]: unknown;
}

/** The registered vector store object. */
export interface VectorStore extends VectorStoreSpec {
  readonly category: 'store';
}

/* ================================================================== 
 * Lifecycle services (Category 4)
 * ================================================================== */

/** Emitted at the start of one agent turn. */
export interface TurnEvent {
  turnId: string;
  prompt: string;
  agent: string;
  /** 0 for the top-level agent, 1+ for a sub-agent. */
  depth: number;
  [extension: string]: unknown;
}

/** Emitted when the model requests a tool call. */
export interface ToolCallEvent {
  tool: string;
  args: Record<string, unknown>;
  agent?: string;
  turnId?: string;
}

/** Emitted after a tool call returns. */
export interface ToolResultEvent {
  tool: string;
  result: unknown;
  /** Milliseconds the tool took, when the host measured it. */
  durationMs?: number;
  agent?: string;
  turnId?: string;
}

/** Emitted at the end of one turn, with the response and its cost. */
export interface TurnEndEvent {
  response: string;
  usage?: ProviderUsage;
  agent?: string;
  turnId?: string;
  [extension: string]: unknown;
}

/** Context handed to onStart/onStop. */
export interface ServiceContext {
  readonly env: Record<string, string | undefined>;
  log(...parts: unknown[]): void;
  [extension: string]: unknown;
}

/**
 * The lifecycle hooks a service may implement.
 *
 * Dispatch status: the generated Service wrapper (plugins.js) forwards
 * start/stop and emits turnStart/turnEnd via sofuu.plugins.emitTurnStart and
 * emitTurnEnd. onToolCall/onToolResult are declared for hosts that emit tool
 * events; if your host never emits them the hooks simply never fire.
 *
 * A hook that throws is caught and logged by the wrapper for turnStart and
 * turnEnd; onStart/onStop propagate so a failed start is visible.
 */
export interface ServiceSpec {
  name: string;
  description?: string;
  onStart?(ctx: ServiceContext): Promise<void> | void;
  /** May rewrite event.prompt in place before the turn is dispatched. */
  onTurnStart?(event: TurnEvent): Promise<void> | void;
  onToolCall?(event: ToolCallEvent): Promise<void> | void;
  onToolResult?(event: ToolResultEvent): Promise<void> | void;
  onTurnEnd?(event: TurnEndEvent): Promise<void> | void;
  onStop?(ctx: ServiceContext): Promise<void> | void;
  [extension: string]: unknown;
}

/** The registered service object. */
export interface Service {
  readonly name: string;
  readonly description: string;
  readonly category: 'service';
  /** The original spec, so hooks can reach their own state. */
  readonly spec: ServiceSpec;
  readonly started: boolean;
  start(ctx?: ServiceContext): Promise<void>;
  stop(ctx?: ServiceContext): Promise<void>;
  turnStart(event: TurnEvent): Promise<void>;
  turnEnd(event: TurnEndEvent): Promise<void>;
}

/* ================================================================== 
 * Protocol bridging (Category 5)
 * ================================================================== */

/** Options accepted by useOpenAPI. */
export interface OpenAPIOptions {
  /** Prefix every generated tool name (e.g. github -> github_list_issues). */
  namespace?: string;
  /** Endpoint root; else servers[0].url, the spec URL origin, or localhost. */
  baseUrl?: string;
  /** Extra headers sent with every generated call. */
  headers?: Record<string, string>;
  /** Keep only the operations this predicate accepts. */
  filter?(operation: Record<string, unknown>): boolean;
}

/* ================================================================== 
 * The runtime namespaces plugins can touch
 * ================================================================== */

/** Chunk yielded by sofuu.ai.stream. */
export interface AiStreamChunk {
  text?: string;
  think?: string;
}

/** Async stream with a cancel handle (agent.js aborts it on cancel). */
export interface AiStream extends AsyncIterable<AiStreamChunk> {
  abort(): void;
}

/** Options for sofuu.ai.complete and sofuu.ai.stream. */
export interface AiCompleteOptions {
  messages: ProviderMessage[];
  provider?: string;
  model?: string;
  tools?: ProviderToolDef[];
  temperature?: number;
  max_tokens?: number;
  api_key?: string;
  base_url?: string;
  profile?: string;
  effort?: 'off' | 'low' | 'medium' | 'high' | 'max';
  ctx_window?: number;
  [extension: string]: unknown;
}

/** What sofuu.ai.complete resolves to. */
export interface AiCompleteResult {
  text?: string;
  toolCalls?: ProviderToolCall[];
  usage?: ProviderUsage;
  model?: string;
  [extension: string]: unknown;
}

/**
 * The inference surface. Model capability lookups return JSON strings (the
 * drivers JSON.parse them), so they are typed as string, not object.
 */
export interface SofuuAi {
  complete(options: AiCompleteOptions): Promise<AiCompleteResult>;
  stream(options: AiCompleteOptions): AiStream;
  /** Approximate token count using the runtime estimator. */
  estimateTokens(text: string): number;
  /** JSON string of the built-in capability table for a model. */
  modelCaps(model: string, baseUrl?: string): string;
  /** JSON string of the resolved capability ladder for a model. */
  resolveCaps(model: string, baseUrl?: string): string;
  /** Remote embedding; a flat number[] or a one-row number[][]. */
  embed(text: string, options?: { provider?: string; model?: string }): Promise<number[] | number[][]>;
  /** Offline trigram-hash embedding. */
  embedLocal(text: string): number[];
  /** Offline learned projector embedding. */
  embedLocalSemantic(text: string): number[];
  /** JSON string describing the embedding backends and dimensions. */
  embedInfo(): string;
  [extension: string]: unknown;
}

/** One entry in sofuu.tools.TOOLS. */
export interface ToolDescriptor {
  name: string;
  description?: string;
  parameters?: JSONSchema;
  execute?(args: unknown, ctx?: unknown): Promise<unknown>;
  source?: string;
  selfTimed?: boolean;
  requiresApproval?: boolean;
  [extension: string]: unknown;
}

/** Outcome of sofuu.tools.validateArgs. */
export interface ValidateArgsResult {
  ok: boolean;
  error?: string;
  path?: string;
  data?: unknown;
  [extension: string]: unknown;
}

/** The tool registry shared by built-in and plugin tools. */
export interface SofuuTools {
  TOOLS: Record<string, ToolDescriptor>;
  /** Names the agent loop exposes to the model. */
  GROUP: string[];
  ENABLED?: string[];
  /** Domain toolpacks: coding / web / memory / system. */
  PACKS?: Record<string, string[]>;
  VERSION?: string;
  list?(): ToolDescriptor[];
  describe?(name: string): unknown;
  validateArgs?(schema: JSONSchema, args: unknown): ValidateArgsResult;
  execute?(name: string, args: unknown, options?: unknown): Promise<unknown>;
  /** Normalises an MCP tool result into the agent text/data envelope. */
  mcpResult?(result: unknown, options?: { maxChars?: number }): unknown;
  endRunScope?(runId: string): void;
  killActive?(): void;
  [extension: string]: unknown;
}

/** A handle on one on-device memory container. */
export interface MemoryHandle {
  remember(vector: number[], text: string, role?: string, turn?: number): number;
  recall(query: unknown): unknown;
  search(query: unknown): unknown;
  count?(): number;
  close?(): void;
  [extension: string]: unknown;
}

/** The on-device vector memory surface. */
export interface SofuuMemory {
  /** Opens (or creates) a .qtsq container. */
  open(path: string, dim?: number, backend?: string): MemoryHandle | null;
  remember?(input: unknown): unknown;
  recall?(input: unknown): unknown;
  search?(input: unknown): unknown;
  learn?(input: unknown): unknown;
  inspect?(input: unknown): unknown;
  [extension: string]: unknown;
}

/** A connected MCP client. */
export interface McpClient {
  listTools?(): Promise<unknown>;
  callTool?(name: string, args?: Record<string, unknown>): Promise<unknown>;
  close?(): void | Promise<void>;
  [extension: string]: unknown;
}

/** MCP ingestion. */
export interface SofuuMcp {
  connect(spec: Record<string, unknown>, options?: { server?: string }): Promise<McpClient>;
  loadConfig?(options?: unknown): unknown;
  [extension: string]: unknown;
}

/** The policy evaluator (Shield 2 reads the deny decision). */
export interface SofuuPolicy {
  /** JSON string: { decision: 'allow' | 'deny', reason?: string }. */
  evaluate(request: Record<string, unknown>): string;
  get?(): string;
  set?(json: string): boolean;
  load?(file: string): boolean;
  secret?(name: string): string | undefined;
  [extension: string]: unknown;
}

/** The filesystem driver, adding the host primitives the jail wraps. */
export interface SofuuFs extends JailedFs {
  appendFile(path: string, data: string): Promise<void>;
  mkdir(path: string, options?: { recursive?: boolean }): Promise<void>;
  realpath(path: string): Promise<string>;
  chmod?(path: string, mode: number): Promise<void>;
  rm?(path: string, options?: { recursive?: boolean; force?: boolean }): Promise<void>;
  isSymlink?(path: string): Promise<boolean>;
}

/** Query options for sofuu.storage. */
export interface StorageQueryOptions {
  filterCol?: string;
  filterVal?: string;
  limit?: number;
  offset?: number;
  orderDesc?: boolean;
  minTimestamp?: number;
  maxTimestamp?: number;
  [extension: string]: unknown;
}

/** One table/collection in the structured store. */
export interface DatabaseTable {
  readonly name: string;
  insert(row: Record<string, unknown>): string | number;
  query(options?: StorageQueryOptions): Promise<Array<Record<string, unknown>>>;
  count(): number;
}

/** Key-value facade over the structured store. */
export interface StorageKeyValue {
  set(key: string, value: unknown): Promise<boolean>;
  get(key: string): Promise<unknown>;
  del(key: string): Promise<boolean>;
}

/** An open structured store (src/js/storage.js). */
export interface SofuuDatabase {
  readonly path: string;
  readonly messages: DatabaseTable;
  readonly traces: DatabaseTable;
  readonly kv: StorageKeyValue;
  insert(table: string, row: Record<string, unknown>): string | number;
  query(table: string, options?: StorageQueryOptions): Promise<Array<Record<string, unknown>>>;
  count(table: string): number;
  table(name: string): DatabaseTable;
  collection(name: string): DatabaseTable;
  flush(): boolean;
  close(): void;
  readonly activeSegment: string | null;
}

/** The structured store factory. */
export interface SofuuStorage {
  open(path: string, options?: { password?: string }): SofuuDatabase;
  Database: new (options?: string | { path?: string; file?: string; password?: string }) => SofuuDatabase;
}

/** A registered circuit breaker. */
export interface CircuitRecord {
  state: CircuitState;
  failures: number;
  lastFailure: number;
  lastTripped: number;
  failureThreshold: number;
  resetTimeoutMs: number;
}

/** One row of 'sofuu plugin list'. */
export interface PluginListing {
  name: string;
  category: 'tool' | 'provider' | 'store' | 'service';
  circuit: CircuitState;
  timeoutMs: number;
  description: string;
  requiresApproval?: boolean | null;
}

/** Result of one directory discovery pass. */
export interface DiscoveryResult {
  loaded: Array<{ file: string; items: string[] }>;
  broken: Array<{ file: string; error: string }>;
}

/** Result of the three-way discovery (local, global, manifest). */
export interface DiscoveryReport {
  local: DiscoveryResult | null;
  global: DiscoveryResult | null;
  manifest: { loaded: number; error?: string } | null;
}

/** Counters for one write-behind mirror queue (vector store or storage). */
export interface MirrorQueueStats {
  queued: number;
  written: number;
  failed: number;
  lastError: string;
  enqueued: number;
  batch: number;
  flushMs: number;
}

/** Result of one `sofuu.plugins.ingestMcp()` call. */
export type IngestMcpResult =
  | { ok: true; server: string; tools: string[]; count: number; source: 'mcp' }
  | { ok: false; error: { code: SofuuErrorCode; message: string } };

/** What this engine build can actually do — feature detection for plugins. */
export interface PluginCapabilities {
  categories: {
    tool: boolean;
    provider: boolean;
    store: boolean;
    service: boolean;
    openapi: boolean;
    mcp: boolean;
  };
  shields: string[];
  zod: boolean;
  [extra: string]: unknown;
}

/** A provider as reported by listProviders() (no callables). */
export interface ProviderListing {
  name: string;
  kind: 'ollama' | 'llamacpp' | 'openai' | 'custom' | string;
  baseUrl: string;
  model: string;
  description: string;
  hasStream: boolean;
  circuit: CircuitState;
}

/**
 * The plugin engine (sofuu.plugins). Every member is callable from a plugin
 * at runtime; the CLI helpers are also the entry points main.rs evaluates.
 */
export interface SofuuPlugins {
  readonly VERSION: string;
  readonly _loaded: true;
  TOOLS: Record<string, Tool>;
  PROVIDERS: Record<string, Provider>;
  STORES: Record<string, Storage | VectorStore>;
  SERVICES: Record<string, Service>;
  CIRCUITS: Record<string, CircuitRecord>;
  defineTool: SofuuSdk['defineTool'];
  defineProvider: SofuuSdk['defineProvider'];
  defineStorage: SofuuSdk['defineStorage'];
  defineVectorStore: SofuuSdk['defineVectorStore'];
  defineService: SofuuSdk['defineService'];
  useOpenAPI: SofuuSdk['useOpenAPI'];
  emitTurnStart(event: TurnEvent): Promise<void>;
  emitTurnEnd(event: TurnEndEvent): Promise<void>;
  discover(dir?: string): Promise<DiscoveryResult>;
  loadManifest(manifestPath?: string): Promise<{ loaded: number; error?: string }>;
  discoverAll(): Promise<DiscoveryReport>;
  list(): PluginListing[];
  invoke(name: string, args?: Record<string, unknown>, ctx?: ToolContextOverrides): Promise<ToolResult>;
  cliList(wantJson?: boolean): Promise<void>;
  cliValidate(file: string): Promise<void>;
  cliRun(toolName: string, rawArgs?: string): Promise<void>;
  /**
   * Isolated single-tool execution for 'sofuu plugin test'. Optional because
   * it ships with the plugin-driver refresh: the CLI falls back to cliRun
   * when the loaded bytecode does not define it yet.
   */
  cliTest?(toolName: string, rawArgs?: string): Promise<void>;
  cliInit(name: string, category?: 'tool' | 'service' | 'provider' | 'store' | 'vectorstore' | 'openapi'): Promise<void>;

  /* ── provider runtime (category 2) ──────────────────────────────────── */
  registerProvider(spec: ProviderSpec): ProviderSpec & Provider;
  getProvider(name: string): Provider | null;
  /** Alias of getProvider(); resolves by exact, then case- and
   *  punctuation-insensitive name, then sofuu.providers. */
  resolveProvider(name: string): Provider | null;
  listProviders(): ProviderListing[];
  /** Run one completion through a registered provider adapter, with the
   *  provider's own breaker and watchdog applied. */
  completeVia(name: string, req: ProviderRequest, ctx?: ProviderContext): Promise<ToolResult<ProviderResult>>;
  /** Buffered replay of a streaming provider: the adapter emits deltas in
   *  order and one terminal done event (no incremental host delivery). */
  streamVia(
    name: string,
    req: ProviderRequest,
    ctx?: ProviderContext,
    onEvent?: ProviderEventSink,
  ): Promise<ToolResult<ProviderResult & { buffered?: boolean; streamed?: boolean }>>;

  /* ── storage + dual-write mirror (category 3) ────────────────────────── */
  /** Mirror every QTSQ memory write into a registered vector store.
   *  Local writes stay synchronous; the remote write is write-behind. */
  attachVectorStore(
    brain: MemoryHandle,
    storeName: string,
    opts?: { namespace?: string; sync?: boolean; batch?: number; flushMs?: number },
  ): unknown;
  attachStorage(
    storeName: string,
    opts?: { namespace?: string; sync?: boolean; batch?: number; flushMs?: number },
  ): unknown;
  mirrorStats(): Record<string, MirrorQueueStats>;
  flushMirror(): Promise<Record<string, MirrorQueueStats>>;

  /* ── service hooks (category 4) ──────────────────────────────────────── */
  /** Called by the agent loop before a tool runs. A service that throws
   *  blocks the call (E_DENIED / E_TOOL travels back as a tool result). */
  emitToolCall(event: ToolCallEvent): Promise<unknown>;
  emitToolResult(event: ToolResultEvent): Promise<unknown>;
  startServices(ctx?: ServiceContext): Promise<unknown[]>;
  stopServices(ctx?: ServiceContext): Promise<unknown[]>;

  /* ── MCP ingestion (category 5b) ─────────────────────────────────────── */
  ingestMcp(server: string, opts?: {
    /** stdio transport. */
    command?: string;
    args?: string[];
    env?: Record<string, string>;
    cwd?: string;
    /** HTTP/SSE transport (mutually exclusive with command). */
    url?: string;
    transport?: 'auto' | 'http' | 'stdio';
    /** Prefix tool names with the server name (default true). */
    prefix?: boolean;
    timeoutMs?: number;
    requiresApproval?: string;
  }): Promise<IngestMcpResult>;

  /* ── introspection and circuit control ───────────────────────────────── */
  capabilities(): PluginCapabilities;
  /** Reset one breaker, or every breaker when called with no argument. */
  resetCircuit(name?: string): true;
  circuitState(name: string): CircuitRecord & { grown?: number; trialInFlight?: boolean };

  /* ── internals (tooling only; not part of the authoring surface) ─────── */
  readonly modules: readonly string[];
  /** The textual TypeScript -> JavaScript stripper used by discovery. */
  _stripTs?(src: string): string;
  _yaml?: { parse(src: string): unknown };
}

/* ================================================================== 
 * Observability (rt/observability.rs + src/js/observability.js)
 * ================================================================== */

/**
 * The terminal-failure policy a run uses when something throws
 * (OBSERVABILITY-SPEC §3). "interactive" prompts [r]etry/[s]kip/[i]nspect/
 * [h]alt on a TTY and falls back to "fail-stop" when there is none.
 */
export type OnErrorMode = 'auto-retry' | 'fail-stop' | 'interactive';

/**
 * One classified failure: the taxonomy code the agent loop acts on, plus the
 * operation that produced it. classifyError never throws - an unknown throw
 * comes back as E_INTERNAL carrying the original message.
 */
export interface SofuuClassifiedError {
  /** A code from the normative taxonomy; never invented by the classifier. */
  code: ErrorCodeLike;
  /** True when retrying the same operation is worth another attempt. */
  retryable?: boolean;
  /** What failed: provider, tool:<name>, memory, ... */
  op?: string;
  message: string;
  /** The classified value, when it was not already an Error. */
  cause?: unknown;
}

/**
 * The agent driver's error-policy surface (src/js/agent.js).
 *
 * Advisory: run() already accepts opts.onError, and the two helpers below
 * ship with the agent driver's observability refresh (OBSERVABILITY-SPEC §3)
 * - until then they are absent, which is why they are declared optional.
 */
export interface SofuuAgent {
  classifyError?(err: unknown): SofuuClassifiedError;
  /** The mode a run defaults to: interactive on a TTY, auto-retry otherwise. */
  onErrorMode?(opts?: { tty?: boolean }): OnErrorMode;
  [extension: string]: unknown;
}

/** One row of sofuu.observability.runs(). */
export interface ObservabilityRun {
  run_id: string;
  /** .sofuu/logs/agent-<run_id>.jsonl, or a session's events.ndjson. */
  path: string;
  /** agent = a headless agent run, session = a chat session. */
  kind: 'agent' | 'session' | (string & {});
  bytes: number;
  /** Epoch seconds or milliseconds, whichever the host reports. */
  mtime: number;
  steps: number;
  /** OBSERVABILITY-SPEC §5: OK | HALTED | RETRYING | RUNNING. */
  status: string;
}

/** The rehydration unit a resume seeds the loop from (OBSERVABILITY-SPEC §3). */
export interface AgentCheckpoint {
  step: number;
  messages: unknown[];
  task: string;
  model?: string;
  provider?: string;
  tokens?: number;
  last_tool?: string;
}

/** sofuu.observability.lastCheckpoint(...), parsed. */
export interface LastCheckpointResult {
  ok: boolean;
  found: boolean;
  step?: number;
  seq?: number;
  ts?: string;
  checkpoint?: AgentCheckpoint;
}

/** The post-mortem card sofuu.observability.postmortem writes. */
export interface CrashCard {
  crash_id: string;
  run_id: string;
  timestamp: string;
  failed_step?: number;
  error: { code?: string; message?: string; details?: unknown; stack?: string };
  context_snapshot?: {
    model?: string;
    provider?: string;
    context_tokens?: number;
    last_user_message?: string;
    last_tool?: string;
  };
  recovery_hint?: string;
  trace_path?: string;
}

/**
 * The observability surface.
 *
 * The four host functions (runs / lastCheckpoint / postmortem / crashCard)
 * take ONE JSON string and answer with JSON TEXT, the convention
 * sofuu.events.append and sofuu.policy.evaluate already use - parse the
 * answer, or go through the renderer. cli/renderRun are the shipped JS
 * renderer (src/js/observability.js); both are pure formatting, so neither
 * writes or validates anything.
 */
export interface SofuuObservability {
  /** { ok, runs: ObservabilityRun[] } as JSON text. Args: { project?, path? }. */
  runs(args?: string | Record<string, unknown>): string;
  /** { ok, found, step, seq, ts, checkpoint } as JSON text. */
  lastCheckpoint(args?: string | Record<string, unknown>): string;
  /** Writes .sofuu/crashes/<crash_id>.json: { ok, crash_id, path, card }. */
  postmortem(args?: string | Record<string, unknown>): string;
  /** The newest crash card as JSON text: { ok, card }. */
  crashCard(args?: string | Record<string, unknown>): string;
  /**
   * The `sofuu logs` renderer. argv[0] is the subcommand: list |
   * show <run_id> | follow [run_id] | last-error. Resolves to the process
   * exit code (0 ok, 1 the named run is missing or unreadable, 2 usage).
   * Prints through the host console and never throws.
   */
  cli(argv: string[]): Promise<number>;
  /**
   * One run rendered as a fixed-width trace (clock, kind, step, payload
   * summary, duration). Accepts envelope objects or raw JSONL text.
   */
  renderRun(events: unknown[] | string): string;
  readonly VERSION?: string;
}

/**
 * The Sofuu runtime as a plugin sees it. Only namespaces the drivers actually
 * install are listed; anything else is undefined at runtime.
 */
export interface SofuuRuntime {
  plugins: SofuuPlugins;
  ai: SofuuAi;
  tools: SofuuTools;
  memory: SofuuMemory;
  /** Present when the MCP driver is loaded. */
  mcp?: SofuuMcp;
  fs: SofuuFs;
  /** Present when a policy file has been loaded or the driver is built in. */
  policy?: SofuuPolicy;
  /** The QTSQ structured store (src/js/storage.js). */
  storage?: SofuuStorage;
  /** The agent runtime (src/js/agent.js) and its error policy. */
  agent?: SofuuAgent;
  /** Run logs, checkpoints and crash cards (src/js/observability.js). */
  observability?: SofuuObservability;
  /** The deterministic in-process execution sandbox (rt/sandbox.rs). */
  sandbox: SofuuSandbox;
  /** Runtime semver, e.g. 0.3.0-beta. */
  version: string;
}

/* ================================================================== 
 * sofuu.sandbox — in-process deterministic execution (SANDBOX_UPGRADE_PLAN)
 * ================================================================== */

/**
 * Machine-readable sandbox failure codes carried on a failed result (`code`)
 * and on the Error thrown by `eval` / `throwOnError: true`.
 *   E_TIMEOUT  the wall-clock budget ran out (also `SandboxTimeoutError`)
 *   E_FUEL     the instruction meter ran out
 *   E_OOM      the memory ceiling was hit
 *   E_STACK    the call stack ceiling was hit
 *   E_SYNTAX   the source did not parse (after the TypeScript retry)
 *   E_GLOBALS  the `globals` payload could not be injected
 *   E_BUSY     a stateful session still has a run parked on an async bridge
 *   E_BRIDGE   a run parked on a host bridge that can never be resumed here
 *   E_SANDBOX_CLOSED  the session was closed
 *   E_SANDBOX  any other sandbox failure
 */
export type SandboxErrorCode =
  | 'E_TIMEOUT'
  | 'E_FUEL'
  | 'E_OOM'
  | 'E_STACK'
  | 'E_SYNTAX'
  | 'E_GLOBALS'
  | 'E_BUSY'
  | 'E_BRIDGE'
  | 'E_SANDBOX_CLOSED'
  | 'E_SANDBOX';

/** Where captured `console.*` output goes. */
export type SandboxConsoleMode = 'capture' | 'inherit' | 'ignore';

/** Presets from the plan table; explicit options always override them. */
export type SandboxPreset = 'strict' | 'standard' | 'compute' | 'worker';

/** TypeScript ingestion: 'auto' parses as JS first and retries stripped. */
export type SandboxTsMode = 'auto' | 'always' | 'never' | true | false;

/** Console levels passed to a custom `console` callback. */
export type SandboxConsoleLevel = 'log' | 'info' | 'warn' | 'error' | 'debug';

export interface SandboxOptions {
  /** strict | standard | compute | worker — see docs/SANDBOX.md. */
  preset?: SandboxPreset;
  /**
   * Instruction budget. Default 5,000,000. `'unlimited'`, `0` or `null`
   * disables the meter entirely. There is no upper clamp.
   */
  fuel?: number | 'unlimited' | null;
  /**
   * Wall-clock budget in ms. Default 2000. `0` or `null` disables the
   * deadline. There is no upper clamp.
   */
  timeoutMs?: number | null;
  /** Memory ceiling in MB. Default 64. No upper clamp. */
  maxMemoryMB?: number;
  /** Call-stack ceiling in KB. Default 512. No upper clamp. */
  maxStackKB?: number;
  /**
   * Console routing. Default 'capture' (into `result.logs`). A function
   * receives (level, ...stringifiedArgs) and runs on the host.
   */
  console?: SandboxConsoleMode | ((level: SandboxConsoleLevel, ...args: string[]) => void);
  /** 0.2 alias: false is the same as console: 'ignore'. */
  captureConsole?: boolean;
  /** Max captured log lines. Default 512; 0 = unbounded (8 MB byte cap). */
  maxLogs?: number;
  /** Freeze Object/Array/Function/... and their prototypes at realm creation. */
  freezePrototypes?: boolean;
  /** TypeScript ingestion. Default 'auto'. */
  ts?: SandboxTsMode;
  /** Transpile JSX to `h()` calls and install `h`/`Fragment` in the realm. */
  jsx?: boolean;
  /** Pump the returned promise before returning. Default true. */
  allowAsync?: boolean;
  /** Throw (or reject) instead of returning a failed result object. */
  throwOnError?: boolean;
  /**
   * Values installed as globals in the realm. A top-level *function* is a
   * callable host bridge: it runs on the host, its arguments and result cross
   * the boundary as JSON, and a promise-returning function is awaited.
   */
  globals?: { [name: string]: unknown };
}

export interface SandboxResult<T = unknown> {
  success: boolean;
  /** JSON round-trip of the value the script produced (undefined when none). */
  result?: T;
  error?: string;
  /** 'timeout' | 'fuel' | 'oom' | 'stack' | 'syntax' | 'runtime' | ... */
  errorType?: string;
  code?: SandboxErrorCode;
  interrupted: boolean;
  fuelUsed: number;
  elapsedMs: number;
  logs: string[];
}

/**
 * A stateful sandbox realm. Declarations persist across `run` calls, except
 * that code using top-level `await` executes inside an async wrapper (see
 * docs/SANDBOX.md) and therefore returns a value only via `return`.
 */
export interface SandboxSession {
  /**
   * A result object normally; a Promise only when the run parked on a
   * promise-returning host bridge. `await` handles both shapes.
   */
  run<T = unknown>(code: string, options?: SandboxOptions): SandboxResult<T> | Promise<SandboxResult<T>>;
  /** The value itself (or a Promise of it); throws on failure. */
  eval<T = unknown>(code: string, options?: SandboxOptions): T | Promise<T>;
  /** Close the realm. A parked run defers teardown to its completion. */
  close(): void;
}

export interface SofuuSandbox {
  /** One-shot realm: create, run, destroy. */
  run<T = unknown>(code: string, options?: SandboxOptions): SandboxResult<T> | Promise<SandboxResult<T>>;
  /** One-shot realm returning the value itself; throws on failure. */
  eval<T = unknown>(code: string, options?: SandboxOptions): T | Promise<T>;
  /** Stateful realm for multi-step work. */
  create(options?: SandboxOptions): SandboxSession;
}

/* ================================================================== 
 * The SDK factories
 * ================================================================== */

/** Shape of the five define* factories plus useOpenAPI. */
export interface SofuuSdk {
  defineTool<TSchema extends ZodTypeAny, TOutput = unknown>(spec: ToolSpec<TSchema, TOutput>): Tool<TSchema, TOutput>;
  defineProvider<TSpec extends ProviderSpec>(spec: TSpec): TSpec & Provider;
  defineStorage<TSpec extends StorageSpec>(spec: TSpec): TSpec & Storage;
  defineVectorStore<TSpec extends VectorStoreSpec>(spec: TSpec): TSpec & VectorStore;
  defineService<TSpec extends ServiceSpec>(spec: TSpec): Service;
  /**
   * Auto-bridges an OpenAPI 3.x document into 5-Shield tools. Reached as
   * sofuu.plugins.useOpenAPI (it is not a global), and re-exported here so
   * plugin files can import it.
   */
  useOpenAPI(spec: string | Record<string, unknown>, options?: OpenAPIOptions): Promise<Array<Tool>>;
}

/* ================================================================== 
 * Module exports (what 'import { ... } from sofuu' binds)
 * ================================================================== */

/** Action-plugin factory. */
export declare const defineTool: SofuuSdk['defineTool'];
/** Inference-adapter factory. */
export declare const defineProvider: SofuuSdk['defineProvider'];
/** Storage-adapter factory. */
export declare const defineStorage: SofuuSdk['defineStorage'];
/** Vector-store factory. */
export declare const defineVectorStore: SofuuSdk['defineVectorStore'];
/** Lifecycle-service factory. */
export declare const defineService: SofuuSdk['defineService'];
/** OpenAPI bridge (sofuu.plugins.useOpenAPI). */
export declare const useOpenAPI: SofuuSdk['useOpenAPI'];

/* ================================================================== 
 * Globals the runtime installs
 * ================================================================== */

declare global {
  /** The runtime namespace. */
  const sofuu: SofuuRuntime;
  /** The micro-Zod builders registered by src/js/plugins.js. */
  const z: MicroZod;
  /** Type side of the global z, so global-only files can still write z.infer. */
  namespace z {
    type infer<T extends ZodTypeAny> = Infer<T>;
  }
  /** Global aliases installed by plugins.js (usable without an import). */
  const defineTool: SofuuSdk['defineTool'];
  const defineProvider: SofuuSdk['defineProvider'];
  const defineStorage: SofuuSdk['defineStorage'];
  const defineVectorStore: SofuuSdk['defineVectorStore'];
  const defineService: SofuuSdk['defineService'];
}


