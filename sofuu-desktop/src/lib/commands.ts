// lib/commands.ts — the slash-command registry.
//
// ONE list feeds two surfaces: the composer's "/" palette (Composer.tsx)
// and the /help output (store.ts handleCommand). Add a command here and
// both pick it up; handleCommand still owns what each command DOES.

export interface SlashCommand {
  name: string;
  /** Argument placeholder, when the command accepts one ("[id]"). */
  arg?: string;
  desc: string;
}

export const SLASH_COMMANDS: SlashCommand[] = [
  { name: "new", desc: "Fresh session — clears the chat" },
  { name: "compact", desc: "Fold old history into a summary" },
  { name: "ctx", arg: "[tokens]", desc: "Show context usage, or set window" },
  { name: "maxout", arg: "[tokens]", desc: "Show/set max output tokens" },
  { name: "model", arg: "[name]", desc: "Show or set the model" },
  { name: "provider", arg: "[name]", desc: "Show or set the provider" },
  { name: "effort", arg: "[low|medium|high|max|off]", desc: "Show or set reasoning effort" },
  { name: "brain", arg: "[on|off]", desc: "Show or toggle brain memory" },
  { name: "remember", arg: "<fact>", desc: "Pin a fact to the brain" },
  { name: "why", desc: "Which memories shaped the last answer" },
  { name: "cost", desc: "Session tokens + spend" },
  { name: "resume", arg: "[id]", desc: "List sessions, or resume one" },
  { name: "permissions", arg: "[profile]", desc: "Show or set tool permissions" },
  { name: "tools", desc: "List built-in + MCP tools" },
  { name: "agents", desc: "List loaded agents" },
  { name: "rlm", arg: "[on|off|auto]", desc: "Show or set long-context mode" },
  { name: "ml", arg: "[on|off|info]", desc: "Context-economy gates" },
  { name: "sessions", desc: "List sessions in this workspace" },
  { name: "context", desc: "Show current engine context" },
  { name: "help", desc: "This list" },
];

/** Palette/HELP signature: "/name [arg]" — padded rows come from here. */
export function slashSignature(c: SlashCommand): string {
  return `/${c.name}${c.arg ? " " + c.arg : ""}`;
}
