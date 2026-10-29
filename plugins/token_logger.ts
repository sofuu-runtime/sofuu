// plugins/token_logger.ts - per-turn token accounting service (cookbook plugin).
//
// Demonstrates Category 4 (defineService): a lifecycle hook that runs after
// every turn, appends one JSONL line to .sofuu/token_log.jsonl, keeps
// cumulative prompt/completion totals per agent, and flushes on stop.
//
// Two rules this service follows, both non-negotiable for a background hook:
//   1. It never throws into the turn. Every hook body is wrapped, and a
//      failed write keeps the line buffered for the next flush instead of
//      failing the turn that produced it.
//   2. It never invents numbers. When the host reports no usage it estimates
//      completion tokens with sofuu.ai.estimateTokens and marks the line
//      estimated: true, so a report can separate measured from guessed.
//
// Read the totals from a service handle:
//   sofuu.plugins.SERVICES.token_logger.spec.getTotals()
//
// Standalone: no import beyond the SDK primitives.

import { defineService } from 'sofuu';
import type { ServiceContext, SofuuFs, TurnEndEvent } from 'sofuu';

// -- local types (this block is stripped before the runtime sees the file) -

interface TokenTotals {
  turns: number;
  promptTokens: number;
  completionTokens: number;
  totalTokens: number;
  estimatedTurns: number;
  missingUsageTurns: number;
  lastTurnAt: number;
}

interface AgentTotals {
  agent: string;
  turns: number;
  promptTokens: number;
  completionTokens: number;
}

type MaybeFs = SofuuFs | null;

// -- state ----------------------------------------------------------------

var LOG_PATH = '.sofuu/token_log.jsonl';

var totals: TokenTotals = {
  turns: 0,
  promptTokens: 0,
  completionTokens: 0,
  totalTokens: 0,
  estimatedTurns: 0,
  missingUsageTurns: 0,
  lastTurnAt: 0
};

var perAgent: Record<string, AgentTotals> = {};
var buffer: string[] = [];

function fsDriver(): MaybeFs {
  if (typeof sofuu === 'undefined' || !sofuu.fs) return null;
  return sofuu.fs;
}

function dirOf(filePath: string): string {
  var cut = filePath.lastIndexOf('/');
  return cut > 0 ? filePath.slice(0, cut) : '.';
}

function estimateTokens(text: string): number {
  var body = String(text || '');
  if (!body) return 0;
  try {
    if (typeof sofuu !== 'undefined' && sofuu.ai && typeof sofuu.ai.estimateTokens === 'function') {
      return Math.max(0, sofuu.ai.estimateTokens(body) | 0);
    }
  } catch (e) { /* fall through to the character heuristic */ }
  return Math.max(1, Math.round(body.length / 4));
}

// -- accounting -----------------------------------------------------------

function recordTurn(event: TurnEndEvent): void {
  var response = (event && typeof event.response === 'string') ? event.response : '';
  var usage = event ? event.usage : undefined;
  var promptTokens = (usage && typeof usage.promptTokens === 'number') ? usage.promptTokens : 0;
  var completionTokens = (usage && typeof usage.completionTokens === 'number') ? usage.completionTokens : 0;
  var reported = !!(usage && typeof usage.completionTokens === 'number');
  var estimated = false;
  if (!reported) {
    totals.missingUsageTurns++;
    completionTokens = estimateTokens(response);
    estimated = completionTokens > 0;
    if (estimated) totals.estimatedTurns++;
  }
  var agent = (event && event.agent) ? String(event.agent) : 'main';
  var turnId = (event && event.turnId) ? String(event.turnId) : '';

  totals.turns++;
  totals.promptTokens += promptTokens;
  totals.completionTokens += completionTokens;
  totals.totalTokens = totals.promptTokens + totals.completionTokens;
  totals.lastTurnAt = Date.now();

  var agentRow = perAgent[agent];
  if (!agentRow) {
    agentRow = { agent: agent, turns: 0, promptTokens: 0, completionTokens: 0 };
    perAgent[agent] = agentRow;
  }
  agentRow.turns++;
  agentRow.promptTokens += promptTokens;
  agentRow.completionTokens += completionTokens;

  // One line per turn: cheap to append, trivial to tail or aggregate.
  var line: Record<string, unknown> = {};
  line.t = new Date().toISOString();
  line.turnId = turnId;
  line.agent = agent;
  line.promptTokens = promptTokens;
  line.completionTokens = completionTokens;
  line.totalTokens = promptTokens + completionTokens;
  line.estimated = estimated;
  line.responseChars = response.length;
  buffer.push(JSON.stringify(line) + '\n');
}

// -- writing --------------------------------------------------------------

// Resolves even on failure: a full disk or a jailed path leaves the line in
// the buffer, it never surfaces as a turn error.
async function flushBuffer(): Promise<void> {
  if (!buffer.length) return;
  var driver = fsDriver();
  if (!driver) return;
  var pending = buffer.join('');
  buffer = [];
  try {
    await driver.mkdir(dirOf(LOG_PATH), { recursive: true });
  } catch (e) { /* the directory usually already exists */ }
  try {
    await driver.appendFile(LOG_PATH, pending);
  } catch (e) {
    buffer.push(pending);
  }
}

function totalsSnapshot(): TokenTotals {
  return {
    turns: totals.turns,
    promptTokens: totals.promptTokens,
    completionTokens: totals.completionTokens,
    totalTokens: totals.totalTokens,
    estimatedTurns: totals.estimatedTurns,
    missingUsageTurns: totals.missingUsageTurns,
    lastTurnAt: totals.lastTurnAt
  };
}

function agentSnapshot(): AgentTotals[] {
  var out: AgentTotals[] = [];
  var names = Object.keys(perAgent).sort();
  for (var i = 0; i < names.length; i++) {
    var row = perAgent[names[i]];
    out.push({ agent: row.agent, turns: row.turns, promptTokens: row.promptTokens, completionTokens: row.completionTokens });
  }
  return out;
}

// -- the service ----------------------------------------------------------

export default defineService({
  name: 'token_logger',
  description: 'Appends one JSONL token line per turn and keeps cumulative totals',
  async onStart(ctx: ServiceContext): Promise<void> {
    try {
      ctx.log('token_logger: writing ' + LOG_PATH);
    } catch (e) { /* logging is best effort */ }
  },
  async onTurnEnd(event: TurnEndEvent): Promise<void> {
    try {
      recordTurn(event);
      await flushBuffer();
    } catch (e) {
      // Accounting is never worth a failed turn.
    }
  },
  async onStop(ctx: ServiceContext): Promise<void> {
    try {
      await flushBuffer();
    } catch (e) { /* nothing left to do at shutdown */ }
  },
  getTotals(): TokenTotals {
    return totalsSnapshot();
  },
  getAgentTotals(): AgentTotals[] {
    return agentSnapshot();
  },
  getLogPath(): string {
    return LOG_PATH;
  }
});
