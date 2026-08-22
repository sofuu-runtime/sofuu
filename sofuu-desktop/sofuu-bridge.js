// sofuu-bridge.js (H3 desktop dogfood — TTY-free)
//
// Runs inside the `./sofuu` JS runtime. Reads JSON config from stdin,
// calls sofuu.agent.run (the same loop the chat UI uses — PLAN-AGENTS A1),
// outputs JSON lines.
//
// H3 changes (PLAN-HEADLESS.md):
//   - Deleted __ttyRaw() / __ttyNormal() / __readline() — the bridge no
//     longer drags terminal plumbing into a non-TTY consumer.
//   - Switched from sofuu.ai.stream (the primitive) to sofuu.agent.run
//     (the composed behavior that A1 down-leveled to runtime JS).
//   - Reads stdin via process.stdin (not __readline) — no raw mode.
//
// The bridge still runs as a subprocess (v1 per the plan); in-process
// linking via libsofuu is v2 per sofuu-desktop's own roadmap.

'use strict';

var fs = require('fs');

async function main() {
  // Read JSON config from stdin (no raw mode, no __readline).
  var chunks = [];
  await new Promise(function (resolve) {
    process.stdin.setEncoding('utf8');
    process.stdin.on('data', function (data) { chunks.push(data); });
    process.stdin.on('end', resolve);
    process.stdin.on('error', resolve);
    // Timeout fallback: if no data arrives in 10s, proceed with empty config.
    setTimeout(resolve, 10000);
  });

  var configText = chunks.join('').trim();
  if (!configText) {
    process.stdout.write(JSON.stringify({ type: 'error', message: 'no config received on stdin' }) + '\n');
    return;
  }

  var config = JSON.parse(configText);
  var messages = config.messages || [];
  var task = messages.length > 0
    ? messages.map(function (m) { return (m.role === 'user' ? 'User: ' : 'Assistant: ') + m.content; }).join('\n')
    : config.task || '';

  // ─── QTSQ Memory (best-effort) ────────────────────────────────────
  var brain = null;
  var memCount = 0;
  if (config.brainId && config.homedir) {
    var brainPath = config.homedir + '/.sofuu_brains/' + config.brainId + '.qtsq';
    try {
      if (sofuu.memory && sofuu.memory.open) {
        brain = sofuu.memory.open(brainPath, 768);
        if (brain) memCount = brain.count();
      }
    } catch (e) { /* brain not available */ }
  }

  // ─── Define a transient agent and run via the ONE loop (A1) ───────
  // The chat driver uses the same sofuu.agent.run; we define a transient
  // agent here rather than calling the primitive sofuu.ai.stream directly.
  var agentDef = {
    name: 'desktop-bridge',
    system: config.system || 'You are a helpful assistant.',
    provider: config.provider,
    model: config.model,
    memory: brain ? 'shared' : 'off',
    budget: { maxSteps: 1 }  // single-turn: one LLM call, no tool loop
  };
  if (config.apiKey) agentDef.api_key = config.apiKey;
  if (config.think) agentDef.think = true;

  // Register the agent (ignore duplicates from prior runs).
  try { sofuu.agent.define(agentDef); } catch (e) { /* already defined */ }

  // ─── Run with streaming events ─────────────────────────────────────
  var answer = '';
  try {
    var result = await sofuu.agent.run('desktop-bridge', task, {
      onStep: function (evt) {
        // Forward answer_delta events to the desktop UI as chunks.
        if (evt.kind === 'answer_delta' && evt.payload && evt.payload.text) {
          answer += evt.payload.text;
          process.stdout.write(JSON.stringify({ type: 'chunk', content: evt.payload.text }) + '\n');
        }
      }
    });

    // If no streaming happened (e.g. non-streaming provider), use the final answer.
    if (!answer && result.answer) {
      answer = result.answer;
      process.stdout.write(JSON.stringify({ type: 'chunk', content: answer }) + '\n');
    }

    var usage = result.usage || {};
    var stats = {
      promptTokens: usage.promptTokens || 0,
      completionTokens: usage.completionTokens || 0,
      memoryRecords: brain ? brain.count() : 0,
      steps: result.steps || 0
    };

    // ─── Save answer to brain (best-effort) ─────────────────────────
    if (brain && answer) {
      try {
        // Quick low-dim hash vector so memory works even without ollama
        var vec = new Float32Array(768);
        for (var i = 0; i < answer.length && i < 768; i++) {
          vec[i % 768] += answer.charCodeAt(i) / 255;
        }
        brain.remember(vec, answer, 'assistant', 0);
        brain.flush();
        stats.memoryRecords = brain.count();
      } catch (e) {}
    }

    process.stdout.write(JSON.stringify({ type: 'done', stats: stats }) + '\n');

  } catch (err) {
    process.stdout.write(JSON.stringify({ type: 'error', message: String(err.message || err) }) + '\n');
  }
}

main().catch(function (e) {
  process.stdout.write(JSON.stringify({ type: 'error', message: String(e.message || e) }) + '\n');
});
