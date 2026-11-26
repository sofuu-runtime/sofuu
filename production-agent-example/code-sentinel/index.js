#!/usr/bin/env sofuu
/**
 * production-agent-example/index.js
 *
 * Production CLI Runner & Programmatic SDK for Code Sentinel.
 * Demonstrates Sofuu's generalized per-agent model config (`models.json`)
 * and built-in `--setup-model` / `--edit-model` binary commands.
 */

import fs from 'fs';
import path from 'path';
import { defineSentinelAgent } from './lib/sentinel.js';
import './lib/tools/ast_scan.js';
import './lib/tools/report_gen.js';
import './lib/tools/git_diff.js';

function defaultAgentConfigPath() {
  const local = path.resolve(process.cwd(), 'agent.config.json');
  if (fs.existsSync(local)) return local;
  return path.resolve(process.cwd(), 'production-agent-example/code-sentinel/agent.config.json');
}

export function printHelp() {
  console.log(`
Usage:
  sofuu run production-agent-example/code-sentinel/index.js [options] [target_dir]
  ./code-sentinel [options] [target_dir]

Autonomous Production Code Intelligence & Security Sentinel.

Model & Provider Configuration (Two Generalized Ways):
  1. Per-Agent Config File: ./models.json (or ./<binary>.models.json)
  2. Binary/CLI Commands:
     --setup-model          Interactive wizard to add a provider & model
     --edit-model [name]    Edit, switch active, add, or delete providers/models
     --list-models          List configured providers & models for this agent
     --models-config <path> Path to a specific models.json file

Runtime Options:
  --interactive, -i        Launch interactive multi-turn chat session
  --target, --dir <path>   Directory or file to audit (or pass as positional arg)
  --task <prompt>          Specific audit instructions or focus area
  --provider, -p <name>    Override provider or profile name from models.json
  --model, -m <name>       Override model identifier
  --api-key <key>          Override API key for this run
  --base-url <url>         Override endpoint base URL
  --report <path>          Output path for markdown report (default: ./security-audit-report.md)
  --notify                 Send the completed audit summary through channels.json
  --listen                 Stay up and answer inbound channel messages with the sentinel
  --json                   Output pure machine-readable JSON (ideal for CI/CD)
  --config, -c <path>      Path to agent.config.json (budgets & audit settings)
  --channels-config <path> Path to a specific channels.json file
  --channel <name>         Override the active channel profile for this run
  --help, -h               Show this help message and exit
`);
}

export function parseArgs(rawArgs) {
  const args = {
    target: '.',
    task: null,
    provider: null,
    model: null,
    apiKey: null,
    baseUrl: null,
    report: 'security-audit-report.md',
    json: false,
    interactive: false,
    listen: false,
    notify: false,
    config: defaultAgentConfigPath(),
  };

  for (let i = 0; i < rawArgs.length; i++) {
    const a = rawArgs[i];
    if (a === '--') continue;
    if (a === '--help' || a === '-h') {
      printHelp();
      process.exit(0);
    }
    if (a === '--json') {
      args.json = true;
      continue;
    }
    if (a === '--interactive' || a === '-i') {
      args.interactive = true;
      continue;
    }
    if (a === '--listen') {
      args.listen = true;
      continue;
    }
    if (a === '--notify') {
      args.notify = true;
      continue;
    }
    if ((a === '--target' || a === '--dir') && i + 1 < rawArgs.length) {
      args.target = rawArgs[++i];
      continue;
    }
    if (a === '--task' && i + 1 < rawArgs.length) {
      args.task = rawArgs[++i];
      continue;
    }
    if ((a === '--provider' || a === '-p') && i + 1 < rawArgs.length) {
      args.provider = rawArgs[++i];
      continue;
    }
    if ((a === '--model' || a === '-m') && i + 1 < rawArgs.length) {
      args.model = rawArgs[++i];
      continue;
    }
    if (a === '--api-key' && i + 1 < rawArgs.length) {
      args.apiKey = rawArgs[++i];
      continue;
    }
    if (a === '--base-url' && i + 1 < rawArgs.length) {
      args.baseUrl = rawArgs[++i];
      continue;
    }
    if (a === '--report' && i + 1 < rawArgs.length) {
      args.report = rawArgs[++i];
      continue;
    }
    if ((a === '--config' || a === '-c') && i + 1 < rawArgs.length) {
      args.config = rawArgs[++i];
      continue;
    }
    if (a === '--models-config' && i + 1 < rawArgs.length) {
      i++;
      continue;
    }
    if ((a === '--channels-config' || a === '--channel') && i + 1 < rawArgs.length) {
      i++;
      continue;
    }
    if (!a.startsWith('-')) {
      args.target = a;
    }
  }

  return args;
}

export function loadConfigFile(configPath) {
  try {
    if (fs.existsSync(configPath)) {
      return JSON.parse(fs.readFileSync(configPath, 'utf8'));
    }
  } catch (_) {}
  return {};
}

/**
 * Resolves runtime model settings from Sofuu's generalized agent model config
 * (`models.json` + CLI overrides) with optional programmatic overrides.
 */
export function resolveActiveModel(options = {}) {
  const mc = typeof sofuu.agent.getModelConfig === 'function' ? sofuu.agent.getModelConfig() : {};
  const provider = options.provider || mc.provider || process.env.SOFUU_PROVIDER || '';
  const model = options.model || mc.model || process.env.SOFUU_MODEL || '';
  const baseUrl = options.baseUrl || mc.baseUrl || null;
  const apiKey = options.apiKey || null;
  return {
    configPath: mc.configPath || null,
    activeProfile: mc.active || null,
    provider,
    model,
    baseUrl,
    apiKey,
    hasApiKey: mc.hasApiKey || !!apiKey,
  };
}

/**
 * Programmatic execution interface for external tools or scripts.
 */
export async function runAudit(options = {}) {
  const cfg = loadConfigFile(
    options.config || defaultAgentConfigPath()
  );
  const resolved = resolveActiveModel(options);

  const targetDir = path.resolve(process.cwd(), options.target || '.');
  const reportPath = path.resolve(
    process.cwd(),
    options.report || cfg.audit?.reportOutput || 'security-audit-report.md'
  );

  // Define the sentinel supervisor agent and specialist sub-agents
  const sentinel = defineSentinelAgent(cfg.budget || {});

  const defaultPrompt = [
    `Perform a comprehensive security, architecture, and code quality audit of: "${targetDir}".`,
    'Step 1: Check git_diff for recent changes or modified files.',
    'Step 2: Run ast_scan to identify secret leaks, injection vectors, and prototype pollution.',
    'Step 3: If critical or high vulnerabilities are detected, delegate to security_scanner to evaluate real exploitability and formulate patches.',
    'Step 4: Synthesize all findings and call report_gen to generate an executive audit report at: "' +
      reportPath +
      '".',
  ].join('\n');

  const finalTask = options.task || defaultPrompt;

  const runOpts = {
    provider: resolved.provider || undefined,
    model: resolved.model || undefined,
    apiKey: resolved.apiKey || undefined,
    baseUrl: resolved.baseUrl || undefined,
    onStep: options.onStep || null,
  };

  return await sofuu.agent.run(sentinel, finalTask, runOpts);
}

function channelProfiles() {
  try {
    if (typeof sofuu !== 'undefined' && sofuu.channels && typeof sofuu.channels.list === 'function') {
      return sofuu.channels.list();
    }
  } catch (_) {}
  return [];
}

/**
 * Send the completed audit through channels.json. Transport, retries and
 * platform formatting are the runtime's job; this example only supplies the
 * domain summary.
 */
async function notifyChannels(result) {
  if (!sofuu.channels || typeof sofuu.channels.send !== 'function') {
    console.warn('⚠️  --notify: sofuu.channels is unavailable in this build.');
    return false;
  }
  if (channelProfiles().length === 0) {
    console.warn('⚠️  --notify: no channels configured (run --setup-channel).');
    return false;
  }
  const answer = String((result && result.answer) || 'Audit completed with no output.');
  const severity = /critical|high severity|vulnerability found/i.test(answer) ? 'critical' : 'info';
  try {
    const sent = await sofuu.channels.send({
      title: 'Code Sentinel audit',
      text: answer,
      severity,
    });
    console.log(`📣 Notified channel: ${sent.channel} (${sent.type})`);
    return !!sent.ok;
  } catch (err) {
    console.warn(`⚠️  Channel notification failed: ${err.message}`);
    return false;
  }
}

/**
 * Two-way mode. One `listen` call covers Telegram polling and the gateway's
 * WhatsApp/webhook routes; verification, allowlists, per-sender history and
 * the remote `plan` permission profile are all runtime defaults.
 */
async function startChannelListener(cfg, resolved) {
  if (!sofuu.channels || typeof sofuu.channels.listen !== 'function') {
    console.error('❌ --listen: sofuu.channels is unavailable in this build.');
    process.exit(2);
  }
  const profiles = channelProfiles();
  if (profiles.length === 0) {
    console.error('❌ --listen: no channels configured. Run --setup-channel first.');
    process.exit(2);
  }
  defineSentinelAgent(cfg.budget || {});
  sofuu.channels.listen({
    permissionProfile: 'plan',
    onMessage: async (msg) => {
      const result = await sofuu.agent.run('code_sentinel', msg.prompt, {
        permissionProfile: 'plan',
        sessionKey: 'channel:' + msg.channel + ':' + msg.senderId,
        provider: resolved.provider || undefined,
        model: resolved.model || undefined,
        apiKey: resolved.apiKey || undefined,
        baseUrl: resolved.baseUrl || undefined,
      });
      return (result && result.answer) || 'Audit completed with no output.';
    },
  });
  console.log('📡 Listening on ' + profiles.map((p) => p.name).join(', '));
  console.log('   Remote senders run with the `plan` permission profile.');
  await new Promise(() => {});
}

// CLI Execution Path
async function main() {
  const rawArgs = Array.prototype.slice.call(process.argv, 2);
  const cliArgs = parseArgs(rawArgs);
  const cfg = loadConfigFile(cliArgs.config);
  const resolved = resolveActiveModel(cliArgs);

  if (!cliArgs.json) {
    console.log('\n🛡️  Sofuu Code Sentinel — Production AI Agent');
    console.log('─────────────────────────────────────────────');
    console.log(`• Target Directory : ${path.resolve(process.cwd(), cliArgs.target)}`);
    console.log(`• Model Config     : ${resolved.configPath || '(none — run --setup-model)'}`);
    console.log(
      `• Active Model     : ${resolved.provider || '(unset)'} / ${resolved.model || '(unset)'}${resolved.activeProfile ? ` [${resolved.activeProfile}]` : ''}`
    );
    if (resolved.baseUrl) {
      console.log(`• Custom Endpoint  : ${resolved.baseUrl}`);
    }
    console.log(`• Report Output    : ${path.resolve(process.cwd(), cliArgs.report)}`);
    console.log('─────────────────────────────────────────────\n');
  }

  if (!resolved.provider || !resolved.model) {
    console.error('❌ No provider/model configured for this agent.');
    console.error('   Configure using either:');
    console.error('     1. Edit or create models.json alongside this agent');
    console.error('     2. Run with --setup-model to launch the setup wizard\n');
    process.exit(2);
  }

  // Two-Way Channel Mode
  if (cliArgs.listen) {
    await startChannelListener(cfg, resolved);
    return;
  }

  // Interactive Session Handler
  if (cliArgs.interactive) {
    defineSentinelAgent(cfg.budget || {});
    console.log('Launching interactive Sentinel TUI session...\n');
    await sofuu.agent.interact('code_sentinel', {
      provider: resolved.provider,
      model: resolved.model,
      apiKey: resolved.apiKey || undefined,
      baseUrl: resolved.baseUrl || undefined,
      task: cliArgs.task || null,
    });
    return;
  }

  // Headless Step Logging
  const onStep = cliArgs.json
    ? null
    : (evt) => {
        if (evt.type === 'tool_start') {
          console.log(
            `⚙️  [Step ${evt.step || '?'}] Calling tool: \x1b[36m${evt.toolName}\x1b[0m`
          );
        } else if (evt.type === 'tool_end') {
          console.log(`   ✓ Tool completed in ${evt.durationMs || 0} ms`);
        } else if (evt.type === 'delegation_start') {
          console.log(
            `\n🔀 [Delegation] Routing task to specialist: \x1b[35m${evt.targetAgent}\x1b[0m`
          );
        } else if (evt.type === 'delegation_end') {
          console.log(`   ✓ Specialist ${evt.targetAgent} returned report\n`);
        }
      };

  try {
    const result = await runAudit({
      target: cliArgs.target,
      task: cliArgs.task,
      provider: resolved.provider,
      model: resolved.model,
      apiKey: resolved.apiKey,
      baseUrl: resolved.baseUrl,
      report: cliArgs.report,
      config: cliArgs.config,
      onStep,
    });

    if (cliArgs.notify) await notifyChannels(result);

    if (cliArgs.json) {
      console.log(JSON.stringify(result, null, 2));
    } else {
      console.log('\n📋 Audit Completed Successfully:');
      console.log('─────────────────────────────────────────────');
      console.log(result.answer || 'Audit completed with no output.');
      console.log('─────────────────────────────────────────────');
      const stepsDisplay = Array.isArray(result.steps)
        ? result.steps.length
        : (typeof result.steps === 'number' ? result.steps : 'N/A');
      const inTok = result.usage?.promptTokens ?? result.usage?.input ?? 0;
      const outTok = result.usage?.completionTokens ?? result.usage?.output ?? 0;
      console.log(`Total Steps Taken : ${stepsDisplay}`);
      console.log(`Token Usage        : In=${inTok}, Out=${outTok}`);
    }

    process.exit(0);
  } catch (err) {
    if (cliArgs.json) {
      console.log(
        JSON.stringify(
          {
            ok: false,
            error: {
              code: err.code || 'ERR_SENTINEL_FAILED',
              message: err.message,
            },
          },
          null,
          2
        )
      );
    } else {
      console.error('\n❌ Sentinel Execution Failed:');
      console.error(`• Error Code : ${err.code || 'ERR_GENERAL'}`);
      console.error(`• Reason     : ${err.message}`);
    }
    process.exit(err.code === 'E_SECURITY' || err.code === 'E_SAFETY_VIOLATION' ? 1 : 2);
  }
}

// Top-level entry
main();
