#!/usr/bin/env sofuu
/**
 * production-agent-example/index.js
 *
 * Production CLI Runner & Programmatic SDK for Code Sentinel.
 * Supports headless CI/CD execution, interactive TUI sessions,
 * and 12-factor configuration (CLI Flags > Env Vars > Config File).
 */

import fs from 'fs';
import path from 'path';
import { defineSentinelAgent } from './lib/sentinel.js';
import { astScanTool } from './lib/tools/ast_scan.js';
import { reportGenTool } from './lib/tools/report_gen.js';
import { gitDiffTool } from './lib/tools/git_diff.js';
import { resolveProviderConfig, runInteractiveSetup } from './lib/provider.js';

export function printHelp() {
  console.log(`
Usage: sofuu run production-agent-example/index.js [options]

Autonomous Production Code Intelligence & Security Sentinel.

Options:
  --interactive, -i        Launch interactive multi-turn session with Sentinel
  --setup                  Interactive wizard to select/configure AI cloud providers
  --target, --dir <path>   Directory or file to audit (default: current directory)
  --task <prompt>          Specific audit instructions or focus area
  --provider <name>        LLM provider (e.g. 0g, openrouter, anthropic, openai, ollama)
  --model <name>           Model identifier or alias (e.g. sonnet, haiku, 4o, 4o-mini, deepseek)
  --api-key <key>          Provider API key (defaults to active Sofuu config / env)
  --base-url <url>         Custom endpoint base URL (e.g. https://api.deepseek.com/v1)
  --report <path>          Output path for generated markdown report (default: ./security-audit-report.md)
  --json                   Output pure machine-readable JSON (ideal for CI/CD gates)
  --config <path>          Path to agent.config.json configuration file
  --help, -h               Show this help message and exit

Shorthand Aliases:
  --model sonnet           Claude 3.5 Sonnet (Anthropic)
  --model haiku            Claude 3.5 Haiku (Anthropic)
  --model 4o               GPT-4o (OpenAI)
  --model 4o-mini          GPT-4o Mini (OpenAI)
  --model deepseek         DeepSeek-V3 Chat (Custom Endpoint)
  --model qwen             Qwen 2.5 Coder 7B (Ollama)
  --provider <name>        Any named provider from ~/.sofuu/config.json (e.g. 0g, openrouter)

Environment Variables (Auto-detected from shell or .env):
  ANTHROPIC_API_KEY        Default key for Anthropic provider
  OPENAI_API_KEY           Default key for OpenAI / OpenRouter provider
  SOFUU_PROVIDER           Active provider override
  SOFUU_MODEL              Active model override
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
    setup: false,
    config: path.resolve(process.cwd(), 'production-agent-example/agent.config.json'),
  };

  for (let i = 0; i < rawArgs.length; i++) {
    const a = rawArgs[i];
    if (a === '--help' || a === '-h') {
      printHelp();
      process.exit(0);
    }
    if (a === '--setup') {
      args.setup = true;
      continue;
    }
    if (a === '--json') {
      args.json = true;
      continue;
    }
    if (a === '--interactive' || a === '-i') {
      args.interactive = true;
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
    if (a === '--provider' && i + 1 < rawArgs.length) {
      args.provider = rawArgs[++i];
      continue;
    }
    if (a === '--model' && i + 1 < rawArgs.length) {
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
    if (a === '--config' && i + 1 < rawArgs.length) {
      args.config = rawArgs[++i];
      continue;
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
 * Programmatic execution interface for external tools or scripts.
 */
export async function runAudit(options = {}) {
  const cfg = loadConfigFile(
    options.config || path.resolve(process.cwd(), 'production-agent-example/agent.config.json')
  );

  const resolved = resolveProviderConfig(options, cfg);

  const provider = resolved.provider;
  const model = resolved.model;
  const apiKey = resolved.apiKey;
  const baseUrl = resolved.baseUrl;

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
    provider,
    model,
    apiKey,
    baseUrl,
    onStep: options.onStep || null,
  };

  return await sofuu.agent.run(sentinel, finalTask, runOpts);
}

// CLI Execution Path
async function main() {
  const rawArgs = Array.prototype.slice.call(process.argv, 2);
  const cliArgs = parseArgs(rawArgs);

  // Dedicated Setup Command
  if (cliArgs.setup) {
    await runInteractiveSetup(cliArgs.config);
    process.exit(0);
  }

  const cfg = loadConfigFile(cliArgs.config);
  const resolved = resolveProviderConfig(cliArgs, cfg);

  if (!cliArgs.json) {
    console.log('\n🛡️  Sofuu Code Sentinel — Production AI Agent');
    console.log('─────────────────────────────────────────────');
    console.log(`• Target Directory : ${path.resolve(process.cwd(), cliArgs.target)}`);
    console.log(`• AI Provider      : \x1b[32m${resolved.provider}\x1b[0m`);
    console.log(`• Model Identifier : \x1b[36m${resolved.model}\x1b[0m`);
    if (resolved.baseUrl) {
      console.log(`• Custom Endpoint  : ${resolved.baseUrl}`);
    }
    console.log(`• Report Output    : ${path.resolve(process.cwd(), cliArgs.report)}`);
    console.log('─────────────────────────────────────────────\n');
  }

  // Interactive Session Handler
  if (cliArgs.interactive) {
    defineSentinelAgent(cfg.budget || {});
    console.log('Launching interactive Sentinel TUI session...\n');
    await sofuu.agent.interact('code_sentinel', {
      provider: resolved.provider,
      model: resolved.model,
      apiKey: resolved.apiKey,
      baseUrl: resolved.baseUrl,
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
      provider: cliArgs.provider,
      model: cliArgs.model,
      apiKey: cliArgs.apiKey,
      baseUrl: cliArgs.baseUrl,
      report: cliArgs.report,
      config: cliArgs.config,
      onStep,
    });

    if (cliArgs.json) {
      console.log(JSON.stringify(result, null, 2));
    } else {
      console.log('\n📋 Audit Completed Successfully:');
      console.log('─────────────────────────────────────────────');
      console.log(result.answer || 'Audit completed with no output.');
      console.log('─────────────────────────────────────────────');
      console.log(
        `Total Steps Taken : ${result.steps ? result.steps.length : 'N/A'}`
      );
      console.log(
        `Token Usage        : In=${result.usage?.input || 0}, Out=${result.usage?.output || 0}`
      );
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
