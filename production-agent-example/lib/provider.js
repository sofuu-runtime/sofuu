/**
 * production-agent-example/lib/provider.js
 *
 * Dead-Simple Cloud Provider & Model Auto-Configuration for Sofuu Agents.
 *
 * Eliminates config mess:
 * 1. Automatically inherits active provider & keys from ~/.sofuu/config.json
 * 2. Auto-detects local .env files
 * 3. Supports instant model aliases ('sonnet', 'haiku', '4o', 'deepseek', etc.)
 * 4. Supports named provider shortcuts (--provider 0g, --provider openrouter)
 * 5. Interactive setup wizard (--setup)
 */

import fs from 'fs';
import path from 'path';

// Known shorthand model aliases
export const MODEL_ALIASES = {
  sonnet: {
    provider: 'anthropic',
    model: 'claude-3-5-sonnet-20241022',
  },
  haiku: {
    provider: 'anthropic',
    model: 'claude-3-5-haiku-20241022',
  },
  opus: {
    provider: 'anthropic',
    model: 'claude-3-opus-20240229',
  },
  '4o': {
    provider: 'openai',
    model: 'gpt-4o',
  },
  '4o-mini': {
    provider: 'openai',
    model: 'gpt-4o-mini',
  },
  o1: {
    provider: 'openai',
    model: 'o1-preview',
  },
  o3: {
    provider: 'openai',
    model: 'o3-mini',
  },
  deepseek: {
    provider: 'custom',
    model: 'deepseek-chat',
    baseUrl: 'https://api.deepseek.com/v1',
  },
  qwen: {
    provider: 'ollama',
    model: 'qwen2.5-coder:7b',
    baseUrl: 'http://localhost:11434/v1',
  },
};

/**
 * Automatically loads key-value pairs from .env if present.
 */
export function loadDotEnv() {
  const envPaths = [
    path.resolve(process.cwd(), '.env'),
    path.resolve(process.cwd(), 'production-agent-example/.env'),
  ];

  for (const envPath of envPaths) {
    if (fs.existsSync(envPath)) {
      try {
        const lines = fs.readFileSync(envPath, 'utf8').split('\n');
        for (const rawLine of lines) {
          const line = rawLine.trim();
          if (!line || line.startsWith('#')) continue;
          const idx = line.indexOf('=');
          if (idx > 0) {
            const key = line.slice(0, idx).trim();
            let val = line.slice(idx + 1).trim();
            if (
              (val.startsWith('"') && val.endsWith('"')) ||
              (val.startsWith("'") && val.endsWith("'"))
            ) {
              val = val.slice(1, -1);
            }
            if (!process.env[key]) {
              process.env[key] = val;
            }
          }
        }
      } catch (_) {}
    }
  }
}

/**
 * Loads ~/.sofuu/config.json to discover existing user providers and active setup.
 */
export function loadSofuuGlobalConfig() {
  const home = process.env.HOME || process.env.USERPROFILE || '';
  const cfgPath = path.join(home, '.sofuu', 'config.json');
  if (fs.existsSync(cfgPath)) {
    try {
      return JSON.parse(fs.readFileSync(cfgPath, 'utf8'));
    } catch (_) {}
  }
  return {};
}

/**
 * Resolves the final provider, model, apiKey, and baseUrl with zero friction.
 */
export function resolveProviderConfig(cliOpts = {}, projectConfig = {}) {
  loadDotEnv();
  const globalCfg = loadSofuuGlobalConfig();

  let provider = cliOpts.provider || null;
  let model = cliOpts.model || null;
  let apiKey = cliOpts.apiKey || null;
  let baseUrl = cliOpts.baseUrl || null;

  // 1. Expand model shorthand alias (e.g. --model sonnet or --model 4o)
  if (model && MODEL_ALIASES[model.toLowerCase()]) {
    const alias = MODEL_ALIASES[model.toLowerCase()];
    model = alias.model;
    if (!provider) provider = alias.provider;
    if (!baseUrl && alias.baseUrl) baseUrl = alias.baseUrl;
  }

  // 2. Check if provider matches a named registered provider in ~/.sofuu/config.json
  const knownProviders = Array.isArray(globalCfg.providers) ? globalCfg.providers : [];
  if (provider) {
    const match = knownProviders.find(
      (p) => p.name && p.name.toLowerCase() === provider.toLowerCase()
    );
    if (match) {
      if (!baseUrl) baseUrl = match.endpoint;
      if (!apiKey) apiKey = match.api_key;
      if (!model) model = match.model;
      if (provider !== 'anthropic' && provider !== 'openai') {
        provider = 'custom';
      }
    }
  }

  // 3. Fall back to agent.config.json
  if (!provider && projectConfig.provider) {
    provider = projectConfig.provider;
  }
  if (!model && projectConfig.model) {
    model = projectConfig.model;
  }
  if (!baseUrl && projectConfig.baseUrl) {
    baseUrl = projectConfig.baseUrl;
  }
  if (!apiKey && projectConfig.apiKey) {
    apiKey = projectConfig.apiKey;
  }

  // 4. Fall back to environment variables
  if (!provider && process.env.SOFUU_PROVIDER) {
    provider = process.env.SOFUU_PROVIDER;
  }
  if (!model && process.env.SOFUU_MODEL) {
    model = process.env.SOFUU_MODEL;
  }
  if (!apiKey) {
    if (provider === 'anthropic' && process.env.ANTHROPIC_API_KEY) {
      apiKey = process.env.ANTHROPIC_API_KEY;
    } else if (provider === 'openai' && process.env.OPENAI_API_KEY) {
      apiKey = process.env.OPENAI_API_KEY;
    } else if (process.env.OPENAI_API_KEY) {
      apiKey = process.env.OPENAI_API_KEY;
    } else if (process.env.ANTHROPIC_API_KEY) {
      apiKey = process.env.ANTHROPIC_API_KEY;
    }
  }

  // 5. Zero-Config Fallback: Inherit active provider from ~/.sofuu/config.json
  if ((!provider || provider === 'auto') && globalCfg.active) {
    const activeMatch = knownProviders.find((p) => p.name === globalCfg.active);
    if (activeMatch) {
      provider = globalCfg.active === 'anthropic' || globalCfg.active === 'openai' ? globalCfg.active : 'custom';
      if (!model) model = activeMatch.model || globalCfg.model;
      if (!apiKey) apiKey = activeMatch.api_key || globalCfg.api_key;
      if (!baseUrl) baseUrl = activeMatch.endpoint || globalCfg.base_url;
    } else if (globalCfg.provider) {
      provider = globalCfg.provider === 'anthropic' || globalCfg.provider === 'openai' ? globalCfg.provider : 'custom';
      if (!model) model = globalCfg.model;
      if (!apiKey) apiKey = globalCfg.api_key;
      if (!baseUrl) baseUrl = globalCfg.base_url;
    }
  }

  // 6. Absolute fallback if still nothing
  if (!provider) provider = 'anthropic';
  if (!model) model = 'claude-3-5-sonnet-20241022';

  return {
    provider,
    model,
    apiKey,
    baseUrl,
  };
}

/**
 * Quick 2-second connectivity probe against the configured provider.
 */
export async function testConnection(resolvedCfg) {
  try {
    const res = await sofuu.ai.complete('Ping', {
      provider: resolvedCfg.provider,
      model: resolvedCfg.model,
      apiKey: resolvedCfg.apiKey,
      baseUrl: resolvedCfg.baseUrl,
      max_tokens: 5,
    });
    return { ok: true, text: res.text || res.answer || 'Connected' };
  } catch (err) {
    return { ok: false, error: err.message || String(err) };
  }
}

/**
 * Interactive Provider Setup Wizard.
 */
export async function runInteractiveSetup(configPath) {
  const promptFn = typeof globalThis.__prompt === 'function' ? globalThis.__prompt : null;
  const globalCfg = loadSofuuGlobalConfig();
  const knownProviders = Array.isArray(globalCfg.providers) ? globalCfg.providers : [];

  console.log('\n┌────────────────────────────────────────────────────────┐');
  console.log('│ 🚀 Sofuu AI Provider Setup Wizard                      │');
  console.log('└────────────────────────────────────────────────────────┘');

  console.log('\nDiscovered Sofuu Configuration:');
  const choices = [];

  // Add saved providers from ~/.sofuu/config.json
  knownProviders.forEach((p, idx) => {
    const isActive = p.name === globalCfg.active ? ' (Active in Sofuu)' : '';
    console.log(`  [${idx + 1}] ${p.name}${isActive} · ${p.model || 'default'} · ${p.endpoint || 'cloud'}`);
    choices.push(p);
  });

  const nextNum = choices.length + 1;
  console.log(`  [${nextNum}] Anthropic Cloud (Claude 3.5 Sonnet)`);
  console.log(`  [${nextNum + 1}] OpenAI Cloud (GPT-4o)`);
  console.log(`  [${nextNum + 2}] Custom Endpoint (OpenRouter, DeepSeek, Groq, Ollama)`);

  if (!promptFn) {
    console.log('\nNon-interactive environment detected. To configure, set agent.config.json or pass --provider / --model.');
    return;
  }

  const selection = promptFn(`\nSelect provider [1-${nextNum + 2}] (press Enter to keep active): `);
  const choiceIdx = parseInt((selection || '1').trim(), 10) - 1;

  let chosen = {};

  if (choiceIdx >= 0 && choiceIdx < choices.length) {
    const p = choices[choiceIdx];
    chosen = {
      provider: p.name === 'anthropic' || p.name === 'openai' ? p.name : 'custom',
      model: p.model,
      baseUrl: p.endpoint,
      apiKey: p.api_key,
    };
  } else if (choiceIdx === choices.length) {
    // Anthropic
    const key = promptFn('Enter ANTHROPIC_API_KEY (press Enter for env var): ');
    chosen = {
      provider: 'anthropic',
      model: 'claude-3-5-sonnet-20241022',
      apiKey: key.trim() || undefined,
    };
  } else if (choiceIdx === choices.length + 1) {
    // OpenAI
    const key = promptFn('Enter OPENAI_API_KEY (press Enter for env var): ');
    chosen = {
      provider: 'openai',
      model: 'gpt-4o',
      apiKey: key.trim() || undefined,
    };
  } else {
    // Custom
    const url = promptFn('Enter base URL (e.g. https://api.deepseek.com/v1): ');
    const model = promptFn('Enter model name (e.g. deepseek-chat): ');
    const key = promptFn('Enter API key: ');
    chosen = {
      provider: 'custom',
      model: model.trim(),
      baseUrl: url.trim(),
      apiKey: key.trim(),
    };
  }

  console.log('\nVerifying connectivity...');
  const testRes = await testConnection(chosen);
  if (testRes.ok) {
    console.log('✓ Success! Provider responded cleanly.\n');
  } else {
    console.log(`⚠️  Warning: Probe failed: ${testRes.error}`);
    console.log('Saving configuration anyway...\n');
  }

  // Save to agent.config.json
  const targetPath = configPath || path.resolve(process.cwd(), 'production-agent-example/agent.config.json');
  let currentJson = {};
  if (fs.existsSync(targetPath)) {
    try {
      currentJson = JSON.parse(fs.readFileSync(targetPath, 'utf8'));
    } catch (_) {}
  }

  currentJson.provider = chosen.provider;
  currentJson.model = chosen.model;
  if (chosen.baseUrl) currentJson.baseUrl = chosen.baseUrl;
  if (chosen.apiKey) currentJson.apiKey = chosen.apiKey;

  fs.writeFileSync(targetPath, JSON.stringify(currentJson, null, 2), 'utf8');
  console.log(`Saved configuration to: ${targetPath}`);
}
