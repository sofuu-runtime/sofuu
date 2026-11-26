/**
 * production-agent-example/lib/market/notifier.js
 *
 * Target-price & algorithmic opportunity trigger evaluator + multi-channel
 * notification dispatcher with persistent cooldown deduplication.
 */

import fs from 'fs';
import path from 'path';

const DEFAULT_STATE_PATH = path.resolve(process.cwd(), '.sofuu/market-cooldowns.json');

export function loadCooldownState(statePath = DEFAULT_STATE_PATH) {
  try {
    if (fs.existsSync(statePath)) {
      return JSON.parse(fs.readFileSync(statePath, 'utf8')) || {};
    }
  } catch (_) {}
  return {};
}

export function saveCooldownState(state, statePath = DEFAULT_STATE_PATH) {
  try {
    const dir = path.dirname(statePath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    fs.writeFileSync(statePath, JSON.stringify(state, null, 2), 'utf8');
  } catch (_) {}
}

/**
 * Evaluates target prices + algorithmic opportunity triggers against cooldown state.
 */
export function detectTriggers(assetCfg, evalResult, opts = {}) {
  const triggers = [];
  const price = evalResult.price;
  const targets = assetCfg.targets || {};
  const cooldownMs = Math.max(0, Number(opts.cooldownMinutes ?? 30)) * 60 * 1000;
  const now = opts.nowMs || Date.now();
  const statePath = opts.statePath || DEFAULT_STATE_PATH;
  const state = opts.stateObj || loadCooldownState(statePath);

  function shouldFire(key) {
    if (opts.ignoreCooldown) return true;
    const last = state[key];
    return !last || now - last >= cooldownMs;
  }

  function markFired(key) {
    state[key] = now;
  }

  // 1. Target price crossed above
  if (typeof targets.above === 'number' && price >= targets.above) {
    const key = `${evalResult.symbol}:TARGET_ABOVE:${targets.above}`;
    if (shouldFire(key)) {
      triggers.push({
        kind: 'TARGET_ABOVE',
        key,
        symbol: evalResult.symbol,
        price,
        target: targets.above,
        headline: `${evalResult.symbol} reached upper target ${targets.above} (now ${price})`,
      });
      markFired(key);
    }
  }

  // 2. Target price crossed below
  if (typeof targets.below === 'number' && price <= targets.below) {
    const key = `${evalResult.symbol}:TARGET_BELOW:${targets.below}`;
    if (shouldFire(key)) {
      triggers.push({
        kind: 'TARGET_BELOW',
        key,
        symbol: evalResult.symbol,
        price,
        target: targets.below,
        headline: `${evalResult.symbol} reached lower target ${targets.below} (now ${price})`,
      });
      markFired(key);
    }
  }

  // 3. Algorithmic Opportunity Trigger
  if (evalResult.isOpportunity) {
    const dir = evalResult.direction;
    const key = `${evalResult.symbol}:OPPORTUNITY:${dir}`;
    if (shouldFire(key)) {
      triggers.push({
        kind: 'ALGO_OPPORTUNITY',
        key,
        symbol: evalResult.symbol,
        price,
        score: evalResult.compositeScore,
        direction: dir,
        headline: `${evalResult.symbol} ${dir} opportunity (score ${evalResult.compositeScore}/100 at ${price})`,
      });
      markFired(key);
    }
  }

  if (triggers.length > 0 && !opts.stateObj) {
    saveCooldownState(state, statePath);
  }

  return triggers;
}

function escapeAppleScript(s) {
  return String(s || '').replace(/\\/g, '\\\\').replace(/"/g, '\\"').replace(/\r?\n/g, ' ');
}

function sendDesktopNotification(title, body) {
  return new Promise((resolve) => {
    if (typeof sofuu === 'undefined' || typeof sofuu.spawn !== 'function') {
      resolve(false);
      return;
    }
    const plat = typeof process !== 'undefined' ? process.platform : '';
    let cmd = null;
    let args = [];
    if (plat === 'darwin') {
      cmd = 'osascript';
      args = [
        '-e',
        `display notification "${escapeAppleScript(body)}" with title "${escapeAppleScript(title)}"`,
      ];
    } else if (plat === 'linux') {
      cmd = 'notify-send';
      args = [title, body];
    } else {
      resolve(false);
      return;
    }

    try {
      sofuu.spawn({
        command: cmd,
        args,
        onExit: (code) => resolve(code === 0),
      });
    } catch (_) {
      resolve(false);
    }
  });
}

/**
 * Dispatches an alert payload (triggers + AI synthesis) across configured channels.
 */
export async function dispatchNotification(alertPacket, notifyCfg = {}, opts = {}) {
  const channels = { console: false, desktop: false, webhook: false, logFile: false, memory: false };
  const title = `Market Sentinel: ${alertPacket.symbol}`;
  const headline = alertPacket.triggers.map((t) => t.headline).join(' | ');
  const summaryText = alertPacket.aiBriefing || headline;

  // 1. Console Banner + Optional Bell
  if (notifyCfg.console !== false && !opts.silent) {
    const bell = notifyCfg.bell ? '\x07' : '';
    console.log(`\n${bell}\x1b[1;33m╔══════════════════════════════════════════════════════════════╗\x1b[0m`);
    console.log(`\x1b[1;33m║ 🚨 ALERT: ${alertPacket.symbol} (${alertPacket.type.toUpperCase()}) · Price: ${alertPacket.price}\x1b[0m`);
    for (const tr of alertPacket.triggers) {
      console.log(`\x1b[1;33m║\x1b[0m  • \x1b[1m${tr.kind}\x1b[0m: ${tr.headline}`);
    }
    console.log(`\x1b[1;33m╟──────────────────────────────────────────────────────────────╢\x1b[0m`);
    console.log(summaryText);
    console.log(`\x1b[1;33m╚══════════════════════════════════════════════════════════════╝\x1b[0m\n`);
    channels.console = true;
  }

  // 2. Native OS Desktop Notification
  if (notifyCfg.desktop && !opts.skipDesktop) {
    channels.desktop = await sendDesktopNotification(title, `${headline} — ${summaryText.slice(0, 180)}`);
  }

  // 3. Optional Webhook POST (Slack / Discord / ntfy / custom URL)
  if (notifyCfg.webhookUrl && typeof fetch === 'function') {
    try {
      const res = await fetch(notifyCfg.webhookUrl, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          title,
          symbol: alertPacket.symbol,
          type: alertPacket.type,
          price: alertPacket.price,
          triggers: alertPacket.triggers,
          compositeScore: alertPacket.compositeScore,
          aiBriefing: alertPacket.aiBriefing,
          timestamp: alertPacket.timestamp,
        }),
      });
      channels.webhook = res.ok;
    } catch (_) {
      channels.webhook = false;
    }
  }

  // 4. Append-only JSONL Audit Log
  if (notifyCfg.logFile) {
    try {
      const logPath = path.resolve(process.cwd(), notifyCfg.logFile);
      const dir = path.dirname(logPath);
      if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
      fs.appendFileSync(logPath, JSON.stringify(alertPacket) + '\n', 'utf8');
      channels.logFile = true;
    } catch (_) {}
  }

  // 5. Sofuu CMA Vector Memory persistence
  if (
    typeof sofuu !== 'undefined' &&
    sofuu.memory &&
    typeof sofuu.memory.remember === 'function' &&
    !opts.skipMemory
  ) {
    try {
      await sofuu.memory.remember(
        `[${alertPacket.timestamp}] ${alertPacket.symbol} @ ${alertPacket.price}: ${headline}. AI Analysis: ${summaryText.slice(0, 300)}`,
        { entity: 'agent:market_sentinel', category: 'market_alert', tags: [alertPacket.symbol, alertPacket.type] }
      );
      channels.memory = true;
    } catch (_) {}
  }

  return channels;
}
