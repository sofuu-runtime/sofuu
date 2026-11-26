/**
 * production-agent-example/lib/market/notifier.js
 *
 * Target-price & algorithmic opportunity trigger evaluator + multi-channel
 * notification dispatcher with persistent cooldown deduplication.
 */

import fs from 'fs';
import path from 'path';

/* Operational cooldowns live in the runtime's atomic state store
 * (sofuu.state) instead of a hand-rolled JSON file: check-and-stamp is one
 * operation, so two overlapping scan cycles cannot both alert. The store is
 * opened once per process and persists under .sofuu/state/. */
const COOLDOWN_NAMESPACE = 'market-sentinel';
let cooldownStore = null;

function store() {
  if (cooldownStore) return cooldownStore;
  try {
    if (typeof sofuu !== 'undefined' && sofuu.state) {
      cooldownStore = sofuu.state.open(COOLDOWN_NAMESPACE);
    }
  } catch (_) {
    cooldownStore = null;
  }
  return cooldownStore;
}

/* Tests (and embedded callers) may inject a plain object plus a fixed clock;
 * keep that path dependency-free and deterministic. */
function injectedStore(obj, now) {
  return {
    allowAfterCooldown(key, cooldownMs) {
      const last = obj[key];
      if (!last || cooldownMs <= 0 || now - last >= cooldownMs) {
        obj[key] = now;
        return true;
      }
      return false;
    },
  };
}

/**
 * Evaluates target prices + algorithmic opportunity triggers against cooldown state.
 */
export function detectTriggers(assetCfg, evalResult, opts = {}) {
  const triggers = [];
  const price = evalResult.price;
  const targets = assetCfg.targets || {};
  const cooldownMs = Math.max(0, Number(opts.cooldownMinutes ?? 30)) * 60 * 1000;
  const clock = opts.nowMs || Date.now();
  const state = opts.stateObj ? injectedStore(opts.stateObj, clock) : store();

  // ignoreCooldown asks the same atomic gate for a zero window, so the stamp
  // is still taken and the next real scan does not double-fire.
  const shouldFire = (key) =>
    !state || state.allowAfterCooldown(key, opts.ignoreCooldown ? 0 : cooldownMs);

  function push(kind, key, extra) {
    if (!shouldFire(key)) return;
    triggers.push({
      kind,
      key,
      symbol: evalResult.symbol,
      price,
      ...extra,
    });
  }

  // 1. Target price crossed above
  if (typeof targets.above === 'number' && price >= targets.above) {
    push('TARGET_ABOVE', `${evalResult.symbol}:TARGET_ABOVE:${targets.above}`, {
      target: targets.above,
      headline: `${evalResult.symbol} reached upper target ${targets.above} (now ${price})`,
    });
  }

  // 2. Target price crossed below
  if (typeof targets.below === 'number' && price <= targets.below) {
    push('TARGET_BELOW', `${evalResult.symbol}:TARGET_BELOW:${targets.below}`, {
      target: targets.below,
      headline: `${evalResult.symbol} reached lower target ${targets.below} (now ${price})`,
    });
  }

  // 3. Algorithmic Opportunity Trigger
  if (evalResult.isOpportunity) {
    const dir = evalResult.direction;
    push('ALGO_OPPORTUNITY', `${evalResult.symbol}:OPPORTUNITY:${dir}`, {
      score: evalResult.compositeScore,
      direction: dir,
      headline: `${evalResult.symbol} ${dir} opportunity (score ${evalResult.compositeScore}/100 at ${price})`,
    });
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
  const channels = { console: false, desktop: false, webhook: false, logFile: false, memory: false, channels: false };
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

  // 2. Sofuu channel profiles (preferred): Telegram, WhatsApp, generic
  // webhook or desktop through one domain-agnostic API. No transport code
  // lives in this example.
  let channelDelivered = false;
  if (notifyCfg.channels !== false && typeof sofuu !== 'undefined' &&
      sofuu.channels && typeof sofuu.channels.send === 'function') {
    let profiles = [];
    try { profiles = sofuu.channels.list(); } catch (_) { profiles = []; }
    if (profiles.length > 0) {
      try {
        const severity = alertPacket.triggers.some((t) => t.kind === 'TARGET_BELOW') ? 'warn' : 'info';
        const sent = await sofuu.channels.send({
          title,
          text: headline + '\n\n' + summaryText,
          severity,
          actions: [
            { id: 'ack:' + alertPacket.symbol, label: 'Acknowledge' },
            { id: 'mute:' + alertPacket.symbol, label: 'Mute' },
          ],
        });
        channels.channels = !!(sent && sent.ok);
        channelDelivered = channels.channels;
      } catch (err) {
        channels.channels = false;
        channels.channelError = err.message;
      }
    }
  }

  // 3. Legacy fallbacks — only when no channel profile carried the alert.
  if (!channelDelivered && notifyCfg.desktop && !opts.skipDesktop) {
    channels.desktop = await sendDesktopNotification(title, `${headline} — ${summaryText.slice(0, 180)}`);
  }

  if (!channelDelivered && notifyCfg.webhookUrl && typeof fetch === 'function') {
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
