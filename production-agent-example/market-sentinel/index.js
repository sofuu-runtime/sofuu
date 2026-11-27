#!/usr/bin/env sofuu
/**
 * production-agent-example/market-sentinel/index.js
 *
 * 24/7 Real-Time Market Sentinel Agent (Pure JS + Sofuu).
 *
 * Monitors any selected stocks, cryptocurrencies, and currencies (FX),
 * evaluates built-in + user-added custom algorithms, combines mathematical
 * signals with the active AI model (`models.json`), and dispatches
 * target-price & opportunity alerts 24/7.
 */

import fs from 'fs';
import path from 'path';
import { fetchAssetData } from './lib/feeds.js';
import { loadAlgorithms, evaluateAssetAlgorithms } from './lib/algorithms.js';
import { detectTriggers, dispatchNotification } from './lib/notifier.js';
import {
  loadWatchlist,
  saveWatchlist,
  synthesizeAlertWithAI,
  defineMarketAgent,
} from './lib/agent.js';
import { createMarketDashboard } from './lib/ui.js';

function defaultWatchlistPath() {
  const local = path.resolve(process.cwd(), 'watchlist.json');
  if (fs.existsSync(local)) return local;
  return 'production-agent-example/market-sentinel/watchlist.json';
}

function printHelp() {
  console.log(`
Usage:
  sofuu run production-agent-example/market-sentinel/index.js [mode] [options]
  ./market-sentinel [mode] [options]

24/7 Real-Time Stock, Crypto & Currency Sentinel (Algorithms + AI).

Modes:
  --watch, --daemon        Run continuous 24/7 monitoring loop
  --once                   Run one full live scan cycle and exit (default)
  --interactive, -i        Launch interactive AI chat with live market tools
  --listen                 Stay up and answer inbound channel messages (implies --watch)
  --list-algos             List active built-in and custom algorithms (algos/*.js)

Watchlist Management:
  --add <symbol>           Add or update an asset (pair with --type, --above, --below)
  --type <kind>            stock | crypto | currency (default: stock)
  --above <price>          Upper target price trigger
  --below <price>          Lower target price trigger
  --remove <symbol>        Remove an asset from watchlist.json
  --watchlist <path>       Custom watchlist JSON file

Watch & Notification Options:
  --every, --interval <d>  Poll interval (e.g. 15s, 1m, 5m; default: watchlist.json)
  --force-alert            Ignore cooldowns during --once scan
  --no-ai                  Skip LLM synthesis (pure deterministic algo scan)
  --json                   Output structured JSON (for --once)
  --ui                     Open the native 2D GPU dashboard (sofuu run --ui ...)
  --auto-ui                Use Zero-Boilerplate Auto-Composer instead of custom UI

Model & Provider Options (shared Sofuu generalized system):
  --setup-model            Configure provider & model into models.json
  --edit-model [name]      Edit, switch, add, or remove providers/models
  --list-models            Show configured providers & models

Messaging Channels (shared Sofuu generalized system):
  --setup-channel          Configure Telegram/WhatsApp/webhook/desktop into channels.json
  --edit-channel [name]    Edit, switch, add, or remove channel profiles
  --list-channels          Show configured channels (secrets masked)
  --channels-config <path> Use a specific channels.json file
  --channel <name>         Override the active channel for this run
`);
}

function parseDurationMs(raw) {
  if (!raw) return null;
  const s = String(raw).trim().toLowerCase();
  if (s.endsWith('ms')) return Math.max(1000, parseInt(s, 10) || 1000);
  if (s.endsWith('s')) return Math.max(1000, (parseFloat(s) || 10) * 1000);
  if (s.endsWith('m')) return Math.max(1000, (parseFloat(s) || 1) * 60 * 1000);
  if (s.endsWith('h')) return Math.max(1000, (parseFloat(s) || 1) * 3600 * 1000);
  const num = parseFloat(s);
  return Number.isFinite(num) ? Math.max(1000, num * 1000) : null;
}

function parseCli(rawArgs) {
  const args = {
    mode: 'once',
    watchlist: defaultWatchlistPath(),
    everyMs: null,
    forceAlert: false,
    noAi: false,
    json: false,
    autoUi: false,
    listen: false,
    addSymbol: null,
    removeSymbol: null,
    type: 'stock',
    above: undefined,
    below: undefined,
    provider: null,
    model: null,
    apiKey: null,
    baseUrl: null,
  };

  for (let i = 0; i < rawArgs.length; i++) {
    const a = rawArgs[i];
    if (a === '--') continue;
    if (a === '--help' || a === '-h') {
      printHelp();
      process.exit(0);
    }
    if (a === '--watch' || a === '--daemon' || a === '-d') {
      args.mode = 'watch';
      continue;
    }
    if (a === '--once') {
      args.mode = 'once';
      continue;
    }
    if (a === '--interactive' || a === '-i') {
      args.mode = 'interactive';
      continue;
    }
    if (a === '--list-algos') {
      args.mode = 'list_algos';
      continue;
    }
    if (a === '--listen') {
      args.listen = true;
      args.mode = 'watch';
      continue;
    }
    if (a === '--force-alert') {
      args.forceAlert = true;
      continue;
    }
    if (a === '--no-ai') {
      args.noAi = true;
      continue;
    }
    if (a === '--json') {
      args.json = true;
      continue;
    }
    if (a === '--auto-ui') {
      args.autoUi = true;
      continue;
    }
    if (a === '--watchlist' && i + 1 < rawArgs.length) {
      args.watchlist = rawArgs[++i];
      continue;
    }
    if ((a === '--every' || a === '--interval') && i + 1 < rawArgs.length) {
      args.everyMs = parseDurationMs(rawArgs[++i]);
      continue;
    }
    if (a === '--add' && i + 1 < rawArgs.length) {
      args.mode = 'add';
      args.addSymbol = rawArgs[++i];
      continue;
    }
    if (a === '--remove' && i + 1 < rawArgs.length) {
      args.mode = 'remove';
      args.removeSymbol = rawArgs[++i];
      continue;
    }
    if (a === '--type' && i + 1 < rawArgs.length) {
      args.type = rawArgs[++i];
      continue;
    }
    if (a === '--above' && i + 1 < rawArgs.length) {
      args.above = Number(rawArgs[++i]);
      continue;
    }
    if (a === '--below' && i + 1 < rawArgs.length) {
      args.below = Number(rawArgs[++i]);
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
    if (a === '--models-config' && i + 1 < rawArgs.length) {
      i++;
      continue;
    }
    if ((a === '--channels-config' || a === '--channel') && i + 1 < rawArgs.length) {
      i++;
      continue;
    }
  }
  return args;
}

function resolveModelOpts(cliArgs) {
  const mc =
    typeof sofuu !== 'undefined' && sofuu.agent && typeof sofuu.agent.getModelConfig === 'function'
      ? sofuu.agent.getModelConfig()
      : {};
  return {
    configPath: mc.configPath || null,
    activeProfile: mc.active || null,
    provider: cliArgs.provider || mc.provider || process.env.SOFUU_PROVIDER || '',
    model: cliArgs.model || mc.model || process.env.SOFUU_MODEL || '',
    baseUrl: cliArgs.baseUrl || mc.baseUrl || null,
    apiKey: cliArgs.apiKey || null,
  };
}

function formatPrice(val) {
  if (typeof val !== 'number') return String(val);
  if (val >= 1000) return val.toFixed(2);
  if (val >= 10) return val.toFixed(3);
  return val.toFixed(4);
}

/**
 * Deterministic first stage: fetch every asset and evaluate the algorithm
 * suite. Returns the cycle report plus the alerts that deserve AI synthesis.
 * This function performs zero model calls.
 */
export async function scanWatchlist(watchlistPath, cliArgs) {
  const wl = loadWatchlist(watchlistPath);
  const algos = await loadAlgorithms(wl.algorithms?.customDir, wl.algorithms?.enabled);
  const threshold = Number(wl.opportunityScoreThreshold ?? 40);
  const cooldownMin = Number(wl.cooldownMinutes ?? 30);

  const cycleReport = {
    timestamp: new Date().toISOString(),
    algorithmsLoaded: algos.length,
    assets: [],
    alerts: [],
  };
  const pending = [];
  const snapshots = Object.create(null);

  for (const asset of wl.assets || []) {
    try {
      const marketData = await fetchAssetData(asset);
      snapshots[marketData.symbol] = marketData;
      const evalRes = await evaluateAssetAlgorithms(marketData, algos, threshold);
      const triggers = detectTriggers(asset, evalRes, {
        cooldownMinutes: cooldownMin,
        ignoreCooldown: cliArgs.forceAlert,
      });

      const entry = {
        symbol: evalRes.symbol,
        type: evalRes.type,
        price: evalRes.price,
        currency: marketData.currency,
        changePct24h: evalRes.changePct24h,
        targets: asset.targets || {},
        compositeScore: evalRes.compositeScore,
        direction: evalRes.direction,
        isOpportunity: evalRes.isOpportunity,
        signals: evalRes.signals,
        triggers,
        source: marketData.source,
      };
      cycleReport.assets.push(entry);

      if (triggers.length > 0) {
        pending.push({
          asset,
          marketData,
          evalRes,
          triggers,
          alertPacket: {
            timestamp: new Date().toISOString(),
            symbol: evalRes.symbol,
            type: evalRes.type,
            price: evalRes.price,
            currency: marketData.currency,
            compositeScore: evalRes.compositeScore,
            direction: evalRes.direction,
            triggers,
            signals: evalRes.signals,
            aiBriefing: '',
          },
        });
      }
    } catch (err) {
      cycleReport.assets.push({
        symbol: asset.symbol,
        type: asset.type,
        error: err.message,
      });
    }
  }

  return { report: cycleReport, pending, notifications: wl.notifications || {}, snapshots };
}

/**
 * Second stage: synthesize and dispatch the alerts a gate let through.
 */
export async function deliverAlerts(pending, report, notifications, modelOpts, cliArgs, uiDashboard) {
  for (const item of pending) {
    let aiBriefing = '';
    if (!cliArgs.noAi && modelOpts.provider && modelOpts.model) {
      aiBriefing = await synthesizeAlertWithAI(
        item.asset,
        item.marketData,
        item.evalRes,
        item.triggers,
        modelOpts
      );
    } else {
      aiBriefing = item.evalRes.signals
        .filter((s) => s.signal !== 'NEUTRAL')
        .map((s) => `• ${s.name}: ${s.reason}`)
        .join('\n');
    }
    item.alertPacket.aiBriefing = aiBriefing;
    await dispatchNotification(item.alertPacket, notifications, {
      silent: cliArgs.json,
    });
    report.alerts.push(item.alertPacket);
    if (uiDashboard) uiDashboard.pushAlertBriefing(item.alertPacket);
  }
}

/**
 * Executes one full live market scan cycle across all watchlist assets.
 */
export async function runScanCycle(watchlistPath, modelOpts, cliArgs, uiDashboard) {
  const { report, pending, notifications, snapshots } = await scanWatchlist(watchlistPath, cliArgs);
  if (uiDashboard) uiDashboard.updateFromScan({ report, snapshots });
  await deliverAlerts(pending, report, notifications, modelOpts, cliArgs, uiDashboard);
  return report;
}

/**
 * Two-way channel bridge. This is the ONLY inbound code in the example: the
 * transport, verification, allowlist and per-sender session are all supplied
 * by `sofuu.channels`. Commands are intentionally domain-level (market scan),
 * never platform boilerplate.
 */
function startChannelListener(watchlistPath, modelOpts, cliArgs) {
  if (typeof sofuu === 'undefined' || !sofuu.channels || typeof sofuu.channels.listen !== 'function') {
    console.warn('[Market Sentinel] sofuu.channels is unavailable in this build.');
    return null;
  }
  let profiles = [];
  try { profiles = sofuu.channels.list(); } catch (err) { profiles = []; }
  if (!profiles.length) {
    console.warn('[Market Sentinel] No channels configured; run --setup-channel to enable two-way chat.');
    return null;
  }
  try { defineMarketAgent(watchlistPath); } catch (err) {
    /* Already registered is the normal case when the chat surface ran first. */
  }

  const sub = sofuu.channels.listen({
    permissionProfile: 'plan',
    onMessage: async (msg) => {
      const text = String(msg.text || '').trim();
      const lower = text.toLowerCase();
      if (!text || lower === 'help' || lower === 'start') {
        return [
          'Market Sentinel commands:',
          '  status  — watchlist and engine state',
          '  scan    — run one live scan cycle',
          '  <text>  — ask the market agent',
        ].join('\n');
      }
      if (lower === 'status') {
        const wl = loadWatchlist(watchlistPath);
        return `Market Sentinel\nAssets: ${(wl.assets || []).length}\nInterval: ${Math.max(5, Number(wl.intervalSeconds || 60))}s\nModel: ${modelOpts.provider || 'disabled'} / ${modelOpts.model || 'none'}`;
      }
      if (lower === 'scan') {
        const report = await runScanCycle(watchlistPath, modelOpts, cliArgs);
        const hits = report.alerts.map((a) => `${a.symbol} @ ${a.price}`).join(', ');
        return `Scan: ${report.assets.length} assets, ${report.alerts.length} alert(s)${hits ? ' — ' + hits : ''}`;
      }
      if (!modelOpts.provider || !modelOpts.model) {
        return 'No model configured on this agent; use `status`, `scan`, or configure one with --setup-model.';
      }
      const result = await sofuu.agent.run('market_sentinel', msg.prompt, {
        permissionProfile: 'plan',
        sessionKey: 'channel:' + msg.channel + ':' + msg.senderId,
      });
      return (result && result.answer) || 'No answer.';
    },
  });
  console.log('[Market Sentinel] Listening on ' + profiles.map((p) => p.name).join(', '));
  return sub;
}

function printCycleSummary(report) {
  const ts = new Date().toLocaleTimeString();
  console.log(`\n📊 Market Scan [${ts}] · ${report.algorithmsLoaded} algorithm(s) active`);
  console.log('─────────────────────────────────────────────────────────────────────────────');
  for (const a of report.assets) {
    if (a.error) {
      console.log(`  \x1b[31m✗ ${a.symbol.padEnd(10)}\x1b[0m (${a.type}) error: ${a.error}`);
      continue;
    }
    const chg = a.changePct24h >= 0 ? `\x1b[32m+${a.changePct24h}%\x1b[0m` : `\x1b[31m${a.changePct24h}%\x1b[0m`;
    const scoreColor = a.compositeScore >= 25 ? '\x1b[32m' : a.compositeScore <= -25 ? '\x1b[31m' : '\x1b[90m';
    const tgt = `above:${a.targets?.above ?? '-'} below:${a.targets?.below ?? '-'}`;
    const flag = a.triggers.length > 0 ? ' \x1b[1;33m🚨 TRIGGERED\x1b[0m' : '';
    console.log(
      `  \x1b[1m${a.symbol.padEnd(10)}\x1b[0m [${a.type.slice(0, 6).padEnd(6)}]  ${formatPrice(a.price).padStart(10)} ${a.currency}  (${chg})  score:${scoreColor}${String(a.compositeScore).padStart(4)}\x1b[0m  ${tgt}${flag}`
    );
    const activeSigs = (a.signals || []).filter((s) => s.signal !== 'NEUTRAL');
    for (const s of activeSigs) {
      console.log(`      └─ ${s.name}: ${s.reason}`);
    }
  }
  console.log('─────────────────────────────────────────────────────────────────────────────');
}

async function main() {
  const rawArgs = Array.prototype.slice.call(process.argv, 2);
  const cliArgs = parseCli(rawArgs);
  const modelOpts = resolveModelOpts(cliArgs);

  // Add/update watchlist asset
  if (cliArgs.mode === 'add') {
    const wl = loadWatchlist(cliArgs.watchlist);
    const sym = String(cliArgs.addSymbol || '').trim().toUpperCase();
    wl.assets = wl.assets || [];
    const existing = wl.assets.find((a) => a.symbol.toUpperCase() === sym);
    const targets = Object.assign({}, existing?.targets || {});
    if (Number.isFinite(cliArgs.above)) targets.above = cliArgs.above;
    if (Number.isFinite(cliArgs.below)) targets.below = cliArgs.below;
    if (existing) {
      existing.type = cliArgs.type || existing.type;
      existing.targets = targets;
    } else {
      wl.assets.push({ symbol: sym, type: cliArgs.type || 'stock', targets });
    }
    saveWatchlist(cliArgs.watchlist, wl);
    console.log(`✓ Saved ${sym} (${cliArgs.type}) targets=${JSON.stringify(targets)} in ${cliArgs.watchlist}`);
    return;
  }

  // Remove watchlist asset
  if (cliArgs.mode === 'remove') {
    const wl = loadWatchlist(cliArgs.watchlist);
    const sym = String(cliArgs.removeSymbol || '').trim().toUpperCase();
    const before = (wl.assets || []).length;
    wl.assets = (wl.assets || []).filter((a) => a.symbol.toUpperCase() !== sym);
    saveWatchlist(cliArgs.watchlist, wl);
    console.log(`✓ Removed ${before - wl.assets.length} matching asset(s) for ${sym}`);
    return;
  }

  // List loaded algorithms
  if (cliArgs.mode === 'list_algos') {
    const wl = loadWatchlist(cliArgs.watchlist);
    const algos = await loadAlgorithms(wl.algorithms?.customDir, wl.algorithms?.enabled);
    console.log('\nLoaded Quantitative Algorithms:');
    for (const a of algos) {
      const origin = a.builtin ? 'built-in' : `custom (${path.basename(a.file)})`;
      console.log(`  • ${a.id.padEnd(20)} ${a.name} [${origin}]`);
    }
    console.log('');
    return;
  }

  // Interactive Chat Mode
  if (cliArgs.mode === 'interactive') {
    defineMarketAgent(cliArgs.watchlist);
    console.log('\n📈 Sofuu Market Sentinel — Interactive Chat');
    console.log(`• Model Config : ${modelOpts.configPath || '(env / defaults)'}`);
    console.log(`• Active Model : ${modelOpts.provider || '(unset)'} / ${modelOpts.model || '(unset)'}`);
    console.log(`• Watchlist    : ${cliArgs.watchlist}\n`);
    await sofuu.agent.interact('market_sentinel', {
      provider: modelOpts.provider || undefined,
      model: modelOpts.model || undefined,
      apiKey: modelOpts.apiKey || undefined,
      baseUrl: modelOpts.baseUrl || undefined,
    });
    return;
  }

  const uiAttached =
    typeof sofuu !== 'undefined' && sofuu.ui && typeof sofuu.ui.isAttached === 'function' && sofuu.ui.isAttached();

  let uiDashboard = null;
  let activeRunner = null;
  const triggerScanNow = async () => runScanCycle(cliArgs.watchlist, modelOpts, cliArgs, uiDashboard);

  if (uiAttached && !cliArgs.autoUi) {
    try { defineMarketAgent(cliArgs.watchlist); } catch (_) {}
    uiDashboard = createMarketDashboard({
      liveEnabled: cliArgs.mode === 'watch',
      onScanNow: triggerScanNow,
      onClose: () => {
        if (activeRunner) activeRunner.stop('ui_closed');
      },
      onAsk: async (question, dash) => {
        if (question.toLowerCase() === 'scan') {
          await triggerScanNow();
          return;
        }
        if (cliArgs.noAi || !modelOpts.provider || !modelOpts.model) {
          dash.pushFeed(
            'AGENT',
            'info',
            'AI model disabled or unset (--no-ai or empty models.json)',
            'Configure a model with --setup-model or use "scan" to run a live scan cycle.'
          );
          return;
        }
        const res = await sofuu.agent.run('market_sentinel', question, {
          permissionProfile: 'plan',
          provider: modelOpts.provider || undefined,
          model: modelOpts.model || undefined,
          apiKey: modelOpts.apiKey || undefined,
          baseUrl: modelOpts.baseUrl || undefined,
        });
        dash.pushFeed('AGENT', 'ok', 'Market Sentinel Answer', (res && res.answer) || 'No answer.');
      },
    });
  }

  // Single Scan (--once)
  if (cliArgs.mode === 'once') {
    const report = await runScanCycle(cliArgs.watchlist, modelOpts, cliArgs, uiDashboard);
    if (cliArgs.json) {
      console.log(JSON.stringify(report, null, 2));
    } else {
      printCycleSummary(report);
    }
    if (uiDashboard) {
      await uiDashboard.waitUntilClosed();
    }
    return;
  }

  // 24/7 Continuous Watch Loop (--watch / --daemon)
  const wl = loadWatchlist(cliArgs.watchlist);
  const envInterval = (() => {
    try {
      const n = Number(process.env.SOFUU_DAEMON_INTERVAL_MS);
      return Number.isFinite(n) && n > 0 ? n : 0;
    } catch (_) {
      return 0;
    }
  })();
  const intervalMs =
    cliArgs.everyMs || envInterval || Math.max(5, Number(wl.intervalSeconds || 60)) * 1000;

  console.log('\n📡 Sofuu 24/7 Market Sentinel Started');
  console.log('─────────────────────────────────────────────────────────────────────────────');
  console.log(`• Watchlist    : ${cliArgs.watchlist} (${(wl.assets || []).length} assets)`);
  console.log(`• Poll Interval: ${Math.round(intervalMs / 1000)}s`);
  console.log(`• AI Engine    : ${modelOpts.provider || 'disabled'} / ${modelOpts.model || 'none'}`);
  if (uiDashboard) console.log('• GPU UI       : Attached (native 2D window)');
  console.log('• Press Ctrl+C to stop');
  console.log('─────────────────────────────────────────────────────────────────────────────');

  if (cliArgs.listen) startChannelListener(cliArgs.watchlist, modelOpts, cliArgs);

  // Stage 1 (gate) is deterministic market work; stage 2 (onTrigger) is the
  // only place a model is called. A quiet scan therefore costs 0 tokens.
  const runner = sofuu.triggers.register({
    id: 'market-sentinel',
    kind: 'interval',
    every: intervalMs,
    nonOverlapping: true,
    jitterPct: 10,
    backoff: { initialMs: 2000, maxMs: 60_000, multiplier: 2 },
    async gate() {
      if (uiDashboard && !uiDashboard.isLiveEnabled()) return null;
      const scan = await scanWatchlist(cliArgs.watchlist, cliArgs);
      printCycleSummary(scan.report);
      if (uiDashboard) uiDashboard.updateFromScan({ report: scan.report, snapshots: scan.snapshots });
      if (scan.pending.length === 0) return null;
      return scan;
    },
    async onTrigger(payload) {
      await deliverAlerts(payload.pending, payload.report, payload.notifications, modelOpts, cliArgs, uiDashboard);
    },
    onError(err) {
      console.error(`[Market Sentinel] Cycle error (will retry): ${err.message}`);
      if (uiDashboard) uiDashboard.pushFeed('ERR', 'danger', `Cycle error: ${err.message}`);
    },
  });
  activeRunner = runner;

  if (uiDashboard) {
    // Perform immediate initial cycle so candlesticks & watchlist appear instantly on window open.
    await runner._fire({ kind: 'initial', triggerId: runner.id, tick: 0, nowMs: Date.now() });
    await Promise.race([runner.start(), uiDashboard.waitUntilClosed()]);
    runner.stop('ui_closed');
  } else {
    await runner.start();
  }
  console.log('\n[Market Sentinel] Shutting down 24/7 monitor cleanly...');
}

main();
