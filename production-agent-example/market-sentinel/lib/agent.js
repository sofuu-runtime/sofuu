/**
 * production-agent-example/lib/market/agent.js
 *
 * Combines live market feeds + deterministic/custom algorithms with Sofuu's
 * AI agent engine (`market_sentinel`) for 24/7 monitoring & interactive chat.
 */

import fs from 'fs';
import path from 'path';
import { fetchAssetData } from './feeds.js';
import { loadAlgorithms, evaluateAssetAlgorithms } from './algorithms.js';
import { detectTriggers } from './notifier.js';

export function loadWatchlist(watchlistPath) {
  const abs = path.resolve(process.cwd(), watchlistPath || 'production-agent-example/market-sentinel/watchlist.json');
  if (!fs.existsSync(abs)) {
    return {
      intervalSeconds: 60,
      cooldownMinutes: 30,
      opportunityScoreThreshold: 40,
      notifications: { console: true, desktop: true, bell: true, webhookUrl: '', logFile: './market-alerts.jsonl' },
      algorithms: { enabled: ['rsi_reversion', 'ema_crossover', 'bollinger_squeeze', 'zscore_breakout'], customDir: './production-agent-example/market-sentinel/algos' },
      assets: [],
    };
  }
  return JSON.parse(fs.readFileSync(abs, 'utf8'));
}

export function saveWatchlist(watchlistPath, data) {
  const abs = path.resolve(process.cwd(), watchlistPath || 'production-agent-example/market-sentinel/watchlist.json');
  const dir = path.dirname(abs);
  if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
  fs.writeFileSync(abs, JSON.stringify(data, null, 2) + '\n', 'utf8');
}

/**
 * Combines deterministic algorithm outputs + target hits with the active AI
 * model (resolved automatically from `models.json`) to produce a final alert.
 */
export async function synthesizeAlertWithAI(assetCfg, marketData, algoEval, triggers, modelOpts = {}) {
  const signalLines = algoEval.signals
    .map((s) => `- ${s.name} [${s.id}]: signal=${s.signal}, score=${s.score}, metrics=${JSON.stringify(s.metrics)}, reason="${s.reason}"`)
    .join('\n');

  const triggerLines = triggers.map((t) => `- ${t.kind}: ${t.headline}`).join('\n');

  const prompt = [
    `Asset: ${marketData.symbol} (${marketData.type})`,
    `Live Price: ${marketData.price} ${marketData.currency} (24h change: ${algoEval.changePct24h}%)`,
    `Targets: above=${assetCfg.targets?.above ?? 'none'}, below=${assetCfg.targets?.below ?? 'none'}`,
    `Triggered Events:\n${triggerLines}`,
    `Composite Algorithm Score: ${algoEval.compositeScore}/100 (${algoEval.direction})`,
    `Algorithm Breakdown:\n${signalLines}`,
    '',
    'Synthesize a brief, actionable alert notification (under 120 words):',
    '1. Summarize why it triggered (price target and/or algorithm convergence).',
    '2. Combine the indicator numbers into an opportunity vs risk note.',
    '3. Suggest concrete watch levels (not financial advice).',
  ].join('\n');

  const timeoutMs = Number(modelOpts.aiTimeoutMs || 60000);
  try {
    const aiCall = sofuu.ai.complete(prompt, {
      system:
        'You are Market Sentinel, a concise quantitative market analyst. Base every claim strictly on the live numbers and algorithm metrics provided. Never invent prices.',
      provider: modelOpts.provider || undefined,
      model: modelOpts.model || undefined,
      apiKey: modelOpts.apiKey || undefined,
      baseUrl: modelOpts.baseUrl || undefined,
      max_tokens: 2000,
    });
    const timer = new Promise((_, reject) =>
      setTimeout(() => reject(new Error(`AI synthesis timed out after ${timeoutMs}ms`)), timeoutMs)
    );
    const res = await Promise.race([aiCall, timer]);
    const text = res?.text || res?.answer || '';
    if (text.trim()) return text.trim();
  } catch (err) {
    // Fail-open on AI formatting so a critical price alert still reaches the user!
    const fallbackSignals = algoEval.signals
      .filter((s) => s.signal !== 'NEUTRAL')
      .map((s) => `${s.name}: ${s.reason}`)
      .join('; ');
    return `Signal summary (${algoEval.direction}, score ${algoEval.compositeScore}/100): ${fallbackSignals || 'Target threshold reached.'} [AI note unavailable: ${err.message}]`;
  }

  return `Composite ${algoEval.direction} score ${algoEval.compositeScore}/100 at ${marketData.price}.`;
}

/**
 * Defines the interactive `market_sentinel` Sofuu agent with live tools.
 */
export function defineMarketAgent(watchlistPath = 'production-agent-example/market-sentinel/watchlist.json') {
  const fetchMarketTool = {
    name: 'fetch_market_data',
    description: 'Fetches real-time price and hourly OHLCV candle history for a stock, crypto, or currency pair.',
    parameters: {
      type: 'object',
      properties: {
        symbol: { type: 'string', description: 'Ticker or pair (e.g. NVDA, BTC-USD, USD/INR, EUR/USD)' },
        type: { type: 'string', enum: ['stock', 'crypto', 'currency'], description: 'Asset class' },
      },
      required: ['symbol', 'type'],
    },
    async execute({ symbol, type }) {
      const data = await fetchAssetData({ symbol, type });
      const recent = data.candles.slice(-5).map((c) => ({
        t: new Date(c.t).toISOString(),
        o: c.o,
        h: c.h,
        l: c.l,
        c: c.c,
        v: c.v,
      }));
      return JSON.stringify({
        symbol: data.symbol,
        type: data.type,
        price: data.price,
        currency: data.currency,
        changePct24h: Number(data.changePct24h.toFixed(2)),
        candleCount: data.candles.length,
        recentCandles: recent,
        source: data.source,
      });
    },
  };

  const analyzeAssetTool = {
    name: 'analyze_asset',
    description: 'Fetches live data for any stock/crypto/currency and evaluates all built-in + custom algorithms for opportunities.',
    parameters: {
      type: 'object',
      properties: {
        symbol: { type: 'string', description: 'Ticker or pair (e.g. NVDA, BTC-USD, USD/INR)' },
        type: { type: 'string', enum: ['stock', 'crypto', 'currency'], description: 'Asset class' },
      },
      required: ['symbol', 'type'],
    },
    async execute({ symbol, type }) {
      const wl = loadWatchlist(watchlistPath);
      const algos = await loadAlgorithms(wl.algorithms?.customDir, wl.algorithms?.enabled);
      const data = await fetchAssetData({ symbol, type });
      const evalRes = await evaluateAssetAlgorithms(
        data,
        algos,
        wl.opportunityScoreThreshold || 40
      );
      return JSON.stringify(evalRes);
    },
  };

  const scanWatchlistTool = {
    name: 'scan_watchlist',
    description: 'Scans every asset in watchlist.json, runs all quantitative + custom algorithms, and reports targets and opportunities.',
    parameters: { type: 'object', properties: {} },
    async execute() {
      const wl = loadWatchlist(watchlistPath);
      const algos = await loadAlgorithms(wl.algorithms?.customDir, wl.algorithms?.enabled);
      const reports = [];
      for (const asset of wl.assets || []) {
        try {
          const data = await fetchAssetData(asset);
          const evalRes = await evaluateAssetAlgorithms(
            data,
            algos,
            wl.opportunityScoreThreshold || 40
          );
          const triggers = detectTriggers(asset, evalRes, { ignoreCooldown: true, stateObj: {} });
          reports.push({
            symbol: evalRes.symbol,
            type: evalRes.type,
            price: evalRes.price,
            changePct24h: evalRes.changePct24h,
            targets: asset.targets || {},
            compositeScore: evalRes.compositeScore,
            direction: evalRes.direction,
            isOpportunity: evalRes.isOpportunity,
            triggers: triggers.map((t) => t.headline),
            signals: evalRes.signals,
          });
        } catch (err) {
          reports.push({ symbol: asset.symbol, type: asset.type, error: err.message });
        }
      }
      return JSON.stringify({ scannedAt: new Date().toISOString(), assets: reports });
    },
  };

  const manageWatchlistTool = {
    name: 'manage_watchlist',
    description: 'Add, update target prices, or remove an asset in watchlist.json.',
    parameters: {
      type: 'object',
      properties: {
        action: { type: 'string', enum: ['list', 'upsert', 'remove'] },
        symbol: { type: 'string', description: 'Ticker or pair (e.g. ETH-USD, AAPL, EUR/USD)' },
        type: { type: 'string', enum: ['stock', 'crypto', 'currency'] },
        above: { type: 'number', description: 'Upper target price alert' },
        below: { type: 'number', description: 'Lower target price alert' },
      },
      required: ['action'],
    },
    async execute({ action, symbol, type, above, below }) {
      const wl = loadWatchlist(watchlistPath);
      if (action === 'list') {
        return JSON.stringify({ assets: wl.assets || [], intervalSeconds: wl.intervalSeconds });
      }
      if (!symbol) return JSON.stringify({ ok: false, error: 'symbol is required' });
      const normSym = symbol.trim().toUpperCase();
      wl.assets = wl.assets || [];
      if (action === 'remove') {
        const before = wl.assets.length;
        wl.assets = wl.assets.filter((a) => a.symbol.toUpperCase() !== normSym);
        saveWatchlist(watchlistPath, wl);
        return JSON.stringify({ ok: true, removed: before - wl.assets.length });
      }
      const existing = wl.assets.find((a) => a.symbol.toUpperCase() === normSym);
      const targets = Object.assign({}, existing?.targets || {});
      if (typeof above === 'number') targets.above = above;
      if (typeof below === 'number') targets.below = below;
      if (existing) {
        if (type) existing.type = type;
        existing.targets = targets;
      } else {
        wl.assets.push({ symbol: normSym, type: type || 'stock', targets });
      }
      saveWatchlist(watchlistPath, wl);
      return JSON.stringify({ ok: true, symbol: normSym, targets });
    },
  };

  return sofuu.agent.define({
    name: 'market_sentinel',
    identity: {
      role: '24/7 Quantitative Market & Opportunity Sentinel',
      expertise: 'Stocks, cryptocurrencies, forex monitoring, technical algorithms, target-price alerting',
    },
    system: [
      'You are Market Sentinel, a real-time stock, crypto, and currency monitoring assistant.',
      'Always verify live prices and quantitative indicators using your tools (`fetch_market_data`, `analyze_asset`, `scan_watchlist`, `manage_watchlist`) before answering.',
      'Combine exact indicator metrics (RSI, EMA crossover, Bollinger Bands, Z-Score, and any user custom algorithms) with clear target-price context.',
      'Never invent prices or claim triggers occurred without reading tool outputs. Be concise and direct.',
    ].join(' '),
    tools: [
      fetchMarketTool,
      analyzeAssetTool,
      scanWatchlistTool,
      manageWatchlistTool,
      'todo_write',
    ],
    memory: 'agent',
    budget: {
      maxSteps: 12,
      maxDepth: 1,
      maxTokens: 60000,
      maxWallMs: 120000,
    },
  });
}
