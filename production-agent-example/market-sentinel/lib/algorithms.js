/**
 * production-agent-example/lib/market/algorithms.js
 *
 * Deterministic quantitative trading/opportunity algorithms + dynamic loader
 * for user-added custom algorithms in `algos/*.js`.
 *
 * Signal contract returned by every algorithm's evaluate(marketData):
 *   {
 *     signal: 'BUY_OPPORTUNITY' | 'SELL_OPPORTUNITY' | 'WATCH' | 'NEUTRAL',
 *     score: number (-100 to +100; positive = bullish/buy, negative = bearish/sell),
 *     confidence: number (0.0 to 1.0),
 *     metrics: object (exact numbers fed into the AI reasoning model),
 *     reason: string
 *   }
 */

import fs from 'fs';
import path from 'path';

// ── Quantitative Math Helpers ─────────────────────────────────────────────────

export function computeSMA(values, period) {
  if (!values || values.length < period || period <= 0) return null;
  const slice = values.slice(values.length - period);
  let sum = 0;
  for (const v of slice) sum += v;
  return sum / period;
}

export function computeEMA(values, period) {
  if (!values || values.length < period || period <= 0) return [];
  const k = 2 / (period + 1);
  let seed = 0;
  for (let i = 0; i < period; i++) seed += values[i];
  let ema = seed / period;
  const series = [ema];
  for (let i = period; i < values.length; i++) {
    ema = values[i] * k + ema * (1 - k);
    series.push(ema);
  }
  return series;
}

export function computeRSI(closes, period = 14) {
  if (!closes || closes.length <= period) return null;
  let gains = 0;
  let losses = 0;
  for (let i = 1; i <= period; i++) {
    const diff = closes[i] - closes[i - 1];
    if (diff >= 0) gains += diff;
    else losses -= diff;
  }
  let avgGain = gains / period;
  let avgLoss = losses / period;
  for (let i = period + 1; i < closes.length; i++) {
    const diff = closes[i] - closes[i - 1];
    const g = diff > 0 ? diff : 0;
    const l = diff < 0 ? -diff : 0;
    avgGain = (avgGain * (period - 1) + g) / period;
    avgLoss = (avgLoss * (period - 1) + l) / period;
  }
  if (avgLoss === 0) return 100;
  const rs = avgGain / avgLoss;
  return 100 - 100 / (1 + rs);
}

export function computeBollinger(closes, period = 20, mult = 2) {
  if (!closes || closes.length < period) return null;
  const slice = closes.slice(closes.length - period);
  let sum = 0;
  for (const v of slice) sum += v;
  const mid = sum / period;
  let sqSum = 0;
  for (const v of slice) sqSum += (v - mid) * (v - mid);
  const stdDev = Math.sqrt(sqSum / period);
  const upper = mid + mult * stdDev;
  const lower = mid - mult * stdDev;
  const last = slice[slice.length - 1];
  const bandwidth = mid > 0 ? ((upper - lower) / mid) * 100 : 0;
  const pctB = upper > lower ? (last - lower) / (upper - lower) : 0.5;
  return { mid, upper, lower, stdDev, bandwidth, pctB };
}

// ── Built-in Quantitative Algorithms ──────────────────────────────────────────

export const BUILTIN_ALGORITHMS = [
  {
    id: 'rsi_reversion',
    name: 'RSI(14) Mean Reversion',
    builtin: true,
    evaluate(data) {
      const closes = (data.candles || []).map((c) => c.c);
      const rsi = computeRSI(closes, 14);
      if (rsi === null) {
        return {
          signal: 'NEUTRAL',
          score: 0,
          confidence: 0,
          metrics: { bars: closes.length },
          reason: 'Insufficient candle history for RSI(14)',
        };
      }
      const r = Number(rsi.toFixed(2));
      if (rsi <= 30) {
        const strength = Math.min(100, Math.round(60 + (30 - rsi) * 2));
        return {
          signal: 'BUY_OPPORTUNITY',
          score: strength,
          confidence: 0.8,
          metrics: { rsi14: r },
          reason: `Oversold RSI(14)=${r} <= 30 mean-reversion buy zone`,
        };
      }
      if (rsi >= 70) {
        const strength = -Math.min(100, Math.round(60 + (rsi - 70) * 2));
        return {
          signal: 'SELL_OPPORTUNITY',
          score: strength,
          confidence: 0.8,
          metrics: { rsi14: r },
          reason: `Overbought RSI(14)=${r} >= 70 pullback/profit-taking risk`,
        };
      }
      if (rsi <= 38) {
        return {
          signal: 'WATCH',
          score: 25,
          confidence: 0.5,
          metrics: { rsi14: r },
          reason: `Approaching oversold RSI(14)=${r}`,
        };
      }
      if (rsi >= 62) {
        return {
          signal: 'WATCH',
          score: -25,
          confidence: 0.5,
          metrics: { rsi14: r },
          reason: `Approaching overbought RSI(14)=${r}`,
        };
      }
      return {
        signal: 'NEUTRAL',
        score: 0,
        confidence: 0.4,
        metrics: { rsi14: r },
        reason: `Neutral momentum RSI(14)=${r}`,
      };
    },
  },

  {
    id: 'ema_crossover',
    name: 'EMA(9/21) Trend Crossover',
    builtin: true,
    evaluate(data) {
      const closes = (data.candles || []).map((c) => c.c);
      if (closes.length < 24) {
        return {
          signal: 'NEUTRAL',
          score: 0,
          confidence: 0,
          metrics: { bars: closes.length },
          reason: 'Insufficient candle history for EMA(9/21)',
        };
      }
      const emaFast = computeEMA(closes, 9);
      const emaSlow = computeEMA(closes, 21);
      const fNow = emaFast[emaFast.length - 1];
      const fPrev = emaFast[emaFast.length - 2];
      const sNow = emaSlow[emaSlow.length - 1];
      const sPrev = emaSlow[emaSlow.length - 2];
      const spreadPct = sNow > 0 ? Number((((fNow - sNow) / sNow) * 100).toFixed(3)) : 0;

      const crossedUp = fPrev <= sPrev && fNow > sNow;
      const crossedDown = fPrev >= sPrev && fNow < sNow;

      if (crossedUp) {
        return {
          signal: 'BUY_OPPORTUNITY',
          score: 80,
          confidence: 0.85,
          metrics: { ema9: Number(fNow.toFixed(4)), ema21: Number(sNow.toFixed(4)), spreadPct },
          reason: `Bullish EMA(9) crossed above EMA(21) (+${spreadPct}%)`,
        };
      }
      if (crossedDown) {
        return {
          signal: 'SELL_OPPORTUNITY',
          score: -80,
          confidence: 0.85,
          metrics: { ema9: Number(fNow.toFixed(4)), ema21: Number(sNow.toFixed(4)), spreadPct },
          reason: `Bearish EMA(9) crossed below EMA(21) (${spreadPct}%)`,
        };
      }
      const trendScore = Math.max(-35, Math.min(35, Math.round(spreadPct * 15)));
      return {
        signal: Math.abs(trendScore) >= 20 ? 'WATCH' : 'NEUTRAL',
        score: trendScore,
        confidence: 0.55,
        metrics: { ema9: Number(fNow.toFixed(4)), ema21: Number(sNow.toFixed(4)), spreadPct },
        reason: `EMA(9/21) spread ${spreadPct >= 0 ? '+' : ''}${spreadPct}%`,
      };
    },
  },

  {
    id: 'bollinger_squeeze',
    name: 'Bollinger(20, 2) Band Breakout',
    builtin: true,
    evaluate(data) {
      const closes = (data.candles || []).map((c) => c.c);
      const bb = computeBollinger(closes, 20, 2);
      if (!bb) {
        return {
          signal: 'NEUTRAL',
          score: 0,
          confidence: 0,
          metrics: { bars: closes.length },
          reason: 'Insufficient candle history for Bollinger(20,2)',
        };
      }
      const pctB = Number(bb.pctB.toFixed(3));
      const bw = Number(bb.bandwidth.toFixed(2));
      const metrics = {
        lower: Number(bb.lower.toFixed(4)),
        mid: Number(bb.mid.toFixed(4)),
        upper: Number(bb.upper.toFixed(4)),
        pctB,
        bandwidthPct: bw,
      };

      if (bb.pctB <= 0.05) {
        return {
          signal: 'BUY_OPPORTUNITY',
          score: 65,
          confidence: 0.72,
          metrics,
          reason: `Price at/below lower Bollinger band (%B=${pctB}, BW=${bw}%)`,
        };
      }
      if (bb.pctB >= 0.95) {
        return {
          signal: 'SELL_OPPORTUNITY',
          score: -65,
          confidence: 0.72,
          metrics,
          reason: `Price stretched above upper Bollinger band (%B=${pctB}, BW=${bw}%)`,
        };
      }
      if (bb.bandwidth < 2.0) {
        return {
          signal: 'WATCH',
          score: 20,
          confidence: 0.6,
          metrics,
          reason: `Tight Bollinger volatility squeeze (BW=${bw}%) — breakout watch`,
        };
      }
      return {
        signal: 'NEUTRAL',
        score: 0,
        confidence: 0.4,
        metrics,
        reason: `Inside Bollinger bands (%B=${pctB}, BW=${bw}%)`,
      };
    },
  },

  {
    id: 'zscore_breakout',
    name: 'Z-Score & Volume Anomaly',
    builtin: true,
    evaluate(data) {
      const candles = data.candles || [];
      if (candles.length < 20) {
        return {
          signal: 'NEUTRAL',
          score: 0,
          confidence: 0,
          metrics: { bars: candles.length },
          reason: 'Insufficient candle history for Z-Score(20)',
        };
      }
      const closes = candles.map((c) => c.c);
      const bb = computeBollinger(closes, 20, 2);
      const z = bb && bb.stdDev > 0 ? (data.price - bb.mid) / bb.stdDev : 0;
      const vols = candles.slice(candles.length - 20).map((c) => c.v || 0);
      const avgVol = vols.reduce((a, b) => a + b, 0) / vols.length;
      const lastVol = vols[vols.length - 1];
      const volRatio = avgVol > 0 ? lastVol / avgVol : 1;

      const metrics = {
        zScore: Number(z.toFixed(2)),
        volumeRatio: Number(volRatio.toFixed(2)),
      };

      if (z <= -2.0) {
        return {
          signal: 'BUY_OPPORTUNITY',
          score: volRatio >= 1.5 ? 75 : 55,
          confidence: volRatio >= 1.5 ? 0.8 : 0.65,
          metrics,
          reason: `Statistical dip Z=${metrics.zScore} with ${metrics.volumeRatio}x avg volume`,
        };
      }
      if (z >= 2.0) {
        return {
          signal: 'SELL_OPPORTUNITY',
          score: volRatio >= 1.5 ? -75 : -55,
          confidence: volRatio >= 1.5 ? 0.8 : 0.65,
          metrics,
          reason: `Statistical extension Z=+${metrics.zScore} with ${metrics.volumeRatio}x avg volume`,
        };
      }
      return {
        signal: 'NEUTRAL',
        score: 0,
        confidence: 0.4,
        metrics,
        reason: `Normal dispersion Z=${metrics.zScore}`,
      };
    },
  },
];

/**
 * Loads enabled built-in algorithms plus any custom drop-in `.js` files
 * inside `customDir` (e.g. `production-agent-example/algos/*.js`).
 */
export async function loadAlgorithms(customDir, enabledIds = null) {
  const active = [];
  const allowSet =
    Array.isArray(enabledIds) && enabledIds.length > 0 ? new Set(enabledIds) : null;

  for (const algo of BUILTIN_ALGORITHMS) {
    if (!allowSet || allowSet.has(algo.id)) {
      active.push(algo);
    }
  }

  if (customDir) {
    const absDir = path.resolve(process.cwd(), customDir);
    if (fs.existsSync(absDir)) {
      const files = fs.readdirSync(absDir).filter((f) => f.endsWith('.js')).sort();
      for (const file of files) {
        const fullPath = path.join(absDir, file);
        try {
          const mod = await import(fullPath);
          const candidate = mod.default || mod.algorithm || mod;
          if (
            candidate &&
            typeof candidate.id === 'string' &&
            typeof candidate.evaluate === 'function'
          ) {
            if (!allowSet || allowSet.has(candidate.id) || candidate.alwaysEnabled !== false) {
              active.push({
                id: candidate.id,
                name: candidate.name || candidate.id,
                builtin: false,
                file: fullPath,
                evaluate: candidate.evaluate,
              });
            }
          }
        } catch (err) {
          console.error(`[algos] Failed to load custom algorithm ${file}: ${err.message}`);
        }
      }
    }
  }

  return active;
}

/**
 * Evaluates all loaded algorithms against fetched OHLCV market data
 * and computes a composite opportunity score (-100 to +100).
 */
export async function evaluateAssetAlgorithms(marketData, algorithms, opportunityThreshold = 40) {
  const results = [];
  let weightedSum = 0;
  let weightTotal = 0;

  for (const algo of algorithms) {
    try {
      const out = await algo.evaluate(marketData);
      if (!out || typeof out !== 'object') continue;
      const score = Number.isFinite(out.score) ? Math.max(-100, Math.min(100, out.score)) : 0;
      const conf = Number.isFinite(out.confidence)
        ? Math.max(0.1, Math.min(1, out.confidence))
        : 0.5;
      results.push({
        id: algo.id,
        name: algo.name,
        builtin: !!algo.builtin,
        signal: out.signal || 'NEUTRAL',
        score,
        confidence: conf,
        metrics: out.metrics || {},
        reason: out.reason || '',
      });
      weightedSum += score * conf;
      weightTotal += conf;
    } catch (err) {
      results.push({
        id: algo.id,
        name: algo.name,
        builtin: !!algo.builtin,
        signal: 'ERROR',
        score: 0,
        confidence: 0,
        metrics: {},
        reason: `Algorithm error: ${err.message}`,
      });
    }
  }

  const compositeScore = weightTotal > 0 ? Math.round(weightedSum / weightTotal) : 0;
  const maxAbsSingle = results.reduce((mx, r) => Math.max(mx, Math.abs(r.score)), 0);
  const direction =
    compositeScore >= 20 ? 'BULLISH' : compositeScore <= -20 ? 'BEARISH' : 'NEUTRAL';
  const isOpportunity =
    Math.abs(compositeScore) >= opportunityThreshold || maxAbsSingle >= 70;

  return {
    symbol: marketData.symbol,
    type: marketData.type,
    price: marketData.price,
    changePct24h: Number((marketData.changePct24h || 0).toFixed(2)),
    compositeScore,
    direction,
    isOpportunity,
    signals: results,
  };
}
