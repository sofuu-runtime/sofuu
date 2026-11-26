/**
 * production-agent-example/algos/example_vwap_dip.js
 *
 * Example User-Extensible Custom Algorithm.
 * Drop any `.js` file into `production-agent-example/algos/` exporting
 * `{ id, name, evaluate(data) }` and Market Sentinel loads it automatically.
 */

export default {
  id: 'vwap_dip',
  name: 'Session VWAP Deviation (Custom)',

  /**
   * @param {object} data Normalized market telemetry: { symbol, type, price, candles, ... }
   */
  evaluate(data) {
    const candles = (data.candles || []).slice(-24);
    if (candles.length < 6) {
      return { signal: 'NEUTRAL', score: 0, confidence: 0, metrics: {}, reason: 'Need >= 6 candles' };
    }

    let pvSum = 0;
    let vSum = 0;
    for (const c of candles) {
      const typical = (c.h + c.l + c.c) / 3;
      const vol = c.v > 0 ? c.v : 1; // fallback equal-weight for FX feeds without volume
      pvSum += typical * vol;
      vSum += vol;
    }

    const vwap = vSum > 0 ? pvSum / vSum : data.price;
    const distPct = vwap > 0 ? ((data.price - vwap) / vwap) * 100 : 0;
    const roundedVwap = Number(vwap.toFixed(4));
    const roundedDist = Number(distPct.toFixed(2));

    if (distPct <= -2.5) {
      return {
        signal: 'BUY_OPPORTUNITY',
        score: 70,
        confidence: 0.75,
        metrics: { vwap24h: roundedVwap, distPct: roundedDist },
        reason: `Trading ${Math.abs(roundedDist)}% below 24h VWAP (${roundedVwap}) — mean-reversion discount`,
      };
    }

    if (distPct >= 3.0) {
      return {
        signal: 'SELL_OPPORTUNITY',
        score: -70,
        confidence: 0.75,
        metrics: { vwap24h: roundedVwap, distPct: roundedDist },
        reason: `Extended +${roundedDist}% above 24h VWAP (${roundedVwap})`,
      };
    }

    return {
      signal: 'NEUTRAL',
      score: 0,
      confidence: 0.5,
      metrics: { vwap24h: roundedVwap, distPct: roundedDist },
      reason: `Near 24h VWAP (${roundedVwap}, ${roundedDist >= 0 ? '+' : ''}${roundedDist}%)`,
    };
  },
};
