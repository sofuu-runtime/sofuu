/**
 * production-agent-example/lib/market/feeds.js
 *
 * Real-time live market data fetchers for stocks, crypto, and currencies (FX).
 * Uses Sofuu's native fetch() with automatic endpoint normalization and fallbacks.
 */

const USER_AGENT = 'SofuuMarketSentinel/1.0';

async function fetchJson(url, headers = {}) {
  const res = await fetch(url, {
    headers: Object.assign({ 'User-Agent': USER_AGENT, Accept: 'application/json' }, headers),
  });
  if (!res.ok) {
    throw new Error(`HTTP ${res.status} from ${url}`);
  }
  return await res.json();
}

function normalizeCryptoPair(symbol) {
  const clean = String(symbol || '').trim().toUpperCase().replace(/\//g, '-');
  return clean.includes('-') ? clean : `${clean}-USD`;
}

function normalizeForexSymbol(symbol) {
  const clean = String(symbol || '').trim().toUpperCase().replace(/[^A-Z]/g, '');
  if (clean.length === 6) {
    return { base: clean.slice(0, 3), quote: clean.slice(3, 6), yahoo: `${clean}=X` };
  }
  return { base: clean, quote: 'USD', yahoo: `${clean}USD=X` };
}

function parseYahooChart(json, symbol, type) {
  const result = json?.chart?.result?.[0];
  if (!result) {
    throw new Error(`No Yahoo chart data returned for ${symbol}`);
  }
  const meta = result.meta || {};
  const timestamps = result.timestamp || [];
  const quote = result.indicators?.quote?.[0] || {};
  const opens = quote.open || [];
  const highs = quote.high || [];
  const lows = quote.low || [];
  const closes = quote.close || [];
  const volumes = quote.volume || [];

  const candles = [];
  for (let i = 0; i < timestamps.length; i++) {
    const c = closes[i];
    if (typeof c !== 'number' || !Number.isFinite(c)) continue;
    candles.push({
      t: timestamps[i] * 1000,
      o: typeof opens[i] === 'number' ? opens[i] : c,
      h: typeof highs[i] === 'number' ? highs[i] : c,
      l: typeof lows[i] === 'number' ? lows[i] : c,
      c,
      v: typeof volumes[i] === 'number' ? volumes[i] : 0,
    });
  }

  const price =
    typeof meta.regularMarketPrice === 'number'
      ? meta.regularMarketPrice
      : candles.length > 0
        ? candles[candles.length - 1].c
        : null;

  if (price === null || !Number.isFinite(price)) {
    throw new Error(`Missing current price for ${symbol}`);
  }

  const prevClose =
    typeof meta.chartPreviousClose === 'number'
      ? meta.chartPreviousClose
      : typeof meta.previousClose === 'number'
        ? meta.previousClose
        : candles.length > 1
          ? candles[0].c
          : price;

  const changePct24h = prevClose > 0 ? ((price - prevClose) / prevClose) * 100 : 0;

  return {
    symbol,
    type,
    price,
    prevClose,
    changePct24h,
    currency: meta.currency || 'USD',
    candles,
    source: 'yahoo',
    fetchedAt: new Date().toISOString(),
  };
}

export async function fetchCrypto(symbol) {
  const pair = normalizeCryptoPair(symbol);

  // Primary: Coinbase Exchange hourly candles (up to 350 bars, reverse chronological)
  try {
    const raw = await fetchJson(
      `https://api.exchange.coinbase.com/products/${encodeURIComponent(pair)}/candles?granularity=3600`
    );
    if (Array.isArray(raw) && raw.length > 0) {
      const candles = raw
        .slice(0, 120)
        .reverse()
        .map((row) => ({
          t: Number(row[0]) * 1000,
          l: Number(row[1]),
          h: Number(row[2]),
          o: Number(row[3]),
          c: Number(row[4]),
          v: Number(row[5] || 0),
        }))
        .filter((k) => Number.isFinite(k.c));

      if (candles.length > 0) {
        const latest = candles[candles.length - 1];
        const refIdx = Math.max(0, candles.length - 25);
        const prev24h = candles[refIdx].c;
        const changePct24h = prev24h > 0 ? ((latest.c - prev24h) / prev24h) * 100 : 0;
        const quote = pair.split('-')[1] || 'USD';
        return {
          symbol: pair,
          type: 'crypto',
          price: latest.c,
          prevClose: prev24h,
          changePct24h,
          currency: quote,
          candles,
          source: 'coinbase-exchange',
          fetchedAt: new Date().toISOString(),
        };
      }
    }
  } catch (_) {}

  // Secondary: Yahoo Finance crypto chart
  try {
    const yJson = await fetchJson(
      `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(pair)}?range=5d&interval=1h`
    );
    return parseYahooChart(yJson, pair, 'crypto');
  } catch (_) {}

  // Last resort: Coinbase spot only
  const spot = await fetchJson(
    `https://api.coinbase.com/v2/prices/${encodeURIComponent(pair)}/spot`
  );
  const price = Number(spot?.data?.amount);
  if (!Number.isFinite(price)) {
    throw new Error(`Unable to fetch crypto price for ${pair}`);
  }
  return {
    symbol: pair,
    type: 'crypto',
    price,
    prevClose: price,
    changePct24h: 0,
    currency: spot?.data?.currency || 'USD',
    candles: [{ t: Date.now(), o: price, h: price, l: price, c: price, v: 0 }],
    source: 'coinbase-spot',
    fetchedAt: new Date().toISOString(),
  };
}

export async function fetchStock(symbol) {
  const ticker = String(symbol || '').trim().toUpperCase();
  const yJson = await fetchJson(
    `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(ticker)}?range=5d&interval=1h`
  );
  return parseYahooChart(yJson, ticker, 'stock');
}

export async function fetchCurrency(symbol) {
  const fx = normalizeForexSymbol(symbol);
  const display = `${fx.base}/${fx.quote}`;

  try {
    const yJson = await fetchJson(
      `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(fx.yahoo)}?range=5d&interval=1h`
    );
    const parsed = parseYahooChart(yJson, display, 'currency');
    parsed.currency = fx.quote;
    return parsed;
  } catch (_) {}

  // Fallback spot rate
  const er = await fetchJson(`https://open.er-api.com/v6/latest/${encodeURIComponent(fx.base)}`);
  const rate = Number(er?.rates?.[fx.quote]);
  if (!Number.isFinite(rate)) {
    throw new Error(`Unable to fetch FX rate for ${display}`);
  }
  return {
    symbol: display,
    type: 'currency',
    price: rate,
    prevClose: rate,
    changePct24h: 0,
    currency: fx.quote,
    candles: [{ t: Date.now(), o: rate, h: rate, l: rate, c: rate, v: 0 }],
    source: 'open-er-api',
    fetchedAt: new Date().toISOString(),
  };
}

/**
 * Fetches live price and OHLCV candle history for any asset entry.
 */
export async function fetchAssetData(asset) {
  const kind = String(asset?.type || 'stock').toLowerCase();
  const sym = String(asset?.symbol || '').trim();
  if (!sym) throw new Error('Asset symbol is required');

  if (kind === 'crypto') return await fetchCrypto(sym);
  if (kind === 'currency' || kind === 'forex' || kind === 'fx') return await fetchCurrency(sym);
  return await fetchStock(sym);
}
