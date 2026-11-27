/**
 * production-agent-example/market-sentinel/lib/ui.js — Market Sentinel GPU
 * dashboard on the V2 `sofuu.ui` surface (auto-IDs, handles, inline handlers,
 * `ui.signal` selection): KPI sparkline tiles + multi-pane OHLC/volume chart +
 * sortable signal table + live alert feed + agent prompt bar.
 */

function precisionOf(price) {
  return typeof price !== 'number' || !Number.isFinite(price) || price >= 10 ? 2 : 4;
}
function pctOf(snap) {
  const chg = Number(snap.changePct24h || 0);
  return { chg, text: (chg >= 0 ? '+' : '') + chg.toFixed(2) + '%' };
}
function plotData(ui, snap) {
  if (!snap) return { title: 'Price & Volume', subtitle: 'Waiting for initial market scan...', panes: [], series: [] };
  const raw = Array.isArray(snap.candles) ? snap.candles : [];
  const prec = precisionOf(snap.price);
  const stamp = (k) => new Date(k.t).toISOString().slice(5, 16).replace('T', ' ');
  const candles = raw.slice(-80).map((k, x) => ({ x, open: Number(k.o), high: Number(k.h),
    low: Number(k.l), close: Number(k.c), label: stamp(k) }));
  const volumes = raw.slice(-80).map((k, x) => ({ x, y: Number(k.v || 0),
    y2: Number(k.c) >= Number(k.o) ? 1 : -1, label: stamp(k) }));
  const thresholds = [];
  if (Number.isFinite(snap.targets?.above)) thresholds.push({ value: Number(snap.targets.above), label: 'Above' });
  if (Number.isFinite(snap.targets?.below)) thresholds.push({ value: Number(snap.targets.below), label: 'Below' });
  const panes = [ui.pane('ohlc_pane', {
    label: `${snap.symbol} (${snap.currency || 'USD'})`, heightWeight: 2.2, yPrecision: prec, thresholds,
    series: candles.length ? [{ id: 'ohlc', name: snap.symbol, kind: 'candles', candles }]
      : [{ id: 'spot', name: snap.symbol, kind: 'line', points: [{ x: 0, y: Number(snap.price || 0) }] }],
  })];
  if (volumes.some((v) => v.y > 0)) panes.push(ui.pane('vol_pane', {
    label: 'Volume', heightWeight: 0.85, yPrecision: 0,
    series: [{ id: 'vol', name: 'Volume', kind: 'histogram', points: volumes }],
  }));
  return {
    title: `${snap.symbol}  ·  ${Number(snap.price || 0).toFixed(prec)} ${snap.currency || 'USD'}`,
    subtitle: `Source: ${snap.source || 'live'}  ·  Score: ${snap.compositeScore ?? 0}/100 (${snap.direction || 'NEUTRAL'})`,
    panes,
  };
}

function tableRows(state) {
  return state.symbols.map((sym) => {
    const s = state.snapshots[sym];
    if (!s || s.error) return { id: sym, tone: 'danger', cells: [sym, s?.type || '-', 'ERR', '-', '-',
      { text: 'ERROR', tone: 'danger', badge: true }] };
    const { chg, text } = pctOf(s);
    const trig = Array.isArray(s.triggers) ? s.triggers.length : 0;
    const badge = trig > 0 ? { text: `TRIG (${trig})`, tone: 'warn', badge: true }
      : s.isOpportunity ? { text: 'OPP', tone: 'ok', badge: true } : { text: 'WATCH', tone: 'info', badge: true };
    return { id: sym, cells: [sym, String(s.type || '').toUpperCase(), Number(s.price || 0).toFixed(precisionOf(s.price)),
      { text, tone: chg >= 0 ? 'ok' : 'danger' }, String(s.compositeScore ?? 0), badge] };
  });
}

export function createMarketDashboard(opts = {}) {
  const ui = sofuu.ui;
  const theme = opts.theme === 'dark' ? 'dark' : 'light';
  const state = {
    title: String(opts.title || 'Market Sentinel // Live GPU'),
    symbols: [], snapshots: Object.create(null), selectedSymbol: null,
    liveEnabled: opts.liveEnabled !== false, scanning: false, keys: '',
  };
  let view = null;
  let askText = '';
  let feedSeq = 0;
  const selected = ui.signal(null);
  const title = ui.title(state.title);
  const status = ui.text(opts.statusLine || 'Initializing market scan...', { size: 12.5 });
  const liveSw = ui.switch(state.liveEnabled, 'Live', { tone: 'ok',
    onChange: (on) => {
      state.liveEnabled = on;
      pushFeed('LIVE', on ? 'ok' : 'warn', `Auto-scan ${on ? 'enabled' : 'paused'}`);
      if (typeof opts.onToggleLive === 'function') opts.onToggleLive(on);
    } });
  const themeBtn = ui.themeToggle();
  const scanBtn = ui.button('Scan Now', { tone: 'info',
    onClick: async () => {
      if (state.scanning || typeof opts.onScanNow !== 'function') return;
      state.scanning = true; status.set('Scanning live feeds...');
      try { await opts.onScanNow(); } finally { state.scanning = false; }
    } });
  let assetSeg = ui.segmented([], 0, { onChange: (i) => selectAsset(state.symbols[i]) });
  const header = ui.row('ms_hdr', [title, status, assetSeg, liveSw, scanBtn, themeBtn], { foldBelow: 860, gap: 10 });
  const statsGrid = ui.grid('ms_stats_grid', [], { minColWidth: 210, gap: 12 });
  const plot = ui.plot('ms_plot', { title: 'Price & Volume', subtitle: 'Waiting for initial market scan...',
    flex: 1.45, h: { fill: 1 }, minHeight: 260, emptyText: 'Run a scan to populate OHLC candles' });
  const table = ui.table('ms_table', {
    title: 'Watchlist & Quantitative Signals', flex: 1, h: { fill: 1 }, minHeight: 260,
    selectedRow: null, emptyText: 'No assets in watchlist',
    columns: ['symbol:Asset:left', 'type:Class:center', 'price:Price:right', 'chg:24h %:right',
      'score:Score:right', 'status:State:center'].map((c) => {
      const [key, label, align] = c.split(':'); return { key, label, align };
    }),
    onSelect: (sym) => selectAsset(sym),
  });
  const feed = ui.feed('ms_feed', { title: 'Sentinel Alerts & Briefings', h: 176, maxItems: 150,
    emptyText: 'No scan events or target triggers yet' });
  const askInput = ui.input('Ask Market Sentinel (e.g. "Why did BTC-USD score this?" or "scan")...', {
    flex: 1, onChange: (v) => { askText = v; }, onSubmit: (v) => ask(v),
  });
  const askBtn = ui.button('Ask Agent', { tone: 'info', onClick: () => ask(askInput.kind.value) });
  const mainRow = ui.row('ms_main', [plot, table], { gap: 12, foldBelow: 820, align: 'stretch', h: { fill: 1 }, minHeight: 260 });
  const askRow = ui.row('ms_ask_row', [askInput, askBtn], { gap: 10, foldBelow: 540, align: 'center' });

  selected.subscribe((sym) => {
    if (!sym || !state.snapshots[sym]) return;
    plot.setData(plotData(ui, state.snapshots[sym]));
    table.selectRow(sym);
    assetSeg.setActive(Math.max(0, state.symbols.indexOf(sym)));
    statsGrid.setChildren(buildCards());
  });

  function buildCards() {
    return state.symbols.map((sym) => {
      const snap = state.snapshots[sym] || { symbol: sym, price: 0 };
      const { chg, text } = pctOf(snap);
      const trig = Array.isArray(snap.triggers) && snap.triggers.length > 0;
      return ui.stat(statIdFor(sym), sym, Number(snap.price || 0), {
        precision: precisionOf(snap.price),
        badge: trig ? 'TRIG' : (snap.type || 'ASSET').slice(0, 5).toUpperCase(),
        delta: `${text}  ·  S:${snap.compositeScore ?? 0}`,
        caption: sym === state.selectedSymbol ? `${snap.source || 'live'} · selected` : (snap.source || 'live'),
        tone: trig ? 'warn' : chg >= 0 ? 'ok' : 'danger', interactive: true,
        sparkline: (snap.candles || []).slice(-28).map((k) => Number(k.c)).filter(Number.isFinite),
        onClick: () => selectAsset(sym),
      });
    });
  }
  function rebuildHeader() {
    const options = state.symbols.length ? state.symbols : ['Watchlist'];
    assetSeg = ui.segmented(options, Math.max(0, options.indexOf(state.selectedSymbol)), { onChange: (i) => selectAsset(state.symbols[i]) });
    header.setChildren([title, status, assetSeg, liveSw, scanBtn, themeBtn]);
  }
  function selectAsset(sym) {
    if (!sym || !state.snapshots[sym] || state.selectedSymbol === sym) return;
    state.selectedSymbol = sym;
    selected.set(sym);
  }
  function pushFeed(badge, tone, headline, body) {
    feed.append({ id: 'ms_ev_' + (++feedSeq), time: new Date().toTimeString().slice(0, 8),
      badge: String(badge || 'INFO'), tone: tone || 'neutral',
      title: String(headline || ''), body: body ? String(body) : null }, 150);
  }
  async function ask(value) {
    const q = String(value !== undefined ? value : askText).trim();
    if (!q) return;
    askText = '';
    askInput.setValue('');
    pushFeed('USER', 'info', q);
    if (typeof opts.onAsk !== 'function') return;
    try { await opts.onAsk(q, dashboard); } catch (err) { pushFeed('ERR', 'danger', `Ask failed: ${err.message || err}`); }
  }
  function buildScene() {
    return ui.app(state.title, { theme }).header(header).add(statsGrid).add(mainRow).add(feed).add(askRow).build();
  }
  function mount() {
    if (!view) { view = ui.surface('market_sentinel', { title: state.title, theme }); view.mount(buildScene()); }
    return view;
  }
  function updateFromScan({ report, snapshots }) {
    const assets = report.assets || [];
    const nextSymbols = assets.map((a) => a.symbol);
    for (const a of assets) {
      const raw = snapshots && snapshots[a.symbol] ? snapshots[a.symbol] : null;
      state.snapshots[a.symbol] = Object.assign({}, a, { candles: raw ? raw.candles : (state.snapshots[a.symbol]?.candles || []) });
    }
    const nextKey = nextSymbols.join('|');
    const symbolsChanged = nextKey !== state.keys;
    state.symbols = nextSymbols;
    state.keys = nextKey;
    if (!state.selectedSymbol || !state.snapshots[state.selectedSymbol]) state.selectedSymbol = nextSymbols[0] || null;
    mount();
    status.set(`${assets.length} asset(s) · ${report.algorithmsLoaded} algos · ${new Date().toTimeString().slice(0, 8)}`);
    if (symbolsChanged) rebuildHeader();
    statsGrid.setChildren(buildCards());
    plot.setData(plotData(ui, state.snapshots[state.selectedSymbol]));
    table.setRows(tableRows(state), state.selectedSymbol);
    assetSeg.setActive(Math.max(0, state.symbols.indexOf(state.selectedSymbol)));
    const trig = assets.filter((a) => (a.triggers || []).length > 0);
    pushFeed('SCAN', trig.length > 0 ? 'warn' : 'ok',
      `Scan complete (${assets.length} assets, ${trig.length} triggered)`,
      assets.filter((a) => !a.error).map((a) =>
        `${a.symbol}: ${Number(a.price).toFixed(precisionOf(a.price))} (${a.compositeScore})`).join('  ·  '));
  }
  const dashboard = {
    buildScene, pushFeed, selectAsset,
    isLiveEnabled: () => state.liveEnabled,
    updateFromScan,
    pushAlertBriefing(alertPacket) {
      const headline = (alertPacket.triggers || []).map((t) => t.headline).join(' | ');
      pushFeed('ALERT', 'warn', `${alertPacket.symbol} @ ${alertPacket.price} · ${headline || 'Trigger'}`, alertPacket.aiBriefing || '');
    },
    waitUntilClosed() { mount(); return view.waitUntilClosed(); },
  };
  return dashboard;
}
function statIdFor(sym) {
  return 'ms_stat_' + String(sym || 'asset').replace(/[^A-Za-z0-9_]/g, '_');
}
