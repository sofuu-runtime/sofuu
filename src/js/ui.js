/* src/js/ui.js — Declarative GPU UI builders, incremental patches, and event bridge (`sofuu.ui`).
 *
 * Keeps `sofuu-core` 100% free of UI bloat: emits plain JSON Scene/ScenePatch envelopes to
 * `__sofuu_ui_push_patch` (backed by dynamically-loaded `libsofuu_gpu`) and polls semantic
 * `UIEvent`s via `__sofuu_ui_poll_event`. Zero domain schemas, zero hardcoding.
 */
(function () {
  'use strict';

  var g = (typeof globalThis !== 'undefined') ? globalThis : this;
  if (!g.sofuu) g.sofuu = {};
  var sofuu = g.sofuu;
  if (sofuu.ui && sofuu.ui._loaded) return;

  var ui = {};
  ui._loaded = true;

  function sizeSpec(v) {
    if (v === undefined || v === null || v === 'auto') return 'auto';
    if (typeof v === 'number') return { px: v };
    if (typeof v === 'object') return v;
    return 'auto';
  }

  function fillSize(v) {
    return { fill: (typeof v === 'number' && v > 0) ? v : 1.0 };
  }

  function makeNode(id, kind, opts) {
    var o = opts || {};
    var pad = { top: 0, right: 0, bottom: 0, left: 0 };
    if (o.padding !== undefined) {
      if (typeof o.padding === 'number') {
        pad = { top: o.padding, right: o.padding, bottom: o.padding, left: o.padding };
      } else {
        pad = {
          top: Number(o.padding.top || 0),
          right: Number(o.padding.right || 0),
          bottom: Number(o.padding.bottom || 0),
          left: Number(o.padding.left || 0),
        };
      }
    }
    return {
      id: String(id || 'node'),
      priority: (o.priority !== undefined) ? Number(o.priority) : 100,
      visible: (o.visible !== undefined) ? !!o.visible : true,
      width: (o.width !== undefined) ? sizeSpec(o.width) : (o.fill ? fillSize(o.fill) : 'auto'),
      height: (o.height !== undefined) ? sizeSpec(o.height) : (o.fill ? fillSize(o.fill) : 'auto'),
      min_width: (o.minWidth !== undefined) ? Number(o.minWidth) : 0,
      min_height: (o.minHeight !== undefined) ? Number(o.minHeight) : 0,
      padding: pad,
      kind: kind,
    };
  }

  function unwrapNode(x) {
    if (!x) return null;
    if (typeof x.toNode === 'function') return x.toNode();
    return x;
  }

  function isContentNode(n) {
    if (!n || !n.kind) return false;
    var t = n.kind.type;
    return t === 'stat' || t === 'plot' || t === 'table' || t === 'feed' || t === 'graph' || t === 'code_text';
  }

  // ── Node & Component Builders ────────────────────────────────────────────

  ui.row = function (id, children, opts) {
    var o = opts || {};
    var kids = (children || []).map(unwrapNode).filter(Boolean);
    return makeNode(id, {
      type: 'container',
      axis: 'row',
      align: o.align || (o.fill ? 'stretch' : 'center'),
      gap: (o.gap !== undefined) ? Number(o.gap) : 10.0,
      fold_width: (o.foldBelow !== undefined) ? o.foldBelow : 680.0,
      scrollable: !!o.scrollable,
      style: o.style || null,
      children: kids,
    }, o);
  };

  ui.col = function (id, children, opts) {
    var o = opts || {};
    var kids = (children || []).map(unwrapNode).filter(Boolean);
    return makeNode(id, {
      type: 'container',
      axis: 'column',
      align: o.align || 'stretch',
      gap: (o.gap !== undefined) ? Number(o.gap) : 12.0,
      fold_width: null,
      scrollable: !!o.scrollable,
      style: o.style || null,
      children: kids,
    }, o);
  };

  ui.grid = function (id, children, opts) {
    var o = opts || {};
    var kids = (children || []).map(unwrapNode).filter(Boolean);
    return makeNode(id, {
      type: 'grid',
      columns: (o.columns !== undefined) ? Number(o.columns) : 0,
      min_col_width: (o.minColWidth !== undefined) ? Number(o.minColWidth) : 220.0,
      gap: (o.gap !== undefined) ? Number(o.gap) : 12.0,
      style: o.style || null,
      children: kids,
    }, o);
  };

  ui.tabs = function (id, items, opts) {
    var o = opts || {};
    var list = (items || []).map(function (it) {
      return { label: String(it.label || ''), child: unwrapNode(it.child) };
    });
    return makeNode(id, {
      type: 'tabs',
      active: Number(o.active || 0),
      items: list,
    }, o);
  };

  ui.text = function (id, content, opts) {
    var o = opts || {};
    return makeNode(id, {
      type: 'text',
      content: String(content !== undefined && content !== null ? content : ''),
      size: Number(o.size || 14.0),
      color: (o.color !== undefined) ? o.color : null,
      mono: !!o.mono,
      wrap: !!o.wrap,
    }, o);
  };

  ui.title = function (content, opts) {
    var o = Object.assign({ size: 18.0, width: { fill: 1.0 } }, opts || {});
    return ui.text(o.id || 'title', content, o);
  };

  ui.rule = function (id, opts) {
    var o = opts || {};
    return makeNode(id || 'rule', {
      type: 'rule',
      thickness: Number(o.thickness || 1.0),
      color: (o.color !== undefined) ? o.color : null,
    }, o);
  };

  ui.spacer = function (id, opts) {
    return makeNode(id || 'spacer', { type: 'spacer' }, Object.assign({ width: { fill: 1.0 } }, opts || {}));
  };

  ui.themeToggle = function (id, opts) {
    var o = opts || {};
    return makeNode(id || 'theme_toggle', {
      type: 'theme_toggle',
      size: Number(o.size || 34.0),
    }, o);
  };

  ui.segmented = function (id, options, active, opts) {
    return makeNode(id, {
      type: 'segmented',
      active: Number(active || 0),
      options: (options || []).map(String),
    }, opts);
  };

  ui.button = function (id, label, opts) {
    var o = opts || {};
    return makeNode(id, {
      type: 'button',
      label: String(label || ''),
      tone: o.tone || 'neutral',
    }, o);
  };

  ui.switch = function (id, checked, label, opts) {
    var o = opts || {};
    return makeNode(id, {
      type: 'switch',
      checked: !!checked,
      label: (label !== undefined && label !== null) ? String(label) : null,
      tone: o.tone || 'neutral',
    }, o);
  };

  ui.input = function (id, placeholder, opts) {
    var o = Object.assign({ height: 34.0 }, opts || {});
    return makeNode(id, {
      type: 'input',
      value: String(o.value || ''),
      placeholder: String(placeholder || ''),
      tone: o.tone || 'neutral',
    }, o);
  };

  ui.stat = function (id, label, value, opts) {
    var o = opts || {};
    var variant = o.variant || (o.range ? 'range_bar' : (o.sparkline ? 'sparkline' : 'compact'));
    return makeNode(id, {
      type: 'stat',
      label: String(label || ''),
      value: Number(value || 0),
      value_text: (o.valueText !== undefined) ? o.valueText : null,
      prefix: String(o.prefix || ''),
      suffix: String(o.suffix || ''),
      precision: (o.precision !== undefined) ? Number(o.precision) : null,
      badge: (o.badge !== undefined) ? o.badge : null,
      delta: (o.delta !== undefined) ? o.delta : null,
      caption: (o.caption !== undefined) ? o.caption : null,
      tone: o.tone || 'neutral',
      variant: variant,
      sparkline: Array.isArray(o.sparkline) ? o.sparkline.map(Number) : [],
      range_bounds: Array.isArray(o.range) && o.range.length === 2 ? [Number(o.range[0]), Number(o.range[1])] : null,
      interactive: !!o.interactive,
    }, o);
  };

  function normalizeSeries(s) {
    var x = s || {};
    var out = {
      id: String(x.id || 'series'),
      name: String(x.name || x.id || 'Series'),
      kind: x.kind || 'line',
      color: (x.color !== undefined) ? x.color : null,
      secondary_color: (x.secondaryColor !== undefined) ? x.secondaryColor : ((x.secondary_color !== undefined) ? x.secondary_color : null),
      visible: (x.visible !== undefined) ? !!x.visible : true,
      dashed: !!x.dashed,
      stroke_width: (x.strokeWidth !== undefined) ? Number(x.strokeWidth) : ((x.stroke_width !== undefined) ? Number(x.stroke_width) : null),
      points: Array.isArray(x.points) ? x.points : [],
      candles: Array.isArray(x.candles) ? x.candles : [],
    };
    return out;
  }

  ui.pane = function (id, opts) {
    var o = opts || {};
    return {
      id: String(id || 'pane'),
      label: (o.label !== undefined) ? o.label : null,
      height_weight: Number(o.heightWeight !== undefined ? o.heightWeight : (o.weight !== undefined ? o.weight : 1.0)),
      series: Array.isArray(o.series) ? o.series.map(normalizeSeries) : [],
      thresholds: Array.isArray(o.thresholds) ? o.thresholds : [],
      stacked_percent: !!o.percent,
      y_prefix: String(o.yPrefix || ''),
      y_suffix: String(o.ySuffix || ''),
      y_precision: (o.yPrecision !== undefined) ? Number(o.yPrecision) : null,
    };
  };

  ui.plot = function (id, opts) {
    var o = opts || {};
    return makeNode(id, {
      type: 'plot',
      title: (o.title !== undefined) ? o.title : null,
      subtitle: (o.subtitle !== undefined) ? o.subtitle : null,
      panes: Array.isArray(o.panes) ? o.panes : [],
      series: Array.isArray(o.series) ? o.series.map(normalizeSeries) : [],
      thresholds: Array.isArray(o.thresholds) ? o.thresholds : [],
      show_grid: (o.showGrid !== undefined) ? !!o.showGrid : true,
      crosshair: (o.showCrosshair !== undefined) ? !!o.showCrosshair : (o.crosshair !== undefined ? !!o.crosshair : true),
      show_legend: (o.showLegend !== undefined) ? !!o.showLegend : true,
      readout_mode: o.readoutMode || 'header_bar',
      stacked_percent: !!o.percent,
      y_prefix: String(o.yPrefix || ''),
      y_suffix: String(o.ySuffix || ''),
      y_precision: (o.yPrecision !== undefined) ? Number(o.yPrecision) : null,
      empty_text: (o.emptyText !== undefined) ? o.emptyText : null,
    }, o);
  };

  ui.table = function (id, opts) {
    var o = opts || {};
    var cols = (o.columns || []).map(function (c) {
      return {
        key: String(c.key || ''),
        label: String(c.label || c.key || ''),
        width: (c.width !== undefined) ? Number(c.width) : null,
        align: c.align || 'left',
        sortable: (c.sortable !== undefined) ? !!c.sortable : true,
      };
    });
    var rows = (o.rows || []).map(function (r) {
      return {
        id: String(r.id || ''),
        tone: r.tone || null,
        cells: (r.cells || []).map(function (cell) {
          if (typeof cell === 'string' || typeof cell === 'number') {
            return { text: String(cell), tone: null, badge: false };
          }
          return {
            text: String(cell && cell.text !== undefined ? cell.text : ''),
            tone: (cell && cell.tone) || null,
            badge: !!(cell && cell.badge),
          };
        }),
      };
    });
    return makeNode(id, {
      type: 'table',
      title: (o.title !== undefined) ? o.title : null,
      columns: cols,
      rows: rows,
      selected_row: o.selectedRow || null,
      row_height: Number(o.rowHeight || 32.0),
      striped: (o.striped !== undefined) ? !!o.striped : true,
      empty_text: (o.emptyText !== undefined) ? o.emptyText : null,
    }, o);
  };

  ui.feed = function (id, opts) {
    var o = opts || {};
    return makeNode(id, {
      type: 'feed',
      title: (o.title !== undefined) ? o.title : null,
      items: Array.isArray(o.items) ? o.items : [],
      max_items: Number(o.maxItems || 200),
      newest_first: (o.newestFirst !== undefined) ? !!o.newestFirst : true,
      empty_text: (o.emptyText !== undefined) ? o.emptyText : null,
    }, o);
  };

  ui.graph = function (id, opts) {
    var o = opts || {};
    return makeNode(id, {
      type: 'graph',
      title: (o.title !== undefined) ? o.title : null,
      nodes: Array.isArray(o.nodes) ? o.nodes : [],
      edges: Array.isArray(o.edges) ? o.edges : [],
      selected_node: o.selectedNode || null,
      empty_text: (o.emptyText !== undefined) ? o.emptyText : null,
    }, o);
  };

  ui.code = function (id, content, opts) {
    var o = opts || {};
    return makeNode(id, {
      type: 'code_text',
      title: (o.title !== undefined) ? o.title : null,
      language: (o.language !== undefined) ? o.language : null,
      content: String(content !== undefined && content !== null ? content : ''),
      show_line_numbers: (o.lineNumbers !== undefined) ? !!o.lineNumbers : true,
      highlights: Array.isArray(o.highlights) ? o.highlights : [],
    }, o);
  };

  // ── App Scene Builder ────────────────────────────────────────────────────

  ui.app = function (title, opts) {
    var o = opts || {};
    var cfg = {
      title: String(title || 'Sofuu'),
      theme: (o.theme === 'dark') ? 'dark' : 'light',
      padding: (o.padding !== undefined) ? Number(o.padding) : 16.0,
      gap: (o.gap !== undefined) ? Number(o.gap) : 12.0,
      hdr: null,
      statNodes: [],
      bodyNodes: [],
    };
    var b = {
      dark: function () { cfg.theme = 'dark'; return b; },
      light: function () { cfg.theme = 'light'; return b; },
      header: function (node) { cfg.hdr = unwrapNode(node); return b; },
      stats: function (list) {
        (list || []).forEach(function (n) { cfg.statNodes.push(unwrapNode(n)); });
        return b;
      },
      add: function (node) {
        var n = unwrapNode(node);
        if (n) cfg.bodyNodes.push(n);
        return b;
      },
      build: function () {
        var rootItems = [];
        if (cfg.hdr) rootItems.push(cfg.hdr);

        if (cfg.statNodes.length === 1 && rootItems.length === 0 && cfg.bodyNodes.length === 0) {
          var solo = Object.assign({}, cfg.statNodes[0]);
          if (!solo.width || solo.width === 'auto') solo.width = { fill: 1.0 };
          if (!solo.height || solo.height === 'auto') solo.height = { fill: 1.0 };
          rootItems.push(solo);
        } else if (cfg.statNodes.length > 0) {
          var g = ui.grid('auto_stats_grid', cfg.statNodes, { minColWidth: 220.0, gap: cfg.gap });
          if (cfg.bodyNodes.length === 0 && (!g.height || g.height === 'auto')) g.height = { fill: 1.0 };
          rootItems.push(g);
        }

        var singleBody = cfg.bodyNodes.length === 1 && isContentNode(cfg.bodyNodes[0]);
        cfg.bodyNodes.forEach(function (c) {
          var copy = Object.assign({}, c);
          if (((copy.kind && copy.kind.type === 'plot') || singleBody) && (!copy.height || copy.height === 'auto')) {
            copy.height = { fill: 1.0 };
          }
          rootItems.push(copy);
        });

        if (rootItems.length === 1) {
          var one = Object.assign({}, rootItems[0]);
          if (!one.width || one.width === 'auto') one.width = { fill: 1.0 };
          if (!one.height || one.height === 'auto') one.height = { fill: 1.0 };
          rootItems[0] = one;
        }

        var root = makeNode('root', {
          type: 'container',
          axis: 'column',
          align: 'stretch',
          gap: cfg.gap,
          fold_width: null,
          scrollable: false,
          style: null,
          children: rootItems,
        }, { padding: cfg.padding });

        return { title: cfg.title, theme: cfg.theme, root: root };
      },
      mount: function () {
        var scene = b.build();
        var view = ui.surface('main', { title: scene.title, theme: scene.theme });
        view.mount(scene);
        return view;
      },
    };
    return b;
  };

  // ── Live Host Bridge (`View`) ────────────────────────────────────────────

  var listeners = [];
  var closedResolvers = [];
  var pollTimer = null;
  var windowClosed = false;

  function stopPump() {
    if (pollTimer !== null && typeof clearInterval === 'function') {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  function drainEventsOnce() {
    if (typeof g.__sofuu_ui_poll_event !== 'function') return;
    while (true) {
      var raw = g.__sofuu_ui_poll_event();
      if (!raw) break;
      var ev = null;
      try { ev = JSON.parse(raw); } catch (e) { continue; }
      for (var i = 0; i < listeners.length; i++) {
        try { listeners[i](ev); } catch (e2) {}
      }
      if (ev && ev.type === 'window_closed') {
        windowClosed = true;
        stopPump();
        var resList = closedResolvers.splice(0, closedResolvers.length);
        for (var j = 0; j < resList.length; j++) resList[j]();
        break;
      }
    }
  }

  function ensurePump() {
    if (pollTimer !== null || windowClosed || !ui.isAttached()) return;
    if (typeof setInterval === 'function') {
      pollTimer = setInterval(drainEventsOnce, 16);
    }
  }

  ui.isAttached = function () {
    return typeof g.__sofuu_ui_is_attached === 'function' && !!g.__sofuu_ui_is_attached();
  };

  function sendPatch(patch) {
    if (typeof g.__sofuu_ui_push_patch !== 'function') return -2;
    return g.__sofuu_ui_push_patch(JSON.stringify(patch));
  }

  ui.surface = function (id, opts) {
    var o = opts || {};
    var defTitle = String(o.title || id || 'Sofuu');
    var defTheme = (o.theme === 'dark') ? 'dark' : 'light';

    var view = {
      id: String(id || 'main'),
      mount: function (sceneOrNode) {
        var target = unwrapNode(sceneOrNode);
        var scene;
        if (target && target.root) {
          scene = {
            title: String(target.title || defTitle),
            theme: (target.theme === 'dark') ? 'dark' : defTheme,
            root: unwrapNode(target.root),
          };
        } else {
          scene = ui.app(defTitle, { theme: defTheme }).add(target).build();
        }
        sendPatch({
          op: 'replace_scene',
          title: scene.title,
          theme: scene.theme,
          root: scene.root,
        });
        return view;
      },
      patch: function (p) { return sendPatch(p); },
      setTheme: function (mode) { return sendPatch({ op: 'set_theme', mode: mode }); },
      setText: function (nodeId, content) {
        return sendPatch({ op: 'set_text', id: String(nodeId), content: String(content) });
      },
      setVisible: function (nodeId, visible) {
        return sendPatch({ op: 'set_visible', id: String(nodeId), visible: !!visible });
      },
      updateStat: function (nodeId, fields) {
        var f = fields || {};
        var p = { op: 'update_stat', id: String(nodeId) };
        if (f.value !== undefined) p.value = Number(f.value);
        if (f.valueText !== undefined) p.value_text = f.valueText;
        if (f.delta !== undefined) p.delta = f.delta;
        if (f.tone !== undefined) p.tone = f.tone;
        if (f.caption !== undefined) p.caption = f.caption;
        if (f.pushSparkline !== undefined) p.push_sparkline = Number(f.pushSparkline);
        if (f.sparklineMaxLen !== undefined) p.sparkline_max_len = Number(f.sparklineMaxLen);
        return sendPatch(p);
      },
      appendPlot: function (plotId, seriesId, point, maxPoints) {
        var p = { op: 'append_plot_point', id: String(plotId), series_id: String(seriesId), point: point };
        if (maxPoints !== undefined) p.max_points = Number(maxPoints);
        return sendPatch(p);
      },
      upsertCandle: function (plotId, seriesId, candle, maxCandles) {
        var p = { op: 'upsert_candle', id: String(plotId), series_id: String(seriesId), candle: candle };
        if (maxCandles !== undefined) p.max_candles = Number(maxCandles);
        return sendPatch(p);
      },
      appendFeed: function (feedId, item, maxItems) {
        var p = { op: 'append_feed_item', id: String(feedId), item: item };
        if (maxItems !== undefined) p.max_items = Number(maxItems);
        return sendPatch(p);
      },
      setCode: function (codeId, content, highlights) {
        var p = { op: 'set_code', id: String(codeId), content: String(content) };
        if (highlights !== undefined) p.highlights = highlights;
        return sendPatch(p);
      },
      setSwitch: function (switchId, checked) {
        return sendPatch({ op: 'set_switch', id: String(switchId), checked: !!checked });
      },
      setInput: function (inputId, value) {
        return sendPatch({ op: 'set_input', id: String(inputId), value: String(value) });
      },
      onEvent: function (fn) {
        if (typeof fn === 'function') {
          listeners.push(fn);
          ensurePump();
        }
        return view;
      },
      waitUntilClosed: function () {
        if (!ui.isAttached() || windowClosed) return Promise.resolve();
        ensurePump();
        return new Promise(function (resolve) {
          closedResolvers.push(resolve);
        });
      },
    };
    return view;
  };

  sofuu.ui = ui;
})();
