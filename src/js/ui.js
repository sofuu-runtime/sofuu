/* src/js/ui.js — declarative GPU UI builders + event bridge (`sofuu.ui`). */
(function () {
  'use strict';

  var g = (typeof globalThis !== 'undefined') ? globalThis : this;
  if (!g.sofuu) g.sofuu = {};
  var sofuu = g.sofuu;
  if (sofuu.ui && sofuu.ui._loaded) return;

  var ui = {};
  ui._loaded = true;

  var uidSeq = 0;
  var mounted = false;
  var nodeHandlers = Object.create(null);

  function isObj(v) {
    return v !== null && typeof v === 'object' && !Array.isArray(v);
  }

  function autoId() {
    return '_u' + (++uidSeq);
  }

  function nodeId(id, o) {
    return String(id || (o && o.id) || autoId());
  }

  function sizeSpec(v) {
    if (v === undefined || v === null || v === 'auto') return 'auto';
    if (typeof v === 'number') return { px: v };
    if (typeof v === 'object') return v;
    return 'auto';
  }

  function fillSize(v) {
    return { fill: (typeof v === 'number' && v > 0) ? v : 1.0 };
  }

  function sizeOf(o, key, alias) {
    var v = (o[key] !== undefined) ? o[key] : (alias ? o[alias] : undefined);
    if (v !== undefined) return sizeSpec(v);
    var flex = (o.flex !== undefined) ? o.flex : o.grow;
    if (flex !== undefined) return fillSize(flex);
    return o.fill ? fillSize(o.fill) : 'auto';
  }

  function put(n, k, v) {
    Object.defineProperty(n, k, { value: v, configurable: true, writable: true });
    return n;
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
    var n = {
      id: nodeId(id, o),
      priority: (o.priority !== undefined) ? Number(o.priority) : 100,
      visible: (o.visible !== undefined) ? !!o.visible : true,
      width: sizeOf(o, 'width', 'w'),
      height: sizeOf(o, 'height', 'h'),
      min_width: (o.minWidth !== undefined) ? Number(o.minWidth) : 0,
      min_height: (o.minHeight !== undefined) ? Number(o.minHeight) : 0,
      padding: pad,
      kind: kind,
    };
    put(n, 'toNode', function () { return n; });
    put(n, 'setVisible', function (v) {
      n.visible = !!v;
      if (mounted) sendPatch({ op: 'set_visible', id: n.id, visible: !!v });
      return n;
    });
    if (o.onClick || o.onChange || o.onSubmit || o.onSelect || o.onSort ||
        o.onItemClick || o.onNodeClick) {
      nodeHandlers[n.id] = { n: n, o: o };
    }
    return n;
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

  function containerNode(id, children, opts, kindFn) {
    var o, kids;
    if (Array.isArray(id)) { o = children || {}; kids = id; id = null; }
    else { o = opts || {}; kids = children || []; }
    var n = makeNode(id, Object.assign({ children: kids.map(unwrapNode).filter(Boolean) }, kindFn(o)), o);
    return put(n, 'setChildren', function (list) {
      var next = (list || []).map(unwrapNode).filter(Boolean);
      n.kind.children = next;
      if (mounted) sendPatch({ op: 'set_children', id: n.id, children: next });
      return n;
    });
  }

  ui.row = function (id, children, opts) {
    return containerNode(id, children, opts, function (o) {
      return {
        type: 'container',
        axis: 'row',
        align: o.align || (o.fill ? 'stretch' : 'center'),
        gap: (o.gap !== undefined) ? Number(o.gap) : 10.0,
        fold_width: (o.foldBelow !== undefined) ? o.foldBelow : 680.0,
        scrollable: !!o.scrollable,
        style: o.style || null,
      };
    });
  };

  ui.col = function (id, children, opts) {
    return containerNode(id, children, opts, function (o) {
      return {
        type: 'container',
        axis: 'column',
        align: o.align || 'stretch',
        gap: (o.gap !== undefined) ? Number(o.gap) : 12.0,
        fold_width: null,
        scrollable: !!o.scrollable,
        style: o.style || null,
      };
    });
  };

  ui.grid = function (id, children, opts) {
    return containerNode(id, children, opts, function (o) {
      return {
        type: 'grid',
        columns: (o.columns !== undefined) ? Number(o.columns) : 0,
        min_col_width: (o.minColWidth !== undefined) ? Number(o.minColWidth) : 220.0,
        gap: (o.gap !== undefined) ? Number(o.gap) : 12.0,
        style: o.style || null,
      };
    });
  };

  function activeMethod(n, op) {
    return put(n, 'setActive', function (i) {
      n.kind.active = Number(i) || 0;
      if (mounted) sendPatch({ op: op, id: n.id, active: n.kind.active });
      return n;
    });
  }

  ui.tabs = function (id, items, opts) {
    var o, list;
    if (Array.isArray(id)) { o = items || {}; list = id; id = null; }
    else { o = opts || {}; list = items || []; }
    var n = makeNode(id, {
      type: 'tabs',
      active: Number(o.active || 0),
      items: list.map(function (it) {
        return { label: String(it.label || ''), child: unwrapNode(it.child) };
      }),
    }, o);
    return activeMethod(n, 'set_tab');
  };

  ui.text = function (id, content, opts) {
    var o, txt;
    if (typeof content === 'string') { o = opts || {}; txt = content; }
    else { o = content || {}; txt = id; id = null; }
    var n = makeNode(id, {
      type: 'text',
      content: String(txt !== undefined && txt !== null ? txt : ''),
      size: Number(o.size || 14.0),
      color: (o.color !== undefined) ? o.color : null,
      mono: !!o.mono,
      wrap: !!o.wrap,
    }, o);
    return put(n, 'set', function (v) {
      n.kind.content = String(v !== undefined && v !== null ? v : '');
      if (mounted) sendPatch({ op: 'set_text', id: n.id, content: n.kind.content });
      return n;
    });
  };

  ui.title = function (content, opts) {
    var o = Object.assign({ size: 18.0, width: { fill: 1.0 } }, opts || {});
    return ui.text(o.id || null, content, o);
  };

  ui.rule = function (id, opts) {
    var o = isObj(id) ? id : (opts || {});
    if (isObj(id)) id = null;
    return makeNode(id, {
      type: 'rule',
      thickness: Number(o.thickness || 1.0),
      color: (o.color !== undefined) ? o.color : null,
    }, o);
  };

  ui.spacer = function (id, opts) {
    var o = isObj(id) ? id : (opts || {});
    if (isObj(id)) id = null;
    return makeNode(id, { type: 'spacer' }, Object.assign({ width: { fill: 1.0 } }, o));
  };

  ui.themeToggle = function (id, opts) {
    var o = isObj(id) ? id : (opts || {});
    if (isObj(id)) id = null;
    return makeNode(id, {
      type: 'theme_toggle',
      size: Number(o.size || 34.0),
    }, o);
  };

  ui.segmented = function (id, options, active, opts) {
    var o, arr, act;
    if (Array.isArray(id)) {
      arr = id;
      if (isObj(options)) { o = options; act = 0; }
      else { act = options; o = isObj(active) ? active : (opts || {}); }
      id = null;
    } else { o = opts || {}; arr = options || []; act = active; }
    var n = makeNode(id, {
      type: 'segmented',
      active: Number(act || 0),
      options: (arr || []).map(String),
    }, o);
    return activeMethod(n, 'set_segmented');
  };

  ui.button = function (id, label, opts) {
    var o, lbl;
    if (typeof label === 'string') { o = opts || {}; lbl = label; }
    else { o = isObj(label) ? label : (opts || {}); lbl = id; id = null; }
    return makeNode(id, {
      type: 'button',
      label: String(lbl || ''),
      tone: o.tone || 'neutral',
    }, o);
  };

  ui.switch = function (id, checked, label, opts) {
    var o, chk, lbl;
    if (typeof id === 'boolean') {
      chk = id;
      if (isObj(checked)) { o = checked; lbl = null; }
      else { o = isObj(label) ? label : (opts || {}); lbl = checked; }
      id = null;
    } else { o = opts || {}; chk = checked; lbl = label; }
    var n = makeNode(id, {
      type: 'switch',
      checked: !!chk,
      label: (lbl !== undefined && lbl !== null) ? String(lbl) : null,
      tone: o.tone || 'neutral',
    }, o);
    Object.defineProperty(n, 'checked', {
      configurable: true,
      get: function () { return !!n.kind.checked; },
    });
    return put(n, 'setChecked', function (v) {
      n.kind.checked = !!v;
      if (mounted) sendPatch({ op: 'set_switch', id: n.id, checked: n.kind.checked });
      return n;
    });
  };

  ui.input = function (id, placeholder, opts) {
    var o, ph;
    if (typeof placeholder === 'string') { o = Object.assign({ height: 34.0 }, opts || {}); ph = placeholder; }
    else if (isObj(placeholder)) { o = Object.assign({ height: 34.0 }, placeholder); ph = id; id = null; }
    else { o = Object.assign({ height: 34.0 }, opts || {}); ph = ''; }
    var n = makeNode(id, {
      type: 'input',
      value: String(o.value || ''),
      placeholder: String(ph || ''),
      tone: o.tone || 'neutral',
    }, o);
    return put(n, 'setValue', function (v) {
      n.kind.value = String(v);
      if (mounted) sendPatch({ op: 'set_input', id: n.id, value: n.kind.value });
      return n;
    });
  };

  function statKind(label, value, o) {
    return {
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
      variant: o.variant || (o.range ? 'range_bar' : (o.sparkline ? 'sparkline' : 'compact')),
      sparkline: Array.isArray(o.sparkline) ? o.sparkline.map(Number) : [],
      range_bounds: Array.isArray(o.range) && o.range.length === 2 ? [Number(o.range[0]), Number(o.range[1])] : null,
      interactive: !!o.interactive,
    };
  }

  function statPatch(id, f) {
    var p = { op: 'update_stat', id: id };
    if (f.value !== undefined) p.value = Number(f.value);
    if (f.valueText !== undefined) p.value_text = f.valueText;
    if (f.delta !== undefined) p.delta = f.delta;
    if (f.tone !== undefined) p.tone = f.tone;
    if (f.caption !== undefined) p.caption = f.caption;
    if (f.pushSparkline !== undefined) p.push_sparkline = Number(f.pushSparkline);
    if (f.sparklineMaxLen !== undefined) p.sparkline_max_len = Number(f.sparklineMaxLen);
    return p;
  }

  ui.stat = function (id, label, value, opts) {
    var o, lbl, val;
    if (typeof label === 'number') { lbl = id; val = label; o = isObj(value) ? value : {}; id = null; }
    else { o = opts || {}; lbl = label; val = value; }
    var n = makeNode(id, statKind(lbl, val, o), o);
    return put(n, 'update', function (f) {
      var k = n.kind, x = f || {};
      if (x.value !== undefined) k.value = Number(x.value);
      if (x.valueText !== undefined) k.value_text = x.valueText;
      if (x.delta !== undefined) k.delta = x.delta;
      if (x.tone !== undefined) k.tone = x.tone;
      if (x.caption !== undefined) k.caption = x.caption;
      if (x.pushSparkline !== undefined) {
        k.sparkline.push(Number(x.pushSparkline));
        var lim = Number(x.sparklineMaxLen || 48);
        if (k.sparkline.length > lim) k.sparkline.splice(0, k.sparkline.length - lim);
      }
      if (mounted) sendPatch(statPatch(n.id, x));
      return n;
    });
  };

  function liveSeries(k, sid) {
    var groups = [k.series || []], i, j;
    for (i = 0; i < (k.panes || []).length; i++) groups.push(k.panes[i].series || []);
    for (i = 0; i < groups.length; i++) {
      for (j = 0; j < groups[i].length; j++) if (groups[i][j].id === sid) return groups[i][j];
    }
    return null;
  }

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

  function plotDataPatch(id, d) {
    var p = { op: 'set_plot_data', id: id }, any = false;
    if (d.title !== undefined) { p.title = d.title; any = true; }
    if (d.subtitle !== undefined) { p.subtitle = d.subtitle; any = true; }
    if (d.series !== undefined) { p.series = (d.series || []).map(normalizeSeries); any = true; }
    if (d.panes !== undefined) { p.panes = d.panes || []; any = true; }
    if (d.thresholds !== undefined) { p.thresholds = d.thresholds || []; any = true; }
    return any ? p : null;
  }

  ui.plot = function (id, opts) {
    var o;
    if (isObj(id)) { o = id; id = null; } else { o = opts || {}; }
    var n = makeNode(id, {
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
    put(n, 'append', function (sid, point, maxPoints) {
      var p = { op: 'append_plot_point', id: n.id, series_id: String(sid), point: point };
      if (maxPoints !== undefined) p.max_points = Number(maxPoints);
      var ser = liveSeries(n.kind, String(sid));
      if (ser) {
        ser.points.push(point);
        var lm = Number(maxPoints || 0);
        if (lm > 0 && ser.points.length > lm) ser.points.splice(0, ser.points.length - lm);
      }
      if (mounted) sendPatch(p);
      return n;
    });
    put(n, 'upsertCandle', function (sid, candle, maxCandles) {
      var p = { op: 'upsert_candle', id: n.id, series_id: String(sid), candle: candle };
      if (maxCandles !== undefined) p.max_candles = Number(maxCandles);
      var ser = liveSeries(n.kind, String(sid));
      if (ser) {
        var last = ser.candles[ser.candles.length - 1];
        if (last && Math.abs(last.x - candle.x) < 1e-6) ser.candles[ser.candles.length - 1] = candle;
        else ser.candles.push(candle);
        var cap = Number(maxCandles || 500);
        if (ser.candles.length > cap) ser.candles.splice(0, ser.candles.length - cap);
      }
      if (mounted) sendPatch(p);
      return n;
    });
    return put(n, 'setData', function (d) {
      var x = d || {}, p = plotDataPatch(n.id, x);
      if (x.title !== undefined) n.kind.title = x.title;
      if (x.subtitle !== undefined) n.kind.subtitle = x.subtitle;
      if (x.series !== undefined) n.kind.series = (x.series || []).map(normalizeSeries);
      if (x.panes !== undefined) n.kind.panes = x.panes || [];
      if (x.thresholds !== undefined) n.kind.thresholds = x.thresholds || [];
      if (p && mounted) sendPatch(p);
      return n;
    });
  };

  function normalizeColumns(cols) {
    return (cols || []).map(function (c) {
      return {
        key: String(c.key || ''),
        label: String(c.label || c.key || ''),
        width: (c.width !== undefined) ? Number(c.width) : null,
        align: c.align || 'left',
        sortable: (c.sortable !== undefined) ? !!c.sortable : true,
      };
    });
  }

  function normalizeRows(rows) {
    return (rows || []).map(function (r) {
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
  }

  ui.table = function (id, opts) {
    var o;
    if (isObj(id)) { o = id; id = null; } else { o = opts || {}; }
    var n = makeNode(id, {
      type: 'table',
      title: (o.title !== undefined) ? o.title : null,
      columns: normalizeColumns(o.columns),
      rows: normalizeRows(o.rows),
      selected_row: o.selectedRow || null,
      row_height: Number(o.rowHeight || 32.0),
      striped: (o.striped !== undefined) ? !!o.striped : true,
      empty_text: (o.emptyText !== undefined) ? o.emptyText : null,
    }, o);
    put(n, 'setRows', function (rows, selectedRow) {
      var rr = normalizeRows(rows);
      n.kind.rows = rr;
      var p = { op: 'set_table_rows', id: n.id, rows: rr };
      if (selectedRow !== undefined) {
        n.kind.selected_row = selectedRow === null ? null : String(selectedRow);
        p.selected_row = n.kind.selected_row;
      }
      if (mounted) sendPatch(p);
      return n;
    });
    return put(n, 'selectRow', function (rowId) {
      n.kind.selected_row = rowId === undefined || rowId === null ? null : String(rowId);
      if (mounted) sendPatch({ op: 'select_table_row', id: n.id, row_id: n.kind.selected_row });
      return n;
    });
  };

  ui.feed = function (id, opts) {
    var o;
    if (isObj(id)) { o = id; id = null; } else { o = opts || {}; }
    var n = makeNode(id, {
      type: 'feed',
      title: (o.title !== undefined) ? o.title : null,
      items: Array.isArray(o.items) ? o.items : [],
      max_items: Number(o.maxItems || 200),
      newest_first: (o.newestFirst !== undefined) ? !!o.newestFirst : true,
      empty_text: (o.emptyText !== undefined) ? o.emptyText : null,
    }, o);
    return put(n, 'append', function (item, maxItems) {
      var it = item || {};
      n.kind.items.push(it);
      var max = Number(maxItems || o.maxItems || 200);
      if (n.kind.items.length > max) n.kind.items.splice(0, n.kind.items.length - max);
      if (mounted) {
        var p = { op: 'append_feed_item', id: n.id, item: it };
        if (maxItems !== undefined) p.max_items = Number(maxItems);
        sendPatch(p);
      }
      return n;
    });
  };

  ui.graph = function (id, opts) {
    var o;
    if (isObj(id)) { o = id; id = null; } else { o = opts || {}; }
    var n = makeNode(id, {
      type: 'graph',
      title: (o.title !== undefined) ? o.title : null,
      nodes: Array.isArray(o.nodes) ? o.nodes : [],
      edges: Array.isArray(o.edges) ? o.edges : [],
      selected_node: o.selectedNode || null,
      empty_text: (o.emptyText !== undefined) ? o.emptyText : null,
    }, o);
    put(n, 'setGraph', function (d) {
      var x = d || {};
      var p = { op: 'set_graph', id: n.id, nodes: x.nodes || [], edges: x.edges || [] };
      if (x.title !== undefined) p.title = x.title;
      if (x.selectedNode !== undefined) p.selected_node = x.selectedNode;
      n.kind.nodes = p.nodes;
      n.kind.edges = p.edges;
      if (x.title !== undefined) n.kind.title = x.title;
      if (x.selectedNode !== undefined) n.kind.selected_node = x.selectedNode;
      if (mounted) sendPatch(p);
      return n;
    });
    return put(n, 'selectNode', function (nodeId) {
      n.kind.selected_node = nodeId === undefined || nodeId === null ? null : String(nodeId);
      if (mounted) sendPatch({ op: 'select_graph_node', id: n.id, node_id: n.kind.selected_node });
      return n;
    });
  };

  ui.code = function (id, content, opts) {
    var o, src;
    if (typeof content === 'string') { o = opts || {}; src = content; }
    else { o = content || {}; src = id; id = null; }
    var n = makeNode(id, {
      type: 'code_text',
      title: (o.title !== undefined) ? o.title : null,
      language: (o.language !== undefined) ? o.language : null,
      content: String(src !== undefined && src !== null ? src : ''),
      show_line_numbers: (o.lineNumbers !== undefined) ? !!o.lineNumbers : true,
      highlights: Array.isArray(o.highlights) ? o.highlights : [],
    }, o);
    return put(n, 'setContent', function (v, highlights) {
      n.kind.content = String(v !== undefined && v !== null ? v : '');
      if (highlights !== undefined) n.kind.highlights = highlights;
      if (mounted) {
        var p = { op: 'set_code', id: n.id, content: n.kind.content };
        if (highlights !== undefined) p.highlights = highlights;
        sendPatch(p);
      }
      return n;
    });
  };

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

  function routeHandlers(ev) {
    var h = nodeHandlers[ev.id];
    if (!h) return;
    var k = h.n.kind, o = h.o;
    try {
      if (ev.type === 'click') {
        if (o.onClick) o.onClick(ev);
      } else if (ev.type === 'switch_toggled') {
        k.checked = !!ev.checked;
        if (o.onChange) o.onChange(!!ev.checked, ev);
      } else if (ev.type === 'text_changed' || ev.type === 'input_submitted') {
        k.value = String(ev.value || '');
        var cb = ev.type === 'text_changed' ? o.onChange : o.onSubmit;
        if (cb) cb(String(ev.value || ''), ev);
      } else if (ev.type === 'segmented_changed') {
        k.active = ev.index;
        if (o.onChange) o.onChange(ev.index, (k.options || [])[ev.index], ev);
      } else if (ev.type === 'tab_changed') {
        k.active = ev.index;
        if (o.onChange) o.onChange(ev.index, ((k.items || [])[ev.index] || {}).label, ev);
      } else if (ev.type === 'row_selected') {
        k.selected_row = ev.row_id;
        if (o.onSelect) o.onSelect(ev.row_id, ev);
      } else if (ev.type === 'table_sorted') {
        if (o.onSort) o.onSort(ev.col_index, ev.asc, ev);
      } else if (ev.type === 'feed_item_clicked') {
        if (o.onItemClick) o.onItemClick(ev.item_id, ev);
      } else if (ev.type === 'graph_node_clicked') {
        k.selected_node = ev.node_id;
        if (o.onNodeClick) o.onNodeClick(ev.node_id, ev);
      }
    } catch (e) { /* one handler must not break the event pump. */ }
  }

  function drainEventsOnce() {
    if (typeof g.__sofuu_ui_poll_event !== 'function') return;
    while (true) {
      var raw = g.__sofuu_ui_poll_event();
      if (!raw) break;
      var ev = null;
      try { ev = JSON.parse(raw); } catch (e) { continue; }
      if (ev && ev.id) routeHandlers(ev);
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

  ui.signal = function (initial) {
    var value = initial, subs = [];
    var s = {
      get: function () { return value; },
      set: function (next) {
        if (next === value) return value;
        value = next;
        var list = subs.slice();
        for (var i = 0; i < list.length; i++) {
          try { list[i](value); } catch (e) { /* one subscriber must not break the rest. */ }
        }
        return value;
      },
      update: function (fn) { return s.set(fn(value)); },
      subscribe: function (fn, immediate) {
        if (typeof fn !== 'function') return function () {};
        subs.push(fn);
        if (immediate) { try { fn(value); } catch (e) {} }
        return function () {
          subs = subs.filter(function (f) { return f !== fn; });
        };
      },
    };
    return s;
  };

  var userMounted = false;
  var autoEnabled = true;
  var autoInstance = null;
  var autoSeen = false;
  var promptHandler = null;
  var activeAgentHint = null;

  function sendPatch(patch) {
    if (typeof g.__sofuu_ui_push_patch !== 'function') return -2;
    return g.__sofuu_ui_push_patch(JSON.stringify(patch));
  }

  function createSurface(id, opts, isInternalAuto) {
    var o = opts || {};
    var defTitle = String(o.title || id || 'Sofuu');
    var defTheme = (o.theme === 'dark') ? 'dark' : 'light';

    var view = {
      id: String(id || 'main'),
      mount: function (sceneOrNode) {
        if (!isInternalAuto) userMounted = true;
        mounted = true;
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
        ensurePump();
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
      setSegmented: function (nodeId, active) {
        return sendPatch({ op: 'set_segmented', id: String(nodeId), active: Number(active) || 0 });
      },
      setTab: function (nodeId, active) {
        return sendPatch({ op: 'set_tab', id: String(nodeId), active: Number(active) || 0 });
      },
      selectTableRow: function (tableId, rowId) {
        return sendPatch({
          op: 'select_table_row',
          id: String(tableId),
          row_id: (rowId === undefined || rowId === null) ? null : String(rowId),
        });
      },
      setTableRows: function (tableId, rows, selectedRow) {
        var p = { op: 'set_table_rows', id: String(tableId), rows: normalizeRows(rows) };
        if (selectedRow !== undefined) {
          p.selected_row = selectedRow === null ? null : String(selectedRow);
        }
        return sendPatch(p);
      },
      setPlotData: function (plotId, data) {
        var p = plotDataPatch(String(plotId), data || {});
        return p ? sendPatch(p) : 0;
      },
      setGraph: function (graphId, data) {
        var x = data || {};
        var p = { op: 'set_graph', id: String(graphId), nodes: x.nodes || [], edges: x.edges || [] };
        if (x.title !== undefined) p.title = x.title;
        if (x.selectedNode !== undefined) p.selected_node = x.selectedNode;
        return sendPatch(p);
      },
      setChildren: function (nodeId, children) {
        return sendPatch({
          op: 'set_children',
          id: String(nodeId),
          children: (children || []).map(unwrapNode).filter(Boolean),
        });
      },
      bindTrigger: function (runner, binds) {
        var b = binds || {};
        var sw = b.liveSwitch || null;
        var paused = false;
        if (sw) {
          paused = sw.checked === false;
          var baseGate = runner.gate;
          runner.gate = function (payload, ctx) {
            if (paused) return false;
            return baseGate(payload, ctx);
          };
          view.onEvent(function (ev) {
            if (ev && ev.type === 'switch_toggled' && ev.id === sw.id) paused = !ev.checked;
          });
        }
        view.onEvent(function (ev) {
          if (ev && ev.type === 'window_closed') {
            try { runner.stop('ui_closed'); } catch (e) {}
          }
        });
        var done = Promise.allSettled([runner.start(), view.waitUntilClosed()]);
        done.isPaused = function () { return paused; };
        return done;
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
  }

  ui.surface = function (id, opts) {
    return createSurface(id, opts, false);
  };

  ui.enableAuto = function (flag) {
    autoEnabled = flag !== false;
  };

  ui.onPrompt = function (fn) {
    promptHandler = (typeof fn === 'function') ? fn : null;
  };

  ui.autoComposer = function (opts) {
    var o = opts || {};
    var title = String(o.title || 'Sofuu Agent Runtime');
    var theme = o.theme === 'dark' ? 'dark' : 'light';
    var view = createSurface('auto_main_surface', { title: title, theme: theme }, true);
    var pending = [];
    var promptValue = '';

    function forward(name, payload) {
      var fn = g[name];
      if (typeof fn !== 'function') return -2;
      return fn(JSON.stringify(payload));
    }

    function resolve(allow, always) {
      var front = pending.shift();
      if (!front) return;
      if (g.sofuu && g.sofuu.agent && typeof g.sofuu.agent.resolveApproval === 'function') {
        g.sofuu.agent.resolveApproval(front.id, allow, always);
      }
      forward('__sofuu_ui_auto_event', {
        kind: 'approval_resolved',
        approval: { id: front.id, tool: front.tool || '', args_preview: '', mode: '' },
      });
    }

    function submit(raw) {
      var text = String(raw !== undefined ? raw : promptValue).trim();
      if (!text) return;
      promptValue = '';
      view.setInput('auto_prompt', '');
      if (promptHandler) {
        try { promptHandler(text, composer); } catch (eP) {}
        return;
      }
      var ag = g.sofuu && g.sofuu.agent;
      if (activeAgentHint && ag && typeof ag.run === 'function') {
        Promise.resolve(ag.run(activeAgentHint, text, { approveInteractive: true })).catch(function () {});
      }
    }

    view.onEvent(function (ev) {
      if (!ev) return;
      if (ev.type === 'text_changed' && ev.id === 'auto_prompt') {
        promptValue = String(ev.value || '');
      } else if ((ev.type === 'input_submitted' && ev.id === 'auto_prompt') ||
                 (ev.type === 'click' && ev.id === 'auto_run_btn')) {
        submit(ev.value);
      } else if (ev.type === 'click' && ev.id === 'auto_approve_btn') {
        resolve(true, false);
      } else if (ev.type === 'click' && ev.id === 'auto_always_btn') {
        resolve(true, true);
      } else if (ev.type === 'click' && ev.id === 'auto_deny_btn') {
        resolve(false, false);
      }
    });

    var composer = {
      view: view,
      buildScene: function () { return JSON.parse(g.__sofuu_ui_auto_scene()); },
      mount: function () {
        forward('__sofuu_ui_auto_mount', { title: title, theme: theme });
        return composer;
      },
      clearApproval: function (id) {
        pending = pending.filter(function (p) { return p.id !== String(id); });
        forward('__sofuu_ui_auto_event', {
          kind: 'approval_resolved',
          approval: { id: String(id), tool: '', args_preview: '', mode: '' },
        });
      },
      updateTriggers: function (list) {
        forward('__sofuu_ui_auto_triggers', Array.isArray(list) ? list : []);
      },
      updateStateRows: function (rows) {
        forward('__sofuu_ui_auto_rows', Array.isArray(rows) ? rows : []);
      },
      ingestEvent: function (ev) {
        if (!ev) return;
        if (ev.agentName) activeAgentHint = String(ev.agentName);
        var kind = String(ev.kind || '');
        var pl = ev.payload || {};
        if (kind === 'approval_request' && pl.id &&
            !pending.some(function (p) { return p.id === String(pl.id); })) {
          pending.push({ id: String(pl.id), tool: String(pl.tool || '') });
        } else if (kind === 'approval_resolved' && pl.id) {
          pending = pending.filter(function (p) { return p.id !== String(pl.id); });
        }
        forward('__sofuu_ui_auto_event', ev);
      },
    };
    return composer;
  };

  function ensureAutoInstance(seedTitle) {
    if (!autoEnabled || userMounted || !ui.isAttached()) return null;
    if (!autoInstance) {
      autoInstance = ui.autoComposer({ title: seedTitle || 'Sofuu Agent Runtime' }).mount();
    }
    return autoInstance;
  }

  ui._onAgentStep = function (ev) {
    autoSeen = true;
    var ac = ensureAutoInstance(ev && ev.agentName ? ('Sofuu // ' + ev.agentName) : null);
    if (ac) ac.ingestEvent(ev);
  };

  ui._onTriggerTick = function (stats) {
    autoSeen = true;
    var ac = ensureAutoInstance('Sofuu // Reactive Runtime');
    if (ac) ac.updateTriggers(stats);
  };

  function consoleEvent(level, text) {
    if (userMounted || autoSeen) return;
    var sink = g.__sofuu_ui_auto_event;
    if (typeof sink === 'function') sink(JSON.stringify({
      kind: level, title: text,
      tone: level === 'error' ? 'danger' : (level === 'warn' ? 'warn' : 'info'),
    }));
  }

  function bootConsole() {
    if (!ui.isAttached() || !g.console) return;
    var levels = ['log', 'info', 'warn', 'error'];
    for (var i = 0; i < levels.length; i++) (function (level) {
      var orig = g.console[level];
      if (typeof orig !== 'function') return;
      g.console[level] = function () {
        consoleEvent(level, [].join.call(arguments, ' '));
        return orig.apply(g.console, arguments);
      };
    })(levels[i]);
  }

  bootConsole();

  sofuu.ui = ui;
})();
