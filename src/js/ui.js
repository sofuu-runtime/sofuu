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
  var activeTheme = 'light';

  function isObj(v) {
    return v !== null && typeof v === 'object' && !Array.isArray(v);
  }

  function autoId() {
    return '_u' + (++uidSeq);
  }

  function nodeId(id, o) {
    return String(id || (o && o.id) || autoId());
  }

  var KEYBOARD_HINTS = {
    text: 'text', email: 'email', number: 'number', phone: 'phone', url: 'url',
    search: 'search', password: 'password', decimal: 'decimal',
  };

  /* Soft-keyboard hint for editable fields; hosts map it to their IME. */
  function keyboardHint(o, secure) {
    var raw = o.keyboard || o.inputMode || o.inputType || (secure ? 'password' : 'text');
    var key = String(raw).toLowerCase();
    return KEYBOARD_HINTS[key] || 'text';
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

  /* Optional layout constraints (serde defaults keep older scenes byte-identical). */
  function layoutHintsOf(o) {
    if (o.layout && typeof o.layout === 'object') return o.layout;
    var h = null;
    function ensure() { if (!h) h = {}; return h; }
    if (o.maxWidth !== undefined) ensure().max_width = Number(o.maxWidth);
    if (o.maxHeight !== undefined) ensure().max_height = Number(o.maxHeight);
    if (o.aspectRatio !== undefined) ensure().aspect_ratio = Number(o.aspectRatio);
    if (o.flexShrink !== undefined) ensure().flex_shrink = Number(o.flexShrink);
    if (o.alignSelf !== undefined) ensure().align_self = o.alignSelf;
    if (o.z !== undefined) ensure().z = Number(o.z) | 0;
    if (o.inset && typeof o.inset === 'object') {
      var ins = {};
      if (o.inset.left !== undefined) ins.left = Number(o.inset.left);
      if (o.inset.right !== undefined) ins.right = Number(o.inset.right);
      if (o.inset.top !== undefined) ins.top = Number(o.inset.top);
      if (o.inset.bottom !== undefined) ins.bottom = Number(o.inset.bottom);
      ensure().inset = ins;
    }
    return h;
  }

  function radiusOf(r, fallback) {
    if (r === undefined || r === null) return fallback;
    if (typeof r === 'number') return [r, r, r, r];
    if (Array.isArray(r)) {
      if (r.length === 1) return [Number(r[0]), Number(r[0]), Number(r[0]), Number(r[0])];
      if (r.length === 2) return [Number(r[0]), Number(r[0]), Number(r[1]), Number(r[1])];
      if (r.length === 4) return [Number(r[0]), Number(r[1]), Number(r[2]), Number(r[3])];
    }
    return fallback;
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
    var hints = layoutHintsOf(o);
    if (hints) n.layout = hints;
    put(n, 'toNode', function () { return n; });
    put(n, 'setVisible', function (v) {
      n.visible = !!v;
      if (mounted) sendPatch({ op: 'set_visible', id: n.id, visible: !!v });
      return n;
    });
    if (o.onClick || o.onChange || o.onSubmit || o.onSelect || o.onSort ||
        o.onItemClick || o.onNodeClick || o.onBack || o.onDrag || o.onRefresh) {
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
    return t === 'stat' || t === 'plot' || t === 'table' || t === 'feed' || t === 'graph' ||
      t === 'code_text' || t === 'list_view' || t === 'nav_stack' || t === 'text_area';
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
    if (typeof placeholder === 'string') { o = Object.assign({ height: 48.0 }, opts || {}); ph = placeholder; }
    else if (isObj(placeholder)) { o = Object.assign({ height: 48.0 }, placeholder); ph = id; id = null; }
    else { o = Object.assign({ height: 48.0 }, opts || {}); ph = ''; }
    var n = makeNode(id, {
      type: 'input',
      value: String(o.value || ''),
      placeholder: String(ph || ''),
      tone: o.tone || 'neutral',
      keyboard: keyboardHint(o, false),
      prefix_icon: o.prefixIcon ? String(o.prefixIcon) : null,
      clearable: !!o.clearable,
      error: (o.error !== undefined && o.error !== null) ? String(o.error) : null,
    }, o);
    return put(n, 'setValue', function (v) {
      n.kind.value = String(v);
      if (mounted) sendPatch({ op: 'set_input', id: n.id, value: n.kind.value });
      return n;
    });
  };

  ui.richText = function (id, spans, opts) {
    var o, list;
    if (Array.isArray(id)) { o = spans || {}; list = id; id = null; }
    else { o = opts || {}; list = spans || []; }
    function normSpans(arr) {
      return (arr || []).map(function (sp) {
        if (typeof sp === 'string') {
          return { text: sp, color: null, weight: 400, mono: false, size: null };
        }
        return {
          text: String(sp.text !== undefined ? sp.text : ''),
          color: (sp.color !== undefined) ? sp.color : null,
          weight: (sp.weight !== undefined) ? Number(sp.weight) : 400,
          mono: !!sp.mono,
          size: (sp.size !== undefined) ? Number(sp.size) : null,
        };
      });
    }
    var n = makeNode(id, {
      type: 'rich_text',
      spans: normSpans(list),
      size: Number(o.size || 14.0),
      wrap: !!o.wrap,
      line_height: (o.lineHeight !== undefined) ? Number(o.lineHeight) : null,
    }, o);
    return put(n, 'setSpans', function (next) {
      n.kind.spans = normSpans(next);
      if (mounted) sendPatch({ op: 'set_rich_text', id: n.id, spans: n.kind.spans });
      return n;
    });
  };

  ui.icon = function (id, name, opts) {
    var o, nm;
    if (typeof name === 'string') { o = opts || {}; nm = name; }
    else { o = isObj(name) ? name : (opts || {}); nm = String(id); id = null; }
    var size = Number(o.size || 20.0);
    if (o.width === undefined) o = Object.assign({}, o, { width: size });
    if (o.height === undefined) o = Object.assign({}, o, { height: size });
    var n = makeNode(id, {
      type: 'icon',
      name: String(nm || ''),
      size: size,
      color: (o.color !== undefined) ? o.color : null,
    }, o);
    return put(n, 'setIcon', function (next, color) {
      n.kind.name = String(next || '');
      if (color !== undefined) n.kind.color = color;
      if (mounted) sendPatch({ op: 'set_icon', id: n.id, name: n.kind.name, color: n.kind.color });
      return n;
    });
  };

  ui.image = function (id, src, opts) {
    var o, s;
    if (typeof src === 'string') { o = opts || {}; s = src; }
    else { o = isObj(src) ? src : (opts || {}); s = String(id); id = null; }
    var n = makeNode(id, {
      type: 'image',
      src: String(s || ''),
      fit: o.fit || 'cover',
      radius: radiusOf(o.radius, [0, 0, 0, 0]),
      tint: (o.tint !== undefined) ? o.tint : null,
      placeholder: (o.placeholder !== undefined) ? o.placeholder : null,
      alt: String(o.alt || ''),
    }, o);
    return put(n, 'setSrc', function (next) {
      n.kind.src = String(next || '');
      if (mounted) sendPatch({ op: 'set_image', id: n.id, src: n.kind.src });
      return n;
    });
  };

  ui.stack = function (id, children, opts) {
    var o, list;
    if (Array.isArray(id)) { o = children || {}; list = id; id = null; }
    else { o = opts || {}; list = children || []; }
    var n = makeNode(id, {
      type: 'stack',
      align: o.align || 'center',
      gap: (o.gap !== undefined) ? Number(o.gap) : 0.0,
      style: o.style || null,
      children: list.map(unwrapNode).filter(Boolean),
    }, o);
    return put(n, 'setChildren', function (next) {
      n.kind.children = (next || []).map(unwrapNode).filter(Boolean);
      if (mounted) sendPatch({ op: 'set_children', id: n.id, children: n.kind.children });
      return n;
    });
  };

  ui.safeArea = function (id, child, opts) {
    var o, c;
    if (child && typeof child === 'object' && !child.kind && !child.toNode && !Array.isArray(child)) {
      o = child; c = null;
    } else { o = opts || {}; c = unwrapNode(child); }
    var ins = o.insets || {};
    if (o.padding !== undefined && !o.insets) {
      ins = (typeof o.padding === 'number')
        ? { top: o.padding, right: o.padding, bottom: o.padding, left: o.padding }
        : o.padding;
    }
    return makeNode(id, {
      type: 'safe_area',
      insets: {
        top: Number(ins.top || 0), right: Number(ins.right || 0),
        bottom: Number(ins.bottom || 0), left: Number(ins.left || 0),
      },
      keyboard_avoidance: (o.keyboardAvoidance !== undefined) ? !!o.keyboardAvoidance : true,
      keyboard_height: Number(o.keyboardHeight || 0),
      child: c || null,
    }, o);
  };

  ui.overlay = function (id, child, opts) {
    var o, c;
    if (child && typeof child === 'object' && !child.kind && !child.toNode) { o = child; c = null; }
    else { o = opts || {}; c = unwrapNode(child); }
    return makeNode(id, {
      type: 'overlay',
      scrim: (o.scrim !== undefined) ? o.scrim : 0x99000000,
      dismissable: !!o.dismissable,
      align: o.align || 'center',
      style: o.style || null,
      child: c || null,
    }, o);
  };

  function normRoutes(list) {
    return (list || []).map(function (r) {
      return {
        id: String(r.id || autoId()),
        title: String(r.title || ''),
        child: unwrapNode(r.child),
      };
    });
  }

  ui.navStack = function (id, routes, opts) {
    var o, list;
    if (Array.isArray(id)) { o = routes || {}; list = id; id = null; }
    else { o = opts || {}; list = routes || []; }
    var kids = normRoutes(list);
    var n = makeNode(id, {
      type: 'nav_stack',
      routes: kids,
      active: Number(o.active || 0),
      progress: 0.0,
      swipe_back: (o.swipeBack !== undefined) ? !!o.swipeBack : true,
    }, o);
    return put(n, 'push', function (route) {
      var r = normRoutes([route])[0];
      if (mounted) sendPatch({ op: 'nav_push', id: n.id, route: r });
      else { n.kind.routes.push(r); n.kind.active = n.kind.routes.length - 1; }
      return n;
    });
    put(n, 'pop', function () {
      if (mounted) sendPatch({ op: 'nav_pop', id: n.id });
      else if (n.kind.active > 0) { n.kind.routes.splice(n.kind.active, 1); n.kind.active--; }
      return n;
    });
    put(n, 'replace', function (route) {
      var r = normRoutes([route])[0];
      if (mounted) sendPatch({ op: 'nav_replace', id: n.id, route: r });
      else { n.kind.routes[n.kind.active] = r; }
      return n;
    });
    put(n, 'popToRoot', function () {
      if (mounted) sendPatch({ op: 'nav_pop_to_root', id: n.id });
      else { n.kind.routes = n.kind.routes.slice(0, 1); n.kind.active = 0; }
      return n;
    });
    put(n, 'setProgress', function (p) {
      n.kind.progress = Math.max(0, Math.min(1, Number(p) || 0));
      if (mounted) sendPatch({ op: 'set_nav_progress', id: n.id, progress: n.kind.progress });
      return n;
    });
    return n;
  };

  function normItems(list) {
    return (list || []).map(function (it) {
      return {
        id: String(it.id || autoId()),
        height: (it.height !== undefined && it.height !== null) ? Number(it.height) : null,
        child: unwrapNode(it.child),
      };
    });
  }

  ui.listView = function (id, items, opts) {
    var o, list;
    if (Array.isArray(id)) { o = items || {}; list = id; id = null; }
    else { o = opts || {}; list = items || []; }
    var n = makeNode(id, {
      type: 'list_view',
      items: normItems(list),
      estimated_height: Number(o.estimatedHeight || 56.0),
      spacing: Number(o.spacing || 0.0),
      overscan: (o.overscan !== undefined) ? Number(o.overscan) : 1.0,
      style: o.style || null,
    }, o);
    return put(n, 'setItems', function (next) {
      n.kind.items = normItems(next);
      if (mounted) sendPatch({ op: 'set_list_items', id: n.id, items: n.kind.items });
      return n;
    });
  };

  ui.textArea = function (id, placeholder, opts) {
    var o, ph;
    if (typeof placeholder === 'string') { o = opts || {}; ph = placeholder; }
    else if (isObj(placeholder)) { o = placeholder; ph = id; id = null; }
    else { o = opts || {}; ph = ''; }
    var n = makeNode(id, {
      type: 'text_area',
      value: String(o.value || ''),
      placeholder: String(ph || ''),
      rows: Number(o.rows || 3),
      tone: o.tone || 'neutral',
      secure: !!o.secure,
      max_len: Number(o.maxLen || 0),
      keyboard: keyboardHint(o, !!o.secure),
    }, o);
    return put(n, 'setValue', function (v) {
      n.kind.value = String(v);
      if (mounted) sendPatch({ op: 'set_text_area', id: n.id, value: n.kind.value });
      return n;
    });
  };

  ui.slider = function (id, opts) {
    var o = isObj(id) ? id : (opts || {});
    if (isObj(id)) id = null;
    var n = makeNode(id, {
      type: 'slider',
      value: Number(o.value || 0),
      min: Number(o.min || 0),
      max: (o.max !== undefined) ? Number(o.max) : 1.0,
      step: Number(o.step || 0),
      tone: o.tone || 'neutral',
      label: (o.label !== undefined) ? String(o.label) : null,
      show_value: (o.showValue !== undefined) ? !!o.showValue : true,
    }, o);
    return put(n, 'setValue', function (v) {
      n.kind.value = Number(v) || 0;
      if (mounted) sendPatch({ op: 'set_slider', id: n.id, value: n.kind.value });
      return n;
    });
  };

  ui.rangeSlider = function (id, opts) {
    var o = isObj(id) ? id : (opts || {});
    if (isObj(id)) id = null;
    var hi = (o.high !== undefined) ? Number(o.high)
      : (o.value !== undefined) ? Number(o.value)
      : ((o.max !== undefined) ? Number(o.max) : 1.0);
    var n = makeNode(id, {
      type: 'range_slider',
      low: Number(o.low || 0),
      high: hi,
      min: Number(o.min || 0),
      max: (o.max !== undefined) ? Number(o.max) : 1.0,
      step: Number(o.step || 0),
      tone: o.tone || 'neutral',
      label: (o.label !== undefined) ? String(o.label) : null,
      show_value: (o.showValue !== undefined) ? !!o.showValue : true,
    }, o);
    return put(n, 'setValue', function (low, high) {
      n.kind.low = Number(low) || 0;
      n.kind.high = (high !== undefined) ? Number(high) : n.kind.high;
      if (mounted) {
        sendPatch({ op: 'set_range_slider', id: n.id, low: n.kind.low, high: n.kind.high });
      }
      return n;
    });
  };

  /* Deterministic meter (`linear`) or ring/spinner (`circular`). */
  ui.progress = function (id, opts) {
    var o = isObj(id) ? id : (opts || {});
    if (isObj(id)) id = null;
    var v = (o.value !== undefined) ? Number(o.value) : 0;
    /* Accept 0..100 for convenience; the IR stores a 0..1 fraction. */
    if (v > 1) v = v / 100;
    var n = makeNode(id, {
      type: 'progress',
      value: Math.max(0, Math.min(1, v)),
      variant: (o.variant === 'circular') ? 'circular' : 'linear',
      tone: o.tone || 'neutral',
      label: (o.label !== undefined) ? String(o.label) : null,
      indeterminate: !!o.indeterminate,
    }, o);
    return put(n, 'setValue', function (value, indeterminate) {
      var next = Number(value);
      if (next > 1) next = next / 100;
      n.kind.value = Math.max(0, Math.min(1, next));
      if (indeterminate !== undefined) n.kind.indeterminate = !!indeterminate;
      if (mounted) {
        sendPatch({
          op: 'set_progress', id: n.id, value: n.kind.value,
          indeterminate: n.kind.indeterminate,
        });
      }
      return n;
    });
  };

  ui.pressable = function (id, child, opts) {
    var o, c;
    if (child && typeof child === 'object' && !child.kind && !child.toNode) { o = child; c = null; }
    else { o = opts || {}; c = unwrapNode(child); }
    return makeNode(id, {
      type: 'pressable',
      hit_slop: Number(o.hitSlop || 0),
      disabled: !!o.disabled,
      child: c || null,
    }, o);
  };

  ui.draggable = function (id, child, opts) {
    var o, c;
    if (child && typeof child === 'object' && !child.kind && !child.toNode) { o = child; c = null; }
    else { o = opts || {}; c = unwrapNode(child); }
    var off = o.offset || [0, 0];
    var n = makeNode(id, {
      type: 'draggable',
      axis: o.axis || 'both',
      offset: [Number(off[0] || 0), Number(off[1] || 0)],
      inertia: (o.inertia !== undefined) ? !!o.inertia : true,
      child: c || null,
    }, o);
    return put(n, 'setOffset', function (x, y) {
      n.kind.offset = [Number(x) || 0, Number(y) || 0];
      if (mounted) sendPatch({ op: 'set_draggable', id: n.id, offset: n.kind.offset });
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

  /* Active scene theme (`"light"` / `"dark"`). Component kits use this to
   * pick readable text/icon colors; apps can use it to theme their own
   * custom nodes. */
  ui.theme = function () { return activeTheme; };

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
    activeTheme = cfg.theme;
    var b = {
      dark: function () { cfg.theme = 'dark'; activeTheme = 'dark'; return b; },
      light: function () { cfg.theme = 'light'; activeTheme = 'light'; return b; },
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
      } else if (ev.type === 'value_changed') {
        k.value = Number(ev.value);
        if (o.onChange) o.onChange(Number(ev.value), ev);
      } else if (ev.type === 'range_changed') {
        k.low = Number(ev.low);
        k.high = Number(ev.high);
        if (o.onChange) o.onChange(Number(ev.low), Number(ev.high), ev);
      } else if (ev.type === 'list_item_clicked') {
        if (o.onItemClick) o.onItemClick(ev.index, ev.item_id, ev);
      } else if (ev.type === 'back_requested') {
        if (k.type === 'nav_stack') {
          k.routes.splice(k.active, 1);
          if (k.active > 0) k.active--;
        }
        if (o.onBack) o.onBack(String(ev.route || ''), ev);
      } else if (ev.type === 'dragged') {
        if (o.onDrag) o.onDrag(Number(ev.dx) || 0, Number(ev.dy) || 0, ev);
      } else if (ev.type === 'refresh_requested') {
        if (o.onRefresh) o.onRefresh(ev);
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
      if (ev && ev.type === 'emu_action') {
        emuHandleAction(ev.action);
        continue;
      }
      if (ev && ev.type === 'click' && emuKeyTap(ev.id)) continue;
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
    if (pollTimer !== null || windowClosed || !ui.isAttached() || ui.isHeadless()) return;
    /* `sofuu ui emu` pumps from its own Rust loop: that loop must not run the
     * libuv event loop (it also watches files and services snapshot
     * requests), so a JS interval here would either never fire or block the
     * reload watcher. */
    if (emuLiveConfig()) return;
    if (typeof setInterval === 'function') {
      pollTimer = setInterval(drainEventsOnce, 16);
    }
  }

  ui.isAttached = function () {
    return typeof g.__sofuu_ui_is_attached === 'function' && !!g.__sofuu_ui_is_attached();
  };

  /* Headless capture reports attached (so apps mount normally) but must not
   * start the 16ms poll timer: `sofuu ui snapshot` drains events explicitly. */
  ui.isHeadless = function () {
    return typeof g.__sofuu_ui_headless === 'function' && !!g.__sofuu_ui_headless();
  };

  /* Headless/emulator hook: drain host events synchronously. The windowed
   * path uses setInterval; capture (`sofuu ui snapshot`, `sofuu.emu.*`) has
   * no timer loop, so it calls this after queueing a synthetic event. */
  ui.pump = function () { drainEventsOnce(); };

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

  /* Derived state: recompute whenever one of `deps` changes, and only then
   * notify subscribers (the value is compared by `signal.set`). */
  ui.computed = function (compute, deps) {
    var out = ui.signal(compute());
    var recompute = function () {
      var next;
      try { next = compute(); } catch (e) { return; }
      out.set(next);
    };
    var list = deps || [];
    for (var i = 0; i < list.length; i++) {
      if (list[i] && typeof list[i].subscribe === 'function') list[i].subscribe(recompute);
    }
    return out;
  };

  /* Run `run` now and again on every dependency change. Returned function
   * stops the effect and any cleanup callback `run` returned. */
  ui.effect = function (run, deps) {
    var stops = [];
    var cleanup = null;
    var fire = function () {
      // A re-run disposes the previous run first, so listeners/timers a
      // producer installed never stack up across dependency changes.
      if (typeof cleanup === 'function') {
        try { cleanup(); } catch (e) {}
        cleanup = null;
      }
      try {
        var next = run();
        if (typeof next === 'function') cleanup = next;
      } catch (e) { /* an effect must not break the event pump. */ }
    };
    fire();
    var list = deps || [];
    for (var i = 0; i < list.length; i++) {
      if (list[i] && typeof list[i].subscribe === 'function') stops.push(list[i].subscribe(fire));
    }
    return function () {
      for (var k = 0; k < stops.length; k++) {
        try { stops[k](); } catch (e) {}
      }
      stops.length = 0;
      if (typeof cleanup === 'function') {
        try { cleanup(); } catch (e) {}
        cleanup = null;
      }
    };
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

  /* ── Interactive emulator shell (PLAN-GPU-PLATFORM-UI-EMULATOR E3) ────
   * `sofuu ui emu` installs globalThis.__sofuu_emu_live (platform, device,
   * profiles, theme, orientation, keyboard) before evaluating the app. Every
   * mounted scene is wrapped in a DevTools toolbar + platform-exact device
   * frame, so the app's own layout runs untouched inside the viewport while
   * the surrounding chrome is pure JS. Headless capture never installs the
   * global, so `sofuu ui snapshot` still renders the app alone. */
  var emuLiveCfg = null;

  function emuLiveConfig() {
    if (emuLiveCfg === null) {
      var cfg = g.__sofuu_emu_live;
      if (cfg && typeof cfg === 'object') emuLiveCfg = cfg;
    }
    return emuLiveCfg;
  }

  function emuRgba(r, gg, b, a) {
    return (((r & 255) | ((gg & 255) << 8) | ((b & 255) << 16) |
      (((a === undefined ? 255 : a) & 255) << 24)) >>> 0);
  }

  function emuProfileById(cfg, id) {
    var list = (cfg && Array.isArray(cfg.profiles)) ? cfg.profiles : [];
    for (var i = 0; i < list.length; i++) {
      if (list[i] && list[i].id === id) return list[i];
    }
    return null;
  }

  function emuDeviceById(profile, id) {
    if (!profile || !Array.isArray(profile.devices) || !profile.devices.length) return null;
    if (id) {
      for (var i = 0; i < profile.devices.length; i++) {
        var d = profile.devices[i];
        if (d && (d.id === id || d.name === id)) return d;
      }
    }
    return profile.devices[0];
  }

  function emuMetrics(cfg) {
    var prof = emuProfileById(cfg, cfg.platform);
    var dev = emuDeviceById(prof, cfg.device);
    var w = dev ? Number(dev.width) : Number(cfg.width || 0);
    var h = dev ? Number(dev.height) : Number(cfg.height || 0);
    if (!(w > 0)) w = 393;
    if (!(h > 0)) h = 852;
    var orientation = (cfg.orientation === 'landscape') ? 'landscape' : 'portrait';
    var kb = dev
      ? Number(orientation === 'landscape'
        ? (dev.keyboard_landscape || 0)
        : (dev.keyboard_portrait || 0))
      : 260;
    var status = dev ? Number(dev.status_bar_height || 0) : 0;
    var titlebar = dev ? Number(dev.titlebar_height || 0) : 0;
    var home = dev ? Number(dev.home_bar_height || 0) : 0;
    var radius = dev ? Number(dev.corner_radius || 0) : 0;
    var scale = dev ? Number(dev.scale || 1) : Number(cfg.scale || 1);
    var punch = dev && dev.punch_hole ? dev.punch_hole : null;
    if (orientation === 'landscape') { var t = w; w = h; h = t; }
    return {
      profile: prof, device: dev, orientation: orientation,
      width: w, height: h, keyboardHeight: kb,
      status: status, titlebar: titlebar, home: home,
      radius: radius, scale: scale,
      safeLeft: dev ? Number(dev.safe_left || 0) : 0,
      safeRight: dev ? Number(dev.safe_right || 0) : 0,
      punch: punch,
    };
  }

  function emuPalette(theme) {
    var dark = theme === 'dark';
    return {
      dark: dark,
      bg: dark ? emuRgba(18, 19, 23) : emuRgba(238, 241, 245),
      screen: dark ? emuRgba(12, 13, 16) : emuRgba(255, 255, 255),
      chrome: dark ? emuRgba(32, 34, 40) : emuRgba(249, 250, 252),
      text: dark ? emuRgba(232, 234, 238) : emuRgba(28, 30, 34),
      muted: dark ? emuRgba(150, 155, 164) : emuRgba(108, 112, 120),
      border: dark ? emuRgba(255, 255, 255, 46) : emuRgba(0, 0, 0, 40),
      shadow: emuRgba(0, 0, 0, dark ? 120 : 64),
      key: dark ? emuRgba(66, 69, 78) : emuRgba(255, 255, 255),
      keyboard: dark ? emuRgba(42, 44, 50) : emuRgba(214, 218, 224),
      island: emuRgba(0, 0, 0),
    };
  }

  function emuPx(v) { return { px: Number(v) }; }
  function emuFill(v) { return { fill: (v === undefined ? 1.0 : Number(v)) }; }

  function emuRaw(id, width, height, kind, padding) {
    return {
      id: id,
      priority: 100,
      visible: true,
      width: width || 'auto',
      height: height || 'auto',
      min_width: 0,
      min_height: 0,
      padding: padding || { top: 0, right: 0, bottom: 0, left: 0 },
      kind: kind,
    };
  }

  function emuCol(id, kids, o) {
    o = o || {};
    return emuRaw(id, o.width, o.height, {
      type: 'container',
      axis: 'column',
      align: o.align || 'stretch',
      gap: (o.gap !== undefined ? Number(o.gap) : 0),
      fold_width: null,
      scrollable: !!o.scrollable,
      style: o.style || null,
      children: kids,
    }, o.padding);
  }

  function emuRow(id, kids, o) {
    o = o || {};
    return emuRaw(id, o.width, o.height, {
      type: 'container',
      axis: 'row',
      align: o.align || 'center',
      gap: (o.gap !== undefined ? Number(o.gap) : 0),
      fold_width: null,
      scrollable: !!o.scrollable,
      style: o.style || null,
      children: kids,
    }, o.padding);
  }

  function emuSpacer(id, w, h) {
    return emuRaw(id || ('__emu_sp_' + (++uidSeq)), w || 'auto', h || 'auto', { type: 'spacer' });
  }

  function emuText(id, content, size, color, mono) {
    return emuRaw(id, 'auto', 'auto', {
      type: 'text',
      content: String(content === undefined || content === null ? '' : content),
      size: Number(size || 12),
      color: color,
      mono: !!mono,
      wrap: false,
    });
  }

  function emuSurface(id, w, h, style, child) {
    return emuRaw(id, w, h, {
      type: 'surface',
      style: style,
      interactive: false,
      child: child || null,
    });
  }

  function emuStatus(st, m, pal) {
    var chrome = m.profile ? m.profile.chrome : 'ios';
    var h = m.status > 0 ? Math.max(m.status, 22) : 0;
    if (!h) return null;
    var left = emuText('__emu_status_time', '9:41', 12, pal.text);
    var right = emuText('__emu_status_right', chrome === 'android' ? 'WiFi  100%' : '5G  100%', 12, pal.text);
    var mid;
    if (m.punch) {
      var d = Number(m.punch.diameter || 12);
      mid = emuSurface('__emu_punch', emuPx(d), emuPx(d), {
        bg: pal.island, border: null, border_width: 0,
        radius: [d / 2, d / 2, d / 2, d / 2], opacity: 1,
      }, null);
    } else if (chrome === 'ios') {
      mid = emuSurface('__emu_island', emuPx(118), emuPx(34), {
        bg: pal.island, border: null, border_width: 0,
        radius: [17, 17, 17, 17], opacity: 1,
      }, null);
    } else {
      mid = emuSpacer('__emu_status_mid', emuFill(1), 'auto');
    }
    var kids = [left];
    if (m.punch || chrome === 'ios') {
      kids.push(emuSpacer('__emu_status_l', emuFill(1), 'auto'));
      kids.push(mid);
      kids.push(emuSpacer('__emu_status_r', emuFill(1), 'auto'));
    } else {
      kids.push(mid);
    }
    kids.push(right);
    return emuRow('__emu_status', kids, {
      height: emuPx(h),
      padding: { top: 0, right: 16, bottom: 0, left: 16 },
      align: 'center',
      style: { bg: pal.screen, border: null, border_width: 0, radius: [0, 0, 0, 0], opacity: 1 },
    });
  }

  function emuTitlebar(st, m, pal) {
    var h = m.titlebar > 0 ? Math.max(m.titlebar, 24) : 0;
    if (!h) return null;
    var title = st.app && st.app.title ? st.app.title : 'Sofuu';
    var chrome = m.profile ? m.profile.chrome : 'macos';
    var kids = [];
    if (chrome === 'macos') {
      var dots = ['__emu_dot_r', '__emu_dot_y', '__emu_dot_g'];
      var colors = [emuRgba(255, 95, 86), emuRgba(255, 189, 46), emuRgba(39, 201, 63)];
      for (var i = 0; i < 3; i++) {
        kids.push(emuSurface(dots[i], emuPx(11), emuPx(11), {
          bg: colors[i], border: null, border_width: 0,
          radius: [6, 6, 6, 6], opacity: 1,
        }, null));
      }
      kids.push(emuSpacer('__emu_tb_l', emuFill(1), 'auto'));
      kids.push(emuText('__emu_tb_title', title, 12, pal.muted));
      kids.push(emuSpacer('__emu_tb_r', emuFill(1), 'auto'));
    } else {
      kids.push(emuText('__emu_tb_title', title, 12, pal.muted));
      kids.push(emuSpacer('__emu_tb_l', emuFill(1), 'auto'));
      kids.push(emuText('__emu_tb_win', '-  []  x', 12, pal.muted));
    }
    return emuRow('__emu_titlebar', kids, {
      height: emuPx(h),
      padding: { top: 0, right: 12, bottom: 0, left: 12 },
      gap: 7,
      align: 'center',
      style: { bg: pal.chrome, border: null, border_width: 0, radius: [0, 0, 0, 0], opacity: 1 },
    });
  }

  var EMU_KEY_ROWS = [
    ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P'],
    ['A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L'],
    ['Z', 'X', 'C', 'V', 'B', 'N', 'M'],
    ['123', 'space', '⌫', 'return'],
  ];

  var EMU_NUM_ROWS = [
    ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
    ['-', '/', ':', ';', '(', ')', '$', '&', '@', '"'],
    ['#+=', '.', ',', '?', '!', "'", '⌫'],
    ['ABC', 'space', 'return'],
  ];

  var EMU_SYM_ROWS = [
    ['[', ']', '{', '}', '#', '%', '^', '*', '+', '='],
    ['_', '\\', '|', '~', '<', '>', '€', '£', '¥', '•'],
    ['123', '.', ',', '?', '!', "'", '⌫'],
    ['ABC', 'space', 'return'],
  ];

  function emuKeyRows(layer) {
    if (layer === 1) return EMU_NUM_ROWS;
    if (layer === 2) return EMU_SYM_ROWS;
    return EMU_KEY_ROWS;
  }

  function emuKeyboard(st, m, pal) {
    var rows = emuKeyRows(st.keyboardLayer);
    var kids = [];
    for (var r = 0; r < rows.length; r++) {
      var keys = [];
      for (var k = 0; k < rows[r].length; k++) {
        var label = rows[r][k];
        var weight = 1;
        if (label === 'space') weight = 3.2;
        else if (label === '123' || label === 'ABC' || label === 'return' || label === '#+=') weight = 1.35;
        var key = emuRaw('__emu_key_' + r + '_' + k, emuFill(weight), emuFill(1), {
          type: 'button', label: label, tone: 'neutral',
        });
        /* Tapping a key must not blur the input it is typing into. */
        key.preserve_focus = true;
        keys.push(key);
      }
      kids.push(emuRow('__emu_kb_row_' + r, keys, { height: emuFill(1), gap: 5 }));
    }
    return emuCol('__emu_keyboard', kids, {
      width: emuFill(1),
      height: emuPx(Math.max(m.keyboardHeight, 160)),
      gap: 5,
      padding: { top: 6, right: 6, bottom: 6, left: 6 },
      style: {
        bg: pal.keyboard, border: null, border_width: 0,
        radius: [0, 0, 0, 0], opacity: 1,
      },
    });
  }

  function emuHomeBar(m, pal) {
    var h = m.home > 0 ? Math.max(m.home, 20) : 0;
    if (!h) return null;
    var bar = emuSurface('__emu_home_indicator', emuPx(134), emuPx(5), {
      bg: pal.text, border: null, border_width: 0,
      radius: [3, 3, 3, 3], opacity: 1,
    }, null);
    return emuRow('__emu_home', [
      emuSpacer('__emu_home_l', emuFill(1), 'auto'),
      bar,
      emuSpacer('__emu_home_r', emuFill(1), 'auto'),
    ], {
      height: emuPx(h),
      align: 'center',
      style: { bg: pal.screen, border: null, border_width: 0, radius: [0, 0, 0, 0], opacity: 1 },
    });
  }

  function emuBuildShell(cfg, st, rebuild) {
    /* `st` carries the mutable toolbar state (platform/device/theme/orient),
     * while the profile manifest and window size live on the root config.
     * Merging them keeps device chrome (status bar, island, safe sides,
     * keyboard height) correct after a live platform/device switch. */
    var m = emuMetrics({
      platform: st.platform,
      device: st.device,
      orientation: st.orientation,
      profiles: cfg.profiles || [],
      width: cfg.width,
      height: cfg.height,
      scale: cfg.scale,
    });
    var pal = emuPalette(st.theme);
    var screenKids = [];
    var status = emuStatus(st, m, pal);
    var titlebar = emuTitlebar(st, m, pal);
    if (status) screenKids.push(status);
    else if (titlebar) screenKids.push(titlebar);

    var appRoot = st.app && st.app.root ? st.app.root : st.app;
    if (appRoot) {
      if (!appRoot.width || appRoot.width === 'auto') appRoot.width = emuFill(1);
      if (!appRoot.height || appRoot.height === 'auto') appRoot.height = emuFill(1);
      screenKids.push(emuCol('__emu_viewport', [appRoot], {
        width: emuFill(1),
        height: emuFill(1),
        padding: { top: 0, right: m.safeRight, bottom: 0, left: m.safeLeft },
      }));
    }
    if (st.keyboard && m.keyboardHeight > 0) screenKids.push(emuKeyboard(st, m, pal));
    var home = emuHomeBar(m, pal);
    if (home) screenKids.push(home);

    var screenBg = pal.screen;
    if (appRoot && appRoot.style && appRoot.style.bg !== undefined && appRoot.style.bg !== null) {
      screenBg = appRoot.style.bg;
    }
    var screen = emuSurface('__emu_screen', emuPx(m.width), emuPx(m.height), {
      bg: screenBg,
      border: pal.border,
      border_width: 1,
      radius: [m.radius, m.radius, m.radius, m.radius],
      shadow_color: pal.shadow,
      shadow_blur: 28,
      shadow_offset_y: 10,
      opacity: 1,
    }, emuCol('__emu_screen_col', screenKids, {
      width: emuFill(1),
      height: emuFill(1),
    }));

    /* E6: the window is the screen. No stage, bezel, toolbar or inspector —
     * the mounted app scene plus the simulated OS chrome is the window root.
     * Controls live in the native attached strip, outside the render path. */
    var title = cfg.title || ('Sofuu Emu - ' + (m.device ? m.device.name : st.platform));
    return { title: String(title), theme: st.theme, root: screen };
  }

  /* On-screen keyboard → engine input. The engine owns the focused field and
   * caret, so keys are injected as real TextInput/KeyDown events instead of
   * rewriting the scene from JS. */
  function emuKeyTap(id) {
    var m = /^__emu_key_(\d+)_(\d+)$/.exec(String(id || ''));
    if (!m) return false;
    var cfg = emuLiveConfig();
    var st = cfg && cfg.__state;
    var layer = st ? (st.keyboardLayer || 0) : 0;
    var row = emuKeyRows(layer)[Number(m[1])];
    if (!row) return false;
    var label = row[Number(m[2])];
    if (!label) return false;
    if (label === '123' || label === 'ABC' || label === '#+=') {
      if (st) {
        st.keyboardLayer = (label === '123') ? 1 : (label === '#+=' ? 2 : 0);
        if (typeof cfg.__rebuild === 'function') cfg.__rebuild();
      }
      return true;
    }
    var payload;
    if (label === 'space') {
      payload = { type: 'text_input', text: ' ' };
    } else if (label === 'return') {
      payload = { type: 'key_down', key: 'Enter', modifiers: 0 };
    } else if (label === '⌫') {
      payload = { type: 'key_down', key: 'Backspace', modifiers: 0 };
    } else {
      payload = { type: 'text_input', text: (layer === 0) ? label.toLowerCase() : label };
    }
    if (typeof g.__sofuu_ui_send_input === 'function') {
      g.__sofuu_ui_send_input(JSON.stringify(payload));
    }
    return true;
  }

  function emuHandleAction(action) {
    var cfg = emuLiveConfig();
    if (!cfg || !cfg.__state) return;
    var st = cfg.__state;
    var text = String(action || '');
    var sep = text.indexOf(':');
    var verb = sep >= 0 ? text.slice(0, sep) : text;
    var arg = sep >= 0 ? text.slice(sep + 1) : '';
    if (verb === 'platform' && arg) {
      st.platform = arg;
      st.device = null;
    } else if (verb === 'device' && arg) {
      st.device = arg;
    } else if (verb === 'theme') {
      st.theme = st.theme === 'dark' ? 'light' : 'dark';
    } else if (verb === 'rotate') {
      st.orientation = st.orientation === 'landscape' ? 'portrait' : 'landscape';
    } else if (verb === 'keyboard') {
      st.keyboard = !st.keyboard;
    } else if (verb === 'bounds') {
      st.bounds = !st.bounds;
      if (typeof g.__sofuu_ui_set_bounds === 'function') {
        g.__sofuu_ui_set_bounds(st.bounds);
      }
    } else if (verb === 'pause') {
      if (typeof g.__sofuu_ui_set_paused === 'function') g.__sofuu_ui_set_paused(true);
      return;
    } else if (verb === 'resume') {
      if (typeof g.__sofuu_ui_set_paused === 'function') g.__sofuu_ui_set_paused(false);
      return;
    } else if (verb === 'step') {
      if (typeof g.__sofuu_ui_step_frame === 'function') g.__sofuu_ui_step_frame();
      return;
    } else if (verb === 'snapshot') {
      if (typeof g.__sofuu_emu_request_capture === 'function') {
        g.__sofuu_emu_request_capture('annotated', JSON.stringify({
          platform: st.platform,
          device: st.device,
          theme: st.theme,
          orientation: st.orientation,
          keyboard: st.keyboard,
        }));
      }
      return;
    } else {
      /* `watch`, `collapse`, `expand` are host-only. */
      return;
    }
    if (typeof cfg.__rebuild === 'function') cfg.__rebuild();
  }

  function emuWrapScene(scene, sendPatch) {
    var cfg = emuLiveConfig();
    if (!cfg) return scene;
    if (!cfg.__state) {
      cfg.__state = {
        platform: cfg.platform || 'ios',
        device: cfg.device || null,
        theme: (cfg.theme === 'dark') ? 'dark' : 'light',
        orientation: cfg.orientation || 'portrait',
        keyboard: !!cfg.keyboardOpen,
        keyboardLayer: 0,
        bounds: false,
        inspector: cfg.inspector !== false,
        app: scene,
      };
    }
    var st = cfg.__state;
    st.app = scene;
    function rebuild() {
      var shell = emuBuildShell(cfg, st, rebuild);
      sendPatch({ op: 'replace_scene', title: shell.title, theme: shell.theme, root: shell.root });
    }
    cfg.__rebuild = rebuild;
    return emuBuildShell(cfg, st, rebuild);
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
        var wrapped = emuWrapScene(scene, sendPatch);
        activeTheme = (wrapped.theme === 'dark') ? 'dark' : 'light';
        sendPatch({
          op: 'replace_scene',
          title: wrapped.title,
          theme: wrapped.theme,
          root: wrapped.root,
        });
        ensurePump();
        return view;
      },
      patch: function (p) { return sendPatch(p); },
      setTheme: function (mode) {
        activeTheme = (mode === 'dark') ? 'dark' : 'light';
        return sendPatch({ op: 'set_theme', mode: mode });
      },
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
      setFx: function (nodeId, specs) {
        return sendPatch({ op: 'set_fx', id: String(nodeId), fx: specs || [] });
      },
      triggerFx: function (kind, opts) {
        var o = opts || {};
        var p = { op: 'fx_trigger', kind: String(kind) };
        if (o.target) p.target = String(o.target.id || o.target);
        if (o.x !== undefined) p.x = Number(o.x);
        if (o.y !== undefined) p.y = Number(o.y);
        if (o.count !== undefined) p.count = Number(o.count);
        if (o.seed !== undefined) p.seed = Number(o.seed);
        return sendPatch(p);
      },
      setFxAmplitude: function (nodeId, value) {
        return sendPatch({
          op: 'set_fx_amplitude',
          id: String(nodeId),
          value: Math.max(0, Math.min(1, Number(value) || 0)),
        });
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

  /* ── sofuu.fx — opt-in GPU effects & choreography ─────────────────────
   * Every builder is a decorator: it appends a declarative spec to the node
   * and returns the original handle. The Rust motion engine rewrites only
   * that node's emitted primitives, so layout/audit/hit-testing never move. */
  var fx = {};
  fx._loaded = true;

  function fxHandle(x) {
    if (!x) return null;
    if (typeof x.toNode === 'function') return x.toNode();
    return x;
  }

  function fxPick(o, camel, snake) {
    if (o[camel] !== undefined) return o[camel];
    if (o[snake] !== undefined) return o[snake];
    return undefined;
  }

  function fxSpec(kind, opts) {
    var o = opts || {};
    var s = { kind: kind };
    var v;
    v = fxPick(o, 'preset', 'preset'); if (v !== undefined) s.preset = String(v);
    v = fxPick(o, 'spring', 'spring'); if (v !== undefined) s.spring = String(v);
    v = fxPick(o, 'delayMs', 'delay_ms'); if (v !== undefined) s.delay_ms = Number(v);
    v = fxPick(o, 'durationMs', 'duration_ms'); if (v !== undefined) s.duration_ms = Number(v);
    v = fxPick(o, 'slideX', 'slide_x'); if (v !== undefined) s.slide_x = Number(v);
    v = fxPick(o, 'slideY', 'slide_y'); if (v !== undefined) s.slide_y = Number(v);
    v = fxPick(o, 'scale', 'scale'); if (v !== undefined) s.scale = Number(v);
    v = fxPick(o, 'from', 'from'); if (v !== undefined) s.from = Number(v);
    v = fxPick(o, 'to', 'to'); if (v !== undefined) s.to = Number(v);
    v = fxPick(o, 'precision', 'precision'); if (v !== undefined) s.precision = Number(v) | 0;
    v = fxPick(o, 'prefix', 'prefix'); if (v !== undefined) s.prefix = String(v);
    v = fxPick(o, 'suffix', 'suffix'); if (v !== undefined) s.suffix = String(v);
    v = fxPick(o, 'tone', 'tone'); if (v !== undefined) s.tone = String(v);
    v = fxPick(o, 'color0', 'color0'); if (v !== undefined) s.color0 = Number(v);
    v = fxPick(o, 'color1', 'color1'); if (v !== undefined) s.color1 = Number(v);
    v = fxPick(o, 'color2', 'color2'); if (v !== undefined) s.color2 = Number(v);
    v = fxPick(o, 'speed', 'speed'); if (v !== undefined) s.speed = Number(v);
    v = fxPick(o, 'intensity', 'intensity'); if (v !== undefined) s.intensity = Number(v);
    v = fxPick(o, 'radius', 'radius'); if (v !== undefined) s.radius = Number(v);
    v = fxPick(o, 'width', 'width'); if (v !== undefined) s.width = Number(v);
    v = fxPick(o, 'amplitude', 'amplitude');
    if (v !== undefined) {
      if (typeof v === 'number') s.amplitude = v;
      else if (v && typeof v.get === 'function') s.amplitude = Number(v.get()) || 0;
    }
    v = fxPick(o, 'active', 'active'); if (v !== undefined) s.active = !!v;
    v = fxPick(o, 'count', 'count'); if (v !== undefined) s.count = Number(v) | 0;
    v = fxPick(o, 'auto', 'auto'); if (v !== undefined) s.auto = !!v;
    v = fxPick(o, 'sharedKey', 'shared_key'); if (v !== undefined) s.shared_key = String(v);
    v = fxPick(o, 'blur', 'blur'); if (v !== undefined) s.blur = Number(v);
    v = fxPick(o, 'rim', 'rim'); if (v !== undefined) s.rim = Number(v);
    v = fxPick(o, 'seed', 'seed'); if (v !== undefined) s.seed = Number(v) | 0;
    v = fxPick(o, 'exit', 'exit'); if (v !== undefined) s.exit = !!v;
    return s;
  }

  function fxAttach(x, spec) {
    var n = fxHandle(x);
    if (!n) return x;
    if (!Array.isArray(n.fx)) n.fx = [];
    n.fx.push(spec);
    if (mounted) sendPatch({ op: 'set_fx', id: n.id, fx: n.fx });
    return x;
  }

  fx.spec = fxSpec;

  fx.enter = function (x, opts) { return fxAttach(x, fxSpec('enter', opts)); };
  fx.fade = function (x, opts) { return fxAttach(x, fxSpec('fade', opts)); };
  fx.slide = function (x, opts) { return fxAttach(x, fxSpec('slide', opts)); };
  fx.scalePop = function (x, opts) {
    var o = Object.assign({ preset: 'scale_pop' }, opts || {});
    return fxAttach(x, fxSpec('enter', o));
  };

  fx.stagger = function (items, opts) {
    var o = opts || {};
    var step = Number(fxPick(o, 'delayMs', 'delay_ms') || 28);
    var slideY = fxPick(o, 'slideY', 'slide_y');
    if (slideY === undefined) slideY = 12;
    var list = items || [];
    for (var i = 0; i < list.length; i++) {
      var item = Object.assign({}, o, {
        preset: o.preset || 'fade_slide',
        delayMs: step * i + Number(o.baseDelayMs || 0),
        slideY: slideY,
      });
      fxAttach(list[i], fxSpec('enter', item));
    }
    return items;
  };

  fx.shared = function (a, b) {
    var key = (typeof a === 'string') ? a : (typeof b === 'string' ? b : (a && a.key) || '');
    var node = (typeof a === 'string') ? b : a;
    return fxAttach(node, fxSpec('shared', { sharedKey: key }));
  };

  fx.shake = function (x) {
    var n = fxHandle(x);
    if (!n) return x;
    fxAttach(x, fxSpec('shake', { auto: true }));
    sendPatch({ op: 'fx_trigger', kind: 'shake', target: n.id });
    return x;
  };

  fx.numberRoll = function (x, opts) { return fxAttach(x, fxSpec('number_roll', opts)); };
  fx.typewriter = function (x, opts) { return fxAttach(x, fxSpec('typewriter', opts)); };
  fx.scramble = function (x, opts) { return fxAttach(x, fxSpec('text_scramble', opts)); };
  fx.textShimmer = function (x, opts) { return fxAttach(x, fxSpec('text_shimmer', opts)); };

  fx.shimmer = function (x, opts) { return fxAttach(x, fxSpec('shimmer', opts)); };
  fx.touchRipple = function (x, opts) { return fxAttach(x, fxSpec('touch_ripple', opts)); };
  fx.borderBeam = function (x, opts) { return fxAttach(x, fxSpec('border_beam', opts)); };
  fx.cursorGlow = function (x, opts) { return fxAttach(x, fxSpec('cursor_glow', opts)); };
  fx.aurora = function (x, opts) { return fxAttach(x, fxSpec('aurora', opts)); };
  fx.pulseRing = function (x, opts) { return fxAttach(x, fxSpec('pulse_ring', opts)); };

  fx.backdropBlur = function (x, opts) {
    var o = opts || {};
    if (o.blur === undefined && o.radius !== undefined) o = Object.assign({}, o, { blur: o.radius });
    return fxAttach(x, fxSpec('backdrop_blur', o));
  };

  fx.voiceOrb = function (x, opts) {
    var node = x;
    var o = opts || {};
    if (!node || (!node.kind && typeof node.toNode !== 'function' && node.id === undefined)) {
      o = x || {};
      node = makeNode(null, { type: 'spacer' }, {
        width: Number(o.size || 88),
        height: Number(o.size || 88),
      });
    }
    var spec = fxSpec('voice_orb', o);
    var amp = o.amplitude;
    if (amp && typeof amp.subscribe === 'function') {
      var id = fxHandle(node).id;
      if (typeof amp.get === 'function') spec.amplitude = Number(amp.get()) || 0;
      amp.subscribe(function (value) {
        if (mounted) sendPatch({ op: 'set_fx_amplitude', id: id, value: Number(value) || 0 });
      }, true);
    }
    return fxAttach(node, spec);
  };

  fx.burst = function (kind, opts) {
    var o = opts || {};
    var k = String(kind);
    if (k === 'sparkBurst') k = 'spark_burst';
    var p = { op: 'fx_trigger', kind: k };
    var origin = o.origin;
    if (origin) {
      var n = fxHandle(origin);
      if (n && n.id) p.target = String(n.id);
      else if (origin.x !== undefined && origin.y !== undefined) {
        p.x = Number(origin.x);
        p.y = Number(origin.y);
      }
    }
    if (o.target) {
      var t = fxHandle(o.target);
      p.target = String((t && t.id) || o.target);
    }
    if (o.x !== undefined) p.x = Number(o.x);
    if (o.y !== undefined) p.y = Number(o.y);
    if (o.count !== undefined) p.count = Number(o.count) | 0;
    if (o.seed !== undefined) p.seed = Number(o.seed) | 0;
    sendPatch(p);
    return origin || o.target || null;
  };
  fx.trigger = fx.burst;

  fx.confetti = function (origin, opts) {
    return fx.burst('confetti', Object.assign({}, opts || {}, { origin: origin }));
  };
  fx.sparkBurst = function (origin, opts) {
    return fx.burst('spark_burst', Object.assign({}, opts || {}, { origin: origin }));
  };

  fx.set = function (x, specs) {
    var n = fxHandle(x);
    if (!n) return x;
    n.fx = (specs || []).slice();
    if (mounted) sendPatch({ op: 'set_fx', id: n.id, fx: n.fx });
    return x;
  };
  fx.clear = function (x) { return fx.set(x, []); };

  fx.sharedHero = fx.shared;

  fx.motion = {
    fade: fx.fade, slide: fx.slide, scalePop: fx.scalePop, enter: fx.enter,
    stagger: fx.stagger, shake: fx.shake, shared: fx.shared,
    sharedHero: fx.shared,
  };
  fx.text = {
    numberRoll: fx.numberRoll, typewriter: fx.typewriter,
    scramble: fx.scramble, shimmer: fx.textShimmer, shimmerText: fx.textShimmer,
  };
  fx.surface = {
    shimmer: fx.shimmer, borderBeam: fx.borderBeam, cursorGlow: fx.cursorGlow,
    aurora: fx.aurora, backdropBlur: fx.backdropBlur,
  };
  fx.bursts = {
    ripple: fx.touchRipple, pulseRing: fx.pulseRing, confetti: fx.confetti,
    sparkBurst: fx.sparkBurst, voiceOrb: fx.voiceOrb,
  };
  // Both `fx.burst(...)` and `fx.burst.confetti(...)` are common in examples.
  fx.burst.ripple = fx.touchRipple;
  fx.burst.pulseRing = fx.pulseRing;
  fx.burst.confetti = fx.confetti;
  fx.burst.sparkBurst = fx.sparkBurst;
  fx.burst.voiceOrb = fx.voiceOrb;

  sofuu.fx = fx;
  sofuu.ui = ui;
})();
