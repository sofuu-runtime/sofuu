/* src/js/ui_kit.js — Tier-2 composable app widgets (`sofuu.uiKit`).
 *
 * Small, overridable recipes built exclusively from Tier-1 `sofuu.ui`
 * primitives, so apps (and AI agents) can copy one function and customize it
 * without touching Rust. Every widget returns a normal node handle with the
 * usual `setVisible`/patch methods.
 */
(function () {
  'use strict';

  var g = (typeof globalThis !== 'undefined') ? globalThis : this;
  var sofuu = g.sofuu;
  if (!sofuu || !sofuu.ui) return;
  var u = sofuu.ui;
  if (sofuu.uiKit && sofuu.uiKit._loaded) return;

  var kit = {};
  kit._loaded = true;

  /* ── design tokens ─────────────────────────────────────────────── */
  var TONES = {
    neutral: 0x64748b,
    info: 0x3b82f6,
    ok: 0x22c55e,
    warn: 0xf59e0b,
    danger: 0xef4444,
    accent: 0x3b82f6,
    violet: 0x8b5cf6,
  };

  /* Pack `0xRRGGBB` + alpha into the renderer's 0xAABBGGRR word. */
  function color(rgb, alpha) {
    var a = (alpha === undefined) ? 255 : alpha;
    return (((rgb & 0xff) << 16) | (rgb & 0xff00) | ((rgb >> 16) & 0xff) |
      ((a & 0xff) << 24)) >>> 0;
  }

  function tone(name, alpha) {
    return color(TONES[name] || TONES.neutral, alpha);
  }

  /* `500`-weight tones are too light for small text on either theme's
   * surface. Readable tints come from the 700 family (light theme) or the
   * 200 family (dark theme); the kit reads the active theme from `ui.theme()`. */
  var DEEP = {
    neutral: 0x334155, info: 0x1d4ed8, ok: 0x15803d, warn: 0xb45309,
    danger: 0xb91c1c, accent: 0x1d4ed8, violet: 0x6d28d9,
  };
  var BRIGHT = {
    neutral: 0xcbd5e1, info: 0x93c5fd, ok: 0x86efac, warn: 0xfcd34d,
    danger: 0xfca5a5, accent: 0x93c5fd, violet: 0xc4b5fd,
  };

  function isDark() {
    return !!(u.theme && u.theme() === 'dark');
  }

  function keyOf(name) {
    return TONES[name] ? name : 'neutral';
  }

  /* Readable accent/text color for the current theme. */
  function ink(name) {
    var k = keyOf(name);
    return color(isDark() ? BRIGHT[k] : DEEP[k], 255);
  }

  /* Solid fill that keeps >= 4.5:1 against white text on light and dark. */
  function deep(name) {
    return color(DEEP[keyOf(name)], 255);
  }

  /* Foreground that stays >= 4.5:1 on a `THREE`-family solid fill. */
  function onFill() {
    return color(0x0b1220, 255);
  }

  /* Minimum interactive box; iOS 44pt / Android 48dp. */
  var TOUCH = 48;

  function radius(r) {
    if (r === undefined) return [14, 14, 14, 14];
    if (typeof r === 'number') return [r, r, r, r];
    if (Array.isArray(r) && r.length === 4) return r;
    if (Array.isArray(r) && r.length === 2) return [r[0], r[0], r[1], r[1]];
    return [14, 14, 14, 14];
  }

  /* Accept both the JS-style camelCase option names and the raw IR
   * snake_case names, so `{ shadow_blur: 0 }` and `{ shadowBlur: 0 }` agree. */
  function pick(o, snake, camel) {
    if (o[snake] !== undefined) return o[snake];
    if (o[camel] !== undefined) return o[camel];
    return undefined;
  }

  function cardStyle(o, fallback) {
    var sb = pick(o, 'shadow_blur', 'shadowBlur');
    var so = pick(o, 'shadow_offset_y', 'shadowOffsetY');
    var sc = pick(o, 'shadow_color', 'shadowColor');
    var bw = pick(o, 'border_width', 'borderWidth');
    var hb = pick(o, 'hover_bg', 'hoverBg');
    var s = {
      bg: (o.bg !== undefined) ? o.bg : null,
      border: (o.border !== undefined) ? o.border : null,
      border_width: Number(bw === undefined ? 0 : bw),
      radius: radius(o.radius),
      shadow_color: (sc !== undefined) ? sc : null,
      shadow_blur: Number(sb === undefined ? 0 : sb),
      shadow_offset_y: Number(so === undefined ? 0 : so),
      hover_bg: (hb !== undefined) ? hb : null,
      opacity: 1,
    };
    if (o.style && typeof o.style === 'object') {
      for (var k in o.style) if (Object.prototype.hasOwnProperty.call(o.style, k)) s[k] = o.style[k];
    }
    return s;
  }

  function label(id, text, size, opts) {
    return u.text(id, String(text === undefined || text === null ? '' : text),
      Object.assign({ size: size || 13, width: { fill: 1 }, wrap: true }, opts || {}));
  }

  function icon(id, name, size, col) {
    return u.icon(id, name, { size: size || 18, color: col === undefined ? null : col });
  }

  /* Kit rows never fold: a list tile / app bar is a row even on a 320px
   * phone, and the Tier-1 `row` defaults to folding below 680px. */
  function row(id, children, opts) {
    var o = opts || {};
    if (o.foldBelow === undefined) o.foldBelow = 0;
    return u.row(id, children, o);
  }

  function panel(id, children, o, align, pad) {
    o = o || {};
    return u.stack(id, children || [], {
      width: o.width, height: o.height,
      fill: (o.width === undefined) ? (o.fill !== undefined ? o.fill : undefined) : undefined,
      align: align || 'top_left',
      gap: 0,
      padding: pad === undefined ? o.padding : pad,
      style: cardStyle(o),
    });
  }

  /* Vertical card body: unlike `panel` (a z-stack for badges/avatars), a
   * card/dialog/sheet stacks its title and content vertically. */
  function columnPanel(id, children, o, pad, gap) {
    o = o || {};
    if (pick(o, 'shadow_blur', 'shadowBlur') === undefined) {
      o = Object.assign({}, o, { shadowBlur: 16 });
    }
    return u.col(id, (children || []).filter(Boolean), {
      width: o.width, height: o.height,
      align: 'stretch',
      gap: gap === undefined ? Number(o.gap === undefined ? 10 : o.gap) : gap,
      padding: pad === undefined ? 0 : pad,
      style: cardStyle(o),
    });
  }

  /* ── Card ──────────────────────────────────────────────────────── */
  kit.Card = function (id, body, opts) {
    var o = opts || {};
    var pad = (o.padding === undefined) ? 14 : Number(o.padding);
    var kids = [];
    if (o.title) {
      kids.push(row(id + '_head', [
        label(id + '_title', o.title, o.titleSize || 14, { color: null, wrap: false }),
        o.action || null,
      ].filter(Boolean), { gap: 8, height: 22, width: { fill: 1 } }));
    }
    var content = body;
    if (Array.isArray(body)) content = u.col(id + '_body', body, { gap: Number(o.gap || 8), width: { fill: 1 } });
    if (content) kids.push(content);
    return columnPanel(id, kids, o, pad, Number(o.gap === undefined ? 10 : o.gap));
  };

  /* ── ListTile ──────────────────────────────────────────────────── */
  kit.ListTile = function (id, opts) {
    var o = opts || {};
    var leading = o.leading || (o.icon ? icon(id + '_icon', o.icon, 20, o.iconColor) : null);
    var title = label(id + '_title', o.title || '', o.titleSize || 13.5, { wrap: false });
    var lines = [title];
    if (o.subtitle) {
      lines.push(label(id + '_sub', o.subtitle, o.subtitleSize || 11.5, { wrap: false }));
    }
    var trailing = o.trailing || (o.chevron === false ? null : icon(id + '_chev', 'chevron-right', 16));
    var tileRow = row(id + '_row', [
      leading,
      u.col(id + '_text', lines, { gap: 2, width: { fill: 1 } }),
      trailing,
    ].filter(Boolean), {
      gap: 10, align: 'center', width: { fill: 1 },
      padding: Number(o.padding === undefined ? 10 : o.padding),
    });
    var n = u.pressable(id, tileRow, {
      width: o.width || { fill: 1 }, minHeight: TOUCH,
      onClick: o.onClick, disabled: o.disabled,
    });
    return n;
  };

  /* ── Badge ─────────────────────────────────────────────────────── */
  kit.Badge = function (id, text, opts) {
    var o = opts || {};
    var bg = o.bg !== undefined ? o.bg : tone(o.tone || 'neutral', 40);
    var fg = o.color !== undefined ? o.color : ink(o.tone || 'neutral');
    return panel(id, [label(id + '_txt', text, o.size || 10.5, { color: fg, wrap: false })], {
      bg: bg, radius: o.radius === undefined ? 999 : o.radius, border: 0, border_width: 0,
      shadow_blur: 0, width: o.width,
    }, 'center', Number(o.padding === undefined ? 6 : o.padding));
  };

  /* ── Chip ──────────────────────────────────────────────────────── */
  kit.Chip = function (id, text, opts) {
    var o = opts || {};
    var selected = !!o.selected;
    var name = o.tone || 'accent';
    var bg = selected ? tone(name, 46) : tone('neutral', 24);
    var fg = selected ? ink(name) : null;
    var inner = label(id + '_txt', text, 12, { color: fg, wrap: false });
    var n = u.pressable(id, panel(id + '_bg', [inner], {
      bg: bg, radius: 999,
      border: tone(name, selected ? 150 : 60), border_width: selected ? 1 : 1,
      shadow_blur: 0,
    }, 'center', Number(o.padding === undefined ? 9 : o.padding)), {
      onClick: o.onClick, minHeight: TOUCH,
    });
    return n;
  };

  /* ── Avatar ────────────────────────────────────────────────────── */
  kit.Avatar = function (id, opts) {
    var o = opts || {};
    var size = Number(o.size || 36);
    if (o.src) {
      return u.image(id, o.src, {
        width: size, height: size, radius: size / 2, fit: 'cover', alt: o.alt || '',
      });
    }
    var initials = String(o.initials || o.name || '?').trim().slice(0, 2).toUpperCase();
    var names = ['accent', 'violet', 'ok', 'warn', 'info', 'danger'];
    var seed = String(o.name || o.initials || initials);
    var hash = 0;
    for (var i = 0; i < seed.length; i++) hash = (hash * 31 + seed.charCodeAt(i)) >>> 0;
    var bg = deep(o.tone || names[hash % names.length]);
    return panel(id, [label(id + '_init', initials, size * 0.36, {
      color: color(0xffffff, 255), wrap: false,
    })], { bg: bg, radius: size / 2, border: 0, border_width: 0, shadow_blur: 0,
      width: size, height: size }, 'center', 0);
  };

  /* ── Dialog ────────────────────────────────────────────────────── */
  kit.Dialog = function (id, opts) {
    var o = opts || {};
    var acts = (o.actions || []).map(function (a, i) {
      return u.button(id + '_act' + i, a.label, {
        tone: a.tone || 'neutral', width: a.width, onClick: a.onClick,
      });
    });
    var body = o.body;
    if (Array.isArray(body)) body = u.col(id + '_body', body, { gap: 10, width: { fill: 1 } });
    var card = columnPanel(id + '_card', [
      o.title ? label(id + '_title', o.title, 16, { wrap: false }) : null,
      body,
      acts.length ? row(id + '_acts', acts, { gap: 8, align: 'end', width: { fill: 1 } }) : null,
    ], {
      bg: o.bg, radius: o.radius === undefined ? 16 : o.radius, border: o.border,
      border_width: o.borderWidth === undefined ? 1 : o.borderWidth,
      shadow_blur: 36, shadow_offset_y: 14, width: o.width === undefined ? 340 : o.width,
    }, 18, 12);
    var n = u.overlay(id, card, {
      scrim: o.scrim, dismissable: o.dismissable !== false, align: 'center',
      onClick: function (ev) { if (o.onDismiss) o.onDismiss(ev); },
    });
    n.toNode().visible = o.visible !== false;
    return n;
  };

  /* ── Toast ─────────────────────────────────────────────────────── */
  kit.Toast = function (id, text, opts) {
    var o = opts || {};
    var fg = o.color !== undefined ? o.color : color(0xffffff, 255);
    var bg = o.bg !== undefined ? o.bg : color(DEEP[keyOf(o.tone || 'neutral')] | 0, 245);
    var card = panel(id + '_card', [
      label(id + '_txt', text, 12.5, { color: fg, wrap: false }),
    ].filter(Boolean), {
      bg: bg, radius: 12, border: 0, border_width: 0, shadow_blur: 22, shadow_offset_y: 8,
      width: o.width,
    }, 'center', 12);
    var n = u.overlay(id, card, {
      scrim: 0, dismissable: false, align: 'bottom_left',
    });
    n.toNode().visible = o.visible !== false;
    if (o.duration !== undefined && typeof setTimeout === 'function' && n.toNode().visible) {
      setTimeout(function () { n.setVisible(false); }, Math.max(0, Number(o.duration)));
    }
    return n;
  };

  /* ── Sheet / bottom sheet ──────────────────────────────────────── */
  kit.Sheet = function (id, body, opts) {
    var o = opts || {};
    var content = Array.isArray(body) ? u.col(id + '_body', body, { gap: 10, width: { fill: 1 } }) : body;
    var card = columnPanel(id + '_card', [
      u.stack(id + '_grab', [], {
        width: 36, height: 4, style: { bg: tone('neutral', 120), radius: [2, 2, 2, 2], border_width: 0, shadow_blur: 0 },
      }),
      o.title ? label(id + '_title', o.title, 16, { wrap: false }) : null,
      content,
    ], {
      bg: o.bg, radius: [18, 18, 0, 0], border: o.border,
      border_width: o.borderWidth === undefined ? 1 : o.borderWidth,
      shadow_blur: 30, shadow_offset_y: -6, width: o.width,
    }, 16, 10);
    var n = u.overlay(id, card, {
      scrim: o.scrim, dismissable: o.dismissable !== false, align: 'bottom',
      onClick: function (ev) { if (o.onDismiss) o.onDismiss(ev); },
    });
    n.toNode().visible = o.visible !== false;
    return n;
  };

  /* ── Accordion ─────────────────────────────────────────────────── */
  kit.Accordion = function (id, items, opts) {
    var o = opts || {};
    var rows = (items || []).map(function (it, i) {
      var bodyId = id + '_body' + i;
      var body = Array.isArray(it.body) ? u.col(bodyId, it.body, { gap: 8 }) : it.body;
      var chev = icon(id + '_chev' + i, 'chevron-down', 16);
      var head = u.pressable(id + '_head' + i, row(id + '_hrow' + i, [
        label(id + '_htitle' + i, it.title || '', 13.5, { wrap: false, width: { fill: 1 } }),
        chev,
      ], { gap: 8, align: 'center', width: { fill: 1 } }), {
        width: { fill: 1 }, minHeight: TOUCH,
        onClick: function () {
          var next = body ? !body.toNode().visible : false;
          if (body) body.setVisible(next);
          chev.setIcon(next ? 'chevron-up' : 'chevron-down');
        },
      });
      if (body) body.setVisible(!!it.open);
      return u.col(id + '_item' + i, [head, body].filter(Boolean), { gap: 8, width: { fill: 1 } });
    });
    return u.col(id, rows, { gap: Number(o.gap === undefined ? 10 : o.gap), width: { fill: 1 } });
  };

  /* ── Bottom navigation ─────────────────────────────────────────── */
  kit.BottomNav = function (id, items, opts) {
    var o = opts || {};
    var active = Number(o.active || 0);
    var cells = (items || []).map(function (it, i) {
      var on = i === active;
      var col = on ? ink(o.activeTone || 'accent') : (o.inactiveColor || null);
      var cell = u.pressable(id + '_tab' + i, u.col(id + '_tabcol' + i, [
        icon(id + '_tabicon' + i, it.icon || 'circle', 20, col),
        label(id + '_tablabel' + i, it.label || '', 10.5, { color: col, wrap: false }),
      ], {
        gap: 3, align: 'center', width: { fill: 1 },
        padding: { top: 3, right: 4, bottom: 3, left: 4 },
        style: on ? {
          bg: tone(o.activeTone || 'accent', 34), radius: [10, 10, 10, 10],
          border_width: 0, shadow_blur: 0, opacity: 1,
        } : null,
      }), {
        minHeight: TOUCH,
        onClick: (function (idx, item) {
          return function (ev) { if (o.onChange) o.onChange(idx, item, ev); };
        })(i, it),
      });
      cell.toNode().width = { fill: 1 };
      return cell;
    });
    return row(id, cells, {
      gap: 0, align: 'center', width: { fill: 1 }, height: o.height || 56,
      padding: { top: 4, right: 8, bottom: 4, left: 8 },
      style: {
        bg: o.bg, border: o.border, border_width: o.border ? 1 : 0,
        radius: o.radius || [0, 0, 0, 0], shadow_blur: 0, opacity: 1,
      },
    });
  };

  /* ── Sidebar ───────────────────────────────────────────────────── */
  kit.Sidebar = function (id, items, opts) {
    var o = opts || {};
    var rows = (items || []).map(function (it, i) {
      var active = !!it.active;
      var cell = u.pressable(id + '_item' + i, row(id + '_row' + i, [
        it.icon ? icon(id + '_icon' + i, it.icon, 16,
          active ? ink('accent') : (o.iconColor || null)) : null,
        label(id + '_label' + i, it.label || '', 12.5,
          { color: active ? ink('accent') : null, wrap: false }),
        it.badge || null,
      ].filter(Boolean), {
        gap: 9, align: 'center', width: { fill: 1 },
        padding: { top: 8, right: 10, bottom: 8, left: 10 },
        style: active ? {
          bg: tone('accent', 30), radius: [9, 9, 9, 9],
          border_width: 0, shadow_blur: 0, opacity: 1,
        } : null,
      }), {
        minHeight: TOUCH,
        onClick: (function (idx, item) {
          return function (ev) { if (o.onSelect) o.onSelect(idx, item, ev); };
        })(i, it),
      });
      cell.toNode().width = { fill: 1 };
      return cell;
    });
    return u.col(id, rows, {
      gap: 2, width: o.width || 240, padding: { top: 10, right: 8, bottom: 10, left: 8 },
      style: {
        bg: o.bg, border: o.border, border_width: o.border ? 1 : 0,
        radius: o.radius || [0, 0, 0, 0], shadow_blur: 0, opacity: 1,
      },
    });
  };

  /* ── AppBar ────────────────────────────────────────────────────── */
  kit.AppBar = function (id, opts) {
    var o = opts || {};
    var left = o.leading || (o.back ? u.pressable(id + '_back', icon(id + '_backicon', 'chevron-left', 20), {
      onClick: o.onBack, hitSlop: 8, width: TOUCH, height: TOUCH,
    }) : null);
    var title = label(id + '_title', o.title || '', o.titleSize || 16, { wrap: false });
    var acts = o.actions || [];
    return row(id, [
      left,
      title,
      row(id + '_acts', acts, { gap: 4, align: 'center' }),
    ].filter(Boolean), {
      gap: 10, align: 'center', width: { fill: 1 }, height: o.height || 52,
      padding: { top: 6, right: 12, bottom: 6, left: 12 },
      style: {
        bg: o.bg, border: o.border, border_width: o.border ? 1 : 0,
        radius: o.radius || [0, 0, 0, 0], shadow_blur: o.shadowBlur || 0, opacity: 1,
      },
    });
  };

  /* ── Checkbox / Radio ──────────────────────────────────────────── */
  kit.Checkbox = function (id, labelText, opts) {
    var o = opts || {};
    var checked = !!o.checked;
    var box = panel(id + '_box', [icon(id + '_check', 'check', 14, color(0xffffff, 255))], {
      bg: checked ? deep(o.tone || 'accent') : tone('neutral', 20),
      radius: o.radius === undefined ? 5 : o.radius,
      border: tone(o.tone || 'accent', 180), border_width: 1.5, shadow_blur: 0,
      width: 20, height: 20,
    }, 'center', 0);
    if (!checked) box.toNode().kind.children[0].visible = false;
    var n = u.pressable(id, row(id + '_row', [
      box,
      labelText ? label(id + '_label', labelText, 13, { wrap: false, width: { fill: 1 } }) : null,
    ].filter(Boolean), { gap: 9, align: 'center', width: { fill: 1 } }), {
      width: { fill: 1 }, minHeight: TOUCH,
      disabled: o.disabled,
      onClick: function (ev) {
        checked = !checked;
        box.toNode().kind.style.bg = checked ? deep(o.tone || 'accent') : tone('neutral', 20);
        box.toNode().kind.children[0].visible = checked;
        if (o.onChange) o.onChange(checked, ev);
      },
    });
    n.checked = function () { return checked; };
    n.setChecked = function (v) {
      checked = !!v;
      box.toNode().kind.style.bg = checked ? deep(o.tone || 'accent') : tone('neutral', 20);
      box.toNode().kind.children[0].visible = checked;
      return n;
    };
    return n;
  };

  kit.Radio = function (id, labelText, opts) {
    var o = opts || {};
    var checked = !!o.checked;
    var dot = panel(id + '_dot', [], {
      bg: deep(o.tone || 'accent'), radius: 999, border: 0, border_width: 0,
      shadow_blur: 0, width: 9, height: 9,
    }, 'center', 0);
    var ring = panel(id + '_ring', [dot], {
      bg: tone('neutral', 16), radius: 999, border: checked ? deep(o.tone || 'accent') : tone(o.tone || 'accent', 120),
      border_width: 1.5, shadow_blur: 0, width: 20, height: 20,
    }, 'center', 0);
    if (!checked) dot.toNode().visible = false;
    var n = u.pressable(id, row(id + '_row', [
      ring,
      labelText ? label(id + '_label', labelText, 13, { wrap: false, width: { fill: 1 } }) : null,
    ].filter(Boolean), { gap: 9, align: 'center', width: { fill: 1 } }), {
      width: { fill: 1 }, minHeight: TOUCH,
      disabled: o.disabled,
      onClick: function (ev) {
        checked = true;
        dot.toNode().visible = true;
        ring.toNode().kind.style.border = deep(o.tone || 'accent');
        if (o.onChange) o.onChange(true, ev);
      },
    });
    n.checked = function () { return checked; };
    return n;
  };

  /* ── Slider ────────────────────────────────────────────────────── */
  kit.Slider = function (id, opts) {
    return u.slider(id, opts || {});
  };

  kit.RangeSlider = function (id, opts) {
    return u.rangeSlider(id, opts || {});
  };

  /* ── Progress ──────────────────────────────────────────────────── */
  kit.Progress = function (id, opts) {
    var o = opts || {};
    return u.progress(id, o);
  };

  /* Indeterminate spinner: a circular ring that sweeps while mounted. */
  kit.Spinner = function (id, opts) {
    var o = opts || {};
    return u.progress(id, {
      variant: 'circular',
      indeterminate: true,
      tone: o.tone || 'info',
      size: o.size || 24,
      width: o.size || 24,
      height: o.size || 24,
    });
  };

  /* ── Select (dropdown) ─────────────────────────────────────────── */
  kit.Select = function (id, options, opts) {
    var o = opts || {};
    var list = options || [];
    var active = Math.max(0, Math.min(Number(o.active || 0), list.length - 1));
    var valueLabel = label(id + '_value', list[active] !== undefined ? list[active].label : (o.placeholder || ''), 12.5, { wrap: false });
    var chev = icon(id + '_chev', 'chevron-down', 16);
    var trigger = u.pressable(id + '_trigger', row(id + '_trow', [
      valueLabel, chev,
    ], { gap: 8, align: 'center', padding: 10, width: { fill: 1 } }), {
      minHeight: TOUCH,
      onClick: function () {
        menu.setVisible(!menu.toNode().visible);
        chev.setIcon(menu.toNode().visible ? 'chevron-up' : 'chevron-down');
      },
    });
    trailer4(id + '_trigger', trigger, o);
    var rows = list.map(function (it, i) {
      return u.pressable(id + '_opt' + i, row(id + '_orow' + i, [
        label(id + '_olabel' + i, it.label || '', 12.5, { wrap: false }),
      ], { padding: 9, width: { fill: 1 } }), {
        minHeight: TOUCH,
        onClick: function (ev) {
          active = i;
          valueLabel.set(it.label || '');
          menu.setVisible(false);
          chev.setIcon('chevron-down');
          if (o.onChange) o.onChange(i, it, ev);
        },
      });
    });
    var menu = panel(id + '_menu', rows, {
      bg: o.menuBg, radius: 10, border: o.border, border_width: 1,
      shadow_blur: 18, shadow_offset_y: 6, width: o.menuWidth || 220,
    }, 'top_left', 4);
    menu.setVisible(false);
    var field = panel(id, [trigger, menu], {
      bg: o.bg, radius: o.radius === undefined ? 10 : o.radius,
      border: o.border || tone('neutral', 70), border_width: 1, shadow_blur: 0,
      width: o.width,
    }, 'top_left', 0);
    field.setValue = function (i) {
      active = Math.max(0, Math.min(Number(i) || 0, list.length - 1));
      if (list[active]) valueLabel.set(list[active].label || '');
      return field;
    };
    return field;
  };

  function trailer4(id, trigger, o) {
    trigger.toNode().width = { fill: 1 };
  }

  /* ── Skeleton ──────────────────────────────────────────────────── */
  kit.Skeleton = function (id, opts) {
    var o = opts || {};
    var n = panel(id, [], {
      bg: o.bg !== undefined ? o.bg : tone('neutral', 34),
      radius: o.radius === undefined ? 8 : o.radius,
      border: 0, border_width: 0, shadow_blur: 0,
      width: o.width === undefined ? { fill: 1 } : o.width,
      height: o.height === undefined ? 12 : o.height,
    }, 'top_left', 0);
    if (sofuu.fx && typeof sofuu.fx.shimmer === 'function') {
      try { sofuu.fx.shimmer(n, { speed: 1.1, intensity: o.intensity || 0.55 }); } catch (e) {}
    }
    return n;
  };

  /* ── AdaptiveShell ─────────────────────────────────────────────── */
  kit.breakpoint = function (width) {
    var w = Number(width || 0);
    if (w < 680) return 'mobile';
    if (w < 1080) return 'tablet';
    return 'desktop';
  };

  /* Choose a scaffold for the current width:
   *   mobile  -> AppBar + body + BottomNav
   *   tablet  -> compact left rail + AppBar + body
   *   desktop -> full sidebar + toolbar + multi-pane body
   * Pass `opts.mode` to force one layout (useful for emu filmstrips/tests). */
  kit.AdaptiveShell = function (id, opts) {
    var o = opts || {};
    var mode = o.mode || kit.breakpoint(o.width);
    var appBar = o.appBar || kit.AppBar(id + '_appbar', {
      title: o.title || '', back: !!o.back, onBack: o.onBack, actions: o.actions,
    });
    var body = Array.isArray(o.body)
      ? u.col(id + '_body', o.body, { gap: 12, scrollable: true })
      : (o.body || u.spacer(id + '_body'));
    if (body && body.toNode && (!body.toNode().width || body.toNode().width === 'auto')) {
      body.toNode().width = { fill: 1 };
    }
    if (body && body.toNode && (!body.toNode().height || body.toNode().height === 'auto')) {
      body.toNode().height = { fill: 1 };
    }
    if (mode === 'mobile') {
      return u.col(id, [
        appBar,
        body,
        o.bottomNav || null,
      ].filter(Boolean), { gap: 0, width: { fill: 1 }, height: { fill: 1 } });
    }
    if (mode === 'tablet') {
      var rail = o.rail || kit.Sidebar(id + '_rail', (o.tabs || []).map(function (t) {
        return { icon: t.icon, label: '', active: t.active };
      }), { width: 64 });
      return row(id, [
        rail,
        u.col(id + '_main', [appBar, body], { gap: 0, width: { fill: 1 }, height: { fill: 1 } }),
      ], { gap: 0, width: { fill: 1 }, height: { fill: 1 } });
    }
    var side = o.sidebar || kit.Sidebar(id + '_side', o.nav || [], { width: o.sidebarWidth || 240 });
    return row(id, [
      side,
      u.col(id + '_main', [appBar, body], { gap: 0, width: { fill: 1 }, height: { fill: 1 } }),
    ], { gap: 0, width: { fill: 1 }, height: { fill: 1 } });
  };

  kit.tokens = { tones: TONES, color: color, tone: tone, radius: radius, modes: ['mobile', 'tablet', 'desktop'] };

  sofuu.uiKit = kit;
})();
