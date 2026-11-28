/*
 * v2_ui/icons.js — geometry-first icons for the GPU shell.
 *
 * sofuu-ui's SDF atlas is ASCII Geist Mono only, so app-bar glyphs that v1
 * drew with Material vectors are composed from rounded quads here. This keeps
 * every icon crisp at any DPI without a sprite/icon node (which the UI
 * library plan adds later).
 *
 * Sizing contract: every builder returns a box whose intrinsic size matches
 * the slot it is used in (44dp app-bar targets, 40dp composer actions), with
 * the painted mark centred inside. Icons are never scaled, so v1's 20dp glyph
 * / 16dp chevron sizes are preserved exactly.
 */
(function () {
  'use strict';
  var g = globalThis;
  var ui = g.sofuu.ui;
  var V = (g.SofuuV2 = g.SofuuV2 || {});

  function radius4(r) {
    return [r, r, r, r];
  }

  /** A bare filled/outlined rounded rectangle. */
  function box(w, h, opts) {
    var o = opts || {};
    var style = { radius: radius4(o.radius !== undefined ? o.radius : 0) };
    if (o.bg !== undefined && o.bg !== null) style.bg = o.bg;
    if (o.border !== undefined && o.border !== null) {
      style.border = o.border;
      style.border_width = o.borderWidth !== undefined ? o.borderWidth : 1;
    }
    return ui.col(null, [], { width: w, height: h, style: style });
  }

  /** A text glyph inside a fixed box. */
  function glyph(ch, size, color, w, h) {
    var padLeft = Math.max(0, (w - size * 0.6) / 2);
    return ui.text(ch, {
      size: size,
      color: color,
      mono: true,
      width: w,
      height: h,
      padding: { left: padLeft }
    });
  }

  /** Fixed-size slot with its content vertically centred (column main axis).
   *  `contentH` is the painted mark's height: v1's 20 dp paperclip and 18 dp
   *  attachment stand-in need a different lead than the 14 dp shield. */
  function slot(w, h, child, contentH) {
    var ch = contentH === undefined ? 14 : contentH;
    var pad = { top: Math.max(0, (h - ch) / 2) };
    return ui.col(null, [child], { width: w, height: h, padding: pad, gap: 0, align: 'center' });
  }

  /** Three-bar hamburger (16 x 10.8 inside a 44 box). */
  function menu(color) {
    return ui.col(null, [
      box(16, 1.6, { bg: color, radius: 0.8 }),
      box(16, 1.6, { bg: color, radius: 0.8 }),
      box(16, 1.6, { bg: color, radius: 0.8 })
    ], { width: 44, height: 44, gap: 3, padding: { top: 16.6 }, align: 'center' });
  }

  /** Vertical ellipsis (3.5 x 17.5 inside a 44 box). */
  function more(color) {
    return ui.col(null, [
      box(3.5, 3.5, { bg: color, radius: 1.75 }),
      box(3.5, 3.5, { bg: color, radius: 1.75 }),
      box(3.5, 3.5, { bg: color, radius: 1.75 })
    ], { width: 44, height: 44, gap: 3.5, padding: { top: 13.25 }, align: 'center' });
  }

  function plus(color) {
    return glyph('+', 26, color, 44, 44);
  }

  /** v1's 16dp ChevronRight. The 26dp-tall slot centres ink on the row. */
  function chevron(color) {
    return glyph('>', 14, color, 16, 26);
  }

  /** 13dp ArrowDropDown (the composer pills' disclosure caret). */
  function caret(color) {
    return glyph('v', 13, color, 13, 20);
  }

  /** ChatBubbleOutline stand-in: 11 x 9 rounded outline + a small tail. */
  function chat(color) {
    return ui.col(null, [
      box(11, 9, { border: color, borderWidth: 1.2, radius: 3 }),
      box(4, 2, { bg: color, radius: 1 })
    ], { width: 14, height: 14, gap: 0, padding: { top: 2 }, align: 'center' });
  }

  /** Shield outline stand-in: a 10 x 12 rounded outline. */
  function shield(color) {
    return slot(14, 14, box(10, 12, { border: color, borderWidth: 1.2, radius: 3 }));
  }

  /** AutoAwesome stand-in: the atlas' four-point asterisk. */
  function spark(color) {
    return glyph('*', 18, color, 14, 20);
  }

  /** Paperclip stand-in: v1's Attach vector ink is 9.1 x 18.3 dp. */
  function attach(color) {
    return slot(40, 40, box(10, 18, { border: color, borderWidth: 1.6, radius: 5 }), 18);
  }

  /** Stacked layers: three hairlines with descending width (10.8 high). */
  function layers(color) {
    return ui.col(null, [
      box(15, 1.6, { bg: color, radius: 0.8 }),
      box(11, 1.6, { bg: color, radius: 0.8 }),
      box(7, 1.6, { bg: color, radius: 0.8 })
    ], { width: 40, height: 40, gap: 3, padding: { top: 14.6 }, align: 'center' });
  }

  /** Microphone capsule + stand (13 + 2 + 1.6 = 16.6 high inside a 44 box). */
  function mic(color) {
    return ui.col(null, [
      box(7, 13, { bg: color, radius: 3.5 }),
      box(11, 1.6, { bg: color, radius: 0.8 })
    ], { width: 44, height: 44, gap: 2, padding: { top: 13.7 }, align: 'center' });
  }

  /** Context-usage ring (16 x 16 inside a 40 box). */
  function ring(color) {
    return slot(40, 40, box(16, 16, { border: color, borderWidth: 2, radius: 8 }));
  }

  /** Icon button container: one square touch target, one pre-sized icon. */
  function iconButton(id, icon, opts) {
    var o = opts || {};
    var size = o.size || 44;
    return ui.col(id, [icon], {
      width: size,
      height: size,
      align: 'center',
      gap: 0,
      onClick: o.onClick,
      style: o.style || null
    });
  }

  /** ArrowUpward: 13 x 13, accent ink on no plate (ghost mode). Composed as a
   *  stepped triangle head over a 5 dp stem, because the IR has no rotated
   *  quads or vector paths yet. The 42 dp slot centres it like v1's icon. */
  function arrowUp(color) {
    return ui.col(null, [
      box(2.4, 1.6, { bg: color, radius: 0.8 }),
      box(5.2, 1.6, { bg: color, radius: 0.8 }),
      box(8.0, 1.6, { bg: color, radius: 0.8 }),
      box(10.8, 1.6, { bg: color, radius: 0.8 }),
      box(13.2, 1.6, { bg: color, radius: 0.8 }),
      box(2.4, 5.0, { bg: color, radius: 1.2 })
    ], { width: 42, height: 42, gap: 0, padding: { top: 14.5 }, align: 'center' });
  }

  /** The composer send target: 42dp hit box, no disc in ghost mode. */
  function send(id, t, onClick) {
    return ui.col(id, [arrowUp(t.accent)], {
      width: 42,
      height: 42,
      align: 'center',
      gap: 0,
      onClick: onClick
    });
  }

  V.ui = {
    box: box,
    glyph: glyph,
    slot: slot,
    menu: menu,
    more: more,
    plus: plus,
    chevron: chevron,
    caret: caret,
    chat: chat,
    shield: shield,
    spark: spark,
    attach: attach,
    layers: layers,
    mic: mic,
    ring: ring,
    iconButton: iconButton,
    arrowUp: arrowUp,
    send: send,
    radius4: radius4
  };
})();
