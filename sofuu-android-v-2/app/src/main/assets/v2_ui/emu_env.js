/*
 * v2_ui/emu_env.js — sofuu-emu -> __v2 adapter.
 *
 * The Android host builds `__v2` directly from WindowInsets (MainActivity
 * `mount()`): `{ w, h, top, bottom, dark, workspace }`, where `w`/`h` are the
 * full canvas in dp and `top`/`bottom` are the system-bar insets the app pads
 * with. The emulator exports the same resolved metrics as
 * `__sofuu_emu_device` plus `__sofuu_emu_options` (theme/orientation), so the
 * real v2 assets run unchanged.
 *
 * The emulator host already lays the app out inside the safe viewport: the
 * status bar and home bar are host siblings, not app padding. `width`/`height`
 * are therefore the post-inset viewport and `top`/`bottom` are 0 here. The
 * exported `safe` set stays available for apps that draw edge-to-edge.
 */
(function () {
  'use strict';
  var g = globalThis;
  var d = g.__sofuu_emu_device;
  if (!d || !(d.width > 0) || !(d.height > 0)) return;
  var o = g.__sofuu_emu_options || {};
  var p = g.__sofuu_emu_props || {};
  g.__v2 = {
    w: d.width,
    h: d.height,
    top: 0,
    bottom: 0,
    dark: o.theme === 'dark',
    workspace: (p && p.workspace) ? String(p.workspace) : ''
  };
})();
