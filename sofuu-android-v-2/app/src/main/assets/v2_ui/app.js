/*
 * v2_ui/app.js — entry point for the GPU shell.
 *
 * The host (MainActivity) evaluates this bundle once per process inside a
 * try/catch, then calls `sofuu.v2.mount()` on every viewport/inset change.
 * The bundle itself mounts once on load so a plain `sofuu eval` of these
 * assets (or sofuu-emu) shows the same scene without host help.
 *
 * There is no Compose fallback and no mock data: an unwired host action goes
 * through SofuuAndroid.invoke and comes back as a typed E_HOST_UNAVAILABLE
 * error the UI could render.
 */
(function () {
  'use strict';
  var g = globalThis;
  var ui = g.sofuu.ui;
  var T = g.SofuuTokens;
  var V = (g.SofuuV2 = g.SofuuV2 || {});

  var surface = null;

  function env() {
    var e = g.__v2;
    if (!e) return { w: 360, h: 780, top: 0, bottom: 0, dark: false };
    return e;
  }

  function tokens(e) {
    return e.dark ? T.dark : T.light;
  }

  function build() {
    var e = env();
    return V.home.build(e, tokens(e));
  }

  /**
   * Build the scene and replace the mounted tree. Idempotent and cheap
   * enough for inset-driven remounts; later phases turn the shell parts into
   * patches (`set_children`) so state survives a resize.
   */
  function mount() {
    var e = env();
    var scene = V.home.build(e, tokens(e));
    if (!surface) surface = ui.surface('main', { title: scene.title, theme: scene.theme });
    surface.mount(scene);
    return scene;
  }

  /** Back-navigation hook. The basic shell has no overlays/routes yet. */
  function back() {
    return false;
  }

  V.env = env;
  V.tokens = tokens;
  V.build = build;
  V.mount = mount;
  V.back = back;

  mount();
  0;
})();
