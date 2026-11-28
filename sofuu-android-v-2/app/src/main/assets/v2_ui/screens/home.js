/*
 * v2_ui/screens/home.js — the v1 home screen, rebuilt in sofuu.ui.
 *
 * Mirrors sofuu-android/.../ui/Shell.kt (top bar + starter prompts) and
 * .../ui/composer/Composer.kt (mode pills + input card + action bar) with the
 * same dp spacing, the same 1.15-scaled sp sizes and the same cream palette.
 * There is no Compose here: every node is a Scene IR node drawn by ui_gles.c.
 *
 * v1 runs every SoapBubbleSurface with `transparent = true` (MainActivity's
 * LocalSoapBubbleSettings), so there are NO plates behind pills, icon buttons
 * or starter rows: only their glyphs and text are visible. The one filled
 * surface is the composer card itself (#F7F0DE). This file mirrors that.
 *
 * Text is `mono: true` everywhere: v1's GeistMono is a monospace face, and its
 * 0.6 em advance is what makes the starter descriptions wrap at the same word
 * and gives the app bar its rhythm. The engine atlas carries the same 0.6 em
 * advance and 1.3 em line height.
 *
 * Deliberate gaps until the UI-library phases land: the text field is static
 * placeholder text (input focus/IME is a later phase), buttons have no press
 * spring (`ui.card`/Surface does not exist yet), and app-bar icons are quad
 * compositions from icons.js because the SDF atlas is ASCII-only.
 */
(function () {
  'use strict';
  var g = globalThis;
  var ui = g.sofuu.ui;
  var T = g.SofuuTokens;
  var V = (g.SofuuV2 = g.SofuuV2 || {});
  var I = V.ui;

  var STARTERS = [
    // v1 wraps this description after "entry" at phone widths; the wrap is
    // part of the screen's vertical rhythm (the first card is one line taller
    // than the other three). The engine's wrapped-line gap is 1.3 em vs the
    // 1.85 em Compose ends up with, so the break is carried as two lines.
    ['Explain this project', ['Walk through architecture and entry', 'points']],
    ['Review git diff', ['Inspect uncommitted code changes']],
    ['Find bugs & tests', ['Analyze test coverage and failure modes']],
    ['Health check', ['Run doctor diagnostics on the runtime']]
  ];

  // v1 store defaults: AppStore.mode = "agent", permissions = "ask",
  // config.effort = "high". The model pill is icon + caret only in v1: its
  // "No model" label is measured to zero width by the flexible-label weight
  // (verified with `uiautomator dump`: the pill is 56.4 dp wide and the
  // effort pill next to it starts at x=90.7 and reads "High").
  var MODE_LABEL = 'Agent';
  var PERMISSION_LABEL = 'Ask me';
  var MODEL_LABEL = '';
  var EFFORT_LABEL = 'High';

  // v1's titleLarge is 17 sp x 1.15 = 19.55 dp, but the face Compose lands on
  // for the Bold heading draws ~5% narrower than the atlas at that size
  // (253.0 dp for 23 chars vs 267.4). 18.5 dp reproduces the exact v1 ink
  // width and height without changing the centred layout.
  var TITLE_SIZE = 18.5;

  function label(id, text, size, color, opts) {
    var o = opts || {};
    o.size = size;
    o.color = color;
    o.mono = true;
    return ui.text(id, text, o);
  }

  function vgap(h) {
    return ui.spacer(null, { width: { fill: 1 }, height: h });
  }

  function hgap(w) {
    return ui.spacer(null, { width: w, height: 1 });
  }

  /** Ghost pill: v1's SoapBubbleButton has no plate in transparent mode, but
   *  keeps its 36 dp min height and 12 dp horizontal content padding. */
  function pillRow(id, children, onClick) {
    return ui.row(id, children, {
      height: 36,
      padding: { left: 12, right: 12 },
      gap: 0,
      align: 'center',
      foldBelow: 0,
      onClick: onClick
    });
  }

  function topBar(t) {
    var ink = t.dark ? t.muted : t.text;
    return ui.row('topbar', [
      I.iconButton('tb_menu', I.menu(ink)),
      ui.row('tb_pills', [
        // v1's null session (the "new chat" shortcut) is the ACTIVE pill, so
        // its label is accent-coloured and bold. Ghost mode has no plate.
        pillRow('tb_new_chat', [
          label('tb_new_chat_label', 'New chat', T.type.pill, t.accent)
        ], function () {})
      ], {
        width: { fill: 1 },
        gap: 12,
        align: 'center',
        padding: { left: 10, right: 10 },
        foldBelow: 0
      }),
      hgap(12),
      I.iconButton('tb_plus', I.plus(ink)),
      hgap(12),
      I.iconButton('tb_more', I.more(ink))
    ], {
      width: { fill: 1 },
      padding: { left: 6, right: 6, top: 4, bottom: 4 },
      gap: 0,
      align: 'center',
      foldBelow: 0
    });
  }

  function starterRow(index, title, desc, t, textWidth) {
    var descNodes = [];
    for (var i = 0; i < desc.length; i++) {
      var opts = { width: textWidth, wrap: true };
      if (i > 0) opts.padding = { top: 7.2 };
      descNodes.push(label('starter_' + index + '_desc' + (i || ''), desc[i],
        T.type.starterDesc, t.dark ? t.faint : t.muted, opts));
    }
    return ui.row('starter_' + index, [
      ui.col(null, [
        label('starter_' + index + '_title', title, T.type.starterTitle, t.text, {
          width: textWidth
        })
      ].concat(descNodes), { width: { fill: 1 }, gap: 0, align: 'stretch' }),
      I.chevron(t.dark ? t.faint : t.muted)
    ], {
      width: { fill: 1 },
      padding: { left: 14, right: 14, top: 10, bottom: 10 },
      gap: 0,
      align: 'center',
      foldBelow: 0,
      style: null,
      onClick: function () {}
    });
  }

  function homeBody(env, t) {
    var workspace = env.workspace ? ('Workspace: ' + env.workspace) : null;
    // The starter rows share one text column: body width minus the 24 dp
    // gutters, the 14 dp card padding and the 16 dp chevron. A px width keeps
    // the wrap measurement honest: the solver measures Fill children at an
    // equal share of the row, which would reserve two lines for every desc.
    var textWidth = Math.max(120, env.w - 92);
    var content = [
      label('home_title', 'What should we work on?', TITLE_SIZE, t.text)
    ];
    if (workspace) {
      content.push(label('home_sub', workspace, T.type.bodySmall, t.faint, {
        padding: { top: 6.7, bottom: 24 }
      }));
    } else {
      content.push(vgap(24));
    }
    var rows = [];
    for (var i = 0; i < STARTERS.length; i++) {
      rows.push(starterRow(i, STARTERS[i][0], STARTERS[i][1], t, textWidth));
    }
    content.push(ui.col('home_cards', rows, {
      width: { fill: 1 },
      gap: 12,
      align: 'stretch'
    }));

    // v1 centres the whole prompt block (Arrangement.Center) inside the body.
    // Two fill spacers reproduce that without a justify property. The fixed
    // 10.2 dp bias compensates for the engine's 1.3 em text boxes: Compose
    // trims a line box to its ink, the SDF node keeps the full em box, so an
    // identical, centred tree would paint every line ~5.1 dp low.
    return ui.col('home', [
      ui.spacer('home_lead', { width: { fill: 1 }, height: { fill: 1 } }),
      ui.col('home_content', content, {
        width: { fill: 1 },
        gap: 0,
        align: 'center'
      }),
      ui.spacer('home_ink_bias', { width: { fill: 1 }, height: 10.2 }),
      ui.spacer('home_tail', { width: { fill: 1 }, height: { fill: 1 } })
    ], {
      width: { fill: 1 },
      height: { fill: 1 },
      padding: { left: 24, right: 24, top: 20, bottom: 20 },
      gap: 0,
      align: 'stretch'
    });
  }

  function actionPill(id, text, t, icon, onClick) {
    var children = [];
    if (icon) {
      children.push(icon);
      children.push(hgap(5));
    }
    children.push(label(id + '_label', text, T.type.pill, t.text));
    children.push(hgap(3));
    children.push(I.caret(t.dark ? t.faint : t.muted));
    return pillRow(id, children, onClick);
  }

  function composer(t) {
    var ink = t.dark ? t.textMuted : t.text;
    return ui.col('composer', [
      ui.row('composer_modes', [
        actionPill('comp_mode', MODE_LABEL, t, I.chat(t.accent), function () {}),
        actionPill('comp_perms', PERMISSION_LABEL, t, I.shield(t.accent), function () {})
      ], { gap: 12, align: 'center', foldBelow: 0 }),
      ui.spacer('composer_gap', { width: { fill: 1 }, height: 6 }),
      ui.col('composer_card', [
        label('composer_ph', 'Ask, @mention files, or type / for commands...', T.type.composerInput, t.faint, {
          width: { fill: 1 },
          height: 47,
          wrap: true
        }),
        ui.spacer('composer_inner_gap', { width: { fill: 1 }, height: 8 }),
        ui.row('composer_actions', [
          actionPill('comp_model', MODEL_LABEL, t, I.spark(t.accent), function () {}),
          hgap(12),
          actionPill('comp_effort', EFFORT_LABEL, t, null, function () {}),
          ui.spacer('composer_flex', { width: { fill: 1 }, height: 1 }),
          I.iconButton('comp_attach', I.attach(ink), { size: 40 }),
          hgap(12),
          I.iconButton('comp_layers', I.layers(ink), { size: 40 }),
          // v1 keeps the 12 dp spacer before the optional mic slot even when
          // the mic is hidden, so layers sits 32 dp left of send, not 20.
          hgap(12),
          hgap(8),
          hgap(12),
          I.send('comp_send', t, function () {})
        ], { width: { fill: 1 }, gap: 0, align: 'center', foldBelow: 0 })
      ], {
        width: { fill: 1 },
        padding: 12,
        gap: 0,
        style: {
          bg: t.card,
          border: t.border,
          border_width: 1,
          radius: I.radius4(16)
        }
      })
    ], {
      width: { fill: 1 },
      padding: { left: 10, right: 10, top: 6, bottom: 6 },
      gap: 0
    });
  }

  /**
   * @param {{w:number,h:number,top:number,bottom:number,dark:boolean,workspace?:string}} env
   * @param {object} t resolved palette
   */
  function build(env, t) {
    // Insets are host-owned scene padding: the root background stays
    // full-bleed (v1's Compose shell did the same), so the status-bar strip is
    // the same cream as the app instead of the engine Theme::bg.
    var root = ui.col('root', [
      topBar(t),
      homeBody(env, t),
      composer(t)
    ], {
      width: { fill: 1 },
      height: { fill: 1 },
      padding: { top: env.top, bottom: env.bottom, left: 0, right: 0 },
      gap: 0,
      align: 'stretch',
      style: { bg: t.bg }
    });

    return {
      title: 'Sofuu',
      theme: env.dark ? 'dark' : 'light',
      root: root
    };
  }

  V.home = { build: build, starters: STARTERS };
})();
