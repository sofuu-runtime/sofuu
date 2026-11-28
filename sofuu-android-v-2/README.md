# sofuu-android-v-2

The pure-GPU rewrite of the Sofuu Android app: **zero Jetpack Compose, zero
Material widgets**. Every visible pixel is a `sofuu.ui` Scene IR node compiled
by `sofuu-ui` and drawn by the four-pipeline GLES 3.0 renderer in
`app/src/main/cpp/ui_gles.c`.

This is the phase-A1 scaffold from
`docs/plans/PLAN-SOFUU-ANDROID-V2-GPU-REWRITE.md`: the v1 home screen,
top bar and composer, rebuilt with current `sofuu.ui` primitives on the real
engine. `sofuu-android/` is not modified by this project (the repo may carry
unrelated in-flight changes from other sessions).

## What is here

```
app/src/main/
  kotlin/com/sofuu/
    MainActivity.kt              # host: runtime + GL surface + JS mount + insets
    engine/SofuuBridge.kt        # libsofuu C ABI (copied from v1)
    render/SofuuGpuSurfaceView.kt# GLSurfaceView host + SofuuUiNative JNI (v1)
  cpp/
    jni_bridge.c                 # Android JNI bridge (copied from v1)
    ui_gles.c                    # ES 3.0 DrawList renderer (copied from v1)
    include/sofuu_embed.h        # stable C ABI header (copied from v1)
  assets/v2_ui/
    theme.js                     # v1 Color.kt tokens, packed RGBA8 + 1.15 sp scale
    icons.js                     # quad-composed app-bar icons (ASCII SDF atlas)
    bridge.js                    # typed JS -> Android channel (not wired yet)
    screens/home.js              # the v1 home/top-bar/composer scene
    app.js                       # mount/remount + back hook
```

## Build

```bash
cd sofuu-android-v-2
./gradlew :app:assembleDebug           # add --offline on this Mac if needed
```

Outputs `app/build/outputs/apk/debug/app-{arm64-v8a,x86_64,universal}-debug.apk`.
The application id is `com.sofuu.v2`, so it installs side by side with v1 while
the Java package stays `com.sofuu` (the JNI symbol names are unchanged).

The engine payload is **not duplicated**. `app/build.gradle.kts` reads
`../sofuu-android/app/src/main/jniLibs` (staged by
`scripts/mobile/build_engine_android.sh`) and links the prebuilt
`libsofuu.so`. Pass `-PsofuuPrebuiltJniDir=<dir>` to build against a
v2-specific staging directory instead.

## How it runs

1. `MainActivity` pins `$HOME` to `filesDir` and creates one `SofuuRuntime`.
2. A `SofuuGpuSurfaceView` owns the `SofuuUiSession`; the runtime is attached
   with `sofuu_ui_attach_runtime`, which installs the JS UI globals.
3. The `assets/v2_ui` bundle is evaluated once; `sofuu.v2.mount()` builds the
   scene and pushes `replace_scene`.
4. The GL thread polls `sofuu_ui_poll_draw_list` on request and renders; the
   UI thread ticks the JS loop every 50 ms for the `sofuu.ui` event pump.
5. Insets are scene padding: the root background stays full-bleed cream, like
   v1's Compose shell, and the content is inset by the status/navigation bars.
   `sofuu_ui_session_set_safe_insets` is deliberately left at 0 because the
   engine paints its own `Theme::bg` behind the root; letting it reserve the
   top inset would show a bluish status-bar strip above the cream root.

## Fidelity notes

- Palette, spacing, sizes and structure come from
  `sofuu-android/.../ui/theme/Color.kt`, `ui/Shell.kt` and
  `ui/composer/Composer.kt`, including the app-wide `UI_FONT_SCALE = 1.15`.
- The captured v1 home state is reproduced exactly: mode `Agent`,
  permissions `Ask me`, effort `High`, and a model pill that is icon + caret
  only (v1's flexible-label weight squeezes the "No model" text to zero
  width). The top-bar `New chat` pill is the accent active style.
- `foldBelow: 0` on every one-line `ui.row`: the default 680 px desktop fold
  threshold would otherwise turn phone rows into columns.
- Colors are packed little-endian RGBA8 (`r | g<<8 | b<<16 | a<<24`), not
  Kotlin's `0xAARRGGBB`; `theme.js` converts once at load.
- Ghost mode is mirrored from v1's app-wide `transparent = true`: starter
  rows, pills and icon buttons draw ink only. The composer card
  (`#F7F0DE` / dark `#2B2620`, 16 dp radius, 1 px border) is the only filled
  surface.

## Verified

- `./gradlew --offline :app:assembleDebug` produces the arm64, x86_64 and
  universal APKs; the APK carries `assets/v2_ui/*`, `ui_shaders_es3.glsl`,
  `libsofuu.so` and `libsofuu_jni.so`, with no Compose dependency.
- Light and dark screenshots at 411.43 x 914.29 dp were compared with v1
  after `cmd uimode` toggles: cream `#FFFBF0` / dark `#211D19` root, exact
  v1 card rect (`x 10, y 763.3, w 391.4, h 121`), and matching home bands
  (heading, subtitle, four starter rows) within ~1.5 dp.
- A C ABI smoke run evaluates the real bundle, gets `{"ok":true}`, and
  checks the draw list: 22 quads / 296 glyphs, full-bleed root, the composer
  card, no white plates, no filled starter cards, accent send glyph, and
  text runs for the heading, starters and composer pills. It passes for both
  themes.

## Known gaps (later plan phases)

- The text area is the v1 placeholder as static text; focus/IME editing needs
  the host input phase. The composer buttons are visual only.
- App-bar icons are quad compositions because the embedded SDF atlas is ASCII
  Geist Mono only; the icon atlas / sprite node lands with the UI-library
  phase. The paperclip/layers stand-ins match v1's ink boxes, but the atlas'
  single regular weight is lighter than v1's Medium/SemiBold faces and its
  glyph advances differ by a few percent (the heading is retuned to 18.5 dp
  to match v1's captured 253 dp ink width).
- Tappable cards and press springs need `ui.card` (the `Surface` node). The
  shared `src/js/ui.js` currently has no builder for it, and that file is being
  edited by the UI-library work; v2 does not fork it.
- `SofuuRuntime` is created on the UI thread in this scaffold. Moving it to a
  dedicated engine worker (the v1 `SofuuEngine` pump) is the next host phase.
- No chat/session store, drawer, history sheet or settings yet: the scaffold
  renders the v1 home state only.
