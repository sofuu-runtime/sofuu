//! `sofuu.fx` showcase and headless frame-budget bench.
//!
//! Windowed scene:
//!   cargo run -p sofuu-gpu --example fx_showcase
//!
//! Headless CPU compile bench (prints ms/frame for the animated scene vs. the same
//! scene with every `fx` spec removed, then proves the reduced-motion idle gate):
//!   SOFUU_FX_BENCH=1 cargo run -p sofuu-gpu --example fx_showcase
//!
//! The bench isolates the CPU spring/timeline/compile layer. GPU frame cost for the
//! effect and Kawase passes is measured by the `sofuu-emu` perf capture with offscreen
//! timestamps; there is no GPU timer in this example.

use std::time::{Duration, Instant};

use sofuu_gpu::{run_window, WindowConfig};
use sofuu_ui::{
    fx, rgba8, ui, Align, BoxStyle, DrawList, FxKind, FxSpec, Insets, Node, NodeKind, Scene,
    ScenePatch, SizeSpec, ThemeMode, Tone, UIEvent, UiEngine,
};

const VIEWPORT: [f32; 2] = [1180.0, 820.0];

fn panel(id: &str, bg: Option<u32>, radius: f32, height: Option<f32>, child: Option<Node>) -> Node {
    let mut node = Node::new(
        id,
        NodeKind::Surface {
            style: BoxStyle {
                bg,
                radius: [radius; 4],
                ..BoxStyle::default()
            },
            interactive: false,
            child: child.map(Box::new),
        },
    );
    if let Some(h) = height {
        node.height = SizeSpec::Px(h);
    }
    node
}

fn padded(mut node: Node, inset: f32) -> Node {
    node.padding = Insets::all(inset);
    node
}

fn glass_sheet(with_fx: bool) -> Node {
    let body: Node = ui::col("glass_body")
        .gap(6.0)
        .add(ui::text("glass_title", "Frosted sheet"))
        .add(ui::text(
            "glass_hint",
            "Kawase half-res backdrop blur over the live framebuffer.",
        ))
        .into();
    let mut sheet = panel(
        "glass_sheet",
        Some(rgba8(18, 25, 36, 214)),
        20.0,
        Some(148.0),
        Some(padded(body, 18.0)),
    );
    sheet.width = SizeSpec::Fill(1.0);
    if with_fx {
        sheet = fx::backdrop_blur(sheet, 26.0, 1.0);
    }
    sheet
}

fn aurora_band(with_fx: bool) -> Node {
    let mut band = panel(
        "aurora_band",
        Some(rgba8(10, 15, 24, 255)),
        18.0,
        Some(148.0),
        None,
    );
    band.width = SizeSpec::Fill(1.0);
    if with_fx {
        band.fx = vec![FxSpec {
            speed: 0.7,
            intensity: 0.85,
            ..FxSpec::new(FxKind::Aurora)
        }];
    }
    band
}

fn voice_orb(with_fx: bool) -> Node {
    let mut orb = panel("voice_orb", None, 48.0, Some(96.0), None);
    orb.width = SizeSpec::Px(96.0);
    if with_fx {
        orb.fx = vec![FxSpec {
            amplitude: 0.45,
            radius: 48.0,
            ..FxSpec::new(FxKind::VoiceOrb)
        }];
    }
    orb
}

fn skeleton_row(with_fx: bool, index: usize) -> Node {
    let mut bar = panel(
        &format!("skeleton_{index}"),
        Some(rgba8(148, 163, 184, 46)),
        7.0,
        Some(14.0),
        None,
    );
    bar.width = SizeSpec::Px(220.0 + (index as f32 * 18.0));
    if with_fx {
        bar.fx = vec![FxSpec {
            speed: 0.9,
            radius: 26.0,
            ..FxSpec::new(FxKind::Shimmer)
        }];
    }
    bar
}

fn feature_card(id: &str, title: &str, hint: &str, with_fx: bool, fx: FxSpec) -> Node {
    let body: Node = ui::col(format!("{id}_body"))
        .gap(4.0)
        .add(ui::text(format!("{id}_title"), title))
        .add(ui::text(format!("{id}_hint"), hint))
        .into();
    let mut card = panel(
        id,
        Some(rgba8(22, 30, 42, 255)),
        14.0,
        Some(106.0),
        Some(padded(body, 14.0)),
    );
    card.width = SizeSpec::Fill(1.0);
    if with_fx {
        card.fx = vec![fx];
    }
    card
}

fn list_row(index: usize) -> Node {
    let row = panel(
        &format!("item_{index}"),
        Some(rgba8(22, 30, 42, 255)),
        10.0,
        Some(42.0),
        Some(padded(
            ui::text(
                format!("item_{index}_label"),
                format!("Queued job #{}", index + 1),
            )
            .into(),
            12.0,
        )),
    );
    row
}

fn build_scene(with_fx: bool) -> Scene {
    let header: Node = ui::row("hdr")
        .align(Align::Center)
        .add(ui::title("sofuu.fx // GPU Effects Suite"))
        .add(ui::button_toned("celebrate", "Celebrate", Tone::Ok))
        .add(ui::button_toned("deny", "Deny", Tone::Danger))
        .into();

    let skeleton: Node = ui::col("skeleton_stack")
        .gap(10.0)
        .add(skeleton_row(with_fx, 0))
        .add(skeleton_row(with_fx, 1))
        .add(skeleton_row(with_fx, 2))
        .into();

    let orb_column: Node = ui::col("orb_column")
        .align(Align::Center)
        .gap(8.0)
        .add(voice_orb(with_fx))
        .add(ui::text("orb_label", "voice orb"))
        .into();

    let hero: Node = ui::row("hero")
        .align(Align::Stretch)
        .gap(16.0)
        .add(aurora_band(with_fx))
        .add(glass_sheet(with_fx))
        .add(orb_column)
        .into();

    let features: Node = ui::row("features")
        .align(Align::Stretch)
        .fold_below(820.0)
        .gap(12.0)
        .add(feature_card(
            "beam_card",
            "BorderBeam",
            "rotating comet perimeter",
            with_fx,
            FxSpec {
                speed: 0.8,
                width: 1.5,
                tone: Some(Tone::Info),
                ..FxSpec::new(FxKind::BorderBeam)
            },
        ))
        .add(feature_card(
            "glow_card",
            "CursorGlow",
            "pointer-tracked specular",
            with_fx,
            FxSpec {
                radius: 120.0,
                intensity: 0.8,
                ..FxSpec::new(FxKind::CursorGlow)
            },
        ))
        .add(feature_card(
            "pulse_card",
            "PulseRing",
            "live agent beacon",
            with_fx,
            FxSpec {
                speed: 0.9,
                radius: 34.0,
                tone: Some(Tone::Ok),
                ..FxSpec::new(FxKind::PulseRing)
            },
        ))
        .into();

    let mut queue_rows: Vec<Node> = (0..5).map(list_row).collect();
    if with_fx {
        fx::stagger(queue_rows.iter_mut(), 45.0, 12.0);
    }
    let mut queue_builder = ui::col("queue")
        .gap(8.0)
        .add(ui::text("queue_title", "Staggered arrival"));
    for row in queue_rows {
        queue_builder = queue_builder.add(row);
    }
    let queue: Node = queue_builder.into();

    let mut text_shine: Node = ui::text("text_shine", "Streaming a live agent response...").into();
    if with_fx {
        text_shine.fx = vec![FxSpec {
            speed: 0.7,
            ..FxSpec::new(FxKind::TextShimmer)
        }];
    }

    let streaming: Node = ui::row("streaming")
        .align(Align::Center)
        .gap(16.0)
        .add(skeleton)
        .add(text_shine)
        .into();

    let mut root: Node = ui::col("fx_root")
        .gap(16.0)
        .add(header)
        .add(hero)
        .add(features)
        .add(streaming)
        .add(queue)
        .add(ui::text(
            "footer",
            "Celebrate fires confetti; Deny fires a damped shake; every one-shot drains to 0% idle.",
        ))
        .into();
    root.height = SizeSpec::Fill(1.0);
    root = padded(root, 22.0);

    Scene::new("fx-showcase", ThemeMode::Dark, root)
}

fn run_showcase() {
    let cfg = WindowConfig {
        title: "Sofuu FX — Effects Suite".into(),
        width: VIEWPORT[0],
        height: VIEWPORT[1],
        min_width: 760.0,
        min_height: 560.0,
        ..WindowConfig::default()
    };

    if let Err(err) = run_window(cfg, build_scene(true), |handle| {
        let (sender, events) = handle.split();
        std::thread::spawn(move || {
            let started = Instant::now();
            loop {
                let ev = events.recv_timeout(Duration::from_millis(60));
                match ev {
                    Ok(UIEvent::WindowClosed) => return,
                    Ok(UIEvent::Click { id }) if id == "celebrate" => {
                        let _ = sender.push_patch(ScenePatch::FxTrigger {
                            target: Some("celebrate".into()),
                            kind: FxKind::Confetti,
                            count: Some(64),
                            x: None,
                            y: None,
                            seed: Some(7),
                        });
                    }
                    Ok(UIEvent::Click { id }) if id == "deny" => {
                        let _ = sender.push_patch(ScenePatch::FxTrigger {
                            target: Some("deny".into()),
                            kind: FxKind::Shake,
                            count: None,
                            x: None,
                            y: None,
                            seed: None,
                        });
                    }
                    Ok(_) | Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
                    Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return,
                }
                let t = started.elapsed().as_secs_f32();
                let amplitude = 0.5 + 0.45 * (t * 2.4).sin();
                let _ = sender.push_patch(ScenePatch::SetFxAmplitude {
                    id: "voice_orb".into(),
                    value: amplitude,
                });
            }
        });
    }) {
        eprintln!("fx_showcase: {err}");
        std::process::exit(1);
    }
}

fn bench_frames(scene: Scene, frames: usize) -> (f64, usize, bool) {
    let mut engine = UiEngine::new(scene, VIEWPORT);
    let mut dl = DrawList::new();

    // Compile every text/glyph once before the timed loop.
    for _ in 0..4 {
        engine.tick_and_compile(1.0 / 120.0, &mut dl);
    }

    let start = Instant::now();
    let mut compiled = 0;
    for _ in 0..frames {
        if engine.tick_and_compile(1.0 / 120.0, &mut dl).compiled {
            compiled += 1;
        }
    }
    let ms = start.elapsed().as_secs_f64() * 1000.0 / frames as f64;
    (ms, compiled, engine.fx.is_animating())
}

fn run_bench() {
    let frames = 240;
    let (fx_ms, fx_compiled, animating) = bench_frames(build_scene(true), frames);
    let (base_ms, base_compiled, _) = bench_frames(build_scene(false), frames);

    println!("sofuu.fx headless bench (CPU compile, {frames} frames @ 120Hz)");
    println!("  baseline (no fx):  {base_ms:.4} ms/frame, {base_compiled}/{frames} compiles");
    println!("  effects scene:     {fx_ms:.4} ms/frame, {fx_compiled}/{frames} compiles");
    println!("  fx delta:          {:+.4} ms/frame", fx_ms - base_ms);
    assert!(animating, "ambient effects must keep requesting frames");

    // Reduced motion stops ambient effects; one-shots still settle, then the engine
    // must return `compiled == false, animating == false` (the 0% idle gate).
    let mut engine = UiEngine::new(build_scene(true), VIEWPORT);
    let mut dl = DrawList::new();
    engine.tick_and_compile(1.0 / 120.0, &mut dl);
    engine.apply_patch(ScenePatch::FxTrigger {
        target: Some("celebrate".into()),
        kind: FxKind::Confetti,
        count: Some(64),
        x: None,
        y: None,
        seed: Some(7),
    });
    engine.set_reduced_motion(true);

    let settle_start = Instant::now();
    let mut settle_frames = 0;
    loop {
        let status = engine.tick_and_compile(1.0 / 120.0, &mut dl);
        settle_frames += 1;
        if !status.animating {
            break;
        }
        assert!(
            settle_frames <= 144,
            "one-shots must reach 0% idle within 1.2s (stopped after {settle_frames} frames)"
        );
    }
    let idle = engine.tick_and_compile(1.0 / 120.0, &mut dl);
    assert!(
        !idle.compiled && !idle.animating,
        "idle frame must do no work"
    );
    println!(
        "  idle gate:         settled in {settle_frames} frames ({:.1} ms), next frame compiled=false animating=false",
        settle_start.elapsed().as_secs_f64() * 1000.0
    );
}

fn main() {
    if std::env::var("SOFUU_FX_BENCH").is_ok() {
        run_bench();
    } else {
        run_showcase();
    }
}
