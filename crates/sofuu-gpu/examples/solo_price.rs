//! Minimal single-component live price window (`cargo run -p sofuu-gpu --example solo_price`).
//!
//! Shows how a developer can mount a single auto-expanding `Stat` component in just a few lines
//! (Light theme by default; scales up automatically into Solo Hero Focus layout).

use std::thread;
use std::time::Duration;

use sofuu_gpu::{run_window, WindowConfig};
use sofuu_ui::{ui, ScenePatch, Tone, UIEvent};

fn main() {
    let scene = ui::app("Spot Price")
        .add(
            ui::stat("price", "Live Index", 94_210.50)
                .badge("LIVE")
                .prefix("$")
                .precision(2)
                .delta("+1.42%")
                .ok()
                .spark(vec![92_800.0, 93_120.0, 93_490.0, 93_850.0, 94_210.5]),
        )
        .build();

    let cfg = WindowConfig {
        title: "Solo Price".into(),
        width: 540.0,
        height: 360.0,
        min_width: 240.0,
        min_height: 160.0,
    };

    let _ = run_window(cfg, scene, |handle| {
        let (sender, events) = handle.split();
        thread::spawn(move || {
            let mut price = 94_210.50_f64;
            let mut step = 0u64;
            loop {
                while let Ok(UIEvent::WindowClosed) = events.try_recv() {
                    return;
                }
                thread::sleep(Duration::from_millis(800));
                step += 1;
                price += (step as f64 * 0.6).sin() * 110.0;
                let pct = ((price - 92_800.0) / 92_800.0) * 100.0;
                let tone = if pct >= 0.0 { Tone::Ok } else { Tone::Danger };
                if !sender.push_patch(ScenePatch::UpdateStat {
                    id: "price".into(),
                    value: Some(price),
                    value_text: None,
                    delta: Some(Some(format!("{pct:+.2}%"))),
                    tone: Some(tone),
                    caption: None,
                    push_sparkline: Some(price as f32),
                    sparkline_max_len: Some(40),
                }) {
                    break;
                }
            }
        });
    });
}
