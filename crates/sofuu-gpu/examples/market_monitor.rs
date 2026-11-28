//! Pro Multi-Variant Market Monitor built with the fluent `sofuu_ui::ui` API.
//! Run with: `cargo run -p sofuu-gpu --example market_monitor`
//!
//! Defaults to **Light theme** unless `.dark()` is passed explicitly (toggle via the top-right Sun/Moon icon).

use std::thread;
use std::time::Duration;

use sofuu_gpu::{run_window, WindowConfig};
use sofuu_ui::{hex_rgba, ui, CandlePoint, Node, PlotPoint, Scene, ScenePatch, Tone, UIEvent};

struct GeneratedMarketData {
    candles: Vec<CandlePoint>,
    ema: Vec<PlotPoint>,
    volume: Vec<PlotPoint>,
    band_spread: Vec<PlotPoint>,
    stoch: Vec<PlotPoint>,
    alloc_series: [Vec<PlotPoint>; 5],
}

fn generate_series(count: usize, tf_scale: f64) -> GeneratedMarketData {
    let mut candles = Vec::with_capacity(count);
    let mut ema = Vec::with_capacity(count);
    let mut volume = Vec::with_capacity(count);
    let mut band_spread = Vec::with_capacity(count);
    let mut stoch = Vec::with_capacity(count);
    let mut a0 = Vec::with_capacity(count);
    let mut a1 = Vec::with_capacity(count);
    let mut a2 = Vec::with_capacity(count);
    let mut a3 = Vec::with_capacity(count);
    let mut a4 = Vec::with_capacity(count);

    let mut price = 242.0_f64;
    let mut ema_v = price;

    for i in 0..count {
        let t = i as f64 * tf_scale;
        let drift = (t * 0.14).sin() * 4.8 + (t * 0.047).cos() * 3.1;
        let noise = (((i * 37) % 13) as f64 - 6.0) * 0.62;
        let open = price;
        let close = (open + drift * 0.45 + noise).max(165.0);
        let wick_hi = 1.6 + (((i * 19) % 7) as f64) * 0.55;
        let wick_lo = 1.5 + (((i * 11) % 6) as f64) * 0.52;
        let high = open.max(close) + wick_hi;
        let low = (open.min(close) - wick_lo).max(150.0);
        ema_v = ema_v * 0.86 + close * 0.14;
        price = close;

        let months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
        let lbl = format!("{} {:02}", months[(i / 12) % 12], 1 + (i % 28));

        candles.push(CandlePoint { x: i as f64, open, high, low, close, label: Some(lbl.clone()) });
        ema.push(PlotPoint { x: i as f64, y: ema_v, y2: None, label: Some(lbl.clone()) });
        volume.push(PlotPoint {
            x: i as f64,
            y: 120.0 + ((t * 0.31).abs().sin() * 240.0) + (close - open).abs() * 32.0,
            y2: Some(if close >= open { 0.0 } else { 999999.0 }),
            label: Some(lbl.clone()),
        });

        let sig_fast = (40.0 + (t * 0.21).sin() * 14.0 + (t * 0.61).cos() * 5.0).clamp(15.0, 85.0);
        let sig_slow = (38.0 + (t * 0.13).cos() * 11.0 + (t * 0.41).sin() * 4.0).clamp(15.0, 85.0);
        band_spread.push(PlotPoint { x: i as f64, y: sig_fast, y2: Some(sig_slow), label: Some(lbl.clone()) });

        let osc = (50.0 + (t * 0.29).sin() * 32.0 + (t * 0.73).cos() * 11.0).clamp(8.0, 94.0);
        stoch.push(PlotPoint { x: i as f64, y: osc, y2: None, label: Some(lbl.clone()) });

        a0.push(PlotPoint { x: i as f64, y: 38.0 + (t * 0.11).sin() * 7.0, y2: None, label: Some(lbl.clone()) });
        a1.push(PlotPoint { x: i as f64, y: 22.0 + (t * 0.16).cos() * 4.5, y2: None, label: Some(lbl.clone()) });
        a2.push(PlotPoint { x: i as f64, y: 14.0 + (t * 0.22).sin() * 3.5, y2: None, label: Some(lbl.clone()) });
        a3.push(PlotPoint { x: i as f64, y: 18.0 + (t * 0.09).cos() * 3.0, y2: None, label: Some(lbl.clone()) });
        a4.push(PlotPoint { x: i as f64, y: 8.0 + (t * 0.18).sin() * 2.0, y2: None, label: Some(lbl) });
    }

    GeneratedMarketData {
        candles,
        ema,
        volume,
        band_spread,
        stoch,
        alloc_series: [a0, a1, a2, a3, a4],
    }
}

fn build_plot_node(variant_idx: usize, tf_idx: usize) -> Node {
    let tf_scales = [0.65, 1.0, 1.45, 2.1];
    let data = generate_series(96, tf_scales[tf_idx.min(tf_scales.len() - 1)]);
    let ema_col = Some(hex_rgba(0x0284C7, 255));
    let band_b = Some(hex_rgba(0xA855F7, 255));
    let thresh_col = Some(hex_rgba(0x64748B, 160));

    match variant_idx {
        0 => ui::plot("main_plot")
            .title("Market Telemetry // Multi-Pane Suite")
            .subtitle("Scroll: Zoom | Drag: Pan | Right-Click: Reset")
            .pane(
                ui::pane("price")
                    .label("Spot OHLC")
                    .weight(2.2)
                    .y_prefix("$")
                    .precision(2)
                    .candles("ohlc", "OHLC", data.candles)
                    .line("ema", "EMA-12", data.ema, ema_col),
            )
            .pane(
                ui::pane("spread")
                    .label("Momentum Band")
                    .weight(1.05)
                    .precision(2)
                    .threshold(50.0, "50", thresh_col)
                    .band("band", "Fast / Slow", data.band_spread, ema_col, band_b),
            )
            .pane(
                ui::pane("stoch")
                    .label("Oscillator")
                    .weight(0.95)
                    .precision(0)
                    .threshold(80.0, "80", thresh_col)
                    .threshold(20.0, "20", thresh_col)
                    .area("osc", "Signal %", data.stoch, Some(hex_rgba(0xD97706, 255))),
            )
            .into(),
        1 => ui::plot("main_plot")
            .title("Primary Feed // Candles & Volume")
            .subtitle("Scroll: Zoom | Drag: Pan | Right-Click: Reset")
            .pane(
                ui::pane("price")
                    .label("Price")
                    .weight(2.6)
                    .y_prefix("$")
                    .precision(2)
                    .candles("ohlc", "OHLC", data.candles)
                    .area("ema", "EMA-12", data.ema, ema_col),
            )
            .pane(
                ui::pane("vol")
                    .label("Volume")
                    .weight(0.9)
                    .precision(0)
                    .histogram("vol_hist", "Vol", data.volume, Some(hex_rgba(0x2563EB, 255))),
            )
            .into(),
        _ => ui::plot("main_plot")
            .title("Portfolio Exposure // 100% Stacked Share")
            .subtitle("Smooth continuous multi-band allocation")
            .percent()
            .precision(1)
            .stacked("tier_a", "Core", data.alloc_series[0].clone(), hex_rgba(0x06B6D4, 255))
            .stacked("tier_b", "Growth", data.alloc_series[1].clone(), hex_rgba(0x8B5CF6, 255))
            .stacked("tier_c", "Alt", data.alloc_series[2].clone(), hex_rgba(0x10B981, 255))
            .stacked("tier_d", "Reserve", data.alloc_series[3].clone(), hex_rgba(0xF59E0B, 255))
            .stacked("tier_e", "Other", data.alloc_series[4].clone(), hex_rgba(0x94A3B8, 255))
            .into(),
    }
}

fn build_market_scene() -> Scene {
    let seed = generate_series(96, 0.65);
    let last_close = seed.candles.last().map(|c| c.close).unwrap_or(242.0);
    let lo_24 = seed.candles.iter().map(|c| c.low).fold(f64::INFINITY, f64::min);
    let hi_24 = seed.candles.iter().map(|c| c.high).fold(f64::NEG_INFINITY, f64::max);
    let spark: Vec<f32> = seed.candles.iter().rev().take(28).rev().map(|c| c.close as f32).collect();

    // Light theme by default! Pass `.dark()` to opt in to Dark mode.
    ui::app("Sofuu Market Monitor")
        .header(
            ui::row("topbar")
                .add(ui::title("Market Monitor"))
                .add(ui::segmented("mode_seg", 0, ["Multi-Pane", "Candles", "Stacked"]))
                .add(ui::segmented("tf_seg", 0, ["15m", "1h", "4h", "1D"]))
                .add(ui::button("reset_btn", "Fit"))
                .add(ui::theme_toggle("theme_toggle")),
        )
        .stats([
            ui::stat("stat_spot", "Index Price", last_close)
                .badge("LIVE")
                .prefix("$")
                .precision(2)
                .delta("+1.84%")
                .ok()
                .spark(spark)
                .caption("Smooth ribbon sparkline"),
            ui::stat("stat_range", "Session Range", last_close)
                .badge("24H")
                .prefix("$")
                .precision(2)
                .delta("MID")
                .info()
                .range(lo_24, hi_24)
                .caption("High / Low range position"),
            ui::stat("stat_vol", "24h Volume", 1_842_000_000.0)
                .badge("USD")
                .prefix("$")
                .delta("+14.2%")
                .warn()
                .spark(vec![1.2, 1.38, 1.31, 1.54, 1.72, 1.84])
                .caption("Aggregate liquidity"),
        ])
        .add(build_plot_node(0, 0))
        .build()
}

fn main() {
    let cfg = WindowConfig {
        title: "Sofuu Market Monitor".into(),
        width: 1180.0,
        height: 780.0,
        min_width: 360.0,
        min_height: 340.0,
        ..WindowConfig::default()
    };

    let initial_seed = generate_series(96, 0.65);
    let init_price = initial_seed.candles.last().map(|c| c.close).unwrap_or(242.0);
    let init_ema = initial_seed.ema.last().map(|p| p.y).unwrap_or(init_price);
    let init_base = init_price;

    if let Err(err) = run_window(cfg, build_market_scene(), |ui_handle| {
        let (sender, events) = ui_handle.split();
        thread::spawn(move || {
            let mut mode_idx = 0usize;
            let mut tf_idx = 0usize;
            let mut x_idx = 96.0_f64;
            let mut price = init_price;
            let mut ema_v = init_ema;
            let mut step = 0u64;

            loop {
                while let Ok(ev) = events.try_recv() {
                    match ev {
                        UIEvent::WindowClosed => return,
                        UIEvent::SegmentedChanged { id, index } if id == "mode_seg" => {
                            mode_idx = index;
                            let _ = sender.push_patch(ScenePatch::ReplaceNode {
                                id: "main_plot".into(),
                                node: build_plot_node(mode_idx, tf_idx),
                            });
                        }
                        UIEvent::SegmentedChanged { id, index } if id == "tf_seg" => {
                            tf_idx = index;
                            let _ = sender.push_patch(ScenePatch::ReplaceNode {
                                id: "main_plot".into(),
                                node: build_plot_node(mode_idx, tf_idx),
                            });
                        }
                        UIEvent::Click { id } if id == "reset_btn" => {
                            let _ = sender.push_patch(ScenePatch::ResetPlotView {
                                id: "main_plot".into(),
                            });
                        }
                        _ => {}
                    }
                }

                thread::sleep(Duration::from_millis(850));
                step += 1;

                let dp = (step as f64 * 0.55).sin() * 2.4 + (step as f64 * 0.19).cos() * 1.1;
                let open = price;
                let close = (open + dp).max(165.0);
                let high = open.max(close) + 1.4;
                let low = (open.min(close) - 1.3).max(150.0);
                price = close;
                ema_v = ema_v * 0.86 + close * 0.14;
                x_idx += 1.0;

                let pct = ((close - init_base) / init_base) * 100.0;
                let tone = if pct >= 0.0 { Tone::Ok } else { Tone::Danger };

                if !sender.push_patch(ScenePatch::UpdateStat {
                    id: "stat_spot".into(),
                    value: Some(close),
                    value_text: None,
                    delta: Some(Some(format!("{pct:+.2}%"))),
                    tone: Some(tone),
                    caption: None,
                    push_sparkline: Some(close as f32),
                    sparkline_max_len: Some(32),
                }) {
                    break;
                }
                let _ = sender.push_patch(ScenePatch::UpdateStat {
                    id: "stat_range".into(),
                    value: Some(close),
                    value_text: None,
                    delta: None,
                    tone: None,
                    caption: None,
                    push_sparkline: None,
                    sparkline_max_len: None,
                });

                if mode_idx != 2 {
                    let lbl = format!("+{step}");
                    let _ = sender.push_patch(ScenePatch::UpsertCandle {
                        id: "main_plot".into(),
                        series_id: "ohlc".into(),
                        candle: CandlePoint { x: x_idx, open, high, low, close, label: Some(lbl.clone()) },
                        max_candles: Some(140),
                    });
                    let _ = sender.push_patch(ScenePatch::AppendPlotPoint {
                        id: "main_plot".into(),
                        series_id: "ema".into(),
                        point: PlotPoint { x: x_idx, y: ema_v, y2: None, label: Some(lbl.clone()) },
                        max_points: Some(140),
                    });
                    if mode_idx == 0 {
                        let sig_fast = (40.0 + (step as f64 * 0.21).sin() * 14.0).clamp(15.0, 85.0);
                        let sig_slow = (38.0 + (step as f64 * 0.13).cos() * 11.0).clamp(15.0, 85.0);
                        let _ = sender.push_patch(ScenePatch::AppendPlotPoint {
                            id: "main_plot".into(),
                            series_id: "band".into(),
                            point: PlotPoint { x: x_idx, y: sig_fast, y2: Some(sig_slow), label: Some(lbl.clone()) },
                            max_points: Some(140),
                        });
                        let osc_v = (50.0 + (step as f64 * 0.4).sin() * 34.0).clamp(8.0, 94.0);
                        let _ = sender.push_patch(ScenePatch::AppendPlotPoint {
                            id: "main_plot".into(),
                            series_id: "osc".into(),
                            point: PlotPoint { x: x_idx, y: osc_v, y2: None, label: Some(lbl) },
                            max_points: Some(140),
                        });
                    }
                }
            }
        });
    }) {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}
