//! Domain-agnostic GPU showcase exercising all `sofuu-ui` components (`Stat`, `Plot`,
//! `Table`, `Feed`, `Graph`, `CodeText`, `Input`, `Switch`, `Segmented`, `Button`, `ThemeToggle`).
//!
//! Run with:
//!   cargo run -p sofuu-gpu --example components_showcase

use std::time::Duration;

use sofuu_gpu::{run_window, WindowConfig};
use sofuu_ui::{
    ui, Align, FeedItem, GraphEdge, GraphNode, Node, PlotPoint, ScenePatch, SizeSpec, TableAlign,
    TableCell, TableColumn, Tone, UIEvent,
};

fn build_showcase_scene() -> sofuu_ui::Scene {
    let header: Node = ui::row("hdr")
        .align(Align::Center)
        .add(ui::title("Sofuu UI // Component Suite"))
        .add(ui::input("filter_input", "Filter items or type note..."))
        .add(ui::switch("live_switch", true, Some("Live Stream")))
        .add(ui::button("append_btn", "Add Event"))
        .add(ui::theme_toggle("theme_btn"))
        .into();

    let stat_a = ui::stat("st_rate", "Throughput", 1420.5)
        .badge("LIVE")
        .suffix("/s")
        .delta("+6.4%")
        .spark(vec![1180.0, 1240.0, 1310.0, 1295.0, 1380.0, 1420.5])
        .ok();
    let stat_b = ui::stat("st_lat", "P95 Latency", 14.2)
        .badge("MS")
        .suffix(" ms")
        .delta("-2.1 ms")
        .range(5.0, 40.0)
        .info();

    let pts: Vec<PlotPoint> = (0..32)
        .map(|i| PlotPoint {
            x: i as f64,
            y: 50.0 + ((i as f64) * 0.35).sin() * 24.0 + (i as f64) * 0.4,
            y2: None,
            label: Some(format!("T+{i}")),
        })
        .collect();
    let mut plot_node: Node = ui::plot("mini_plot")
        .title("Signal Envelope")
        .area("s0", "Load %", pts, None)
        .into();
    plot_node.width = SizeSpec::Fill(1.0);
    plot_node.height = SizeSpec::Fill(1.0);

    let mut row_top: Node = ui::row("top_band")
        .fold_below(760.0)
        .align(Align::Stretch)
        .add(stat_a)
        .add(stat_b)
        .add(plot_node)
        .into();
    row_top.height = SizeSpec::Px(158.0);

    let table_node: Node = ui::table("jobs_table")
        .title("Worker Queue")
        .columns([
            TableColumn::new("id", "Task").width(96.0),
            TableColumn::new("status", "State").align(TableAlign::Center).width(92.0),
            TableColumn::new("dur", "Elapsed").align(TableAlign::Right),
            TableColumn::new("load", "Score").align(TableAlign::Right),
        ])
        .row("r1", ["task_01".into(), TableCell::new("ACTIVE").tone(Tone::Ok).badge(), "12.4 ms".into(), "98.2".into()])
        .row("r2", ["task_02".into(), TableCell::new("QUEUE").tone(Tone::Info).badge(), "4.1 ms".into(), "84.0".into()])
        .row("r3", ["task_03".into(), TableCell::new("RETRY").tone(Tone::Warn).badge(), "29.8 ms".into(), "61.5".into()])
        .row("r4", ["task_04".into(), TableCell::new("OK").tone(Tone::Ok).badge(), "8.9 ms".into(), "94.7".into()])
        .row("r5", ["task_05".into(), TableCell::new("HALT").tone(Tone::Danger).badge(), "0.0 ms".into(), "12.0".into()])
        .fill()
        .into();

    let graph_node: Node = ui::graph("pipeline_graph")
        .title("Reactive DAG")
        .node(GraphNode::new("src", "Source").subtitle("120Hz Poller").tone(Tone::Info))
        .node(GraphNode::new("gate", "Pre-LLM Gate").subtitle("Deterministic").tone(Tone::Ok))
        .node(GraphNode::new("agent", "Synthesizer").subtitle("Worker Pool").tone(Tone::Warn))
        .node(GraphNode::new("sink", "Channel Out").subtitle("Dispatcher").tone(Tone::Ok))
        .edge(GraphEdge::new("src", "gate").label("batch").active(true))
        .edge(GraphEdge::new("gate", "agent").label("trigger").active(true))
        .edge(GraphEdge::new("gate", "sink").label("fast-path").active(false))
        .edge(GraphEdge::new("agent", "sink").label("notify").active(true))
        .fill()
        .into();

    let mid_band: Node = ui::row("mid_band")
        .fold_below(780.0)
        .fill()
        .add(table_node)
        .add(graph_node)
        .into();

    let feed_node: Node = ui::feed("event_feed")
        .title("Runtime Stream")
        .item(
            FeedItem::new("ev1", "Gate passed composite threshold")
                .time("23:50:12")
                .badge("GATE")
                .tone(Tone::Ok)
                .body("Dispatched worker batch cleanly with zero overlap."),
        )
        .item(
            FeedItem::new("ev2", "Checkpoint suspended awaiting confirm")
                .time("23:49:44")
                .badge("HITL")
                .tone(Tone::Warn)
                .body("Run parked in QTSQ trace store; memory released."),
        )
        .item(
            FeedItem::new("ev3", "Channel delivery confirmed")
                .time("23:49:02")
                .badge("CHAN")
                .tone(Tone::Info)
                .body("Outbound adapter acknowledged receipt in 48ms."),
        )
        .fill()
        .into();

    let code_sample = r#"// Domain-agnostic reactive trigger gate
export async function gate(ctx) {
  const score = ctx.evaluators.score(ctx.sample, rules);
  if (!score.triggered) return null; // 0 tokens spent
  return { composite: score.value, seq: ctx.tick };
}

export async function onTrigger(payload, ctx) {
  await ctx.channels.send({ text: `Signal=${payload.composite}` });
}"#;
    let code_node: Node = ui::code("snippet_view", code_sample)
        .title("trigger_gate.js")
        .language("JS")
        .highlight(3, Tone::Info)
        .highlight(4, Tone::Ok)
        .fill()
        .into();

    let bot_band: Node = ui::row("bot_band")
        .fold_below(780.0)
        .fill()
        .add(feed_node)
        .add(code_node)
        .into();

    ui::app("Sofuu UI // Component Suite")
        .header(header)
        .add(row_top)
        .add(mid_band)
        .add(bot_band)
        .build()
}

fn main() {
    let cfg = WindowConfig {
        title: "Sofuu UI — All Components".into(),
        width: 1180.0,
        height: 820.0,
        min_width: 680.0,
        min_height: 520.0,
    };

    if let Err(err) = run_window(cfg, build_showcase_scene(), |ui_handle| {
        let (sender, events) = ui_handle.split();
        std::thread::spawn(move || {
            let mut live = true;
            let mut seq = 4u64;
            loop {
                while let Ok(ev) = events.try_recv() {
                    match ev {
                        UIEvent::WindowClosed => return,
                        UIEvent::SwitchToggled { checked, .. } => {
                            live = checked;
                        }
                        UIEvent::Click { id } if id == "append_btn" => {
                            seq += 1;
                            let _ = sender.push_patch(ScenePatch::AppendFeedItem {
                                id: "event_feed".into(),
                                item: FeedItem::new(
                                    format!("ev{seq}"),
                                    format!("Manual pulse #{seq}"),
                                )
                                .time("now")
                                .badge("USER")
                                .tone(Tone::Info)
                                .body("Injected live via ScenePatch::AppendFeedItem."),
                                max_items: Some(50),
                            });
                        }
                        UIEvent::InputSubmitted { id, value } => {
                            seq += 1;
                            let _ = sender.push_patch(ScenePatch::AppendFeedItem {
                                id: "event_feed".into(),
                                item: FeedItem::new(
                                    format!("ev{seq}"),
                                    format!("Submitted: {value}"),
                                )
                                .time("now")
                                .badge("INPUT")
                                .tone(Tone::Ok),
                                max_items: Some(50),
                            });
                            let _ = sender.push_patch(ScenePatch::SetInput {
                                id,
                                value: String::new(),
                            });
                        }
                        _ => {}
                    }
                }
                if live {
                    seq += 1;
                    let v = 1350.0 + ((seq as f64) * 0.4).sin() * 90.0;
                    let _ = sender.push_patch(ScenePatch::UpdateStat {
                        id: "st_rate".into(),
                        value: Some(v),
                        value_text: None,
                        delta: None,
                        tone: None,
                        caption: None,
                        push_sparkline: Some(v as f32),
                        sparkline_max_len: Some(32),
                    });
                }
                std::thread::sleep(Duration::from_millis(650));
            }
        });
    }) {
        eprintln!("showcase error: {err}");
    }
}
