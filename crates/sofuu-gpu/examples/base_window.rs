//! Smoke example for the base Sofuu 2D Metal GPU UI engine (`cargo run -p sofuu-gpu --example base_window`).

use std::thread;
use std::time::Duration;

use sofuu_gpu::{run_window, WindowConfig};
use sofuu_ui::{
    Align, Axis, BoxStyle, Insets, Node, NodeKind, Scene, ScenePatch, SizeSpec, TabItem, Theme,
    ThemeMode, UIEvent,
};

fn build_demo_scene() -> Scene {
    let t = Theme::dark();
    Scene::new(
        "Sofuu 2D GPU Base Engine (Metal)",
        ThemeMode::Dark,
        Node::new(
            "root",
            NodeKind::Container {
                axis: Axis::Column,
                align: Align::Stretch,
                gap: 14.0,
                fold_width: None,
                scrollable: false,
                style: None,
                children: vec![
                    Node::new(
                        "header",
                        NodeKind::Surface {
                            style: BoxStyle::card(t.surface_1, t.border, t.radius_card),
                            interactive: false,
                            child: Some(Box::new(Node::new(
                                "header_row",
                                NodeKind::Container {
                                    axis: Axis::Row,
                                    align: Align::Center,
                                    gap: 12.0,
                                    fold_width: Some(540.0),
                                    scrollable: false,
                                    style: None,
                                    children: vec![
                                        Node::new(
                                            "title",
                                            NodeKind::Text {
                                                content: "Sofuu Custom 2D GPU Engine".into(),
                                                size: 18.0,
                                                color: Some(t.text_primary),
                                                mono: false,
                                                wrap: false,
                                            },
                                        )
                                        .with_size(SizeSpec::Fill(1.0), SizeSpec::Auto),
                                        Node::new(
                                            "pulse",
                                            NodeKind::Text {
                                                content: "Agent Tick: 0 | 120Hz Springs + Metal SDF".into(),
                                                size: 13.0,
                                                color: Some(t.info),
                                                mono: true,
                                                wrap: false,
                                            },
                                        ),
                                    ],
                                },
                            ))),
                        },
                    )
                    .with_padding(Insets::all(16.0)),
                    Node::new(
                        "tabs",
                        NodeKind::Tabs {
                            active: 0,
                            items: vec![
                                TabItem {
                                    label: "Live Cards".into(),
                                    child: Node::new(
                                        "grid",
                                        NodeKind::Grid {
                                            columns: 0,
                                            min_col_width: 260.0,
                                            gap: 14.0,
                                            style: None,
                                            children: vec![
                                                Node::new(
                                                    "card_a",
                                                    NodeKind::Surface {
                                                        style: BoxStyle::card(t.surface_1, t.border, t.radius_card),
                                                        interactive: true,
                                                        child: Some(Box::new(Node::new(
                                                            "card_a_txt",
                                                            NodeKind::Text {
                                                                content: "Interactive Surface A\nHover & Click (120Hz Spring)".into(),
                                                                size: 14.0,
                                                                color: None,
                                                                mono: false,
                                                                wrap: true,
                                                            },
                                                        ))),
                                                    },
                                                )
                                                .with_padding(Insets::all(18.0))
                                                .with_size(SizeSpec::Auto, SizeSpec::Px(130.0)),
                                                Node::new(
                                                    "card_b",
                                                    NodeKind::Surface {
                                                        style: BoxStyle {
                                                            bg: Some(t.surface_1),
                                                            bg_bottom: Some(t.surface_2),
                                                            hover_bg: Some(t.surface_3),
                                                            border: Some(t.border),
                                                            border_width: 1.0,
                                                            radius: [t.radius_card; 4],
                                                            shadow_blur: 18.0,
                                                            shadow_offset_y: 6.0,
                                                            opacity: 1.0,
                                                        },
                                                        interactive: true,
                                                        child: Some(Box::new(Node::new(
                                                            "card_b_txt",
                                                            NodeKind::Text {
                                                                content: "Gradient + Analytic Shadow\n0% CPU/GPU when idle".into(),
                                                                size: 14.0,
                                                                color: None,
                                                                mono: false,
                                                                wrap: true,
                                                            },
                                                        ))),
                                                    },
                                                )
                                                .with_padding(Insets::all(18.0))
                                                .with_size(SizeSpec::Auto, SizeSpec::Px(130.0)),
                                            ],
                                        },
                                    ),
                                },
                                TabItem {
                                    label: "Architecture".into(),
                                    child: Node::new(
                                        "arch_box",
                                        NodeKind::Surface {
                                            style: BoxStyle::card(t.surface_1, t.border, t.radius_card),
                                            interactive: false,
                                            child: Some(Box::new(Node::new(
                                                "arch_txt",
                                                NodeKind::Text {
                                                    content: "Thread 0: Direct NSWindow + CAMetalLayer + 3 MSL Shaders\nThread 1: Background 24/7 Agent pushing ScenePatch updates".into(),
                                                    size: 14.0,
                                                    color: None,
                                                    mono: true,
                                                    wrap: true,
                                                },
                                            ))),
                                        },
                                    )
                                    .with_padding(Insets::all(18.0)),
                                },
                            ],
                        },
                    )
                    .with_size(SizeSpec::Fill(1.0), SizeSpec::Fill(1.0)),
                ],
            },
        )
        .with_padding(Insets::all(20.0)),
    )
}

fn main() {
    let cfg = WindowConfig {
        title: "Sofuu Base 2D Metal Window".into(),
        width: 920.0,
        height: 580.0,
        ..WindowConfig::default()
    };

    if let Err(err) = run_window(cfg, build_demo_scene(), |ui| {
        let (sender, events) = ui.split();
        // Simulated background 24/7 agent thread driving UI patches asynchronously:
        thread::spawn(move || {
            let mut tick = 0u64;
            loop {
                while let Ok(ev) = events.try_recv() {
                    if let UIEvent::WindowClosed = ev {
                        return;
                    }
                    if let UIEvent::Click { id } = ev {
                        let _ = sender.push_patch(ScenePatch::SetText {
                            id: format!("{id}_txt"),
                            content: format!("Clicked surface '{id}' at tick #{tick}!"),
                        });
                    }
                }
                thread::sleep(Duration::from_millis(1000));
                tick += 1;
                if !sender.push_patch(ScenePatch::SetText {
                    id: "pulse".into(),
                    content: format!("Agent Tick: {tick} | 120Hz Springs + Metal SDF"),
                }) {
                    break;
                }
            }
        });
    }) {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}
