//! Zero-boilerplate Auto-Composer (`auto_compose`).
//!
//! Adapts Sofuu's generic runtime telemetry (`agent` steps, `triggers` stats,
//! operational `state` tables, and HITL `approval_request` events) into a cohesive,
//! light-first responsive dashboard `Scene` and minimal incremental `ScenePatch` stream.

use std::collections::{BTreeMap, VecDeque};

use serde::{Deserialize, Serialize};

use crate::components::code::CodeHighlight;
use crate::components::feed::FeedItem;
use crate::components::graph::{GraphEdge, GraphNode};
use crate::components::table::{TableAlign, TableCell, TableColumn, TableData, TableRow};
use crate::ir::{Align, Axis, Node, NodeKind, Scene, ScenePatch, SizeSpec, TabItem};
use crate::theme::{ThemeMode, Tone};
use crate::ui;

const MAX_SPARKLINE: usize = 48;
const MAX_FEED_ITEMS: usize = 200;
const MAX_GRAPH_NODES: usize = 24;

/// Structured generic agent/runtime event ingested by [`AutoComposer`].
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct AutoAgentEvent {
    #[serde(default)]
    pub run_id: String,
    #[serde(default)]
    pub agent_name: String,
    #[serde(default)]
    pub profile: Option<String>,
    #[serde(default)]
    pub depth: usize,
    #[serde(default)]
    pub chain: Vec<String>,
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub tool: Option<String>,
    #[serde(default)]
    pub title: Option<String>,
    #[serde(default)]
    pub detail: Option<String>,
    #[serde(default)]
    pub tokens: Option<u64>,
    #[serde(default)]
    pub elapsed_ms: Option<u64>,
    #[serde(default)]
    pub time_label: Option<String>,
    #[serde(default)]
    pub tone: Option<Tone>,
    #[serde(default)]
    pub approval: Option<PendingApproval>,
}

/// Interactive human-in-the-loop approval request surfaced in the HITL action bar.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PendingApproval {
    pub id: String,
    pub tool: String,
    #[serde(default)]
    pub args_preview: String,
    #[serde(default)]
    pub mode: String,
}

/// Generic reactive trigger telemetry matching `sofuu.triggers.stats()`.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AutoTriggerStat {
    pub id: String,
    #[serde(default)]
    pub kind: String,
    #[serde(default)]
    pub running: bool,
    #[serde(default)]
    pub ticks: u64,
    #[serde(default)]
    pub gate_skips: u64,
    #[serde(default)]
    pub gate_hits: u64,
    #[serde(default)]
    pub ai_calls: u64,
    #[serde(default)]
    pub errors: u64,
    #[serde(default)]
    pub last_duration_ms: u64,
}

/// Stateful Auto-Composer converting runtime event/stat streams into a retained
/// [`Scene`] plus incremental [`ScenePatch`] deltas.
#[derive(Clone, Debug)]
pub struct AutoComposer {
    pub title: String,
    pub theme: ThemeMode,
    pub active_agent: String,
    pub active_profile: String,
    pub runs: u64,
    pub steps: u64,
    pub tokens: u64,
    pub tool_calls: u64,
    pub tool_errors: u64,
    pub token_sparkline: Vec<f32>,
    pub latency_sparkline: Vec<f32>,
    pub event_seq: u64,
    pub pending_approvals: VecDeque<PendingApproval>,
    pub triggers: Vec<AutoTriggerStat>,
    pub state_rows: Vec<TableRow>,
    pub graph_nodes: BTreeMap<String, GraphNode>,
    pub graph_order: Vec<String>,
    pub graph_edges: Vec<GraphEdge>,
    pub feed_items: Vec<FeedItem>,
    pub latest_code: String,
    pub latest_code_lang: String,
}

impl AutoComposer {
    /// Create a new Light-theme Auto-Composer dashboard (default unless `ThemeMode::Dark`).
    pub fn new(title: impl Into<String>, theme: ThemeMode) -> Self {
        Self {
            title: title.into(),
            theme,
            active_agent: "agent".to_string(),
            active_profile: "default".to_string(),
            runs: 0,
            steps: 0,
            tokens: 0,
            tool_calls: 0,
            tool_errors: 0,
            token_sparkline: Vec::new(),
            latency_sparkline: Vec::new(),
            event_seq: 0,
            pending_approvals: VecDeque::new(),
            triggers: Vec::new(),
            state_rows: Vec::new(),
            graph_nodes: BTreeMap::new(),
            graph_order: Vec::new(),
            graph_edges: Vec::new(),
            feed_items: Vec::new(),
            latest_code: "// Waiting for agent steps or trigger ticks...".to_string(),
            latest_code_lang: "JSON".to_string(),
        }
    }

    /// Build the complete initial [`Scene`] from current Auto-Composer state.
    pub fn build_scene(&self) -> Scene {
        let mut status_parts = vec![format!("agent: {}", self.active_agent)];
        if !self.active_profile.is_empty() && self.active_profile != "default" {
            status_parts.push(format!("profile: {}", self.active_profile));
        }
        status_parts.push(format!("steps: {}", self.steps));
        let status_label = status_parts.join("  ·  ");

        let header: Node = ui::row("auto_hdr")
            .gap(10.0)
            .fold_below(680.0)
            .align(Align::Center)
            .add(ui::text("auto_title", &self.title).size(18.0).fill())
            .add(ui::text("auto_status", status_label).size(12.5))
            .add(ui::theme_toggle("theme_toggle"))
            .into();

        let (trigger_rate, trigger_delta, trigger_tone) = self.trigger_summary();
        let err_tone = if self.tool_errors > 0 {
            Tone::Warn
        } else {
            Tone::Ok
        };

        let stats: Vec<Node> = vec![
            ui::stat("auto_stat_runs", "Runs / Steps", self.steps as f64)
                .precision(0)
                .badge("LIVE")
                .delta(format!("{} runs", self.runs))
                .info()
                .compact()
                .into(),
            ui::stat("auto_stat_tokens", "Tokens", self.tokens as f64)
                .precision(0)
                .badge("LLM")
                .spark(self.token_sparkline.clone())
                .caption(format!("Profile: {}", self.active_profile))
                .into(),
            ui::stat("auto_stat_gates", "Trigger Pass", trigger_rate)
                .suffix("%")
                .precision(1)
                .badge("GATE")
                .delta(trigger_delta)
                .tone_set(trigger_tone)
                .into(),
            ui::stat("auto_stat_tools", "Tool Calls", self.tool_calls as f64)
                .precision(0)
                .badge("EXEC")
                .delta(if self.tool_errors > 0 {
                    format!("{} err", self.tool_errors)
                } else {
                    "0 err".to_string()
                })
                .spark(self.latency_sparkline.clone())
                .tone_set(err_tone)
                .into(),
        ];

        let mut graph_node = self.build_graph_node();
        graph_node.width = SizeSpec::Fill(1.0);
        graph_node.height = SizeSpec::Fill(1.0);
        graph_node.min_height = 210.0;

        let mut feed_node: Node = ui::feed("auto_feed")
            .title("Runtime Stream")
            .max_items(MAX_FEED_ITEMS)
            .items(self.feed_items.clone())
            .empty_text("Waiting for agent or trigger telemetry...")
            .into();
        feed_node.width = SizeSpec::Fill(1.25);
        feed_node.height = SizeSpec::Fill(1.0);
        feed_node.min_height = 210.0;

        let mut main_split: Node = ui::row("auto_main")
            .gap(12.0)
            .fold_below(740.0)
            .align(Align::Stretch)
            .add(graph_node)
            .add(feed_node)
            .into();
        main_split.height = SizeSpec::Fill(1.0);
        main_split.min_height = 220.0;

        let table_node = self.build_table_node();
        let code_node: Node = ui::code("auto_trace_code", &self.latest_code)
            .title("Latest Event Payload")
            .language(&self.latest_code_lang)
            .highlight(1, Tone::Info)
            .fill()
            .into();

        let mut tabs_node = Node::new(
            "auto_bottom_tabs",
            NodeKind::Tabs {
                active: 0,
                items: vec![
                    TabItem {
                        label: "Triggers & State".to_string(),
                        child: table_node,
                    },
                    TabItem {
                        label: "Latest Payload".to_string(),
                        child: code_node,
                    },
                ],
            },
        );
        tabs_node.width = SizeSpec::Fill(1.0);
        tabs_node.height = SizeSpec::Px(196.0);

        let hitl_bar = self.build_hitl_bar();

        let mut prompt_input = ui::input("auto_prompt", "Send task or follow-up prompt...");
        prompt_input.width = SizeSpec::Fill(1.0);
        let run_btn = ui::button_toned("auto_run_btn", "Run", Tone::Info);
        let prompt_row: Node = ui::row("auto_prompt_row")
            .gap(10.0)
            .fold_below(520.0)
            .align(Align::Center)
            .add(prompt_input)
            .add(run_btn)
            .into();

        ui::app(&self.title)
            .theme(self.theme)
            .header(header)
            .stats(stats)
            .add(main_split)
            .add(tabs_node)
            .add(hitl_bar)
            .add(prompt_row)
            .build()
    }

    /// Ingest a runtime agent event and emit incremental [`ScenePatch`] deltas.
    pub fn ingest_event(&mut self, ev: AutoAgentEvent) -> Vec<ScenePatch> {
        let mut patches = Vec::new();
        self.event_seq = self.event_seq.wrapping_add(1);

        if !ev.agent_name.is_empty() {
            self.active_agent = ev.agent_name.clone();
        }
        if let Some(ref prof) = ev.profile {
            if !prof.is_empty() {
                self.active_profile = prof.clone();
            }
        }

        if ev.kind == "start" {
            self.runs += 1;
        }
        if matches!(
            ev.kind.as_str(),
            "start" | "plan" | "think" | "thought" | "tool" | "tool_result" | "llm" | "delegate" | "answer" | "stop" | "error"
        ) {
            self.steps += 1;
        }
        if ev.kind == "tool" {
            self.tool_calls += 1;
        }
        if ev.kind == "error" || ev.tone == Some(Tone::Danger) {
            self.tool_errors += 1;
        }
        if let Some(tok) = ev.tokens {
            self.tokens = self.tokens.saturating_add(tok);
            push_spark(&mut self.token_sparkline, self.tokens as f32);
        }
        if let Some(ms) = ev.elapsed_ms {
            push_spark(&mut self.latency_sparkline, ms as f32);
        }

        // Update execution topology graph from chain / tool / delegate nodes.
        self.record_topology(&ev);
        patches.push(ScenePatch::ReplaceNode {
            id: "auto_graph".to_string(),
            node: self.build_graph_node(),
        });

        // Update feed stream.
        let item = self.build_feed_item(&ev);
        if self.feed_items.len() >= MAX_FEED_ITEMS {
            self.feed_items.remove(0);
        }
        self.feed_items.push(item.clone());
        patches.push(ScenePatch::AppendFeedItem {
            id: "auto_feed".to_string(),
            item,
            max_items: Some(MAX_FEED_ITEMS),
        });

        // Update header status line & KPI counters.
        patches.push(ScenePatch::SetText {
            id: "auto_status".to_string(),
            content: format!(
                "agent: {}  ·  profile: {}  ·  steps: {}",
                self.active_agent, self.active_profile, self.steps
            ),
        });
        patches.push(ScenePatch::UpdateStat {
            id: "auto_stat_runs".to_string(),
            value: Some(self.steps as f64),
            value_text: None,
            delta: Some(Some(format!("{} runs", self.runs))),
            tone: None,
            caption: None,
            push_sparkline: None,
            sparkline_max_len: None,
        });
        patches.push(ScenePatch::UpdateStat {
            id: "auto_stat_tokens".to_string(),
            value: Some(self.tokens as f64),
            value_text: None,
            delta: None,
            tone: None,
            caption: Some(Some(format!("Profile: {}", self.active_profile))),
            push_sparkline: ev.tokens.map(|_| self.tokens as f32),
            sparkline_max_len: Some(MAX_SPARKLINE),
        });
        patches.push(ScenePatch::UpdateStat {
            id: "auto_stat_tools".to_string(),
            value: Some(self.tool_calls as f64),
            value_text: None,
            delta: Some(Some(if self.tool_errors > 0 {
                format!("{} err", self.tool_errors)
            } else {
                "0 err".to_string()
            })),
            tone: Some(if self.tool_errors > 0 {
                Tone::Warn
            } else {
                Tone::Ok
            }),
            caption: None,
            push_sparkline: ev.elapsed_ms.map(|ms| ms as f32),
            sparkline_max_len: Some(MAX_SPARKLINE),
        });

        // Latest JSON payload in Code tab.
        if let Ok(pretty) = serde_json::to_string_pretty(&ev) {
            self.latest_code = pretty.clone();
            patches.push(ScenePatch::SetCode {
                id: "auto_trace_code".to_string(),
                content: pretty,
                highlights: Some(vec![CodeHighlight {
                    line: 1,
                    tone: event_tone(&ev),
                }]),
            });
        }

        // HITL approval bar handling.
        if ev.kind == "approval_request" {
            if let Some(ap) = ev.approval {
                if !self.pending_approvals.iter().any(|p| p.id == ap.id) {
                    self.pending_approvals.push_back(ap);
                }
                patches.extend(self.hitl_patches());
            }
        } else if ev.kind == "approval_resolved" {
            if let Some(ap) = ev.approval {
                patches.extend(self.clear_approval(&ap.id));
            }
        }

        patches
    }

    /// Update trigger statistics and refresh the Trigger Gate KPI + Table.
    pub fn update_triggers(&mut self, triggers: &[AutoTriggerStat]) -> Vec<ScenePatch> {
        self.triggers = triggers.to_vec();
        let (rate, delta, tone) = self.trigger_summary();
        vec![
            ScenePatch::UpdateStat {
                id: "auto_stat_gates".to_string(),
                value: Some(rate),
                value_text: None,
                delta: Some(Some(delta)),
                tone: Some(tone),
                caption: None,
                push_sparkline: None,
                sparkline_max_len: None,
            },
            ScenePatch::ReplaceNode {
                id: "auto_state_table".to_string(),
                node: self.build_table_node(),
            },
        ]
    }

    /// Update generic operational state rows displayed alongside triggers in `auto_state_table`.
    pub fn update_state_rows(&mut self, rows: Vec<TableRow>) -> Vec<ScenePatch> {
        self.state_rows = rows;
        vec![ScenePatch::ReplaceNode {
            id: "auto_state_table".to_string(),
            node: self.build_table_node(),
        }]
    }

    /// Remove a resolved HITL approval and update visibility/label on the HITL action bar.
    pub fn clear_approval(&mut self, id: &str) -> Vec<ScenePatch> {
        self.pending_approvals.retain(|p| p.id != id);
        self.hitl_patches()
    }

    /// Peek at the currently surfaced HITL approval request (if any).
    pub fn current_approval(&self) -> Option<&PendingApproval> {
        self.pending_approvals.front()
    }

    fn hitl_patches(&self) -> Vec<ScenePatch> {
        if let Some(ap) = self.pending_approvals.front() {
            let summary = if ap.args_preview.is_empty() {
                format!("Approval required: `{}` ({})", ap.tool, ap.id)
            } else {
                format!("Approval required: `{}` · {}", ap.tool, ap.args_preview)
            };
            vec![
                ScenePatch::SetText {
                    id: "auto_hitl_label".to_string(),
                    content: summary,
                },
                ScenePatch::SetVisible {
                    id: "auto_hitl_bar".to_string(),
                    visible: true,
                },
            ]
        } else {
            vec![ScenePatch::SetVisible {
                id: "auto_hitl_bar".to_string(),
                visible: false,
            }]
        }
    }

    fn trigger_summary(&self) -> (f64, String, Tone) {
        if self.triggers.is_empty() {
            return (0.0, "idle".to_string(), Tone::Neutral);
        }
        let mut ticks = 0u64;
        let mut hits = 0u64;
        let mut errs = 0u64;
        let mut running = 0usize;
        for t in &self.triggers {
            ticks = ticks.saturating_add(t.ticks);
            hits = hits.saturating_add(t.gate_hits);
            errs = errs.saturating_add(t.errors);
            if t.running {
                running += 1;
            }
        }
        let pct = if ticks > 0 {
            (hits as f64 * 100.0) / (ticks as f64)
        } else {
            0.0
        };
        let delta = if errs > 0 {
            format!("{running} live · {errs} err")
        } else {
            format!("{running} live · {ticks} ticks")
        };
        let tone = if errs > 0 {
            Tone::Warn
        } else if running > 0 {
            Tone::Ok
        } else {
            Tone::Neutral
        };
        (pct, delta, tone)
    }

    fn record_topology(&mut self, ev: &AutoAgentEvent) {
        let chain: Vec<String> = if !ev.chain.is_empty() {
            ev.chain.clone()
        } else if !ev.agent_name.is_empty() {
            vec![ev.agent_name.clone()]
        } else {
            vec![self.active_agent.clone()]
        };

        let tone = event_tone(ev);
        for (idx, name) in chain.iter().enumerate() {
            let id = format!("ag:{name}");
            let is_leaf = idx + 1 == chain.len();
            let sub = if is_leaf {
                ev.kind.clone()
            } else {
                "supervisor".to_string()
            };
            self.upsert_graph_node(id.clone(), name.clone(), Some(sub), if is_leaf { tone } else { Tone::Info });
            if idx > 0 {
                let prev = format!("ag:{}", chain[idx - 1]);
                self.upsert_graph_edge(prev, id, Some("delegate".to_string()), is_leaf, Some(Tone::Info));
            }
        }

        if let Some(ref tool) = ev.tool {
            if !tool.is_empty() {
                let ag_id = format!(
                    "ag:{}",
                    chain.last().map(String::as_str).unwrap_or(&self.active_agent)
                );
                let tool_id = format!("tl:{tool}");
                let active = ev.kind == "tool" || ev.kind == "approval_request";
                self.upsert_graph_node(
                    tool_id.clone(),
                    tool.clone(),
                    Some("tool".to_string()),
                    tone,
                );
                self.upsert_graph_edge(ag_id, tool_id, Some(ev.kind.clone()), active, Some(tone));
            }
        }
    }

    fn upsert_graph_node(
        &mut self,
        id: String,
        label: String,
        subtitle: Option<String>,
        tone: Tone,
    ) {
        if !self.graph_nodes.contains_key(&id) {
            if self.graph_order.len() >= MAX_GRAPH_NODES {
                if let Some(old_id) = self.graph_order.first().cloned() {
                    self.graph_order.remove(0);
                    self.graph_nodes.remove(&old_id);
                    self.graph_edges.retain(|e| e.from != old_id && e.to != old_id);
                }
            }
            self.graph_order.push(id.clone());
        }
        let mut node = GraphNode::new(id.clone(), label).tone(tone);
        if let Some(sub) = subtitle {
            node = node.subtitle(sub);
        }
        self.graph_nodes.insert(id, node);
    }

    fn upsert_graph_edge(
        &mut self,
        from: String,
        to: String,
        label: Option<String>,
        active: bool,
        tone: Option<Tone>,
    ) {
        for e in &mut self.graph_edges {
            if e.from == from && e.to == to {
                e.label = label;
                e.active = active;
                e.tone = tone;
                return;
            }
            e.active = false;
        }
        self.graph_edges.push(GraphEdge {
            from,
            to,
            label,
            active,
            tone,
        });
    }

    fn build_graph_node(&self) -> Node {
        let mut b = ui::graph("auto_graph").title("Execution Topology").fill();
        for id in &self.graph_order {
            if let Some(n) = self.graph_nodes.get(id) {
                b = b.node(n.clone());
            }
        }
        for e in &self.graph_edges {
            b = b.edge(e.clone());
        }
        b.into()
    }

    fn build_table_node(&self) -> Node {
        let mut rows = Vec::new();
        for t in &self.triggers {
            let status_txt = if t.running { "LIVE" } else { "STOP" };
            let status_tone = if t.errors > 0 {
                Tone::Warn
            } else if t.running {
                Tone::Ok
            } else {
                Tone::Neutral
            };
            rows.push(TableRow::new(
                format!("trg:{}", t.id),
                vec![
                    TableCell::new(&t.id),
                    TableCell::new(status_txt).badge().tone(status_tone),
                    TableCell::new(format!("{}/{}", t.gate_hits, t.ticks)),
                    TableCell::new(format!("{} ms", t.last_duration_ms)),
                ],
            ));
        }
        rows.extend(self.state_rows.clone());
        Node::new(
            "auto_state_table",
            NodeKind::Table(TableData {
                title: Some("Triggers & Operational State".to_string()),
                columns: vec![
                    TableColumn::new("id", "Runner / Key"),
                    TableColumn::new("state", "Status").align(TableAlign::Center),
                    TableColumn::new("hits", "Hits / Ticks").align(TableAlign::Right),
                    TableColumn::new("dur", "Last / Detail").align(TableAlign::Right),
                ],
                rows,
                empty_text: Some("No triggers or state rows active yet".to_string()),
                ..TableData::default()
            }),
        )
        .with_size(SizeSpec::Fill(1.0), SizeSpec::Fill(1.0))
    }

    fn build_hitl_bar(&self) -> Node {
        let has_pending = !self.pending_approvals.is_empty();
        let label = if let Some(ap) = self.pending_approvals.front() {
            format!("Approval required: `{}` ({})", ap.tool, ap.id)
        } else {
            "No pending approval".to_string()
        };
        let mut label_node: Node = ui::text("auto_hitl_label", label).size(13.0).fill().into();
        label_node.min_width = 160.0;
        let mut bar = Node::new(
            "auto_hitl_bar",
            NodeKind::Container {
                axis: Axis::Row,
                align: Align::Center,
                gap: 8.0,
                fold_width: Some(600.0),
                scrollable: false,
                style: None,
                children: vec![
                    label_node,
                    ui::button_toned("auto_approve_btn", "Approve", Tone::Ok),
                    ui::button_toned("auto_always_btn", "Always", Tone::Info),
                    ui::button_toned("auto_deny_btn", "Deny", Tone::Danger),
                ],
            },
        );
        bar.visible = has_pending;
        bar
    }

    fn build_feed_item(&self, ev: &AutoAgentEvent) -> FeedItem {
        let badge = ev.kind.to_uppercase();
        let title = ev
            .title
            .clone()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| {
                if let Some(ref tool) = ev.tool {
                    format!("{} · {}", ev.kind, tool)
                } else {
                    format!("{} ({})", ev.kind, self.active_agent)
                }
            });
        let mut item = FeedItem::new(format!("auto_ev_{}", self.event_seq), title)
            .badge(badge)
            .tone(event_tone(ev));
        if let Some(ref ts) = ev.time_label {
            item = item.time(ts.clone());
        }
        if let Some(ref detail) = ev.detail {
            if !detail.is_empty() {
                item = item.body(detail.clone());
            }
        }
        item
    }
}

trait StatToneExt {
    fn tone_set(self, tone: Tone) -> Self;
}

impl StatToneExt for ui::StatBuilder {
    fn tone_set(self, tone: Tone) -> Self {
        match tone {
            Tone::Ok => self.ok(),
            Tone::Info => self.info(),
            Tone::Warn => self.warn(),
            Tone::Danger => self.danger(),
            Tone::Neutral => self,
        }
    }
}

fn push_spark(buf: &mut Vec<f32>, val: f32) {
    if buf.len() >= MAX_SPARKLINE {
        buf.remove(0);
    }
    buf.push(val);
}

fn event_tone(ev: &AutoAgentEvent) -> Tone {
    if let Some(t) = ev.tone {
        return t;
    }
    match ev.kind.as_str() {
        "answer" | "ok" | "approval_resolved" => Tone::Ok,
        "approval_request" | "warn" => Tone::Warn,
        "error" | "stop" => Tone::Danger,
        "start" | "tool" | "delegate" | "llm" => Tone::Info,
        _ => Tone::Neutral,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::draw::DrawList;
    use crate::UiEngine;

    #[test]
    fn auto_composer_builds_light_scene_and_handles_hitl_lifecycle() {
        let mut ac = AutoComposer::new("Zero-Boilerplate Test", ThemeMode::Light);
        let scene = ac.build_scene();
        assert_eq!(scene.theme, ThemeMode::Light);

        let mut engine = UiEngine::new(scene, [1200.0, 800.0]);
        let patches = ac.ingest_event(AutoAgentEvent {
            run_id: "r1".into(),
            agent_name: "coordinator".into(),
            chain: vec!["coordinator".into(), "worker".into()],
            kind: "approval_request".into(),
            tool: Some("fs_write".into()),
            tokens: Some(420),
            elapsed_ms: Some(12),
            approval: Some(PendingApproval {
                id: "ap-1".into(),
                tool: "fs_write".into(),
                args_preview: "path=notes.txt".into(),
                mode: "interactive".into(),
            }),
            ..AutoAgentEvent::default()
        });
        assert!(!patches.is_empty());
        for p in patches {
            assert!(engine.apply_patch(p), "patch must apply cleanly");
        }

        // Verify HITL bar became visible and trigger stats apply cleanly.
        let hitl = engine.scene.root.find_mut("auto_hitl_bar").expect("hitl bar exists");
        assert!(hitl.visible);

        let t_patches = ac.update_triggers(&[AutoTriggerStat {
            id: "poll_1".into(),
            kind: "every".into(),
            running: true,
            ticks: 10,
            gate_hits: 3,
            gate_skips: 7,
            ai_calls: 3,
            errors: 0,
            last_duration_ms: 8,
        }]);
        for p in t_patches {
            assert!(engine.apply_patch(p));
        }

        // Resolve HITL approval -> bar hides cleanly.
        for p in ac.clear_approval("ap-1") {
            assert!(engine.apply_patch(p));
        }
        let hitl_after = engine.scene.root.find_mut("auto_hitl_bar").expect("hitl bar exists");
        assert!(!hitl_after.visible);

        // Compile frame at both wide desktop and narrow mobile sizes without panics or underflow.
        let mut dl = DrawList::default();
        let _ = engine.tick_and_compile(1.0 / 60.0, &mut dl);
        assert!(!dl.quads.is_empty());

        engine.handle_input(crate::hit_test::InputEvent::Resize {
            width: 390.0,
            height: 844.0,
            scale_factor: 2.0,
        });
        let mut dl2 = DrawList::default();
        let _ = engine.tick_and_compile(1.0 / 60.0, &mut dl2);
        assert!(!dl2.quads.is_empty());
    }
}
