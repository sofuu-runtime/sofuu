//! Sofuu pure-Rust custom 2D GPU UI compositor core (`sofuu-ui`).
//!
//! Zero browser, zero WebView, zero `wgpu`, zero OS window dependencies, zero CPU rasterization.
//! Compiles domain-agnostic Scene IR + 120Hz springs + SDF typography into flat `#[repr(C)]`
//! GPU instance buffers consumed directly by Metal, OpenGL/GLES, and Vulkan backends in `sofuu-gpu`.

pub mod a11y;
pub mod audit;
pub mod auto_compose;
pub mod compiler;
pub mod components;
pub mod draw;
pub mod fx;
pub mod hit_test;
pub mod icons;
pub mod images;
pub mod ir;
pub mod layout;
pub mod shaders;
pub mod spring;
pub mod text;
pub mod theme;
pub mod ui;

pub use audit::{
    audit_scene, audit_scene_with_scrolls, AuditIssue, AuditNode, AuditPerf, AuditReport,
    AuditRules, AuditSummary, IssueKind, Severity,
};
pub use auto_compose::{AutoAgentEvent, AutoComposer, AutoTriggerStat, PendingApproval};

pub use compiler::{CompileStatus, UiEngine};
pub use components::code::{CodeData, CodeHighlight};
pub use components::feed::{FeedData, FeedItem};
pub use components::graph::{GraphData, GraphEdge, GraphNode};
pub use components::plot::{
    CandlePoint, PlotData, PlotPane, PlotPoint, PlotSeries, PlotSeriesKind, PlotThreshold,
    ReadoutMode,
};
pub use components::stat::{StatData, StatVariant};
pub use components::table::{
    TableAlign, TableCell, TableColumn, TableData, TableRow, TableSortDir,
};
pub use draw::{
    hex_rgba, lerp_rgba8, rgba8, scale_alpha, unpack_rgba8, with_alpha, BackdropRegion, ClipRect,
    DrawCmd, DrawList, EffectQuadInstance, FrameUniforms, GlyphInstance, ParticleInstance,
    QuadInstance, RibbonInstance, SpriteInstance, StrokeInstance,
};
pub use fx::{
    DrawMark, FxEngine, FxKind, FxSpec, FxSpring, NodeFx, EFFECT_AURORA, EFFECT_BORDER_BEAM,
    EFFECT_CURSOR_GLOW, EFFECT_PULSE_RING, EFFECT_SHIMMER, EFFECT_TOUCH_RIPPLE, EFFECT_VOICE_ORB,
    PARTICLE_CIRCLE, PARTICLE_RIBBON, PARTICLE_SPARK,
};
pub use hit_test::{InputEvent, UIEvent};
pub use icons::{IconAtlas, IconEntry};
pub use images::{
    decode, decode_async, detect_format, fit_image, DecodedImage, FittedImage, ImageAtlas,
    ImageError, ImageFit, ImageFormat, ImageHandle, ImagePage,
};
pub use ir::{Align, Axis, BoxStyle, Insets, Node, NodeKind, Scene, ScenePatch, SizeSpec, TabItem};
pub use layout::{ComputedRect, LayoutMap};
pub use spring::{SpringBank, SpringConfig, SpringState};
pub use text::{GlyphAtlas, GlyphMetrics};
pub use theme::{Theme, ThemeMode, Tone};
