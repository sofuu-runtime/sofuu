//! Sofuu pure-Rust custom 2D GPU UI compositor core (`sofuu-ui`).
//!
//! Zero browser, zero WebView, zero `wgpu`, zero OS window dependencies, zero CPU rasterization.
//! Compiles domain-agnostic Scene IR + 120Hz springs + SDF typography into flat `#[repr(C)]`
//! GPU instance buffers consumed directly by Metal, OpenGL/GLES, and Vulkan backends in `sofuu-gpu`.

pub mod compiler;
pub mod components;
pub mod draw;
pub mod hit_test;
pub mod ir;
pub mod layout;
pub mod spring;
pub mod text;
pub mod theme;
pub mod ui;

pub use compiler::{CompileStatus, UiEngine};
pub use components::plot::{
    CandlePoint, PlotData, PlotPane, PlotPoint, PlotSeries, PlotSeriesKind, PlotThreshold,
    ReadoutMode,
};
pub use components::stat::{StatData, StatVariant};
pub use draw::{
    hex_rgba, lerp_rgba8, rgba8, scale_alpha, unpack_rgba8, with_alpha, ClipRect, DrawCmd,
    DrawList, FrameUniforms, GlyphInstance, QuadInstance, RibbonInstance, StrokeInstance,
};
pub use hit_test::{InputEvent, UIEvent};
pub use ir::{Align, Axis, BoxStyle, Insets, Node, NodeKind, Scene, ScenePatch, SizeSpec, TabItem};
pub use layout::{ComputedRect, LayoutMap};
pub use spring::{SpringBank, SpringConfig, SpringState};
pub use text::{GlyphAtlas, GlyphMetrics};
pub use theme::{Theme, ThemeMode, Tone};
