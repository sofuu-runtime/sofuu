// crates/sofuu-core/src/frontend/mod.rs
//
// AI-Native Frontend Engine & Micro-Metaframework.
// Pure Rust JSX/TSX transpiler, embedded micro-runtime, and native unbundled dev server.

pub mod dev_server;
pub mod runtime;
pub mod transpile;

pub use dev_server::{DevServerConfig, start_dev_server};
pub use transpile::transpile;
