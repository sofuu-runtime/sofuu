//! Subsystem domain capsules for Sofuu.
//!
//! Each domain capsule encapsulates its own AST, grammar, hardware protocols, or state
//! machines with strict visibility boundaries to prevent monolithic coupling.

pub mod infer;
pub mod kaze;
pub mod memory;
pub mod robotics;
pub mod subsystem;

pub use subsystem::{init_subsystems, Subsystem};
