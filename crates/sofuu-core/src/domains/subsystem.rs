//! Subsystem registration and lifecycle contract.
//!
//! Provides a decoupled trait so that domain capsules (Kaze, Robotics, Memory, etc.)
//! can register their JavaScript globals, drivers, and health checks without
//! coupling their internal types into `rt/engine.rs`.

use sofuu_ffi::qjs::JSContext;

pub trait Subsystem: Send + Sync {
    /// Human-readable identifier for logging and doctor checks
    fn name(&self) -> &'static str;

    /// Register JavaScript globals, classes, or modules into the QuickJS context
    fn register_js(&self, ctx: *mut JSContext) -> Result<(), &'static str> {
        let _ = ctx;
        Ok(())
    }

    /// Health diagnostic check invoked by `sofuu doctor`
    fn health_check(&self) -> Result<(), &'static str> {
        Ok(())
    }
}

/// Declarative registration of all domain subsystems into QuickJS.
/// Replaces monolithic scattered registration calls in `rt/engine.rs`.
pub fn init_subsystems(ctx: *mut JSContext) -> Result<(), &'static str> {
    let subsystems: &[&dyn Subsystem] = &[
        &crate::domains::robotics::RoboticsSubsystem,
        &crate::domains::memory::MemorySubsystem,
        &crate::domains::kaze::KazeSubsystem,
        &crate::domains::infer::InferSubsystem,
    ];

    for sub in subsystems {
        sub.register_js(ctx)?;
    }
    Ok(())
}
