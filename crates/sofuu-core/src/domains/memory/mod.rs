// sofuu-core — memory subsystem (CMA cognitive memory).
pub mod cma;
pub mod hnsw;
/// B5.4: the recall evaluation harness (tests only — it exists to measure
/// ranking changes, not to ship).
#[cfg(test)]
pub mod recall_eval;
/// Brain Core II M0: the model-based invariant fuzz suite (the precision
/// floor every brain change is gated on).
#[cfg(test)]
pub mod invariant_test;
/// Brain Core II M2: the decay survival harness (weekly-use twins,
/// reinforcement growth caps, the DeathLedger).
#[cfg(test)]
pub mod decay_eval;

use sofuu_ffi::qjs::JSContext;
use crate::domains::Subsystem;

pub struct MemorySubsystem;

impl Subsystem for MemorySubsystem {
    fn name(&self) -> &'static str {
        "memory"
    }

    fn register_js(&self, ctx: *mut JSContext) -> Result<(), &'static str> {
        #[cfg(has_qtsq)]
        unsafe {
            crate::rt::memory::mod_memory_register(ctx);
            crate::rt::memory::mod_kv_register(ctx);
            crate::rt::memory::mod_agent_register(ctx);
            crate::rt::storage::mod_storage_register(ctx);
        }
        #[cfg(not(has_qtsq))]
        unsafe {
            crate::capabilities::mod_memory_unavailable_register(ctx);
        }
        Ok(())
    }
}
