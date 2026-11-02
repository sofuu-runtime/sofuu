//! Robotics, Autonomous Drones & Ambient Hardware Subsystem
//!
//! Encapsulates MAVLink v2 streaming, R7 hardware safety veto gates,
//! Linux GPIO/Raspberry Pi 5 line ABI, and drone agent operations.

pub mod mavlink;
pub mod mavlink_gate;
pub mod mavlink_link;
pub mod mavlink_js;
pub mod gpio;
pub mod drone_ops;
pub mod hardware;
pub mod kaze_live;

use sofuu_ffi::qjs::JSContext;
use crate::domains::Subsystem;

pub struct RoboticsSubsystem;

impl Subsystem for RoboticsSubsystem {
    fn name(&self) -> &'static str {
        "robotics"
    }

    fn register_js(&self, ctx: *mut JSContext) -> Result<(), &'static str> {
        if mavlink_js::hardware_enabled() {
            unsafe {
                mavlink_js::mod_mavlink_register(ctx);
                let (mavlink_src, drone_src) = mavlink_js::driver_js();
                if let Some(src) = mavlink_src {
                    crate::shipped::eval_shipped_with_bytecode(
                        ctx as *mut std::os::raw::c_void,
                        &src,
                        None,
                        "<mavlink-driver>",
                    );
                }
                if let Some(src) = drone_src {
                    crate::shipped::eval_shipped_with_bytecode(
                        ctx as *mut std::os::raw::c_void,
                        &src,
                        None,
                        "<drone-driver>",
                    );
                }
            }
        }
        Ok(())
    }
}
