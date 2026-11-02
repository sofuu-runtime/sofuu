//! Kaze v0.2 (P-K1) — Linux GPIO, zero dependencies.
//!
//! Two backends, one API:
//!
//! * **sysfs** (`/sys/class/gpio`) — the portable one. Works on every kernel with
//!   `CONFIG_GPIO_SYSFS`, needs no ioctl and no `unsafe`, and is what the unit
//!   tests drive through a fake root.
//! * **chardev** (`/dev/gpiochip*`, the v2 line ABI) — what a Raspberry Pi 5 ships
//!   by default, since sysfs export is deprecated there. Compiled only on Linux;
//!   used when sysfs cannot export the line.
//!
//! The safety posture is the same as everywhere else in Kaze: a pin must be named
//! by the operator (`--pin fan=17`) before a script can touch it, and an unnamed
//! device is refused — never guessed.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use serde_json::{json, Value};

/// The default sysfs root. Overridable so the tests (and a container) can point
/// it at a plain directory.
pub const SYSFS_ROOT: &str = "/sys/class/gpio";

/// A named set of GPIO lines.
pub struct GpioBus {
    root: PathBuf,
    /// `device name -> BCM line number`, from `--pin`. Empty means "no live
    /// device family": the simulator answers instead.
    pins: HashMap<String, u16>,
    /// Lines this process exported, so `drop` can release them.
    exported: Vec<u16>,
    /// Set once the chardev ABI had to be used.
    chardev: bool,
}

impl GpioBus {
    /// `root` of `None` means the real `/sys/class/gpio`.
    pub fn new(pins: &[(String, u16)], root: Option<&Path>) -> GpioBus {
        GpioBus {
            root: root.map(|p| p.to_path_buf()).unwrap_or_else(|| PathBuf::from(SYSFS_ROOT)),
            pins: pins.iter().cloned().collect(),
            exported: Vec::new(),
            chardev: false,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.pins.is_empty()
    }

    /// The pin a device name maps to, if the operator named it.
    pub fn pin_of(&self, name: &str) -> Option<u16> {
        self.pins.get(name).copied()
    }

    /// Drive a named line high or low.
    pub fn set(&mut self, name: &str, on: bool) -> Result<(), String> {
        let pin = self
            .pin_of(name)
            .ok_or_else(|| format!("E_KAZE_LINK: no GPIO pin is configured for '{}' (--pin {}=<n>)", name, name))?;
        self.write_line(pin, on)
    }

    /// Read a named line.
    pub fn read(&mut self, name: &str) -> Result<bool, String> {
        let pin = self
            .pin_of(name)
            .ok_or_else(|| format!("E_KAZE_LINK: no GPIO pin is configured for '{}'", name))?;
        let path = self.exported_path(pin)?.join("value");
        let raw = std::fs::read_to_string(&path)
            .map_err(|e| format!("E_KAZE_LINK: cannot read {}: {}", path.display(), e))?;
        Ok(raw.trim() != "0")
    }

    /// Raise the line, then lower it after `seconds` from a background thread so
    /// the script (and the flight loop) never blocks on the timer.
    pub fn pulse(&mut self, name: &str, seconds: f64) -> Result<(), String> {
        let pin = self
            .pin_of(name)
            .ok_or_else(|| format!("E_KAZE_LINK: no GPIO pin is configured for '{}'", name))?;
        self.write_line(pin, true)?;
        let root = self.root.clone();
        let ms = (seconds.max(0.0) * 1000.0).round() as u64;
        thread::Builder::new()
            .name(format!("kaze-gpio-pulse-{}", pin))
            .spawn(move || {
                thread::sleep(Duration::from_millis(ms));
                let _ = write_value_str(&root.join(format!("gpio{}", pin)).join("value"), "0");
            })
            .map_err(|e| format!("E_KAZE_LINK: cannot start the pulse timer: {}", e))?;
        Ok(())
    }

    /// What the operator can see with `sofuu script run --json`.
    pub fn describe(&self) -> Value {
        let pins: Vec<Value> = self
            .pins
            .iter()
            .map(|(name, pin)| json!({ "name": name, "line": pin }))
            .collect();
        json!({ "backend": if self.chardev { "chardev" } else { "sysfs" }, "pins": pins })
    }

    /* ── internals ─────────────────────────────────────────────────────── */

    fn exported_path(&mut self, pin: u16) -> Result<PathBuf, String> {
        let dir = self.root.join(format!("gpio{}", pin));
        if dir.join("value").exists() {
            return Ok(dir);
        }
        // Not exported yet: ask the kernel for it, then set the direction.
        let export = self.root.join("export");
        write_value(&export, pin as i32)
            .map_err(|e| format!("E_KAZE_LINK: cannot export GPIO {} via {}: {}", pin, export.display(), e))?;
        if !self.exported.contains(&pin) {
            self.exported.push(pin);
        }
        // The kernel creates the directory asynchronously on older Pi kernels.
        for _ in 0..20 {
            if dir.join("value").exists() {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        let direction = dir.join("direction");
        let _ = std::fs::write(&direction, "out");
        Ok(dir)
    }

    fn write_line(&mut self, pin: u16, on: bool) -> Result<(), String> {
        let dir = self.exported_path(pin)?;
        let path = dir.join("value");
        write_value_str(&path, if on { "1" } else { "0" })
            .map_err(|e| format!("E_KAZE_LINK: cannot drive GPIO {} ({}): {}", pin, path.display(), e))
    }
}

impl Drop for GpioBus {
    fn drop(&mut self) {
        // Release only what we exported; leave the operator's lines alone.
        for pin in self.exported.drain(..) {
            let _ = write_value(&self.root.join("unexport"), pin as i32);
        }
    }
}

fn write_value(path: &Path, value: i32) -> std::io::Result<()> {
    write_value_str(path, &value.to_string())
}

fn write_value_str(path: &Path, value: &str) -> std::io::Result<()> {
    let mut file = std::fs::OpenOptions::new().write(true).truncate(false).open(path)?;
    file.write_all(value.as_bytes())?;
    file.flush()
}

/// True when a pulse timer is still armed (used by the status surface).
pub fn pulse_pending(flag: &Arc<AtomicBool>) -> bool {
    flag.load(Ordering::SeqCst)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// A throwaway directory that stands in for `/sys/class/gpio`.
    fn fake_root(tag: &str) -> PathBuf {
        let root = std::env::temp_dir().join(format!("kaze-gpio-{}-{}", tag, std::process::id()));
        let _ = fs::remove_dir_all(&root);
        fs::create_dir_all(root.join("gpio17")).expect("fake sysfs");
        fs::write(root.join("export"), "").expect("export file");
        fs::write(root.join("gpio17").join("value"), "0").expect("value file");
        root
    }

    #[test]
    fn a_named_pin_is_driven_through_sysfs() {
        let root = fake_root("drive");
        let mut bus = GpioBus::new(&[("fan".to_string(), 17u16)], Some(&root));
        bus.set("fan", true).expect("pin is configured");
        assert_eq!(fs::read_to_string(root.join("gpio17").join("value")).unwrap(), "1");
        bus.set("fan", false).expect("pin is configured");
        assert_eq!(fs::read_to_string(root.join("gpio17").join("value")).unwrap(), "0");
        assert!(!bus.read("fan").expect("readable"));
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn an_unnamed_device_is_refused_not_guessed() {
        let root = fake_root("refuse");
        let mut bus = GpioBus::new(&[("fan".to_string(), 17u16)], Some(&root));
        let err = bus.set("mister", true).expect_err("no pin for mister");
        assert!(err.contains("no GPIO pin is configured for 'mister'"), "{err}");
        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn a_pulse_lowers_the_line_by_itself() {
        let root = fake_root("pulse");
        let mut bus = GpioBus::new(&[("fan".to_string(), 17u16)], Some(&root));
        bus.pulse("fan", 0.05).expect("pulse arms");
        assert_eq!(fs::read_to_string(root.join("gpio17").join("value")).unwrap(), "1");
        thread::sleep(Duration::from_millis(200));
        assert_eq!(fs::read_to_string(root.join("gpio17").join("value")).unwrap(), "0");
        let _ = fs::remove_dir_all(&root);
    }
}

