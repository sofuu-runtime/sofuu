//! Five per-platform frontend profiles (Android, iOS, macOS, Windows, Linux).
//!
//! A profile is *data*, not a widget library: device geometry, safe areas,
//! chrome style, keyboard heights and audit rules. The shared emulator shell
//! (`host.rs`) renders every profile from the same IR primitives, so adding a
//! sixth platform is a table entry rather than a second emulator.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Orientation {
    Portrait,
    Landscape,
}

impl Orientation {
    pub fn as_str(self) -> &'static str {
        match self {
            Orientation::Portrait => "portrait",
            Orientation::Landscape => "landscape",
        }
    }
    pub fn toggle(self) -> Self {
        match self {
            Orientation::Portrait => Orientation::Landscape,
            Orientation::Landscape => Orientation::Portrait,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChromeStyle {
    Android,
    Ios,
    Macos,
    Windows,
    Linux,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NavBar {
    None,
    Gesture,
    ThreeButton,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PunchHole {
    /// Diameter in logical px.
    pub diameter: f32,
    /// Distance from the top of the screen to the hole centre.
    pub top: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DevicePreset {
    pub id: String,
    pub name: String,
    /// Logical screen width in portrait.
    pub width: f32,
    /// Logical screen height in portrait.
    pub height: f32,
    pub scale: f32,
    pub safe_top: f32,
    pub safe_bottom: f32,
    pub safe_left: f32,
    pub safe_right: f32,
    pub corner_radius: f32,
    /// Height of the painted status bar (0 for desktops).
    pub status_bar_height: f32,
    /// Height of a desktop titlebar (0 for phones).
    pub titlebar_height: f32,
    pub home_bar_height: f32,
    pub keyboard_portrait: f32,
    pub keyboard_landscape: f32,
    pub punch_hole: Option<PunchHole>,
    pub nav_bar: NavBar,
    /// Per-device override (e.g. the Raspberry Pi touch panel).
    #[serde(default)]
    pub touch: Option<bool>,
    #[serde(default)]
    pub min_touch: Option<f32>,
}

impl DevicePreset {
    pub fn is_touch(&self, profile: &PlatformProfile) -> bool {
        self.touch.unwrap_or(profile.touch)
    }

    pub fn min_touch(&self, profile: &PlatformProfile) -> f32 {
        self.min_touch.unwrap_or(profile.min_touch)
    }

    pub fn geometry(&self, orientation: Orientation) -> DeviceGeometry {
        let (width, height) = match orientation {
            Orientation::Portrait => (self.width, self.height),
            Orientation::Landscape => (self.height, self.width),
        };
        // In landscape the notch / island moves to a side edge. Keep the
        // nominal top/bottom clearance on the long edge so content still
        // clears the status area, and swap the physical safe sides.
        let (safe_top, safe_bottom, safe_left, safe_right) = match orientation {
            Orientation::Portrait => (
                self.safe_top,
                self.safe_bottom,
                self.safe_left,
                self.safe_right,
            ),
            Orientation::Landscape => (
                0.0,
                self.safe_bottom,
                self.safe_top.max(self.safe_left),
                self.safe_top.max(self.safe_right),
            ),
        };
        let keyboard_height = match orientation {
            Orientation::Portrait => self.keyboard_portrait,
            Orientation::Landscape => self.keyboard_landscape,
        };
        DeviceGeometry {
            width,
            height,
            scale: self.scale,
            safe_top,
            safe_bottom,
            safe_left,
            safe_right,
            corner_radius: self.corner_radius,
            status_bar_height: if self.status_bar_height > 0.0 {
                self.status_bar_height.max(safe_top)
            } else {
                0.0
            },
            titlebar_height: self.titlebar_height,
            home_bar_height: self.home_bar_height,
            keyboard_height,
            punch_hole: self.punch_hole.clone(),
            nav_bar: self.nav_bar,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DeviceGeometry {
    pub width: f32,
    pub height: f32,
    pub scale: f32,
    pub safe_top: f32,
    pub safe_bottom: f32,
    pub safe_left: f32,
    pub safe_right: f32,
    pub corner_radius: f32,
    pub status_bar_height: f32,
    pub titlebar_height: f32,
    pub home_bar_height: f32,
    pub keyboard_height: f32,
    pub punch_hole: Option<PunchHole>,
    pub nav_bar: NavBar,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlatformProfile {
    pub id: String,
    pub name: String,
    pub chrome: ChromeStyle,
    pub touch: bool,
    /// 0 on desktop profiles; 44 iOS / 48 Android & Linux kiosk.
    pub min_touch: f32,
    pub hover: bool,
    pub devices: Vec<DevicePreset>,
}

impl PlatformProfile {
    pub fn default_device(&self) -> &DevicePreset {
        &self.devices[0]
    }

    pub fn device(&self, id: &str) -> Option<&DevicePreset> {
        self.devices
            .iter()
            .find(|d| d.id == id)
            .or_else(|| self.devices.iter().find(|d| d.name == id))
    }
}

fn dev(
    id: &str,
    name: &str,
    width: f32,
    height: f32,
    scale: f32,
    safe_top: f32,
    safe_bottom: f32,
    corner_radius: f32,
    status_bar_height: f32,
    home_bar_height: f32,
    keyboard_portrait: f32,
    keyboard_landscape: f32,
    punch_hole: Option<PunchHole>,
    nav_bar: NavBar,
) -> DevicePreset {
    DevicePreset {
        id: id.to_string(),
        name: name.to_string(),
        width,
        height,
        scale,
        safe_top,
        safe_bottom,
        safe_left: 0.0,
        safe_right: 0.0,
        corner_radius,
        status_bar_height,
        titlebar_height: 0.0,
        home_bar_height,
        keyboard_portrait,
        keyboard_landscape,
        punch_hole,
        nav_bar,
        touch: None,
        min_touch: None,
    }
}

fn desktop(
    id: &str,
    name: &str,
    width: f32,
    height: f32,
    scale: f32,
    titlebar_height: f32,
) -> DevicePreset {
    DevicePreset {
        id: id.to_string(),
        name: name.to_string(),
        width,
        height,
        scale,
        safe_top: titlebar_height,
        safe_bottom: 0.0,
        safe_left: 0.0,
        safe_right: 0.0,
        corner_radius: 10.0,
        status_bar_height: 0.0,
        titlebar_height,
        home_bar_height: 0.0,
        keyboard_portrait: 0.0,
        keyboard_landscape: 0.0,
        punch_hole: None,
        nav_bar: NavBar::None,
        touch: None,
        min_touch: None,
    }
}

/// All five profiles in the canonical shortcut order (`⌘1..⌘5`).
pub fn profiles() -> Vec<PlatformProfile> {
    vec![
        PlatformProfile {
            id: "ios".to_string(),
            name: "iOS".to_string(),
            chrome: ChromeStyle::Ios,
            touch: true,
            min_touch: 44.0,
            hover: false,
            devices: vec![
                dev(
                    "iphone-16-pro",
                    "iPhone 16 Pro",
                    393.0,
                    852.0,
                    3.0,
                    59.0,
                    34.0,
                    47.0,
                    59.0,
                    34.0,
                    336.0,
                    210.0,
                    Some(PunchHole {
                        diameter: 126.0,
                        top: 30.0,
                    }),
                    NavBar::Gesture,
                ),
                dev(
                    "iphone-se",
                    "iPhone SE",
                    375.0,
                    667.0,
                    2.0,
                    20.0,
                    0.0,
                    0.0,
                    20.0,
                    0.0,
                    260.0,
                    180.0,
                    None,
                    NavBar::None,
                ),
                dev(
                    "iphone-16-pro-max",
                    "iPhone 16 Pro Max",
                    440.0,
                    956.0,
                    3.0,
                    62.0,
                    34.0,
                    55.0,
                    62.0,
                    34.0,
                    346.0,
                    220.0,
                    Some(PunchHole {
                        diameter: 140.0,
                        top: 32.0,
                    }),
                    NavBar::Gesture,
                ),
                dev(
                    "ipad-air-11",
                    "iPad Air 11\"",
                    820.0,
                    1180.0,
                    2.0,
                    24.0,
                    20.0,
                    18.0,
                    24.0,
                    20.0,
                    300.0,
                    260.0,
                    None,
                    NavBar::Gesture,
                ),
            ],
        },
        PlatformProfile {
            id: "android".to_string(),
            name: "Android".to_string(),
            chrome: ChromeStyle::Android,
            touch: true,
            min_touch: 48.0,
            hover: false,
            devices: vec![
                dev(
                    "pixel-9",
                    "Pixel 9",
                    412.0,
                    915.0,
                    2.625,
                    32.0,
                    24.0,
                    28.0,
                    32.0,
                    24.0,
                    300.0,
                    200.0,
                    Some(PunchHole {
                        diameter: 14.0,
                        top: 16.0,
                    }),
                    NavBar::Gesture,
                ),
                /* v1-parity preset: 1080x2400 at 2.625x is the exact
                 * sofuu-android capture size (411.43 x 914.29 dp). No punch
                 * hole so the status bar is a clean cream band, and 24/24
                 * safe insets match v1's status/navigation bars. */
                dev(
                    "sofuu-phone-411",
                    "Sofuu Phone 411",
                    411.43,
                    914.29,
                    2.625,
                    24.0,
                    24.0,
                    0.0,
                    24.0,
                    24.0,
                    300.0,
                    200.0,
                    None,
                    NavBar::Gesture,
                ),
                dev(
                    "pixel-9-pro-fold",
                    "Pixel 9 Pro Fold (Unfolded)",
                    841.0,
                    701.0,
                    2.0,
                    32.0,
                    24.0,
                    16.0,
                    32.0,
                    24.0,
                    300.0,
                    220.0,
                    Some(PunchHole {
                        diameter: 12.0,
                        top: 16.0,
                    }),
                    NavBar::Gesture,
                ),
                dev(
                    "budget-small",
                    "Budget Small",
                    360.0,
                    640.0,
                    2.0,
                    24.0,
                    48.0,
                    0.0,
                    24.0,
                    48.0,
                    280.0,
                    180.0,
                    None,
                    NavBar::ThreeButton,
                ),
                dev(
                    "android-tablet",
                    "Android Tablet",
                    1280.0,
                    800.0,
                    2.0,
                    32.0,
                    24.0,
                    20.0,
                    32.0,
                    24.0,
                    320.0,
                    240.0,
                    None,
                    NavBar::Gesture,
                ),
            ],
        },
        PlatformProfile {
            id: "macos".to_string(),
            name: "macOS".to_string(),
            chrome: ChromeStyle::Macos,
            touch: false,
            min_touch: 0.0,
            hover: true,
            devices: vec![
                desktop("macbook-air-13", "MacBook Air 13\"", 1280.0, 800.0, 2.0, 28.0),
                desktop("desktop-compact", "Desktop Compact", 900.0, 600.0, 2.0, 28.0),
                desktop("narrow-split-pane", "Narrow Split Pane", 480.0, 720.0, 2.0, 28.0),
            ],
        },
        PlatformProfile {
            id: "windows".to_string(),
            name: "Windows".to_string(),
            chrome: ChromeStyle::Windows,
            touch: false,
            min_touch: 0.0,
            hover: true,
            devices: vec![
                // Logical sizes are physical / scale, exactly what the app sees.
                desktop("laptop-1080p-125", "Laptop 1080p @125%", 1536.0, 864.0, 1.25, 32.0),
                desktop("classic-1280x720", "Classic 1280x720 @100%", 1280.0, 720.0, 1.0, 32.0),
                desktop("uhd-4k-200", "4K HiDPI @200%", 1920.0, 1080.0, 2.0, 32.0),
                desktop("snap-half-screen", "Snap Half-Screen", 720.0, 864.0, 1.0, 32.0),
            ],
        },
        PlatformProfile {
            id: "linux".to_string(),
            name: "Linux".to_string(),
            chrome: ChromeStyle::Linux,
            touch: false,
            min_touch: 0.0,
            hover: true,
            devices: vec![
                desktop("wayland-125", "Wayland 1.25x Fractional", 1280.0, 800.0, 1.25, 28.0),
                desktop("tiling-half", "Tiling WM Half", 640.0, 1040.0, 1.0, 28.0),
                DevicePreset {
                    touch: Some(true),
                    min_touch: Some(48.0),
                    ..desktop("pi-touch", "Raspberry Pi 5 Touch Display", 800.0, 480.0, 1.0, 0.0)
                },
            ],
        },
    ]
}

pub fn profile(id: &str) -> Option<PlatformProfile> {
    let id = id.to_ascii_lowercase().replace('-', "");
    profiles()
        .into_iter()
        .find(|p| p.id.replace('-', "") == id)
}

/// Resolve `(platform, device?)` with a clear error for typos.
pub fn resolve(platform: &str, device: Option<&str>) -> Result<(PlatformProfile, DevicePreset), String> {
    let p = profile(platform)
        .ok_or_else(|| format!("unknown platform `{platform}` (ios | android | macos | windows | linux)"))?;
    let d = match device {
        Some(id) => p
            .device(id)
            .cloned()
            .ok_or_else(|| format!("unknown device `{id}` for {}", p.name))?,
        None => p.default_device().clone(),
    };
    Ok((p, d))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_plan_preset_exists() {
        let expected = [
            ("ios", "iphone-16-pro"),
            ("ios", "iphone-se"),
            ("ios", "iphone-16-pro-max"),
            ("ios", "ipad-air-11"),
            ("android", "pixel-9"),
            ("android", "sofuu-phone-411"),
            ("android", "pixel-9-pro-fold"),
            ("android", "budget-small"),
            ("android", "android-tablet"),
            ("macos", "macbook-air-13"),
            ("macos", "desktop-compact"),
            ("macos", "narrow-split-pane"),
            ("windows", "laptop-1080p-125"),
            ("windows", "classic-1280x720"),
            ("windows", "uhd-4k-200"),
            ("windows", "snap-half-screen"),
            ("linux", "wayland-125"),
            ("linux", "tiling-half"),
            ("linux", "pi-touch"),
        ];
        for (p, d) in expected {
            resolve(p, Some(d)).unwrap_or_else(|e| panic!("{p}/{d}: {e}"));
        }
    }

    #[test]
    fn landscape_swaps_dimensions_and_side_insets() {
        let (_, iphone) = resolve("ios", Some("iphone-16-pro")).unwrap();
        let g = iphone.geometry(Orientation::Landscape);
        assert_eq!(g.width, 852.0);
        assert_eq!(g.height, 393.0);
        assert_eq!(g.safe_left, 59.0);
        assert_eq!(g.safe_right, 59.0);
        assert_eq!(g.keyboard_height, 210.0);
    }

    #[test]
    fn v1_parity_phone_is_exact() {
        let (_, phone) = resolve("android", Some("sofuu-phone-411")).unwrap();
        let g = phone.geometry(Orientation::Portrait);
        assert_eq!(g.width, 411.43);
        assert_eq!(g.height, 914.29);
        assert_eq!(g.scale, 2.625);
        assert_eq!(g.safe_top, 24.0);
        assert_eq!(g.safe_bottom, 24.0);
        assert_eq!(g.status_bar_height, 24.0);
        assert_eq!(g.home_bar_height, 24.0);
        assert_eq!((g.width * g.scale).round(), 1080.0);
        assert_eq!((g.height * g.scale).round(), 2400.0);
    }

    #[test]
    fn touch_rules_match_platform_guidance() {
        assert_eq!(resolve("ios", None).unwrap().0.min_touch, 44.0);
        assert_eq!(resolve("android", None).unwrap().0.min_touch, 48.0);
        assert_eq!(resolve("macos", None).unwrap().0.min_touch, 0.0);
    }
}
