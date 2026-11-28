//! Shared emulator chrome around an isolated app viewport.
//!
//! The app scene is nested inside `emu_app_viewport`, so the app's own layout
//! runs at the platform-exact logical size and receives shell events without
//! knowing it is being emulated.
//!
//! * [`EmulatorHost::wrap_device_scene`] — device only (status bar / island /
//!   titlebar / home bar / optional IME). This is what screenshots render.
//! * [`EmulatorHost::wrap_shell_scene`] — legacy in-scene toolbar + inspector.
//!   The screen-first window model (E6) no longer calls it; controls live in
//!   the native attached strip and the app scene is the window root.

use serde::{Deserialize, Serialize};

use sofuu_ui::audit::AuditRules;
use sofuu_ui::draw::{hex_rgba, rgba8};
use sofuu_ui::ir::{Align, Axis, BoxStyle, Insets, Node, NodeKind, Scene, SizeSpec};
use sofuu_ui::theme::{Theme, ThemeMode, Tone};
use sofuu_ui::UIEvent;

use crate::profiles::{
    profiles, resolve, ChromeStyle, DeviceGeometry, DevicePreset, NavBar, Orientation,
    PlatformProfile,
};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmulatorConfig {
    #[serde(default = "default_platform")]
    pub platform: String,
    #[serde(default)]
    pub device: Option<String>,
    #[serde(default)]
    pub theme: ThemeMode,
    #[serde(default = "default_orientation")]
    pub orientation: Orientation,
    #[serde(default)]
    pub keyboard_open: bool,
    #[serde(default)]
    pub scenario: Option<String>,
    /// Scenario names shown in the inspector. Apps can pass their
    /// `preview.scenarios` keys here.
    #[serde(default)]
    pub scenarios: Vec<String>,
    /// Legacy in-scene inspector toggle (see [`EmulatorHost::wrap_shell_scene`]).
    #[serde(default = "default_true")]
    pub inspector: bool,
    #[serde(default = "default_slowmo")]
    pub slowmo: f32,
    #[serde(default)]
    pub mock: bool,
    /// Optional cap on the screenshot scale (e.g. `1.0` for fast CI).
    #[serde(default)]
    pub scale_limit: Option<f32>,
    #[serde(default)]
    pub title: Option<String>,
}

fn default_platform() -> String {
    "ios".to_string()
}
fn default_orientation() -> Orientation {
    Orientation::Portrait
}
fn default_true() -> bool {
    true
}
fn default_slowmo() -> f32 {
    1.0
}

impl Default for EmulatorConfig {
    fn default() -> Self {
        Self {
            platform: default_platform(),
            device: None,
            theme: ThemeMode::Light,
            orientation: default_orientation(),
            keyboard_open: false,
            scenario: None,
            scenarios: Vec::new(),
            inspector: true,
            slowmo: 1.0,
            mock: false,
            scale_limit: None,
            title: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShellAction {
    /// Host state changed; the caller should recompose and remount.
    Changed,
    /// Capture a screenshot (toolbar / ⌘S).
    Screenshot,
    /// Platform back gesture (Android hardware back / iOS edge swipe).
    Back,
    Close,
}

#[derive(Clone, Debug)]
pub struct EmulatorHost {
    pub config: EmulatorConfig,
    pub profile: PlatformProfile,
    pub device: DevicePreset,
    pub theme: ThemeMode,
    pub orientation: Orientation,
    pub keyboard_open: bool,
    pub inspector: bool,
    pub scenario: Option<String>,
}

impl EmulatorHost {
    pub fn new(config: EmulatorConfig) -> Result<Self, String> {
        let (profile, device) = resolve(&config.platform, config.device.as_deref())?;
        let theme = config.theme;
        let orientation = config.orientation;
        let keyboard_open = config.keyboard_open;
        let inspector = config.inspector;
        let scenario = config.scenario.clone();
        Ok(Self {
            config,
            profile,
            device,
            theme,
            orientation,
            keyboard_open,
            inspector,
            scenario,
        })
    }

    pub fn geometry(&self) -> DeviceGeometry {
        self.device.geometry(self.orientation)
    }

    pub fn is_touch(&self) -> bool {
        self.device.is_touch(&self.profile)
    }

    pub fn min_touch(&self) -> f32 {
        self.device.min_touch(&self.profile)
    }

    pub fn chrome(&self) -> Theme {
        Theme::for_mode(self.theme)
    }

    /// App viewport size. When `keyboard_in_flow` is true the IME steals
    /// layout height (Android `adjustResize`); screenshots keep the full
    /// height and composite the keyboard as an overlay instead.
    pub fn app_size(&self, keyboard_in_flow: bool) -> [f32; 2] {
        let g = self.geometry();
        let top = if g.status_bar_height > 0.0 {
            g.status_bar_height
        } else {
            g.titlebar_height
        };
        let kb = if keyboard_in_flow && self.keyboard_open && g.keyboard_height > 0.0 {
            g.keyboard_height
        } else {
            0.0
        };
        [
            (g.width - g.safe_left - g.safe_right).max(1.0),
            (g.height - top - g.home_bar_height - kb).max(1.0),
        ]
    }

    pub fn audit_rules(&self, keyboard_open: bool) -> AuditRules {
        let g = self.geometry();
        let size = self.app_size(false);
        AuditRules {
            platform: self.profile.id.clone(),
            viewport: size,
            touch: self.is_touch(),
            min_touch: self.min_touch(),
            safe: Insets {
                // The status bar / titlebar and home bar are siblings of the
                // app viewport, so the app coordinate space starts at (0, 0)
                // and already excludes hardware safe areas. Reporting those
                // bands again would flag every bottom-aligned control.
                top: 0.0,
                right: 0.0,
                bottom: 0.0,
                left: 0.0,
            },
            keyboard_open,
            keyboard_height: if keyboard_open && g.keyboard_height > 0.0 {
                g.keyboard_height
            } else {
                0.0
            },
            ..AuditRules::default()
        }
    }

    /// Logical position of the app viewport inside the device frame. Used to
    /// offset audit rectangles when compositing the annotated overlay.
    pub fn app_origin(&self) -> [f32; 2] {
        let g = self.geometry();
        let top = if g.status_bar_height > 0.0 {
            g.status_bar_height
        } else {
            g.titlebar_height
        };
        [g.safe_left, top]
    }

    /// Standalone IME node, exported so screenshots can composite it on top of
    /// a full-height app instead of shrinking layout.
    pub fn keyboard_node(&self) -> Node {
        self.keyboard()
    }

    /// Standalone home-indicator / navigation node for screenshot overlays.
    pub fn home_bar_node(&self) -> Node {
        self.home_bar()
    }

    /// Device-only scene used by screenshots.
    pub fn wrap_device_scene(&self, app: Scene, keyboard_in_flow: bool) -> Scene {
        let root = self.device_root(app.root, keyboard_in_flow);
        Scene::new(
            self.config
                .title
                .clone()
                .unwrap_or_else(|| format!("{} - {}", self.profile.name, self.device.name)),
            self.theme,
            root,
        )
    }

    /// Legacy in-scene shell: toolbar + centered device + inspector.
    ///
    /// Retained for API compatibility; the screen-first window (E6) renders
    /// [`Self::wrap_device_scene`] directly and puts controls in the native
    /// attached strip, so captures and the live window never include this.
    pub fn wrap_shell_scene(&self, app: Scene) -> Scene {
        let device = self.wrap_device_scene(app, true);
        let stage = framed(
            "emu_stage",
            device.root,
            self.geometry().width + 36.0,
            self.geometry().height + 36.0,
            self.stage_style(),
            18.0,
        );
        let stage_wrap = col(
            "emu_stage_wrap",
            Align::Center,
            0.0,
            Insets::all(12.0),
            None,
            vec![stage],
        )
        .with_size(SizeSpec::Fill(1.0), SizeSpec::Fill(1.0));

        let toolbar = self.toolbar();
        let left = col(
            "emu_left",
            Align::Stretch,
            0.0,
            Insets::ZERO,
            None,
            vec![toolbar, stage_wrap],
        )
        .with_size(SizeSpec::Fill(1.0), SizeSpec::Fill(1.0));

        let mut children = vec![left];
        if self.inspector {
            children.push(self.inspector());
        }
        let root = row(
            "emu_root",
            Align::Stretch,
            0.0,
            Insets::ZERO,
            None,
            children,
        )
        .with_size(SizeSpec::Fill(1.0), SizeSpec::Fill(1.0));
        Scene::new(
            self.config
                .title
                .clone()
                .unwrap_or_else(|| format!("Sofuu Emu - {}", self.device.name)),
            self.theme,
            root,
        )
    }

    fn device_root(&self, app_root: Node, keyboard_in_flow: bool) -> Node {
        let g = self.geometry();
        let mut children = Vec::new();
        if g.status_bar_height > 0.0 {
            children.push(self.status_bar());
        } else if g.titlebar_height > 0.0 {
            children.push(self.titlebar());
        }

        /* The screen behind the host chrome follows the app's root background
         * when it has one. A full-bleed app (v2's cream root) then gets a
         * matching status-bar band instead of the emulator theme's neutral
         * background; apps without a root fill keep the theme default. */
        let screen_bg = node_bg(&app_root).unwrap_or_else(|| self.chrome().bg);
        let mut app = app_root;
        if app.width == SizeSpec::Auto {
            app.width = SizeSpec::Fill(1.0);
        }
        if app.height == SizeSpec::Auto {
            app.height = SizeSpec::Fill(1.0);
        }
        let viewport = col(
            "emu_app_viewport",
            Align::Stretch,
            0.0,
            Insets {
                top: 0.0,
                right: g.safe_right,
                bottom: 0.0,
                left: g.safe_left,
            },
            None,
            vec![app],
        )
        .with_size(SizeSpec::Fill(1.0), SizeSpec::Fill(1.0));

        children.push(viewport);
        if keyboard_in_flow && self.keyboard_open && g.keyboard_height > 0.0 {
            children.push(self.keyboard());
        }
        if g.home_bar_height > 0.0 {
            children.push(self.home_bar());
        }

        let mut screen = col(
            "emu_screen",
            Align::Stretch,
            0.0,
            Insets::ZERO,
            Some(self.screen_style(screen_bg)),
            children,
        )
        .with_size(SizeSpec::Px(g.width), SizeSpec::Px(g.height));
        screen.min_width = g.width;
        screen.min_height = g.height;
        screen
    }

    fn screen_style(&self, bg: u32) -> BoxStyle {
        let radius = self.geometry().corner_radius;
        BoxStyle {
            bg: Some(bg),
            bg_bottom: None,
            hover_bg: None,
            border: if radius > 0.0 {
                Some(if self.theme == ThemeMode::Dark {
                    rgba8(255, 255, 255, 26)
                } else {
                    rgba8(0, 0, 0, 22)
                })
            } else {
                None
            },
            border_width: if radius > 0.0 { 1.0 } else { 0.0 },
            radius: [radius; 4],
            ..BoxStyle::default()
        }
    }

    fn stage_style(&self) -> BoxStyle {
        // Physical bezel: always a dark shell regardless of UI theme.
        BoxStyle {
            bg: Some(hex_rgba(0x0A0D12, 255)),
            bg_bottom: Some(hex_rgba(0x161B23, 255)),
            hover_bg: None,
            border: Some(rgba8(255, 255, 255, 22)),
            border_width: 1.0,
            radius: [40.0; 4],
            shadow_color: Some(rgba8(0, 0, 0, 150)),
            shadow_blur: 22.0,
            shadow_offset_y: 8.0,
            opacity: 1.0,
        }
    }

    fn status_bar(&self) -> Node {
        let t = self.chrome();
        let g = self.geometry();
        let h = g.status_bar_height.max(20.0);
        let mut kids = Vec::new();
        match self.profile.chrome {
            ChromeStyle::Ios => {
                kids.push(text("emu_time", "9:41", 12.0, t.text_secondary, false));
                kids.push(spacer("emu_status_spacer"));
                if let Some(ref hole) = g.punch_hole {
                    kids.push(
                        Node::new(
                            "emu_island",
                            NodeKind::Surface {
                                style: BoxStyle {
                                    bg: Some(hex_rgba(0x05070A, 255)),
                                    radius: [hole.diameter.min(35.0) * 0.5; 4],
                                    ..BoxStyle::default()
                                },
                                interactive: false,
                                child: None,
                            },
                        )
                        .with_size(SizeSpec::Px(hole.diameter), SizeSpec::Px(35.0)),
                    );
                }
                kids.push(spacer("emu_status_spacer2"));
                kids.push(text("emu_status_right", "5G  100%", 12.0, t.text_secondary, false));
            }
            _ => {
                kids.push(text("emu_time", "12:00", 12.0, t.text_secondary, false));
                kids.push(spacer("emu_tb_spacer"));
                if let Some(ref hole) = g.punch_hole {
                    kids.push(
                        Node::new(
                            "emu_punch_hole",
                            NodeKind::Surface {
                                style: BoxStyle {
                                    bg: Some(hex_rgba(0x05070A, 255)),
                                    radius: [hole.diameter * 0.5; 4],
                                    ..BoxStyle::default()
                                },
                                interactive: false,
                                child: None,
                            },
                        )
                        .with_size(
                            SizeSpec::Px(hole.diameter),
                            SizeSpec::Px(hole.diameter),
                        ),
                    );
                    kids.push(spacer("emu_title_spacer"));
                }
                kids.push(text("emu_status_right", "Wi-Fi  100%", 12.0, t.text_secondary, false));
            }
        }
        row(
            "emu_status",
            Align::Center,
            6.0,
            Insets {
                top: 0.0,
                right: 18.0,
                bottom: 0.0,
                left: 18.0,
            },
            None,
            kids,
        )
        .with_size(SizeSpec::Fill(1.0), SizeSpec::Px(h))
    }

    fn titlebar(&self) -> Node {
        let t = self.chrome();
        let g = self.geometry();
        let h = g.titlebar_height.max(28.0);
        let mut kids = Vec::new();
        match self.profile.chrome {
            ChromeStyle::Macos => {
                kids.push(dot("emu_tl_red", hex_rgba(0xFF5F57, 255)));
                kids.push(dot("emu_tl_yellow", hex_rgba(0xFEBC2E, 255)));
                kids.push(dot("emu_tl_green", hex_rgba(0x28C840, 255)));
                kids.push(spacer("emu_title_spacer2"));
                kids.push(text("emu_title", &self.device.name, 12.0, t.text_muted, false));
                kids.push(spacer("emu_toolbar_spacer"));
                kids.push(text("emu_title_r", "  ", 12.0, t.text_muted, false));
            }
            ChromeStyle::Windows => {
                kids.push(text("emu_title", &self.device.name, 12.0, t.text_secondary, false));
                kids.push(spacer("emu_win_spacer"));
                kids.push(text("emu_win_min", "  -  ", 12.0, t.text_muted, false));
                kids.push(text("emu_win_max", "  []  ", 12.0, t.text_muted, false));
                kids.push(text("emu_win_close", "  x  ", 12.0, t.text_muted, false));
            }
            _ => {
                kids.push(text("emu_title", &self.device.name, 12.0, t.text_secondary, false));
                kids.push(spacer("emu_linux_spacer"));
                kids.push(text("emu_win_close", "  x  ", 12.0, t.text_muted, false));
            }
        }
        row(
            "emu_titlebar",
            Align::Center,
            6.0,
            Insets {
                top: 0.0,
                right: 12.0,
                bottom: 0.0,
                left: 12.0,
            },
            None,
            kids,
        )
        .with_size(SizeSpec::Fill(1.0), SizeSpec::Px(h))
    }

    fn home_bar(&self) -> Node {
        let t = self.chrome();
        let g = self.geometry();
        let h = g.home_bar_height.max(20.0);
        let pill = Node::new(
            "emu_home_pill",
            NodeKind::Surface {
                style: BoxStyle {
                    bg: Some(t.text_muted),
                    radius: [3.0; 4],
                    ..BoxStyle::default()
                },
                interactive: false,
                child: None,
            },
        )
        .with_size(SizeSpec::Px(134.0), SizeSpec::Px(5.0));
        let inner = match g.nav_bar {
            NavBar::ThreeButton => row(
                "emu_nav",
                Align::Center,
                8.0,
                Insets::ZERO,
                None,
                vec![
                    button("emu_back", "Back").with_size(SizeSpec::Fill(1.0), SizeSpec::Px(36.0)),
                    button("emu_home_btn", "Home").with_size(SizeSpec::Fill(1.0), SizeSpec::Px(36.0)),
                    button("emu_recent", "Apps").with_size(SizeSpec::Fill(1.0), SizeSpec::Px(36.0)),
                ],
            ),
            _ => row(
                "emu_nav",
                Align::Center,
                0.0,
                Insets::ZERO,
                None,
                vec![pill],
            ),
        };
        col(
            "emu_home",
            Align::Center,
            0.0,
            Insets {
                top: 6.0,
                right: 12.0,
                bottom: 8.0,
                left: 12.0,
            },
            None,
            vec![inner],
        )
        .with_size(SizeSpec::Fill(1.0), SizeSpec::Px(h))
    }

    fn keyboard(&self) -> Node {
        let t = self.chrome();
        let g = self.geometry();
        let mut rows = Vec::new();
        rows.push(
            col(
                "emu_kb_handle_row",
                Align::Center,
                0.0,
                Insets {
                    top: 2.0,
                    right: 0.0,
                    bottom: 4.0,
                    left: 0.0,
                },
                None,
                vec![Node::new(
                    "emu_kb_handle",
                    NodeKind::Surface {
                        style: BoxStyle {
                            bg: Some(t.text_muted),
                            radius: [2.0; 4],
                            ..BoxStyle::default()
                        },
                        interactive: false,
                        child: None,
                    },
                )
                .with_size(SizeSpec::Px(36.0), SizeSpec::Px(4.0))],
            )
            .with_size(SizeSpec::Fill(1.0), SizeSpec::Auto),
        );
        for (i, keys) in ["QWERTYUIOP", "ASDFGHJKL", "ZXCVBNM"].iter().enumerate() {
            let mut r = Vec::new();
            for c in keys.chars() {
                r.push(
                    button(&format!("emu_key_{c}"), &c.to_string())
                        .with_size(SizeSpec::Fill(1.0), SizeSpec::Px(38.0)),
                );
            }
            rows.push(
                row(
                    &format!("emu_kb_row_{i}"),
                    Align::Stretch,
                    3.0,
                    Insets::ZERO,
                    None,
                    r,
                )
                .with_size(SizeSpec::Fill(1.0), SizeSpec::Px(38.0)),
            );
        }
        rows.push(
            row(
                "emu_kb_row_bottom",
                Align::Stretch,
                4.0,
                Insets::ZERO,
                None,
                vec![
                    button("emu_kb_123", "123").with_size(SizeSpec::Px(54.0), SizeSpec::Px(38.0)),
                    button("emu_kb_space", "space").with_size(SizeSpec::Fill(1.0), SizeSpec::Px(38.0)),
                    button("emu_kb_return", "return").with_size(SizeSpec::Px(74.0), SizeSpec::Px(38.0)),
                ],
            )
            .with_size(SizeSpec::Fill(1.0), SizeSpec::Px(38.0)),
        );
        col(
            "emu_keyboard",
            Align::Stretch,
            4.0,
            Insets::all(6.0),
            Some(BoxStyle {
                bg: Some(t.surface_0),
                border: Some(t.border),
                border_width: 1.0,
                radius: [10.0; 4],
                ..BoxStyle::default()
            }),
            rows,
        )
        .with_size(SizeSpec::Fill(1.0), SizeSpec::Px(g.keyboard_height.max(160.0)))
    }

    fn toolbar(&self) -> Node {
        let t = self.chrome();
        let names: Vec<String> = profiles().iter().map(|p| p.name.clone()).collect();
        let active = profiles()
            .iter()
            .position(|p| p.id == self.profile.id)
            .unwrap_or(0);
        let platform = Node::new(
            "emu_platform",
            NodeKind::Segmented {
                active,
                options: names,
            },
        )
        .with_size(SizeSpec::Auto, SizeSpec::Px(28.0));
        let mut kids = vec![
            platform,
            text("emu_device_name", &self.device.name, 12.0, t.text_muted, false),
            spacer("emu_toolbar_spacer2"),
            small_button("emu_dev_prev", "<"),
            small_button("emu_dev_next", ">"),
            small_button("emu_rotate", "Rot"),
            small_button("emu_kb", if self.keyboard_open { "KB On" } else { "KB" }),
            small_button("emu_theme", "Theme"),
            small_button("emu_inspect", if self.inspector { "Hide" } else { "Inspect" }),
            small_button("emu_shot", "Shot"),
        ];
        if self.profile.touch {
            kids.push(small_button("emu_back", "Back"));
        }
        row(
            "emu_toolbar",
            Align::Center,
            6.0,
            Insets {
                top: 0.0,
                right: 10.0,
                bottom: 0.0,
                left: 10.0,
            },
            Some(BoxStyle {
                bg: Some(t.surface_1),
                border: Some(t.border),
                border_width: 1.0,
                ..BoxStyle::default()
            }),
            kids,
        )
        .with_size(SizeSpec::Fill(1.0), SizeSpec::Px(46.0))
    }

    fn inspector(&self) -> Node {
        let t = self.chrome();
        let names = if self.config.scenarios.is_empty() {
            crate::scenarios::ScenarioSet::builtin().names()
        } else {
            self.config.scenarios.clone()
        };
        let mut kids = vec![
            text("emu_insp_title", "Scenarios", 13.0, t.text_primary, false),
            text(
                "emu_insp_sub",
                &format!("{} - {} @{:.2}x", self.profile.name, self.device.name, self.device.scale),
                11.0,
                t.text_muted,
                false,
            ),
        ];
        for (i, name) in names.iter().enumerate() {
            kids.push(
                button(&format!("emu_scenario_{i}"), name)
                    .with_size(SizeSpec::Fill(1.0), SizeSpec::Px(30.0)),
            );
        }
        kids.push(text(
            "emu_insp_audit",
            "Geometry Audit",
            13.0,
            t.text_primary,
            false,
        ));
        kids.push(text(
            "emu_insp_hint",
            "Run `sofuu ui snapshot` to write PNG + audit.json. Errors are printed on stdout and fail --fail-on-error.",
            11.0,
            t.text_muted,
            true,
        ));
        kids.push(spacer("emu_insp_spacer"));
        kids.push(
            button("emu_shot_big", "Capture AI Screenshot")
                .with_size(SizeSpec::Fill(1.0), SizeSpec::Px(36.0)),
        );
        col(
            "emu_inspector",
            Align::Stretch,
            8.0,
            Insets::all(12.0),
            Some(BoxStyle {
                bg: Some(t.surface_0),
                border: Some(t.border),
                border_width: 1.0,
                ..BoxStyle::default()
            }),
            kids,
        )
        .with_size(SizeSpec::Px(300.0), SizeSpec::Fill(1.0))
    }

    /// Route a semantic UI event to the legacy in-scene shell controls.
    /// Returns `None` when the event belongs to the app. The screen-first
    /// window (E6) routes strip actions natively instead.
    pub fn handle_event(&mut self, ev: &UIEvent) -> Option<ShellAction> {
        match ev {
            UIEvent::Click { id } => match id.as_str() {
                "emu_rotate" => {
                    self.orientation = self.orientation.toggle();
                    self.config.orientation = self.orientation;
                    Some(ShellAction::Changed)
                }
                "emu_kb" => {
                    self.keyboard_open = !self.keyboard_open;
                    self.config.keyboard_open = self.keyboard_open;
                    Some(ShellAction::Changed)
                }
                "emu_theme" => {
                    self.theme = match self.theme {
                        ThemeMode::Light => ThemeMode::Dark,
                        ThemeMode::Dark => ThemeMode::Light,
                    };
                    self.config.theme = self.theme;
                    Some(ShellAction::Changed)
                }
                "emu_inspect" => {
                    self.inspector = !self.inspector;
                    self.config.inspector = self.inspector;
                    Some(ShellAction::Changed)
                }
                "emu_shot" | "emu_shot_big" => Some(ShellAction::Screenshot),
                "emu_back" | "emu_home_btn" => Some(ShellAction::Back),
                "emu_dev_prev" => {
                    self.cycle_device(-1);
                    Some(ShellAction::Changed)
                }
                "emu_dev_next" => {
                    self.cycle_device(1);
                    Some(ShellAction::Changed)
                }
                other if other.starts_with("emu_scenario_") => {
                    if let Ok(i) = other["emu_scenario_".len()..].parse::<usize>() {
                        let names = if self.config.scenarios.is_empty() {
                            crate::scenarios::ScenarioSet::builtin().names()
                        } else {
                            self.config.scenarios.clone()
                        };
                        if let Some(name) = names.get(i) {
                            self.scenario = Some(name.clone());
                            self.config.scenario = self.scenario.clone();
                            return Some(ShellAction::Changed);
                        }
                    }
                    None
                }
                _ => None,
            },
            UIEvent::SegmentedChanged { id, index } if id == "emu_platform" => {
                let all = profiles();
                if let Some(p) = all.get(*index) {
                    self.set_platform(&p.id);
                    return Some(ShellAction::Changed);
                }
                None
            }
            UIEvent::WindowClosed => Some(ShellAction::Close),
            _ => None,
        }
    }

    pub fn set_platform(&mut self, platform: &str) {
        if let Ok((p, d)) = resolve(platform, None) {
            self.profile = p;
            self.device = d;
            self.config.platform = self.profile.id.clone();
            self.config.device = Some(self.device.id.clone());
        }
    }

    pub fn cycle_device(&mut self, delta: i32) {
        if self.profile.devices.is_empty() {
            return;
        }
        let idx = self
            .profile
            .devices
            .iter()
            .position(|d| d.id == self.device.id)
            .unwrap_or(0) as i32;
        let next = (idx + delta).rem_euclid(self.profile.devices.len() as i32) as usize;
        self.device = self.profile.devices[next].clone();
        self.config.device = Some(self.device.id.clone());
    }

    pub fn set_device(&mut self, id: &str) -> Result<(), String> {
        let d = self
            .profile
            .device(id)
            .cloned()
            .ok_or_else(|| format!("unknown device `{id}` for {}", self.profile.name))?;
        self.device = d;
        self.config.device = Some(self.device.id.clone());
        Ok(())
    }

    pub fn toggle_theme(&mut self) {
        self.theme = match self.theme {
            ThemeMode::Light => ThemeMode::Dark,
            ThemeMode::Dark => ThemeMode::Light,
        };
        self.config.theme = self.theme;
    }

    pub fn toggle_keyboard(&mut self) {
        self.keyboard_open = !self.keyboard_open;
        self.config.keyboard_open = self.keyboard_open;
    }

    pub fn rotate(&mut self) {
        self.orientation = self.orientation.toggle();
        self.config.orientation = self.orientation;
    }

    pub fn set_scenario(&mut self, name: Option<String>) {
        self.scenario = name.clone();
        self.config.scenario = name;
    }
}

fn spacer(id: &str) -> Node {
    let mut n = Node::new(id, NodeKind::Spacer);
    n.width = SizeSpec::Fill(1.0);
    n
}

fn text(id: &str, content: &str, size: f32, color: u32, wrap: bool) -> Node {
    Node::new(
        id,
        NodeKind::Text {
            content: content.to_string(),
            size,
            color: Some(color),
            mono: false,
            wrap,
        },
    )
}

fn button(id: &str, label: &str) -> Node {
    Node::new(
        id,
        NodeKind::Button {
            label: label.to_string(),
            tone: Tone::Neutral,
        },
    )
}

fn small_button(id: &str, label: &str) -> Node {
    button(id, label).with_size(SizeSpec::Auto, SizeSpec::Px(28.0))
}

fn dot(id: &str, color: u32) -> Node {
    Node::new(
        id,
        NodeKind::Surface {
            style: BoxStyle {
                bg: Some(color),
                radius: [6.0; 4],
                ..BoxStyle::default()
            },
            interactive: false,
            child: None,
        },
    )
    .with_size(SizeSpec::Px(12.0), SizeSpec::Px(12.0))
}

fn row(
    id: &str,
    align: Align,
    gap: f32,
    padding: Insets,
    style: Option<BoxStyle>,
    children: Vec<Node>,
) -> Node {
    Node::new(
        id,
        NodeKind::Container {
            axis: Axis::Row,
            align,
            gap,
            fold_width: None,
            scrollable: false,
            style,
            children,
        },
    )
    .with_padding(padding)
}

fn col(
    id: &str,
    align: Align,
    gap: f32,
    padding: Insets,
    style: Option<BoxStyle>,
    children: Vec<Node>,
) -> Node {
    Node::new(
        id,
        NodeKind::Container {
            axis: Axis::Column,
            align,
            gap,
            fold_width: None,
            scrollable: false,
            style,
            children,
        },
    )
    .with_padding(padding)
}

fn framed(
    id: &str,
    child: Node,
    width: f32,
    height: f32,
    style: BoxStyle,
    padding: f32,
) -> Node {
    col(
        id,
        Align::Center,
        0.0,
        Insets::all(padding),
        Some(style),
        vec![child],
    )
    .with_size(SizeSpec::Px(width), SizeSpec::Px(height))
}

/// The background an app root paints, if any. Used to tint the host screen
/// behind the status/title bar so a full-bleed app's chrome band matches.
fn node_bg(node: &Node) -> Option<u32> {
    match &node.kind {
        NodeKind::Container { style, .. } | NodeKind::Grid { style, .. } => {
            style.as_ref().and_then(|s| s.bg)
        }
        NodeKind::Surface { style, .. } => style.bg,
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sofuu_ui::layout::LayoutMap;
    use sofuu_ui::text::GlyphAtlas;

    fn app() -> Scene {
        let root = col(
            "app_root",
            Align::Stretch,
            8.0,
            Insets::all(16.0),
            None,
            vec![
                text("title", "Wallet", 20.0, 0, false),
                button("checkout_submit_btn", "Pay"),
            ],
        );
        Scene::new("Wallet", ThemeMode::Light, root)
    }

    #[test]
    fn wraps_all_five_platforms_and_keeps_app_viewport_exact() {
        for platform in ["ios", "android", "macos", "windows", "linux"] {
            let host = EmulatorHost::new(EmulatorConfig {
                platform: platform.to_string(),
                ..EmulatorConfig::default()
            })
            .unwrap();
            let scene = host.wrap_device_scene(app(), false);
            let mut layout = LayoutMap::default();
            layout.solve(
                &scene.root,
                [host.geometry().width, host.geometry().height],
                &GlyphAtlas::builtin(),
            );
            let vp = layout.get("emu_app_viewport").expect("app viewport");
            let expected = host.app_size(false);
            assert!(
                (vp.h - expected[1]).abs() < 1.0,
                "{platform}: viewport height {} != {}",
                vp.h,
                expected[1]
            );
            assert!((vp.w - host.geometry().width).abs() < 1.0);
        }
    }

    #[test]
    fn keyboard_shrinks_only_the_interactive_flow() {
        let mut host = EmulatorHost::new(EmulatorConfig::default()).unwrap();
        let full = host.app_size(false)[1];
        host.keyboard_open = true;
        let shrunk = host.app_size(true)[1];
        assert!((full - shrunk - host.geometry().keyboard_height).abs() < 0.5);
    }

    #[test]
    fn shell_events_mutate_host_state() {
        let mut host = EmulatorHost::new(EmulatorConfig::default()).unwrap();
        let before = host.orientation;
        assert_eq!(
            host.handle_event(&UIEvent::Click {
                id: "emu_rotate".to_string()
            }),
            Some(ShellAction::Changed)
        );
        assert_ne!(host.orientation, before);

        let before_kb = host.keyboard_open;
        host.handle_event(&UIEvent::Click {
            id: "emu_kb".to_string(),
        });
        assert_ne!(host.keyboard_open, before_kb);

        let platform_before = host.profile.id.clone();
        host.handle_event(&UIEvent::SegmentedChanged {
            id: "emu_platform".to_string(),
            index: 1,
        });
        assert_ne!(host.profile.id, platform_before);
    }

    #[test]
    fn screenshot_audit_rules_use_home_bar_not_keyboard_shrink() {
        let mut host = EmulatorHost::new(EmulatorConfig::default()).unwrap();
        host.keyboard_open = true;
        let rules = host.audit_rules(true);
        assert!(rules.keyboard_open);
        assert!((rules.keyboard_height - host.geometry().keyboard_height).abs() < 0.1);
        assert!((rules.viewport[1] - host.app_size(false)[1]).abs() < 0.1);
        assert_eq!(rules.min_touch, 44.0);
    }
}
