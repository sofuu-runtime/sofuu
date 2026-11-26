//! Cohesive visual design tokens for Sofuu's 2D GPU UI engine.
//!
//! Dark-first precision surface palette + matching high-contrast light palette,
//! shared spacing/radius scales, and semantic status tones.

use serde::{Deserialize, Serialize};

use crate::draw::{hex_rgba, rgba8};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThemeMode {
    Dark,
    #[default]
    Light,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Tone {
    #[default]
    Neutral,
    Info,
    Ok,
    Warn,
    Danger,
}

#[derive(Clone, Copy, Debug)]
pub struct Theme {
    pub mode: ThemeMode,
    /// Root window canvas (`#0B0F14` dark / `#F5F7FA` light).
    pub bg: u32,
    /// Recessed inset surface (scroll tracks, code wells).
    pub surface_0: u32,
    /// Primary card / panel surface.
    pub surface_1: u32,
    /// Elevated interactive surface / hover.
    pub surface_2: u32,
    /// Floating / active / pressed highlight surface.
    pub surface_3: u32,
    /// Subtle hairline border (1px crisp outline).
    pub border: u32,
    /// Highlighted / focused border.
    pub border_focus: u32,
    /// Primary high-contrast foreground text.
    pub text_primary: u32,
    /// Secondary label / metadata text.
    pub text_secondary: u32,
    /// Muted caption / placeholder text.
    pub text_muted: u32,
    /// Brand / primary interactive accent.
    pub accent: u32,
    pub info: u32,
    pub ok: u32,
    pub warn: u32,
    pub danger: u32,
    /// Analytic soft card shadow.
    pub shadow: u32,
    /// Base spacing unit in logical px (`8.0`).
    pub space_unit: f32,
    /// Default corner radii `[card, control, pill]`.
    pub radius_card: f32,
    pub radius_control: f32,
    pub radius_pill: f32,
}

impl Theme {
    pub const fn dark() -> Self {
        Self {
            mode: ThemeMode::Dark,
            bg: hex_rgba(0x0B0F14, 255),
            surface_0: hex_rgba(0x0E141C, 255),
            surface_1: hex_rgba(0x131A24, 255),
            surface_2: hex_rgba(0x192331, 255),
            surface_3: hex_rgba(0x212E40, 255),
            border: rgba8(255, 255, 255, 22),
            border_focus: hex_rgba(0x3B82F6, 180),
            text_primary: hex_rgba(0xE6EDF3, 255),
            text_secondary: hex_rgba(0x9BA7B4, 255),
            text_muted: hex_rgba(0x637080, 255),
            accent: hex_rgba(0x3B82F6, 255),
            info: hex_rgba(0x38BDF8, 255),
            ok: hex_rgba(0x22C55E, 255),
            warn: hex_rgba(0xF59E0B, 255),
            danger: hex_rgba(0xEF4444, 255),
            shadow: rgba8(0, 0, 0, 115),
            space_unit: 8.0,
            radius_card: 12.0,
            radius_control: 8.0,
            radius_pill: 999.0,
        }
    }

    pub const fn light() -> Self {
        Self {
            mode: ThemeMode::Light,
            bg: hex_rgba(0xF4F6F9, 255),
            surface_0: hex_rgba(0xEAEEF4, 255),
            surface_1: hex_rgba(0xFFFFFF, 255),
            surface_2: hex_rgba(0xF1F5F9, 255),
            surface_3: hex_rgba(0xE2E8F0, 255),
            border: rgba8(15, 23, 42, 24),
            border_focus: hex_rgba(0x2563EB, 190),
            text_primary: hex_rgba(0x0F172A, 255),
            text_secondary: hex_rgba(0x475569, 255),
            text_muted: hex_rgba(0x64748B, 255),
            accent: hex_rgba(0x2563EB, 255),
            info: hex_rgba(0x0284C7, 255),
            ok: hex_rgba(0x16A34A, 255),
            warn: hex_rgba(0xD97706, 255),
            danger: hex_rgba(0xDC2626, 255),
            shadow: rgba8(15, 23, 42, 28),
            space_unit: 8.0,
            radius_card: 12.0,
            radius_control: 8.0,
            radius_pill: 999.0,
        }
    }

    pub const fn for_mode(mode: ThemeMode) -> Self {
        match mode {
            ThemeMode::Dark => Self::dark(),
            ThemeMode::Light => Self::light(),
        }
    }

    pub const fn tone_color(&self, tone: Tone) -> u32 {
        match tone {
            Tone::Neutral => self.text_secondary,
            Tone::Info => self.info,
            Tone::Ok => self.ok,
            Tone::Warn => self.warn,
            Tone::Danger => self.danger,
        }
    }
}

impl Default for Theme {
    fn default() -> Self {
        Self::light()
    }
}
