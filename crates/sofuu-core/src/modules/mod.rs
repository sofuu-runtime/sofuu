// modules — Rust ports of the JS-visible module shells (PLAN-RUST-MIGRATION
// M2+). The register functions replace the retired C `mod_*_register`
// symbols; engine.c calls them unchanged.

pub mod console;
pub mod process; // M3: process object + TTY readline + chat bridges
