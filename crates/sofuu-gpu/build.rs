use std::env;

fn main() {
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    println!("cargo:rerun-if-changed=native/macos_metal.m");
    println!("cargo:rerun-if-changed=src/shaders/msl/ui_shaders.metal");

    if target_os == "macos" {
        cc::Build::new()
            .file("native/macos_metal.m")
            .flag("-fobjc-arc")
            .flag_if_supported("-Wno-deprecated-declarations")
            .compile("sofuu_gpu_macos");

        println!("cargo:rustc-link-lib=framework=AppKit");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Metal");
        println!("cargo:rustc-link-lib=framework=QuartzCore");
    }
}
