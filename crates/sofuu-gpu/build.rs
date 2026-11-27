use std::env;
use std::path::{Path, PathBuf};

/// Native window/GPU shims are compiled here rather than in the Rust backends so that every
/// display-server / GPU-API dependency stays behind `dlopen`/`LoadLibrary` at run time.
///
/// Set `SOFUU_GPU_SKIP_NATIVE=1` to compile only the Rust side (cross-`cargo check` on a host
/// without a target C toolchain). That produces a library whose backends cannot open a window.
fn main() {
    println!("cargo:rerun-if-env-changed=SOFUU_GPU_SKIP_NATIVE");
    println!("cargo:rerun-if-env-changed=ANDROID_NDK_HOME");
    println!("cargo:rerun-if-env-changed=ANDROID_NDK_ROOT");
    println!("cargo:rerun-if-env-changed=ANDROID_HOME");
    println!("cargo:rerun-if-changed=native/macos_metal.m");
    println!("cargo:rerun-if-changed=native/linux_wayland_egl.c");
    println!("cargo:rerun-if-changed=native/windows_win32_gl.c");
    println!("cargo:rerun-if-changed=native/android_egl.c");
    println!("cargo:rerun-if-changed=src/shaders/msl/ui_shaders.metal");
    println!("cargo:rerun-if-changed=src/shaders/glsl/ui_shaders_330.glsl");
    println!("cargo:rerun-if-changed=src/shaders/glsl/ui_shaders_es3.glsl");
    println!("cargo:rerun-if-changed=src/shaders/glsl/ui_shaders_450.glsl");
    println!("cargo:rerun-if-changed=src/shaders/spirv");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let skip_native = env::var("SOFUU_GPU_SKIP_NATIVE")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    if skip_native {
        println!(
            "cargo:warning=sofuu-gpu: SOFUU_GPU_SKIP_NATIVE is set; native window shims were not \
             compiled. This library can be cargo-checked but cannot open a window."
        );
        return;
    }

    match target_os.as_str() {
        "macos" => build_macos(),
        "linux" | "freebsd" | "openbsd" | "netbsd" | "dragonfly" => build_linux(&target_os),
        "windows" => build_windows(),
        "android" => build_android(),
        _ => {}
    }
}

fn base_build() -> cc::Build {
    let mut build = cc::Build::new();
    build
        .flag_if_supported("-std=c11")
        .flag_if_supported("-fno-common")
        .flag_if_supported("-Wall")
        .flag_if_supported("-Wextra")
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-unused-function");
    build
}

fn build_macos() {
    let mut build = base_build();
    build
        .file("native/macos_metal.m")
        .flag("-fobjc-arc")
        .flag_if_supported("-Wno-deprecated-declarations")
        .compile("sofuu_gpu_macos");

    println!("cargo:rustc-link-lib=framework=AppKit");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=QuartzCore");
}

fn build_linux(target_os: &str) {
    base_build()
        .file("native/linux_wayland_egl.c")
        .compile("sofuu_gpu_linux");

    // `dlopen` lives in libc on the BSDs.
    if target_os == "linux" {
        println!("cargo:rustc-link-lib=dl");
    }
    println!("cargo:rustc-link-lib=m");
}

fn build_windows() {
    let mut build = base_build();
    // Win32 loads wgl/DWM entry points through GetProcAddress, which returns a generic PROC.
    build.flag_if_supported("-Wno-cast-function-type");
    build
        .file("native/windows_win32_gl.c")
        .compile("sofuu_gpu_windows");

    println!("cargo:rustc-link-lib=user32");
    println!("cargo:rustc-link-lib=gdi32");
    println!("cargo:rustc-link-lib=opengl32");
}

fn build_android() {
    // The `cc` crate finds `aarch64-linux-android21-clang` etc. on PATH. Add the NDK's
    // prebuilt toolchain bin directory when the caller has not already done so.
    if env::var_os("CC").is_none() && env::var_os("TARGET_CC").is_none() {
        if let Some(bin) = android_ndk_bin() {
            let mut paths = vec![bin];
            if let Some(existing) = env::var_os("PATH") {
                paths.extend(env::split_paths(&existing));
            }
            if let Ok(joined) = env::join_paths(paths) {
                env::set_var("PATH", joined);
            }
        }
    }

    base_build()
        .file("native/android_egl.c")
        .compile("sofuu_gpu_android");

    println!("cargo:rustc-link-lib=dl");
    println!("cargo:rustc-link-lib=m");
}

fn android_ndk_bin() -> Option<PathBuf> {
    for key in ["ANDROID_NDK_HOME", "ANDROID_NDK_ROOT"] {
        if let Some(root) = env::var_os(key) {
            if let Some(bin) = ndk_host_bin(Path::new(&root)) {
                return Some(bin);
            }
        }
    }
    if let Some(home) = env::var_os("ANDROID_HOME") {
        if let Some(bin) = newest_ndk_bin(Path::new(&home).join("ndk")) {
            return Some(bin);
        }
    }
    if let Some(home) = env::var_os("HOME") {
        if let Some(bin) = newest_ndk_bin(Path::new(&home).join("Library/Android/sdk/ndk")) {
            return Some(bin);
        }
        if let Some(bin) = newest_ndk_bin(Path::new(&home).join("Android/Sdk/ndk")) {
            return Some(bin);
        }
    }
    None
}

fn newest_ndk_bin(root: PathBuf) -> Option<PathBuf> {
    let mut versions: Vec<PathBuf> = std::fs::read_dir(&root)
        .ok()?
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    versions.sort();
    versions.iter().rev().find_map(|p| ndk_host_bin(p))
}

fn ndk_host_bin(ndk: &Path) -> Option<PathBuf> {
    let prebuilt = ndk.join("toolchains/llvm/prebuilt");
    std::fs::read_dir(prebuilt)
        .ok()?
        .flatten()
        .map(|e| e.path().join("bin"))
        .find(|p| p.is_dir())
}
