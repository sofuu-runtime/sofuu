// crates/sofuu-core/examples/cuda_probe.rs — minimal CUDA bring-up probe.
//
// Keeps the native CUDA backend reachable in a real binary (and therefore
// linked) before model wiring lands: without a call site, an unused public
// module is dead-code-eliminated and the loader/PTX never enter the image.
//
// Linux + NVIDIA: prints device identity, loads/JITs the embedded PTX, and
// reports every kernel entry point. No device: prints a clean "not available"
// line and exits 0 (the CPU engine is the fallback, not an error).
//
//   cargo run --release -p sofuu-core --example cuda_probe

fn main() {
    use sofuu_core::infer::cuda;

    println!(
        "driver_present={} (SOFUU_CUDA=0 disables)",
        cuda::driver_present()
    );

    match cuda::probe() {
        Some(info) => {
            println!(
                "device={} cc{}.{} vram={:.2} GiB integrated={} unified={} driver={} lib={}",
                info.name,
                info.cc_major,
                info.cc_minor,
                info.total_memory as f64 / (1024.0 * 1024.0 * 1024.0),
                info.integrated,
                info.unified_addressing,
                info.driver_version,
                info.library
            );
        }
        None => {
            println!("device=none (no libcuda.so.1 / no NVIDIA device on this host)");
        }
    }

    match cuda::CudaEngine::try_init() {
        Some(engine) => {
            let info = engine.device_info();
            println!(
                "engine=ready unified_memory={} ptx_entries={}",
                engine.is_unified_memory(),
                cuda::PTX_ENTRIES.len()
            );
            println!("kernels={}", cuda::PTX_ENTRIES.join(","));
            println!(
                "engine_device={} cc{}.{}",
                info.name, info.cc_major, info.cc_minor
            );
        }
        None => {
            println!("engine=unavailable (CPU fallback is active)");
        }
    }
}
