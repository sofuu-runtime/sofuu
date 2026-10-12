use sofuu_core::infer::metal::MetalEngine;

#[test]
fn test_metal_engine_init_and_gemv() {
    let engine = match MetalEngine::new() {
        Some(e) => e,
        None => {
            eprintln!("Metal not available on this platform, skipping.");
            return;
        }
    };
    println!("Metal GPU Engine successfully initialized on Apple Silicon!");

    // Test zero-copy buffer registration with page-aligned memory
    let mut weights = vec![0u16; 1024 * 1024]; // 2MB buffer
    // Fill with BF16 values (e.g. 1.0f in BF16 = 0x3F80)
    for w in weights.iter_mut() {
        *w = 0x3F80; // 1.0 in BF16
    }
    let u8_slice = unsafe {
        std::slice::from_raw_parts(weights.as_ptr() as *const u8, weights.len() * 2)
    };

    let w_buf = engine.register_zerocopy_buffer(u8_slice)
        .expect("failed to register zero-copy buffer");

    let cols = 1024;
    let rows = 1024;
    let x = vec![1.0f32; cols];
    let mut y = vec![0.0f32; rows];

    // Compute on Apple Silicon GPU via Metal zero-copy
    engine.gemv_bf16(&w_buf, 0, &x, &mut y, rows, cols);

    println!("y[0] = {}, y[500] = {}", y[0], y[500]);
    assert!((y[0] - 1024.0).abs() < 1e-3, "expected 1024.0, got {}", y[0]);
    assert!((y[500] - 1024.0).abs() < 1e-3, "expected 1024.0, got {}", y[500]);
    println!("✓ Metal BF16 GEMV computation verified on Apple Silicon GPU!");

    // Test SwiGLU on GPU
    let mut g = vec![2.0f32; 1024];
    let u = vec![3.0f32; 1024];
    engine.swiglu(&mut g, &u);
    // silu(2.0) = 2.0 / (1 + exp(-2)) ≈ 2 * 0.880797 = 1.76159
    // 1.76159 * 3 = 5.28478
    println!("swiglu g[0] = {}", g[0]);
    assert!((g[0] - 5.28478).abs() < 1e-2, "expected ~5.28478, got {}", g[0]);
    println!("✓ Metal SwiGLU verified on Apple Silicon GPU!");

    // Test buffer creation with fallback
    let mut raw_buf = Vec::with_capacity(128 * 2);
    for _ in 0..128 {
        raw_buf.push(0x80u8);
        raw_buf.push(0x3fu8);
    }
    let m_buf = engine.create_buffer(&raw_buf).expect("failed to create metal buffer");

    // Test Batched QKV Multi
    let mut q_out = vec![0.0f32; 32];
    let mut k_out = vec![0.0f32; 16];
    let mut v_out = vec![0.0f32; 16];
    let x_small = vec![1.0f32; 4];
    engine.gemv_qkv_multi(
        &m_buf, 0, 32,
        &m_buf, 0, 16,
        &m_buf, 0, 16,
        &x_small,
        &mut q_out, &mut k_out, &mut v_out,
        4,
        0, // BF16
    );
    assert_eq!(q_out.len(), 32);
    assert_eq!(k_out.len(), 16);
    assert_eq!(v_out.len(), 16);
    assert!((q_out[0] - 4.0).abs() < 1e-3, "expected 4.0, got {}", q_out[0]);
    assert!((k_out[0] - 4.0).abs() < 1e-3, "expected 4.0, got {}", k_out[0]);
    assert!((v_out[0] - 4.0).abs() < 1e-3, "expected 4.0, got {}", v_out[0]);
    println!("✓ Batched QKV multi verified on Apple Silicon GPU!");

    // Test Asynchronous GEMV with CPU-GPU overlap
    let mut y_async = vec![0.0f32; rows];
    engine.gemv_async(&w_buf, 0, &x, &mut y_async, rows, cols, 0);

    // CPU does concurrent work while GPU executes
    let mut cpu_work_sum = 0.0f32;
    for i in 0..10_000 {
        cpu_work_sum += (i as f32).sqrt();
    }
    assert!(cpu_work_sum > 0.0);

    // Sync with GPU
    engine.sync();
    assert!((y_async[0] - 1024.0).abs() < 1e-3, "expected 1024.0 from async GEMV, got {}", y_async[0]);
    assert!((y_async[500] - 1024.0).abs() < 1e-3, "expected 1024.0 from async GEMV, got {}", y_async[500]);
    println!("✓ Asynchronous CPU-GPU overlapped GEMV verified on Apple Silicon GPU!");
}

