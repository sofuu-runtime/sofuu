// crates/sofuu-core/tests/test_universal_formats.rs — Verification for PyTorch .pt, NumPy .npz, and Universal Tensor Registry.

use std::process::Command;

use sofuu_core::infer::npy::{parse_npy_header, NpzFile, TensorDType};
use sofuu_core::infer::pytorch::PyTorchCheckpoint;
use sofuu_core::infer::tensor_table::UnifiedTensorTable;

#[test]
fn test_npy_header_parsing() {
    // Construct valid NPY v1.0 header for shape (2, 3), float32
    let header_str = "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }";
    let mut npy_bytes = Vec::new();
    npy_bytes.extend_from_slice(b"\x93NUMPY\x01\x00"); // magic + version 1.0

    let h_len = header_str.len() + 1;
    let pad = (64 - ((6 + 2 + 2 + h_len) % 64)) % 64;
    let padded = format!("{}{}\n", header_str, " ".repeat(pad));
    let hlen_u16 = padded.len() as u16;

    npy_bytes.extend_from_slice(&hlen_u16.to_le_bytes());
    npy_bytes.extend_from_slice(padded.as_bytes());

    // Append 6 float32 values: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    let floats: [f32; 6] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    for f in floats {
        npy_bytes.extend_from_slice(&f.to_le_bytes());
    }

    let parsed = parse_npy_header(&npy_bytes).expect("failed to parse valid NPY header");
    assert_eq!(parsed.dtype, TensorDType::F32);
    assert_eq!(parsed.shape, vec![2, 3]);
    assert_eq!(parsed.fortran_order, false);
    assert_eq!(parsed.size_bytes, 6 * 4);

    let data_slice = &npy_bytes[parsed.data_offset..parsed.data_offset + parsed.size_bytes];
    let mut read_floats = Vec::new();
    for chunk in data_slice.chunks_exact(4) {
        read_floats.push(f32::from_le_bytes(chunk.try_into().unwrap()));
    }
    assert_eq!(read_floats, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_pytorch_checkpoint_loading() {
    // Use Python to generate a reference PyTorch .pt checkpoint with multiple tensors
    let temp_dir = std::env::temp_dir();
    let pt_path = temp_dir.join("test_model_checkpoint.pt");

    let script = format!(
        r#"
import torch
sd = {{
    'encoder.weight': torch.tensor([[1.5, 2.5], [3.5, 4.5]], dtype=torch.float32),
    'encoder.bias': torch.tensor([0.25, -0.75], dtype=torch.float32),
    'head.weight': torch.tensor([[10.0, 20.0, 30.0]], dtype=torch.float32),
}}
torch.save(sd, '{}')
"#,
        pt_path.display()
    );

    let status = Command::new("python3")
        .arg("-c")
        .arg(&script)
        .status();

    // If python3/torch is not available in test environment, skip gracefully
    if status.is_err() || !status.unwrap().success() {
        eprintln!("python3 or torch not available, skipping live torch test");
        return;
    }

    // Open via PyTorchCheckpoint
    let ckpt = PyTorchCheckpoint::open(&pt_path).expect("failed to open PyTorch checkpoint");
    assert!(ckpt.tensors.contains_key("encoder.weight"));
    assert!(ckpt.tensors.contains_key("encoder.bias"));
    assert!(ckpt.tensors.contains_key("head.weight"));

    // Verify encoder.weight
    let (tv_w, slice_w) = ckpt.get_tensor("encoder.weight").expect("missing encoder.weight");
    assert_eq!(tv_w.shape, vec![2, 2]);
    assert_eq!(tv_w.dtype, TensorDType::F32);
    let floats_w: Vec<f32> = slice_w
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(floats_w, vec![1.5, 2.5, 3.5, 4.5]);

    // Verify encoder.bias
    let (tv_b, slice_b) = ckpt.get_tensor("encoder.bias").expect("missing encoder.bias");
    assert_eq!(tv_b.shape, vec![2]);
    assert_eq!(tv_b.dtype, TensorDType::F32);
    let floats_b: Vec<f32> = slice_b
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(floats_b, vec![0.25, -0.75]);

    // Verify unified table access
    let table = UnifiedTensorTable::open(&pt_path).expect("failed to open via UnifiedTensorTable");
    let (meta, unified_slice) = table.get_tensor("head.weight").expect("missing head.weight");
    assert_eq!(meta.shape, vec![1, 3]);
    let floats_head: Vec<f32> = unified_slice
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(floats_head, vec![10.0, 20.0, 30.0]);

    let _ = std::fs::remove_file(pt_path);
}

#[test]
fn test_npz_archive_loading() {
    let temp_dir = std::env::temp_dir();
    let npz_path = temp_dir.join("test_tensors.npz");

    let script = format!(
        r#"
import zipfile, struct

def make_npy(shape, floats):
    hdr = "{{'descr': '<f4', 'fortran_order': False, 'shape': " + shape + ", }}"
    h_len = len(hdr) + 1
    pad = (64 - ((10 + h_len) % 64)) % 64
    padded = hdr + ' ' * pad + '\n'
    raw = b'\x93NUMPY\x01\x00' + struct.pack('<H', len(padded)) + padded.encode('latin1')
    raw += struct.pack(f'<{{len(floats)}}f', *floats)
    return raw

with zipfile.ZipFile('{}', 'w', compression=zipfile.ZIP_STORED) as zf:
    zf.writestr('w1.npy', make_npy('(2, 2)', [1.0, 2.0, 3.0, 4.0]))
    zf.writestr('b1.npy', make_npy('(2,)', [0.5, -0.5]))
"#,
        npz_path.display()
    );

    let status = Command::new("python3")
        .arg("-c")
        .arg(&script)
        .status();

    if status.is_err() || !status.unwrap().success() {
        eprintln!("python3 not available, skipping npz test");
        return;
    }

    let npz = NpzFile::open(&npz_path).expect("failed to open npz file");
    assert!(npz.tensors.contains_key("w1"));
    assert!(npz.tensors.contains_key("b1"));

    let (tv_w, slice_w) = npz.get_tensor("w1").expect("missing w1");
    assert_eq!(tv_w.shape, vec![2, 2]);
    let floats_w: Vec<f32> = slice_w
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(floats_w, vec![1.0, 2.0, 3.0, 4.0]);

    // Test through UnifiedTensorTable
    let table = UnifiedTensorTable::open(&npz_path).expect("failed to open npz via UnifiedTensorTable");
    let (meta, slice_b) = table.get_tensor("b1").expect("missing b1");
    assert_eq!(meta.shape, vec![2]);
    let floats_b: Vec<f32> = slice_b
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
        .collect();
    assert_eq!(floats_b, vec![0.5, -0.5]);

    let _ = std::fs::remove_file(npz_path);
}
