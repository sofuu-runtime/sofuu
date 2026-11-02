// crates/sofuu-core/src/infer/tensor_table.rs — Universal Zero-Copy Tensor Registry.
//
// Unifies GGUF, SafeTensors, PyTorch (.pt/.pth), and NumPy (.npz/.npy) into a single
// standardized tensor registry with zero memory copies.

use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::gguf::{GgufFile, GGUF_MAGIC};
use super::npy::{NpzFile, TensorDType};
use super::pytorch::PyTorchCheckpoint;
use super::safetensors::{SafeDType, SafeTensorsFile};
use super::quant::{GGML_TYPE_F16, GGML_TYPE_F32};

#[derive(Debug, Clone)]
pub struct UnifiedTensorMeta {
    pub name: String,
    pub dtype: TensorDType,
    pub shape: Vec<usize>,
    pub data_offset: usize, // absolute offset inside mmap
    pub size_bytes: usize,
}

#[derive(Clone)]
pub enum UnifiedTensorData {
    Heap(Arc<Vec<u8>>),
    Mmap {
        mmap: Arc<crate::infer::mmap::MmapFile>,
        offset: usize,
        size: usize,
    },
}

impl UnifiedTensorData {
    #[inline(always)]
    pub fn as_slice(&self) -> &[u8] {
        match self {
            Self::Heap(v) => v.as_slice(),
            Self::Mmap { mmap, offset, size } => &mmap.as_slice()[*offset..*offset + *size],
        }
    }
}

pub enum TensorSource {
    SafeTensors(SafeTensorsFile),
    Gguf(GgufFile),
    PyTorch(PyTorchCheckpoint),
    Npz(NpzFile),
}

pub struct UnifiedTensorTable {
    pub source: TensorSource,
    pub path: Option<PathBuf>,
}

impl UnifiedTensorTable {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path_ref = path.as_ref();
        let path_buf = path_ref.to_path_buf();
        let ext = path_ref.extension().and_then(|s| s.to_str()).unwrap_or("").to_lowercase();

        // 1. Extension-guided dispatch
        if ext == "safetensors" {
            let st = SafeTensorsFile::open(path_ref)?;
            return Ok(Self { source: TensorSource::SafeTensors(st), path: Some(path_buf) });
        } else if ext == "gguf" {
            let gg = GgufFile::open(path_ref).map_err(|e| e.to_string())?;
            return Ok(Self { source: TensorSource::Gguf(gg), path: Some(path_buf) });
        } else if ext == "pt" || ext == "pth" {
            let pt = PyTorchCheckpoint::open(path_ref)?;
            return Ok(Self { source: TensorSource::PyTorch(pt), path: Some(path_buf) });
        } else if ext == "npz" {
            let npz = NpzFile::open(path_ref)?;
            return Ok(Self { source: TensorSource::Npz(npz), path: Some(path_buf) });
        }

        // 2. Magic byte sniffing fallback
        let mut header = [0u8; 8];
        let mut f = File::open(path_ref).map_err(|e| format!("cannot open file {}: {}", path_ref.display(), e))?;
        let bytes_read = f.read(&mut header).unwrap_or(0);

        if bytes_read >= 4 && u32::from_le_bytes(header[0..4].try_into().unwrap()) == GGUF_MAGIC {
            let gg = GgufFile::open(path_ref).map_err(|e| e.to_string())?;
            return Ok(Self { source: TensorSource::Gguf(gg), path: Some(path_buf) });
        }

        if bytes_read >= 4 && &header[0..4] == b"PK\x03\x04" {
            // ZIP archive: try PyTorch first, then NPZ
            if let Ok(pt) = PyTorchCheckpoint::open(path_ref) {
                return Ok(Self { source: TensorSource::PyTorch(pt), path: Some(path_buf) });
            }
            if let Ok(npz) = NpzFile::open(path_ref) {
                return Ok(Self { source: TensorSource::Npz(npz), path: Some(path_buf) });
            }
        }

        // Default try SafeTensors
        if let Ok(st) = SafeTensorsFile::open(path_ref) {
            return Ok(Self { source: TensorSource::SafeTensors(st), path: Some(path_buf) });
        }

        Err(format!("unrecognized or unsupported tensor file format: {}", path_ref.display()))
    }

    #[inline]
    pub fn get_tensor_data(&self, name: &str) -> Option<(UnifiedTensorMeta, UnifiedTensorData)> {
        match &self.source {
            TensorSource::SafeTensors(st) => {
                let (tv, _) = st.get_tensor(name)?;
                let meta = UnifiedTensorMeta {
                    name: name.to_string(),
                    dtype: safe_dtype_to_tensor_dtype(tv.dtype),
                    shape: tv.shape.clone(),
                    data_offset: tv.data_offset,
                    size_bytes: tv.size_bytes,
                };
                let data = UnifiedTensorData::Mmap {
                    mmap: st.mmap.clone(),
                    offset: tv.data_offset,
                    size: tv.size_bytes,
                };
                Some((meta, data))
            }
            TensorSource::Gguf(gg) => {
                let info = gg.tensors.get(name)?;
                let abs_offset = gg.data_offset + info.offset;
                let _slice = gg.mmap.slice(abs_offset, info.size_bytes)?;
                let meta = UnifiedTensorMeta {
                    name: name.to_string(),
                    dtype: ggml_type_to_tensor_dtype(info.tensor_type),
                    shape: info.shape.clone(),
                    data_offset: abs_offset,
                    size_bytes: info.size_bytes,
                };
                Some((meta, UnifiedTensorData::Mmap {
                    mmap: gg.mmap.clone(),
                    offset: abs_offset,
                    size: info.size_bytes,
                }))
            }
            TensorSource::PyTorch(pt) => {
                let (tv, slice) = pt.get_tensor(name)?;
                let meta = UnifiedTensorMeta {
                    name: name.to_string(),
                    dtype: tv.dtype,
                    shape: tv.shape.clone(),
                    data_offset: tv.data_offset,
                    size_bytes: tv.size_bytes,
                };
                Some((meta, UnifiedTensorData::Heap(Arc::new(slice.to_vec()))))
            }
            TensorSource::Npz(npz) => {
                let (tv, slice) = npz.get_tensor(name)?;
                let meta = UnifiedTensorMeta {
                    name: name.to_string(),
                    dtype: tv.dtype,
                    shape: tv.shape.clone(),
                    data_offset: tv.data_offset,
                    size_bytes: tv.size_bytes,
                };
                Some((meta, UnifiedTensorData::Heap(Arc::new(slice.to_vec()))))
            }
        }
    }

    #[inline]
    pub fn get_tensor(&self, name: &str) -> Option<(UnifiedTensorMeta, &[u8])> {
        match &self.source {
            TensorSource::SafeTensors(st) => {
                let (tv, slice) = st.get_tensor(name)?;
                let meta = UnifiedTensorMeta {
                    name: name.to_string(),
                    dtype: safe_dtype_to_tensor_dtype(tv.dtype),
                    shape: tv.shape.clone(),
                    data_offset: tv.data_offset,
                    size_bytes: tv.size_bytes,
                };
                Some((meta, slice))
            }
            TensorSource::Gguf(gg) => {
                let info = gg.tensors.get(name)?;
                let abs_offset = gg.data_offset + info.offset;
                let slice = gg.mmap.slice(abs_offset, info.size_bytes)?;
                let meta = UnifiedTensorMeta {
                    name: name.to_string(),
                    dtype: ggml_type_to_tensor_dtype(info.tensor_type),
                    shape: info.shape.clone(),
                    data_offset: abs_offset,
                    size_bytes: info.size_bytes,
                };
                Some((meta, slice))
            }
            TensorSource::PyTorch(pt) => {
                let (tv, slice) = pt.get_tensor(name)?;
                let meta = UnifiedTensorMeta {
                    name: name.to_string(),
                    dtype: tv.dtype,
                    shape: tv.shape.clone(),
                    data_offset: tv.data_offset,
                    size_bytes: tv.size_bytes,
                };
                Some((meta, slice))
            }
            TensorSource::Npz(npz) => {
                let (tv, slice) = npz.get_tensor(name)?;
                let meta = UnifiedTensorMeta {
                    name: name.to_string(),
                    dtype: tv.dtype,
                    shape: tv.shape.clone(),
                    data_offset: tv.data_offset,
                    size_bytes: tv.size_bytes,
                };
                Some((meta, slice))
            }
        }
    }

    pub fn tensor_names(&self) -> Vec<String> {
        match &self.source {
            TensorSource::SafeTensors(st) => st.tensor_names().cloned().collect(),
            TensorSource::Gguf(gg) => gg.tensors.keys().cloned().collect(),
            TensorSource::PyTorch(pt) => pt.tensor_names().cloned().collect(),
            TensorSource::Npz(npz) => npz.tensor_names().cloned().collect(),
        }
    }
}

fn safe_dtype_to_tensor_dtype(d: SafeDType) -> TensorDType {
    match d {
        SafeDType::F32 => TensorDType::F32,
        SafeDType::F16 => TensorDType::F16,
        SafeDType::BF16 => TensorDType::BF16,
        SafeDType::I64 => TensorDType::I64,
        SafeDType::I32 => TensorDType::I32,
        SafeDType::I16 => TensorDType::I16,
        SafeDType::I8 => TensorDType::I8,
        SafeDType::U64 => TensorDType::U64,
        SafeDType::U32 => TensorDType::U32,
        SafeDType::U16 => TensorDType::U16,
        SafeDType::U8 => TensorDType::U8,
        SafeDType::Bool => TensorDType::Bool,
        SafeDType::Unknown => TensorDType::Unknown,
    }
}

fn ggml_type_to_tensor_dtype(t: u32) -> TensorDType {
    match t {
        GGML_TYPE_F32 => TensorDType::F32,
        GGML_TYPE_F16 => TensorDType::F16,
        16 => TensorDType::I8,
        17 => TensorDType::I16,
        18 => TensorDType::I32,
        30 => TensorDType::BF16,
        _ => TensorDType::Unknown, // Quantized formats retain GGML type in model logic
    }
}
