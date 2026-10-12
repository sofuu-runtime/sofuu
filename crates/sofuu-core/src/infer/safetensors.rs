// crates/sofuu-core/src/infer/safetensors.rs — Zero-copy SafeTensors file parser.
//
// Compliant with the SafeTensors specification (HuggingFace / MLX).
// Parses the 8-byte LE header length, deserializes metadata JSON, and maps
// all tensors directly into memory with zero data copies.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use serde::Deserialize;

use super::mmap::MmapFile;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SafeDType {
    F32,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
    Bool,
    Unknown,
}

impl SafeDType {
    pub fn from_str(s: &str) -> Self {
        match s {
            "F32" => Self::F32,
            "F16" => Self::F16,
            "BF16" => Self::BF16,
            "I64" => Self::I64,
            "I32" => Self::I32,
            "I16" => Self::I16,
            "I8" => Self::I8,
            "U64" => Self::U64,
            "U32" => Self::U32,
            "U16" => Self::U16,
            "U8" => Self::U8,
            "BOOL" => Self::Bool,
            _ => Self::Unknown,
        }
    }

    pub fn item_size_bytes(&self) -> usize {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::BF16 | Self::I16 | Self::U16 => 2,
            Self::I64 | Self::U64 => 8,
            Self::I8 | Self::U8 | Self::Bool => 1,
            Self::Unknown => 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SafeTensorView {
    pub dtype: SafeDType,
    pub shape: Vec<usize>,
    pub data_offset: usize, // absolute offset inside mmap
    pub size_bytes: usize,
}

#[derive(Deserialize)]
struct RawTensorEntry {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

pub struct SafeTensorsFile {
    pub mmap: Arc<MmapFile>,
    pub tensors: HashMap<String, SafeTensorView>,
    pub metadata: HashMap<String, String>,
    pub data_offset: usize,
}

impl SafeTensorsFile {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path_ref = path.as_ref();
        let mmap = Arc::new(MmapFile::open(path_ref)
            .map_err(|e| format!("failed to mmap safetensors file {}: {}", path_ref.display(), e))?);

        if mmap.len() < 8 {
            return Err("safetensors file too small (less than 8 bytes)".to_string());
        }

        let header_len_bytes = mmap.slice(0, 8).ok_or("failed to read header length")?;
        let header_len = u64::from_le_bytes(header_len_bytes.try_into().unwrap()) as usize;

        if 8 + header_len > mmap.len() {
            return Err(format!(
                "invalid safetensors header length {} (exceeds file size {})",
                header_len,
                mmap.len()
            ));
        }

        let header_bytes = mmap.slice(8, header_len).ok_or("failed to read json header")?;
        let header_str = std::str::from_utf8(header_bytes)
            .map_err(|e| format!("invalid UTF-8 in safetensors header: {}", e))?;

        let raw_map: HashMap<String, serde_json::Value> = serde_json::from_str(header_str)
            .map_err(|e| format!("failed to parse safetensors JSON header: {}", e))?;

        let data_offset = 8 + header_len;
        let mut tensors = HashMap::new();
        let mut metadata = HashMap::new();

        for (k, v) in raw_map {
            if k == "__metadata__" {
                if let Some(obj) = v.as_object() {
                    for (mk, mv) in obj {
                        if let Some(s) = mv.as_str() {
                            metadata.insert(mk.clone(), s.to_string());
                        } else {
                            metadata.insert(mk.clone(), mv.to_string());
                        }
                    }
                }
                continue;
            }

            if let Ok(entry) = serde_json::from_value::<RawTensorEntry>(v) {
                let start = entry.data_offsets[0];
                let end = entry.data_offsets[1];
                let size_bytes = end.saturating_sub(start);
                let abs_offset = data_offset + start;

                tensors.insert(
                    k,
                    SafeTensorView {
                        dtype: SafeDType::from_str(&entry.dtype),
                        shape: entry.shape,
                        data_offset: abs_offset,
                        size_bytes,
                    },
                );
            }
        }

        Ok(Self {
            mmap,
            tensors,
            metadata,
            data_offset,
        })
    }

    #[inline]
    pub fn get_tensor(&self, name: &str) -> Option<(&SafeTensorView, &[u8])> {
        let tv = self.tensors.get(name)?;
        let slice = self.mmap.slice(tv.data_offset, tv.size_bytes)?;
        Some((tv, slice))
    }

    #[inline]
    pub fn tensor_names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }
}
