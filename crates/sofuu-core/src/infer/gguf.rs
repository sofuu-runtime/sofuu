// crates/sofuu-core/src/infer/gguf.rs — Zero-copy GGUF v2/v3 file format parser.
//
// Reads header, metadata key-values, tensor info, and maps tensor data slices.

use std::collections::HashMap;
use std::io;

use super::mmap::MmapFile;

pub const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in little-endian

#[derive(Debug, Clone, PartialEq)]
pub enum GgufValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    StringArray(Vec<String>),
    RawArray { item_type: u32, len: usize, data: Vec<u8> },
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub tensor_type: u32,
    pub offset: usize,
    pub size_bytes: usize,
}

pub struct GgufFile {
    pub mmap: std::sync::Arc<MmapFile>,
    pub version: u32,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: HashMap<String, TensorInfo>,
    pub data_offset: usize,
}

impl GgufFile {
    pub fn open<P: AsRef<std::path::Path>>(path: P) -> io::Result<Self> {
        let mmap = std::sync::Arc::new(MmapFile::open(path)?);
        let bytes = mmap.as_slice();
        let mut r = ByteReader::new(bytes);

        // 1. Header
        let magic = r.read_u32()?;
        if magic != GGUF_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid GGUF magic: {magic:#x}"),
            ));
        }

        let version = r.read_u32()?;
        if version != 2 && version != 3 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported GGUF version: {version}"),
            ));
        }

        let tensor_count = r.read_u64()? as usize;
        let metadata_kv_count = r.read_u64()? as usize;

        // 2. Metadata KV
        let mut metadata = HashMap::with_capacity(metadata_kv_count);
        for _ in 0..metadata_kv_count {
            let key = r.read_string()?;
            let val_type = r.read_u32()?;
            let val = read_value(&mut r, val_type)?;
            metadata.insert(key, val);
        }

        // 3. Tensor Info
        let mut tensors = HashMap::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = r.read_string()?;
            let n_dims = r.read_u32()? as usize;
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(r.read_u64()? as usize);
            }
            let tensor_type = r.read_u32()?;
            let offset = r.read_u64()? as usize;

            let n_elements: usize = shape.iter().product();
            let size_bytes = super::quant::type_row_size(tensor_type, n_elements);

            tensors.insert(
                name.clone(),
                TensorInfo {
                    name,
                    shape,
                    tensor_type,
                    offset,
                    size_bytes,
                },
            );
        }

        // Alignment (default 32 bytes)
        let alignment = match metadata.get("general.alignment") {
            Some(&GgufValue::Uint32(a)) => a as usize,
            Some(&GgufValue::Uint64(a)) => a as usize,
            _ => 32,
        };

        let current_pos = r.pos;
        let data_offset = (current_pos + alignment - 1) & !(alignment - 1);

        Ok(Self {
            mmap,
            version,
            metadata,
            tensors,
            data_offset,
        })
    }

    /// Retrieve raw bytes slice of tensor by name.
    pub fn get_tensor_data(&self, name: &str) -> Option<&[u8]> {
        let info = self.tensors.get(name)?;
        let start = self.data_offset + info.offset;
        self.mmap.slice(start, info.size_bytes)
    }

    pub fn get_u32(&self, key: &str) -> Option<u32> {
        match self.metadata.get(key)? {
            GgufValue::Uint32(v) => Some(*v),
            GgufValue::Int32(v) => Some(*v as u32),
            GgufValue::Uint64(v) => Some(*v as u32),
            GgufValue::Int64(v) => Some(*v as u32),
            _ => None,
        }
    }

    pub fn get_f32(&self, key: &str) -> Option<f32> {
        match self.metadata.get(key)? {
            GgufValue::Float32(v) => Some(*v),
            GgufValue::Float64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn get_str(&self, key: &str) -> Option<&str> {
        match self.metadata.get(key)? {
            GgufValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn get_string_array(&self, key: &str) -> Option<&[String]> {
        match self.metadata.get(key)? {
            GgufValue::StringArray(items) => Some(items.as_slice()),
            _ => None,
        }
    }

    pub fn take_string_array(&mut self, key: &str) -> Option<Vec<String>> {
        match self.metadata.remove(key)? {
            GgufValue::StringArray(items) => Some(items),
            GgufValue::Array(items) => {
                let mut out = Vec::with_capacity(items.len());
                for it in items {
                    if let GgufValue::String(s) = it {
                        out.push(s);
                    }
                }
                Some(out)
            }
            _ => None,
        }
    }
}

// ── Fast slice-based reader ──────────────────────────────────────

pub struct ByteReader<'a> {
    pub bytes: &'a [u8],
    pub pos: usize,
}

impl<'a> ByteReader<'a> {
    #[inline(always)]
    pub fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    #[inline(always)]
    pub fn read_bytes(&mut self, len: usize) -> io::Result<&'a [u8]> {
        if self.pos + len > self.bytes.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "unexpected EOF in GGUF file",
            ));
        }
        let slice = &self.bytes[self.pos..self.pos + len];
        self.pos += len;
        Ok(slice)
    }

    #[inline(always)]
    pub fn read_u8(&mut self) -> io::Result<u8> {
        let b = self.read_bytes(1)?;
        Ok(b[0])
    }

    #[inline(always)]
    pub fn read_u16(&mut self) -> io::Result<u16> {
        let b = self.read_bytes(2)?;
        Ok(u16::from_le_bytes(b.try_into().unwrap()))
    }

    #[inline(always)]
    pub fn read_u32(&mut self) -> io::Result<u32> {
        let b = self.read_bytes(4)?;
        Ok(u32::from_le_bytes(b.try_into().unwrap()))
    }

    #[inline(always)]
    pub fn read_u64(&mut self) -> io::Result<u64> {
        let b = self.read_bytes(8)?;
        Ok(u64::from_le_bytes(b.try_into().unwrap()))
    }

    #[inline(always)]
    pub fn read_string(&mut self) -> io::Result<String> {
        let len = self.read_u64()? as usize;
        let slice = self.read_bytes(len)?;
        match std::str::from_utf8(slice) {
            Ok(s) => Ok(s.to_string()),
            Err(_) => Ok(String::from_utf8_lossy(slice).into_owned()),
        }
    }
}

fn read_value(r: &mut ByteReader<'_>, val_type: u32) -> io::Result<GgufValue> {
    match val_type {
        0 => Ok(GgufValue::Uint8(r.read_u8()?)),
        1 => Ok(GgufValue::Int8(r.read_u8()? as i8)),
        2 => Ok(GgufValue::Uint16(r.read_u16()?)),
        3 => Ok(GgufValue::Int16(r.read_u16()? as i16)),
        4 => Ok(GgufValue::Uint32(r.read_u32()?)),
        5 => Ok(GgufValue::Int32(r.read_u32()? as i32)),
        6 => {
            let b = r.read_u32()?;
            Ok(GgufValue::Float32(f32::from_bits(b)))
        }
        7 => Ok(GgufValue::Bool(r.read_u8()? != 0)),
        8 => Ok(GgufValue::String(r.read_string()?)),
        9 => {
            let item_type = r.read_u32()?;
            let len = r.read_u64()? as usize;
            if item_type == 8 {
                let mut arr = Vec::with_capacity(len);
                for _ in 0..len {
                    arr.push(r.read_string()?);
                }
                return Ok(GgufValue::StringArray(arr));
            }
            let elem_size = match item_type {
                0 | 1 | 7 => 1,
                2 | 3 => 2,
                4 | 5 | 6 => 4,
                10 | 11 | 12 => 8,
                _ => 0,
            };
            if elem_size > 0 {
                let total_bytes = len.checked_mul(elem_size).expect("array size overflow");
                let raw_slice = r.read_bytes(total_bytes)?;
                return Ok(GgufValue::RawArray {
                    item_type,
                    len,
                    data: raw_slice.to_vec(),
                });
            }
            let mut arr = Vec::with_capacity(len);
            for _ in 0..len {
                arr.push(read_value(r, item_type)?);
            }
            Ok(GgufValue::Array(arr))
        }
        10 => Ok(GgufValue::Uint64(r.read_u64()?)),
        11 => Ok(GgufValue::Int64(r.read_u64()? as i64)),
        12 => {
            let b = r.read_u64()?;
            Ok(GgufValue::Float64(f64::from_bits(b)))
        }
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unknown GGUF value type {val_type}"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inspect_user_model() {
        let path = "/Users/priyanshuboruah/Downloads/Qwen-3-0.6B-Claude-4.7-Opus-Distilled.i1-Q4_K_M.gguf";
        if !std::path::Path::new(path).exists() {
            println!("Target model not found at {}", path);
            return;
        }
        let gguf = GgufFile::open(path).expect("open gguf");
        println!("=== GGUF Model Info ===");
        println!("Version: {}", gguf.version);
        println!("Tensors count: {}", gguf.tensors.len());
        println!("Data offset: {}", gguf.data_offset);
        if let Some(arch) = gguf.get_str("general.architecture") {
            println!("Architecture: {}", arch);
        }
        for (k, v) in &gguf.metadata {
            if !k.contains("tokens") && !k.contains("merges") && !k.contains("scores") {
                println!("  meta: {} = {:?}", k, v);
            }
        }
        println!("=== Sample Tensors & Types ===");
        let mut type_counts = HashMap::new();
        for (name, info) in &gguf.tensors {
            *type_counts.entry(info.tensor_type).or_insert(0usize) += 1;
            if name.starts_with("blk.0") || name.starts_with("token_embd") || name.starts_with("output") {
                println!("  tensor: {} shape={:?} type={}", name, info.shape, info.tensor_type);
            }
        }
        println!("=== Tensor Type Summary ===");
        for (t, count) in type_counts {
            println!("  GGML Type {}: {} tensors", t, count);
        }
    }
}
