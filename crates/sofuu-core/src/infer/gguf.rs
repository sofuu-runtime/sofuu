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
    /// Full file content in anonymous RSS (`SOFUU_RESIDENT_WEIGHTS=1`):
    /// decode then never page-faults, at the cost of pinning ~model-size RAM.
    pub resident: Option<Vec<u8>>,
    pub version: u32,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: HashMap<String, TensorInfo>,
    pub data_offset: usize,
}

impl GgufFile {
    /// Bytes backing tensor offsets: the resident copy when enabled, else the mmap.
    pub fn as_bytes(&self) -> &[u8] {
        match &self.resident {
            Some(v) => v,
            None => self.mmap.as_slice(),
        }
    }

    pub fn open<P: AsRef<std::path::Path>>(path: P) -> io::Result<Self> {
        let path_ref = path.as_ref();
        let mmap = std::sync::Arc::new(MmapFile::open(path_ref)?);
        let resident = if std::env::var("SOFUU_RESIDENT_WEIGHTS").as_deref() == Ok("1") {
            // Only the qwen35 backend reads this copy; every other backend
            // loads from the mmap, so the flag doubles RSS for nothing. On an
            // 8 GB Pi that is the difference between running and an OOM abort.
            eprintln!(
                "[gguf] SOFUU_RESIDENT_WEIGHTS=1: reading a second copy of the file into RAM (only the qwen35 backend uses it)"
            );
            Some(std::fs::read(path_ref)?)
        } else {
            None
        };
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

        // Counts come straight from the file: bound them by what the file can
        // physically hold before pre-allocating. A corrupt count (2^60) would
        // otherwise abort the load on the HashMap/Vec allocation.
        // Smallest possible record: 8 B name length + 4 B dim count +
        // 4 B type + 8 B offset = 24 B for a tensor; 8 + 4 + 1 = 13 B for a KV.
        if tensor_count > bytes.len() / 24 || metadata_kv_count > bytes.len() / 13 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "corrupt GGUF header: {tensor_count} tensors / {metadata_kv_count} metadata \
                     entries cannot fit in {} bytes",
                    bytes.len()
                ),
            ));
        }

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
            // GGUF tensors are at most 4-D in every model this engine loads;
            // the bound keeps `Vec::with_capacity` away from a corrupt count.
            if n_dims > 8 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("tensor '{name}' claims {n_dims} dimensions (max 8)"),
                ));
            }
            let mut shape = Vec::with_capacity(n_dims);
            for _ in 0..n_dims {
                shape.push(r.read_u64()? as usize);
            }
            let tensor_type = r.read_u32()?;
            let offset = r.read_u64()? as usize;

            // checked product: a wrapped element count makes `size_bytes` a
            // small number, and the loader would then read past the tensor.
            let n_elements = shape
                .iter()
                .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                .filter(|&n| n <= bytes.len().saturating_mul(8))
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("tensor '{name}' has an impossible element count {shape:?}"),
                    )
                })?;
            // Row-aware size when the layout is known (GGUF's dim 0 is the
            // contiguous row width); the generic per-element fallback keeps
            // the old behaviour for types decoded by other backends.
            let row_width = shape.first().copied().unwrap_or(n_elements);
            let rows = if shape.len() > 1 {
                shape[1..]
                    .iter()
                    .try_fold(1usize, |acc, &d| acc.checked_mul(d))
                    .ok_or_else(|| {
                        io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("tensor '{name}' row count overflows"),
                        )
                    })?
            } else {
                1
            };
            let size_bytes = match packed_row_bytes(tensor_type, row_width) {
                Some(row) => row.saturating_mul(rows),
                None => super::quant::type_row_size(tensor_type, n_elements),
            };

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

        // Alignment (default 32 bytes). The round-down mask below is only
        // valid for a power of two, so a corrupt value falls back instead of
        // producing a nonsense data_offset.
        let alignment = match metadata.get("general.alignment") {
            Some(&GgufValue::Uint32(a)) if a != 0 => a as usize,
            Some(&GgufValue::Uint64(a)) if a != 0 => a as usize,
            _ => 32,
        };
        let alignment = if alignment.is_power_of_two() && alignment <= (1 << 20) {
            alignment
        } else {
            32
        };

        let current_pos = r.pos;
        let data_offset = current_pos
            .checked_add(alignment - 1)
            .map(|p| p & !(alignment - 1))
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "GGUF data offset overflow"))?;

        // Every tensor must lie inside the mapped file. This is what turns a
        // truncated download into a clean load error: without it the loader
        // builds fine and the first matmul reads past the end of the mapping.
        let file_len = bytes.len();
        for info in tensors.values() {
            let start = data_offset.checked_add(info.offset).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("tensor '{}' offset overflows", info.name),
                )
            })?;
            let end = start.checked_add(info.size_bytes).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("tensor '{}' size overflows", info.name),
                )
            })?;
            if end > file_len {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "truncated or corrupt GGUF: tensor '{}' needs bytes {start}..{end} but the \
                         file is {file_len} B",
                        info.name
                    ),
                ));
            }
        }

        Ok(Self {
            resident,
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
        let start = self.data_offset.checked_add(info.offset)?;
        self.as_bytes()
            .get(start..start.checked_add(info.size_bytes)?)
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

/// Packed bytes for one row of `row_width` values, for every tensor type a
/// GGUF loader here can decode. `None` means the type is unknown to this
/// module: the tensor-bounds check is then skipped instead of guessed (a
/// wrong row size would reject valid models — the prism codecs 142/143 used
/// to come out 4 bytes per value from the generic fallback).
fn packed_row_bytes(type_id: u32, row_width: usize) -> Option<usize> {
    match type_id {
        0 => Some(row_width.saturating_mul(4)),   // F32
        1 => Some(row_width.saturating_mul(2)),   // F16
        2 => Some((row_width / 32) * 18),         // Q4_0
        8 => Some((row_width / 32) * 34),         // Q8_0
        11 => Some((row_width / 256) * 110),      // Q3_K
        12 => Some((row_width / 256) * 144),      // Q4_K
        14 => Some((row_width / 256) * 210),      // Q6_K
        30 => Some(row_width.saturating_mul(2)),  // BF16
        142 => Some(row_width.div_ceil(128) * 34), // PQ2_0 (prism ternary)
        143 => Some(row_width.div_ceil(128) * 28), // PTQ1_0 (prism ternary)
        _ => None,
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
        // checked_add, not `pos + len`: a corrupt 64-bit string length can wrap
        // usize and slip past the EOF test, and the slice that follows would
        // panic — fatal under the release profile's panic = "abort".
        let end = match self.pos.checked_add(len) {
            Some(end) => end,
            None => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "GGUF length overflows file offset",
                ))
            }
        };
        if end > self.bytes.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "unexpected EOF in GGUF file",
            ));
        }
        let slice = &self.bytes[self.pos..end];
        self.pos = end;
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
    read_value_at(r, val_type, 0)
}

/// Nested metadata arrays recurse; a hand-crafted file of array headers would
/// otherwise overflow the stack before any length check can reject it.
const MAX_VALUE_DEPTH: usize = 8;

fn read_value_at(r: &mut ByteReader<'_>, val_type: u32, depth: usize) -> io::Result<GgufValue> {
    if depth > MAX_VALUE_DEPTH {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("GGUF metadata nesting exceeds {MAX_VALUE_DEPTH} levels"),
        ));
    }
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
            // Each string costs at least its 8-byte length prefix; a corrupt
            // `len` must fail here, before `with_capacity` tries to allocate it.
            if item_type == 8 {
                if len > r.bytes.len().saturating_sub(r.pos) / 8 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("string array length {len} exceeds the file"),
                    ));
                }
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
                let total_bytes = len.checked_mul(elem_size).ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("array of {len} type-{item_type} values overflows its byte size"),
                    )
                })?;
                let raw_slice = r.read_bytes(total_bytes)?;
                return Ok(GgufValue::RawArray {
                    item_type,
                    len,
                    data: raw_slice.to_vec(),
                });
            }
            if len > r.bytes.len().saturating_sub(r.pos) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("array length {len} exceeds the file"),
                ));
            }
            let mut arr = Vec::with_capacity(len);
            for _ in 0..len {
                arr.push(read_value_at(r, item_type, depth + 1)?);
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

    fn put_u32(v: &mut Vec<u8>, x: u32) {
        v.extend_from_slice(&x.to_le_bytes());
    }
    fn put_u64(v: &mut Vec<u8>, x: u64) {
        v.extend_from_slice(&x.to_le_bytes());
    }
    fn put_str(v: &mut Vec<u8>, s: &str) {
        put_u64(v, s.len() as u64);
        v.extend_from_slice(s.as_bytes());
    }

    /// Minimal well-formed GGUF v3: one f32 tensor `w` of `elems` values plus
    /// its payload. `truncate` removes bytes from the end so the file can be
    /// made inconsistent the way an interrupted download is.
    fn synthetic_gguf(elems: usize, truncate: usize) -> Vec<u8> {
        let mut b = Vec::new();
        put_u32(&mut b, GGUF_MAGIC);
        put_u32(&mut b, 3);
        put_u64(&mut b, 1); // tensor count
        put_u64(&mut b, 0); // metadata count
        put_str(&mut b, "w");
        put_u32(&mut b, 1); // n_dims
        put_u64(&mut b, elems as u64);
        put_u32(&mut b, super::super::quant::GGML_TYPE_F32);
        put_u64(&mut b, 0); // tensor offset
        while b.len() % 32 != 0 {
            b.push(0);
        }
        b.extend(std::iter::repeat(0u8).take(elems * 4));
        b.truncate(b.len().saturating_sub(truncate));
        b
    }

    fn write_temp(tag: &str, bytes: &[u8]) -> std::path::PathBuf {
        let p = std::env::temp_dir().join(format!("sofuu-gguf-{tag}-{}.gguf", std::process::id()));
        std::fs::write(&p, bytes).expect("write temp gguf");
        p
    }

    /// A truncated model must fail the load with an error. Before the tensor
    /// bounds check it loaded fine and the first matmul read past the mapping
    /// (SIGSEGV or garbage logits on device).
    #[test]
    fn test_truncated_gguf_is_a_clean_error() {
        let p = write_temp("trunc", &synthetic_gguf(64, 7));
        let res = GgufFile::open(&p);
        let _ = std::fs::remove_file(&p);
        match res {
            Err(e) => assert_eq!(e.kind(), io::ErrorKind::InvalidData, "unexpected error: {e}"),
            Ok(g) => panic!("truncated GGUF opened as valid ({} tensors)", g.tensors.len()),
        }

        // The same builder without truncation loads, and the slice is real.
        let p = write_temp("full", &synthetic_gguf(64, 0));
        let g = GgufFile::open(&p).expect("valid synthetic GGUF");
        let _ = std::fs::remove_file(&p);
        assert_eq!(g.get_tensor_data("w").map(|s| s.len()), Some(256));
    }

    /// Corrupt header fields must be rejected, not panic. Each of these used
    /// to abort the load (capacity overflow / arithmetic overflow under the
    /// release profile's panic = "abort") or slip through silently.
    #[test]
    fn test_corrupt_gguf_header_is_rejected() {
        // 1. Tensor count larger than the file can hold.
        let mut b = Vec::new();
        put_u32(&mut b, GGUF_MAGIC);
        put_u32(&mut b, 3);
        put_u64(&mut b, u64::MAX);
        put_u64(&mut b, 0);
        let p = write_temp("count", &b);
        let res = GgufFile::open(&p);
        let _ = std::fs::remove_file(&p);
        assert!(res.is_err(), "absurd tensor count must be rejected");

        // 2. A metadata string whose 64-bit length wraps the file offset.
        let mut b = Vec::new();
        put_u32(&mut b, GGUF_MAGIC);
        put_u32(&mut b, 3);
        put_u64(&mut b, 0);
        put_u64(&mut b, 1);
        put_u64(&mut b, u64::MAX); // key length
        let p = write_temp("strlen", &b);
        let res = GgufFile::open(&p);
        let _ = std::fs::remove_file(&p);
        assert!(res.is_err(), "overflowing string length must be rejected");

        // 3. A tensor claiming more dimensions than any real model uses.
        let mut b = Vec::new();
        put_u32(&mut b, GGUF_MAGIC);
        put_u32(&mut b, 3);
        put_u64(&mut b, 1);
        put_u64(&mut b, 0);
        put_str(&mut b, "w");
        put_u32(&mut b, 9); // n_dims > 8
        for _ in 0..9 {
            put_u64(&mut b, 1);
        }
        put_u32(&mut b, super::super::quant::GGML_TYPE_F32);
        put_u64(&mut b, 0);
        let p = write_temp("dims", &b);
        let res = GgufFile::open(&p);
        let _ = std::fs::remove_file(&p);
        assert!(res.is_err(), "9-dimensional tensor must be rejected");

        // 4. An array whose element count overflows its byte size.
        let mut b = Vec::new();
        put_u32(&mut b, GGUF_MAGIC);
        put_u32(&mut b, 3);
        put_u64(&mut b, 0);
        put_u64(&mut b, 1); // one KV
        put_str(&mut b, "x");
        put_u32(&mut b, 9); // array
        put_u32(&mut b, 4); // of u32
        put_u64(&mut b, u64::MAX);
        let p = write_temp("arrlen", &b);
        let res = GgufFile::open(&p);
        let _ = std::fs::remove_file(&p);
        assert!(res.is_err(), "overflowing array length must be rejected");
    }

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
