// crates/sofuu-core/src/infer/npy.rs — Zero-copy NumPy .npy and .npz format parser.
//
// Compliant with NumPy NPY v1.0 and v2.0 format specifications.
// Directly indexes into memory-mapped raw tensor arrays with zero copies.

use std::collections::HashMap;
use std::path::Path;

use super::mmap::MmapFile;
use super::zip_reader::ZipArchive;

pub const NPY_MAGIC: &[u8; 6] = b"\x93NUMPY";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TensorDType {
    F32,
    F16,
    BF16,
    F64,
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

impl TensorDType {
    pub fn from_numpy_descr(s: &str) -> Self {
        let trimmed = s.trim().trim_matches(|c| c == '\'' || c == '"');
        match trimmed {
            "<f4" | ">f4" | "f4" | "=f4" | "<f" | "f" => Self::F32,
            "<f2" | ">f2" | "f2" | "=f2" | "<e" | "e" => Self::F16,
            "<f8" | ">f8" | "f8" | "=f8" | "<d" | "d" => Self::F64,
            "<i4" | ">i4" | "i4" | "=i4" | "<i" | "i" => Self::I32,
            "<i8" | ">i8" | "i8" | "=i8" | "<q" | "q" => Self::I64,
            "<i2" | ">i2" | "i2" | "=i2" | "<h" | "h" => Self::I16,
            "|i1" | "i1" | "b" => Self::I8,
            "|u1" | "u1" | "B" => Self::U8,
            "<u2" | ">u2" | "u2" | "=u2" | "<H" | "H" => Self::U16,
            "<u4" | ">u4" | "u4" | "=u4" | "<I" | "I" => Self::U32,
            "<u8" | ">u8" | "u8" | "=u8" | "<Q" | "Q" => Self::U64,
            "|b1" | "b1" | "?" => Self::Bool,
            // PyTorch bfloat16 or custom b2 / V2
            "<b2" | "b2" | "<V2" | "V2" => Self::BF16,
            _ => Self::Unknown,
        }
    }

    pub fn item_size_bytes(&self) -> usize {
        match self {
            Self::F32 | Self::I32 | Self::U32 => 4,
            Self::F16 | Self::BF16 | Self::I16 | Self::U16 => 2,
            Self::F64 | Self::I64 | Self::U64 => 8,
            Self::I8 | Self::U8 | Self::Bool => 1,
            Self::Unknown => 1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct NpyHeader {
    pub dtype: TensorDType,
    pub shape: Vec<usize>,
    pub fortran_order: bool,
    pub data_offset: usize, // relative to the start of the npy slice
    pub size_bytes: usize,
}

pub fn parse_npy_header(bytes: &[u8]) -> Result<NpyHeader, String> {
    if bytes.len() < 10 {
        return Err("NPY header too small (< 10 bytes)".to_string());
    }

    if &bytes[0..6] != NPY_MAGIC {
        return Err("invalid NPY magic number".to_string());
    }

    let major = bytes[6];
    let _minor = bytes[7];

    let (header_len, header_start) = if major == 1 {
        let hlen = u16::from_le_bytes(bytes[8..10].try_into().unwrap()) as usize;
        (hlen, 10)
    } else if major >= 2 {
        if bytes.len() < 12 {
            return Err("NPY v2+ header too small".to_string());
        }
        let hlen = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        (hlen, 12)
    } else {
        return Err(format!("unsupported NPY major version {}", major));
    };

    let data_offset = header_start + header_len;
    if data_offset > bytes.len() {
        return Err(format!(
            "NPY header length {} exceeds buffer size {}",
            header_len,
            bytes.len()
        ));
    }

    let header_str = std::str::from_utf8(&bytes[header_start..data_offset])
        .map_err(|e| format!("invalid UTF-8 in NPY header: {}", e))?;

    // Extract 'descr': '...'
    let descr = parse_dict_string_field(header_str, "descr")
        .ok_or_else(|| "missing 'descr' in NPY header".to_string())?;
    let dtype = TensorDType::from_numpy_descr(&descr);

    // Extract 'fortran_order': True / False
    let fortran_order = parse_dict_bool_field(header_str, "fortran_order").unwrap_or(false);

    // Extract 'shape': (d1, d2, ...)
    let shape = parse_dict_shape_field(header_str)
        .ok_or_else(|| "missing or invalid 'shape' in NPY header".to_string())?;

    let num_elements: usize = if shape.is_empty() {
        1
    } else {
        shape.iter().product()
    };
    let size_bytes = num_elements * dtype.item_size_bytes();

    Ok(NpyHeader {
        dtype,
        shape,
        fortran_order,
        data_offset,
        size_bytes,
    })
}

fn parse_dict_string_field(s: &str, key: &str) -> Option<String> {
    let key_pat = format!("'{}':", key);
    let key_pat_d = format!("\"{}\":", key);
    let (idx, pat_len) = if let Some(i) = s.find(&key_pat) {
        (i, key_pat.len())
    } else if let Some(i) = s.find(&key_pat_d) {
        (i, key_pat_d.len())
    } else {
        return None;
    };
    let rem = &s[idx + pat_len..];
    let start_quote = rem.find(|c| c == '\'' || c == '"')?;
    let quote_char = rem.as_bytes()[start_quote] as char;
    let after_start = &rem[start_quote + 1..];
    let end_quote = after_start.find(quote_char)?;
    Some(after_start[..end_quote].to_string())
}

fn parse_dict_bool_field(s: &str, key: &str) -> Option<bool> {
    let key_pat = format!("'{}':", key);
    let key_pat_d = format!("\"{}\":", key);
    let idx = s.find(&key_pat).or_else(|| s.find(&key_pat_d))?;
    let rem = &s[idx..];
    if rem.contains("True") {
        Some(true)
    } else if rem.contains("False") {
        Some(false)
    } else {
        None
    }
}

fn parse_dict_shape_field(s: &str) -> Option<Vec<usize>> {
    let shape_idx = s.find("'shape':").or_else(|| s.find("\"shape\":"))?;
    let rem = &s[shape_idx..];
    let open_paren = rem.find('(')?;
    let close_paren = rem[open_paren..].find(')')? + open_paren;
    let inside = &rem[open_paren + 1..close_paren].trim();

    if inside.is_empty() {
        return Some(vec![]);
    }

    let mut dims = Vec::new();
    for part in inside.split(',') {
        let p = part.trim();
        if p.is_empty() {
            continue;
        }
        let d: usize = p.parse().ok()?;
        dims.push(d);
    }
    Some(dims)
}

#[derive(Debug, Clone)]
pub struct NpyTensorView {
    pub dtype: TensorDType,
    pub shape: Vec<usize>,
    pub data_offset: usize, // absolute offset inside mmap
    pub size_bytes: usize,
}

pub struct NpzFile {
    pub mmap: MmapFile,
    pub tensors: HashMap<String, NpyTensorView>,
}

impl NpzFile {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, String> {
        let path_ref = path.as_ref();
        let mmap = MmapFile::open(path_ref)
            .map_err(|e| format!("failed to mmap npz file {}: {}", path_ref.display(), e))?;

        let bytes = mmap.as_slice();
        let zip = ZipArchive::parse(bytes)
            .map_err(|e| format!("failed to parse npz archive {}: {}", path_ref.display(), e))?;

        let mut tensors = HashMap::new();

        for name in zip.entry_names() {
            let entry = match zip.get_entry(name) {
                Some(e) => e,
                None => continue,
            };

            // NPZ files store .npy array files for each tensor
            if entry.compression_method != 0 {
                // We currently require STORED (uncompressed) for zero-copy mmap.
                continue;
            }

            let entry_slice = match zip.get_uncompressed_slice(name) {
                Some(s) => s,
                None => continue,
            };

            if let Ok(hdr) = parse_npy_header(entry_slice) {
                // Strip .npy extension for standard tensor naming
                let clean_name = if name.ends_with(".npy") {
                    name[..name.len() - 4].to_string()
                } else {
                    name.clone()
                };

                let abs_offset = entry.data_offset as usize + hdr.data_offset;
                tensors.insert(
                    clean_name,
                    NpyTensorView {
                        dtype: hdr.dtype,
                        shape: hdr.shape,
                        data_offset: abs_offset,
                        size_bytes: hdr.size_bytes,
                    },
                );
            }
        }

        Ok(Self { mmap, tensors })
    }

    #[inline]
    pub fn get_tensor(&self, name: &str) -> Option<(&NpyTensorView, &[u8])> {
        let tv = self.tensors.get(name)?;
        let slice = self.mmap.slice(tv.data_offset, tv.size_bytes)?;
        Some((tv, slice))
    }

    #[inline]
    pub fn tensor_names(&self) -> impl Iterator<Item = &String> {
        self.tensors.keys()
    }
}
