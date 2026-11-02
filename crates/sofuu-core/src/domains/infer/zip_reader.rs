// crates/sofuu-core/src/infer/zip_reader.rs — Zero-dependency pure-Rust ZIP & ZIP64 reader.
//
// Optimized for zero-copy memory-mapped machine learning weights (PyTorch .pt/.pth and NumPy .npz).
// Supports uncompressed (STORED = 0) zero-copy slices and DEFLATE (method = 8) via flate2.

use std::collections::HashMap;
use std::io::Read;

const EOCD_SIGNATURE: u32 = 0x06054b50;
const ZIP64_LOCATOR_SIGNATURE: u32 = 0x07064b50;
const ZIP64_EOCD_SIGNATURE: u32 = 0x06064b50;
const CD_HEADER_SIGNATURE: u32 = 0x02014b50;
const LOCAL_HEADER_SIGNATURE: u32 = 0x04034b50;

#[derive(Debug, Clone)]
pub struct ZipEntry {
    pub name: String,
    pub compression_method: u16, // 0 = stored, 8 = deflate
    pub compressed_size: u64,
    pub uncompressed_size: u64,
    pub local_header_offset: u64,
    pub data_offset: u64,
}

pub struct ZipArchive<'a> {
    data: &'a [u8],
    entries: HashMap<String, ZipEntry>,
}

impl<'a> ZipArchive<'a> {
    pub fn parse(data: &'a [u8]) -> Result<Self, String> {
        if data.len() < 22 {
            return Err("file too small to be a valid ZIP archive".to_string());
        }

        // 1. Locate End of Central Directory (EOCD) by scanning backwards
        let max_search = data.len().min(65536 + 22);
        let search_start = data.len() - max_search;
        let mut eocd_pos = None;

        for i in (search_start..=data.len() - 22).rev() {
            let sig = u32::from_le_bytes(data[i..i + 4].try_into().unwrap());
            if sig == EOCD_SIGNATURE {
                eocd_pos = Some(i);
                break;
            }
        }

        let eocd_idx = eocd_pos.ok_or("End of Central Directory (EOCD) signature not found")?;

        // 2. Read standard EOCD fields
        let mut total_entries = u16::from_le_bytes(data[eocd_idx + 10..eocd_idx + 12].try_into().unwrap()) as u64;
        let mut cd_size = u32::from_le_bytes(data[eocd_idx + 12..eocd_idx + 16].try_into().unwrap()) as u64;
        let mut cd_offset = u32::from_le_bytes(data[eocd_idx + 16..eocd_idx + 20].try_into().unwrap()) as u64;

        // 3. Check for ZIP64 locator immediately preceding EOCD
        if eocd_idx >= 20 {
            let loc_sig = u32::from_le_bytes(data[eocd_idx - 20..eocd_idx - 16].try_into().unwrap());
            if loc_sig == ZIP64_LOCATOR_SIGNATURE {
                let zip64_eocd_offset = u64::from_le_bytes(data[eocd_idx - 12..eocd_idx - 4].try_into().unwrap()) as usize;
                if zip64_eocd_offset + 56 <= data.len() {
                    let zip64_sig = u32::from_le_bytes(data[zip64_eocd_offset..zip64_eocd_offset + 4].try_into().unwrap());
                    if zip64_sig == ZIP64_EOCD_SIGNATURE {
                        total_entries = u64::from_le_bytes(data[zip64_eocd_offset + 32..zip64_eocd_offset + 40].try_into().unwrap());
                        cd_size = u64::from_le_bytes(data[zip64_eocd_offset + 40..zip64_eocd_offset + 48].try_into().unwrap());
                        cd_offset = u64::from_le_bytes(data[zip64_eocd_offset + 48..zip64_eocd_offset + 56].try_into().unwrap());
                    }
                }
            }
        }

        if cd_offset + cd_size > data.len() as u64 {
            return Err(format!(
                "Central Directory exceeds archive bounds (cd_offset={}, cd_size={}, len={})",
                cd_offset, cd_size, data.len()
            ));
        }

        // 4. Iterate Central Directory headers
        let mut entries = HashMap::new();
        let mut pos = cd_offset as usize;
        let cd_end = (cd_offset + cd_size) as usize;

        for _ in 0..total_entries {
            if pos + 46 > cd_end {
                break;
            }

            let sig = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
            if sig != CD_HEADER_SIGNATURE {
                break;
            }

            let compression_method = u16::from_le_bytes(data[pos + 10..pos + 12].try_into().unwrap());
            let mut compressed_size = u32::from_le_bytes(data[pos + 20..pos + 24].try_into().unwrap()) as u64;
            let mut uncompressed_size = u32::from_le_bytes(data[pos + 24..pos + 28].try_into().unwrap()) as u64;
            let filename_len = u16::from_le_bytes(data[pos + 28..pos + 30].try_into().unwrap()) as usize;
            let extra_len = u16::from_le_bytes(data[pos + 30..pos + 32].try_into().unwrap()) as usize;
            let comment_len = u16::from_le_bytes(data[pos + 32..pos + 34].try_into().unwrap()) as usize;
            let mut local_header_offset = u32::from_le_bytes(data[pos + 42..pos + 46].try_into().unwrap()) as u64;

            let fn_start = pos + 46;
            let fn_end = fn_start + filename_len;
            if fn_end > cd_end {
                break;
            }

            let name = String::from_utf8_lossy(&data[fn_start..fn_end]).to_string();

            // Parse ZIP64 extra field if present
            let extra_start = fn_end;
            let extra_end = extra_start + extra_len;
            if extra_end <= cd_end {
                let mut e_pos = extra_start;
                while e_pos + 4 <= extra_end {
                    let header_id = u16::from_le_bytes(data[e_pos..e_pos + 2].try_into().unwrap());
                    let data_size = u16::from_le_bytes(data[e_pos + 2..e_pos + 4].try_into().unwrap()) as usize;
                    e_pos += 4;
                    if e_pos + data_size > extra_end {
                        break;
                    }

                    if header_id == 0x0001 {
                        // ZIP64 extended information extra field
                        let mut field_pos = e_pos;
                        if uncompressed_size == 0xffffffff && field_pos + 8 <= e_pos + data_size {
                            uncompressed_size = u64::from_le_bytes(data[field_pos..field_pos + 8].try_into().unwrap());
                            field_pos += 8;
                        }
                        if compressed_size == 0xffffffff && field_pos + 8 <= e_pos + data_size {
                            compressed_size = u64::from_le_bytes(data[field_pos..field_pos + 8].try_into().unwrap());
                            field_pos += 8;
                        }
                        if local_header_offset == 0xffffffff && field_pos + 8 <= e_pos + data_size {
                            local_header_offset = u64::from_le_bytes(data[field_pos..field_pos + 8].try_into().unwrap());
                        }
                    }

                    e_pos += data_size;
                }
            }

            // Inspect local header to calculate exact data offset
            let local_idx = local_header_offset as usize;
            if local_idx + 30 > data.len() {
                return Err(format!("local header offset {} beyond archive size", local_header_offset));
            }

            let local_sig = u32::from_le_bytes(data[local_idx..local_idx + 4].try_into().unwrap());
            if local_sig != LOCAL_HEADER_SIGNATURE {
                return Err(format!("invalid local header signature at offset {}", local_header_offset));
            }

            let local_fn_len = u16::from_le_bytes(data[local_idx + 26..local_idx + 28].try_into().unwrap()) as u64;
            let local_extra_len = u16::from_le_bytes(data[local_idx + 28..local_idx + 30].try_into().unwrap()) as u64;
            let data_offset = local_header_offset + 30 + local_fn_len + local_extra_len;

            entries.insert(
                name.clone(),
                ZipEntry {
                    name,
                    compression_method,
                    compressed_size,
                    uncompressed_size,
                    local_header_offset,
                    data_offset,
                },
            );

            pos = fn_start + filename_len + extra_len + comment_len;
        }

        Ok(Self { data, entries })
    }

    #[inline]
    pub fn get_entry(&self, name: &str) -> Option<&ZipEntry> {
        self.entries.get(name)
    }

    #[inline]
    pub fn entry_names(&self) -> impl Iterator<Item = &String> {
        self.entries.keys()
    }

    /// Returns zero-copy slice for uncompressed (STORED) entries.
    #[inline]
    pub fn get_uncompressed_slice(&self, name: &str) -> Option<&'a [u8]> {
        let entry = self.entries.get(name)?;
        if entry.compression_method != 0 {
            return None; // entry is compressed (e.g. Deflated)
        }
        let start = entry.data_offset as usize;
        let end = start + entry.uncompressed_size as usize;
        if end <= self.data.len() {
            Some(&self.data[start..end])
        } else {
            None
        }
    }

    /// Reads entry bytes, decompressing via DEFLATE if necessary.
    pub fn read_entry(&self, name: &str) -> Result<Vec<u8>, String> {
        let entry = self.entries.get(name).ok_or_else(|| format!("entry '{}' not found in zip", name))?;
        let start = entry.data_offset as usize;
        let end = start + entry.compressed_size as usize;
        if end > self.data.len() {
            return Err(format!("entry '{}' compressed slice exceeds archive length", name));
        }
        let comp_bytes = &self.data[start..end];

        match entry.compression_method {
            0 => Ok(comp_bytes.to_vec()),
            8 => {
                let mut decoder = flate2::read::DeflateDecoder::new(comp_bytes);
                let mut out = Vec::with_capacity(entry.uncompressed_size as usize);
                decoder
                    .read_to_end(&mut out)
                    .map_err(|e| format!("failed to decompress entry '{}': {}", name, e))?;
                Ok(out)
            }
            other => Err(format!("unsupported compression method {} for entry '{}'", other, name)),
        }
    }
}
