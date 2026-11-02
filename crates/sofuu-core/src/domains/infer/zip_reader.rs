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

/// Bounds-checked little-endian reads.
///
/// Every field below comes from a file the user did not write, so the reads
/// must fail as data rather than as a panic: the release profile is
/// panic = "abort", where a slice out of range is instant process death in
/// the middle of loading a model. `get` + `try_into` turns a truncated or
/// hostile archive into the error string `parse` already returns.
fn u16_at(data: &[u8], at: usize) -> Result<u16, String> {
    read_at::<2>(data, at).map(u16::from_le_bytes)
}

fn u32_at(data: &[u8], at: usize) -> Result<u32, String> {
    read_at::<4>(data, at).map(u32::from_le_bytes)
}

fn u64_at(data: &[u8], at: usize) -> Result<u64, String> {
    read_at::<8>(data, at).map(u64::from_le_bytes)
}

fn read_at<const N: usize>(data: &[u8], at: usize) -> Result<[u8; N], String> {
    let end = at.checked_add(N).ok_or_else(|| format!("zip: offset {at} overflows"))?;
    data.get(at..end)
        .ok_or_else(|| format!("zip: truncated reading {N} bytes at {at} (len {})", data.len()))?
        .try_into()
        .map_err(|_| format!("zip: short read of {N} bytes at {at}"))
}

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
            let sig = u32_at(data, i)?;
            if sig == EOCD_SIGNATURE {
                eocd_pos = Some(i);
                break;
            }
        }

        let eocd_idx = eocd_pos.ok_or("End of Central Directory (EOCD) signature not found")?;

        // 2. Read standard EOCD fields
        let mut total_entries = u16_at(data, eocd_idx + 10)? as u64;
        let mut cd_size = u32_at(data, eocd_idx + 12)? as u64;
        let mut cd_offset = u32_at(data, eocd_idx + 16)? as u64;

        // 3. Check for ZIP64 locator immediately preceding EOCD
        if eocd_idx >= 20 {
            let loc_sig = u32_at(data, eocd_idx - 20)?;
            if loc_sig == ZIP64_LOCATOR_SIGNATURE {
                // The field is 64-bit and attacker-controlled: widening to
                // usize and adding 56 overflows on a hostile archive, and a
                // panic here is an abort in release builds.
                let zip64_eocd_offset = u64_at(data, eocd_idx - 12)? as usize;
                if zip64_eocd_offset
                    .checked_add(56)
                    .is_some_and(|end| end <= data.len())
                {
                    let zip64_sig = u32_at(data, zip64_eocd_offset)?;
                    if zip64_sig == ZIP64_EOCD_SIGNATURE {
                        total_entries = u64_at(data, zip64_eocd_offset + 32)?;
                        cd_size = u64_at(data, zip64_eocd_offset + 40)?;
                        cd_offset = u64_at(data, zip64_eocd_offset + 48)?;
                    }
                }
            }
        }

        let cd_end_u64 = cd_offset.checked_add(cd_size).ok_or_else(|| {
            format!("Central Directory size overflows (cd_offset={cd_offset}, cd_size={cd_size})")
        })?;
        if cd_end_u64 > data.len() as u64 {
            return Err(format!(
                "Central Directory exceeds archive bounds (cd_offset={}, cd_size={}, len={})",
                cd_offset, cd_size, data.len()
            ));
        }

        // 4. Iterate Central Directory headers
        let mut entries = HashMap::new();
        let mut pos = cd_offset as usize;
        let cd_end = cd_end_u64 as usize;

        for _ in 0..total_entries {
            if pos + 46 > cd_end {
                break;
            }

            let sig = u32_at(data, pos)?;
            if sig != CD_HEADER_SIGNATURE {
                break;
            }

            let compression_method = u16_at(data, pos + 10)?;
            let mut compressed_size = u32_at(data, pos + 20)? as u64;
            let mut uncompressed_size = u32_at(data, pos + 24)? as u64;
            let filename_len = u16_at(data, pos + 28)? as usize;
            let extra_len = u16_at(data, pos + 30)? as usize;
            let comment_len = u16_at(data, pos + 32)? as usize;
            let mut local_header_offset = u32_at(data, pos + 42)? as u64;

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
                    let header_id = u16_at(data, e_pos)?;
                    let data_size = u16_at(data, e_pos + 2)? as usize;
                    e_pos += 4;
                    if e_pos + data_size > extra_end {
                        break;
                    }

                    if header_id == 0x0001 {
                        // ZIP64 extended information extra field
                        let mut field_pos = e_pos;
                        if uncompressed_size == 0xffffffff && field_pos + 8 <= e_pos + data_size {
                            uncompressed_size = u64_at(data, field_pos)?;
                            field_pos += 8;
                        }
                        if compressed_size == 0xffffffff && field_pos + 8 <= e_pos + data_size {
                            compressed_size = u64_at(data, field_pos)?;
                            field_pos += 8;
                        }
                        if local_header_offset == 0xffffffff && field_pos + 8 <= e_pos + data_size {
                            local_header_offset = u64_at(data, field_pos)?;
                        }
                    }

                    e_pos += data_size;
                }
            }

            // Inspect local header to calculate exact data offset
            let local_idx = local_header_offset as usize;
            // local_header_offset is a 64-bit number from the file; adding 30 to
            // a widened hostile value overflows before the bounds check sees it.
            if local_idx.checked_add(30).is_none_or(|end| end > data.len()) {
                return Err(format!("local header offset {} beyond archive size", local_header_offset));
            }

            let local_sig = u32_at(data, local_idx)?;
            if local_sig != LOCAL_HEADER_SIGNATURE {
                return Err(format!("invalid local header signature at offset {}", local_header_offset));
            }

            let local_fn_len = u16_at(data, local_idx + 26)? as u64;
            let local_extra_len = u16_at(data, local_idx + 28)? as u64;
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

#[cfg(test)]
mod hardening {
    use super::*;

    /// A truncated archive is data from the network, not a bug in this file: it
    /// must come back as Err. Before this, 26 slice reads in `parse` unwrapped,
    /// so a hostile or half-downloaded zip aborted the process (panic = abort).
    #[test]
    fn every_truncation_of_a_plausible_archive_returns_an_error() {
        // An EOCD near the end with plausible field values, then every prefix
        // of it: each length must be refused as data, never panic.
        let mut base = vec![0u8; 64];
        base[42..46].copy_from_slice(&EOCD_SIGNATURE.to_le_bytes());
        base[52..54].copy_from_slice(&1u16.to_le_bytes());
        base[54..58].copy_from_slice(&18u32.to_le_bytes());
        base[58..62].copy_from_slice(&22u32.to_le_bytes());
        for end in 0..base.len() {
            let _ = ZipArchive::parse(&base[..end]);
        }
    }

    #[test]
    fn a_hostile_central_directory_size_is_an_error_not_an_overflow() {
        let mut data = vec![0u8; 22];
        data[0..4].copy_from_slice(&EOCD_SIGNATURE.to_le_bytes());
        data[10..12].copy_from_slice(&1u16.to_le_bytes());
        data[12..16].copy_from_slice(&u32::MAX.to_le_bytes());
        data[16..20].copy_from_slice(&u32::MAX.to_le_bytes());
        let err = ZipArchive::parse(&data).err().expect("must be refused");
        let low = err.to_lowercase();
        assert!(
            low.contains("overflow") || low.contains("exceeds"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn a_hostile_zip64_locator_offset_is_an_error_not_a_panic() {
        let mut data = vec![0u8; 86];
        let eocd = 64usize;
        data[eocd..eocd + 4].copy_from_slice(&EOCD_SIGNATURE.to_le_bytes());
        let loc = eocd - 20;
        data[loc..loc + 4].copy_from_slice(&ZIP64_LOCATOR_SIGNATURE.to_le_bytes());
        data[loc + 8..loc + 16].copy_from_slice(&u64::MAX.to_le_bytes());
        let _ = ZipArchive::parse(&data);
    }
}
