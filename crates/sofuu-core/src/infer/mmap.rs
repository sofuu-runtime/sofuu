// crates/sofuu-core/src/infer/mmap.rs — Zero-copy memory mapped file reader.
//
// Implemented via libc::mmap to avoid external crate dependencies and keep
// binary size strictly under the 5 MB limit.

use std::fs::File;
use std::os::unix::io::AsRawFd;
use std::path::Path;
use std::ptr;

pub struct MmapFile {
    ptr: *const u8,
    len: usize,
}

// SAFETY: MmapFile provides read-only access to immutable mapped bytes.
unsafe impl Send for MmapFile {}
unsafe impl Sync for MmapFile {}

impl MmapFile {
    pub fn open<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        let len = file.metadata()?.len() as usize;
        if len == 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "cannot mmap empty file",
            ));
        }

        let fd = file.as_raw_fd();
        let ptr = unsafe {
            libc::mmap(
                ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_SHARED,
                fd,
                0,
            )
        };

        if ptr == libc::MAP_FAILED {
            return Err(std::io::Error::last_os_error());
        }

        // Eagerly prefetch the metadata header (first 32 MB) to eliminate page faults on startup
        unsafe {
            let header_len = len.min(32 * 1024 * 1024);
            libc::madvise(ptr, header_len, libc::MADV_WILLNEED);
        }

        Ok(Self {
            ptr: ptr as *const u8,
            len,
        })
    }

    #[inline]
    pub fn as_slice(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn slice(&self, offset: usize, length: usize) -> Option<&[u8]> {
        if offset.checked_add(length)? <= self.len {
            unsafe {
                Some(std::slice::from_raw_parts(self.ptr.add(offset), length))
            }
        } else {
            None
        }
    }
}

impl Drop for MmapFile {
    fn drop(&mut self) {
        if !self.ptr.is_null() && self.len > 0 {
            unsafe {
                libc::munmap(self.ptr as *mut libc::c_void, self.len);
            }
        }
    }
}
