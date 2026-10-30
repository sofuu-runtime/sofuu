// rt/view_guard.rs — R7.P2: one validated window from JS memory to a native kernel.
//
// JavaScript hands native code a `Float32Array` as (base pointer, byte offset,
// byte length). QuickJS maintains those three consistently, so a real view
// cannot be out of bounds today — but every kernel below trusts that invariant,
// and one engine bug (or one future bridge that fabricates a view) turns it
// into an out-of-bounds read in the SIMD path. This module is the single place
// that re-derives the invariant before a pointer is used:
//
//   * lane alignment — a float32 view must sit on a 4-byte boundary and be a
//     whole number of lanes (a `subarray` of a Float32Array always is; anything
//     else means the value did not come from the class it claims);
//   * end within the buffer — `offset + length <= buffer size`, computed with
//     checked arithmetic so a wrapped sum can never pass;
//   * optional exact dimension — the caller that knows the expected width
//     refuses a mismatch before any element loop runs.
//
// Fail-closed: every violation is an error the caller turns into a typed
// throw. Nothing is clamped and no partial window is returned.
//
// The byte-path extractors (fs/crypto/zerocopy) already check
// `offset + length <= size` at their own sites; they are byte-aligned by
// construction and do not need the lane rule.

/// A validated window: `offset` bytes into the buffer, `count` lanes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ViewWindow {
    pub offset: usize,
    pub count: usize,
}

/// Why a view was refused. Callers map these to their own error vocabulary
/// (a coded JS error in the memory bridge, a TypeError in `sofuu.ai`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewError {
    /// Offset or length is not a whole number of elements.
    Unaligned,
    /// The window ends past the end of its buffer (or the sum overflowed).
    OutOfBounds,
    /// The window is in bounds but the element count is not the one required.
    Dimension { expected: usize, got: usize },
}

impl ViewError {
    /// A short, stable description. Kept here so every surface reports the
    /// same words for the same refusal.
    pub fn describe(self) -> &'static str {
        match self {
            ViewError::Unaligned => "typed-array view is not element-aligned",
            ViewError::OutOfBounds => "typed-array view extends past the end of its buffer",
            ViewError::Dimension { .. } => "typed-array view has the wrong number of elements",
        }
    }
}

const F32_BYTES: usize = std::mem::size_of::<f32>();

/// Validate a float32 window against its backing buffer.
///
/// `buf_size` is the whole `ArrayBuffer`'s byte length; `byte_offset` and
/// `byte_length` are what the engine reported for the view. `expected_dim`,
/// when given, must equal the resulting lane count.
#[inline]
pub fn check_f32_view(
    buf_size: usize,
    byte_offset: usize,
    byte_length: usize,
    expected_dim: Option<usize>,
) -> Result<ViewWindow, ViewError> {
    check_view(F32_BYTES, buf_size, byte_offset, byte_length, expected_dim)
}

/// The unit-generic form. `unit` is the element width in bytes.
#[inline]
pub fn check_view(
    unit: usize,
    buf_size: usize,
    byte_offset: usize,
    byte_length: usize,
    expected_units: Option<usize>,
) -> Result<ViewWindow, ViewError> {
    if unit == 0 || byte_offset % unit != 0 || byte_length % unit != 0 {
        return Err(ViewError::Unaligned);
    }
    let end = byte_offset
        .checked_add(byte_length)
        .ok_or(ViewError::OutOfBounds)?;
    if end > buf_size {
        return Err(ViewError::OutOfBounds);
    }
    let count = byte_length / unit;
    if let Some(expected) = expected_units {
        if count != expected {
            return Err(ViewError::Dimension {
                expected,
                got: count,
            });
        }
    }
    Ok(ViewWindow {
        offset: byte_offset,
        count,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const BUFFER: usize = 4096;

    #[test]
    fn whole_buffer_is_accepted() {
        let w = check_f32_view(BUFFER, 0, BUFFER, None).unwrap();
        assert_eq!(w.offset, 0);
        assert_eq!(w.count, BUFFER / 4);
    }

    #[test]
    fn subarray_window_is_accepted() {
        let w = check_f32_view(BUFFER, 16, 64, None).unwrap();
        assert_eq!(w.offset, 16);
        assert_eq!(w.count, 16);
    }

    #[test]
    fn exact_fit_at_the_end_is_accepted() {
        let w = check_f32_view(64, 32, 32, Some(8)).unwrap();
        assert_eq!(w.count, 8);
    }

    #[test]
    fn empty_view_is_accepted() {
        let w = check_f32_view(0, 0, 0, Some(0)).unwrap();
        assert_eq!(w.count, 0);
        let w = check_f32_view(64, 64, 0, None).unwrap();
        assert_eq!(w.count, 0);
    }

    #[test]
    fn length_past_the_end_is_refused() {
        assert_eq!(
            check_f32_view(64, 0, 68, None),
            Err(ViewError::OutOfBounds)
        );
        assert_eq!(
            check_f32_view(64, 60, 8, None),
            Err(ViewError::OutOfBounds)
        );
    }

    #[test]
    fn offset_past_the_end_is_refused() {
        assert_eq!(
            check_f32_view(64, 68, 0, None),
            Err(ViewError::OutOfBounds)
        );
    }

    #[test]
    fn offset_plus_length_overflow_is_refused_not_wrapped() {
        /* The classic wrap: an aligned offset near usize::MAX plus a length
         * that lands back inside the buffer. Untreated this is an
         * out-of-bounds read at `buf + MAX - 3`, so the checked add must
         * reject it. */
        assert_eq!(
            check_f32_view(64, usize::MAX - 3, 4, None),
            Err(ViewError::OutOfBounds)
        );
        assert_eq!(
            check_f32_view(64, usize::MAX - 7, 8, None),
            Err(ViewError::OutOfBounds)
        );
        /* An offset beyond the buffer needs no length to be refused. */
        assert_eq!(
            check_f32_view(64, usize::MAX - 3, 0, None),
            Err(ViewError::OutOfBounds)
        );
    }

    #[test]
    fn unaligned_offset_is_refused() {
        for off in [1usize, 2, 3, 5, 4095] {
            assert_eq!(
                check_f32_view(BUFFER, off, 4, None),
                Err(ViewError::Unaligned),
                "offset {off}"
            );
        }
    }

    #[test]
    fn unaligned_length_is_refused() {
        for len in [1usize, 2, 3, 6, 4095] {
            assert_eq!(
                check_f32_view(BUFFER, 0, len, None),
                Err(ViewError::Unaligned),
                "length {len}"
            );
        }
    }

    #[test]
    fn dimension_mismatch_reports_both_numbers() {
        assert_eq!(
            check_f32_view(64, 0, 64, Some(8)),
            Err(ViewError::Dimension {
                expected: 8,
                got: 16
            })
        );
    }

    #[test]
    fn byte_width_accepts_odd_lengths_and_offsets() {
        let w = check_view(1, 10, 3, 5, None).unwrap();
        assert_eq!(w.offset, 3);
        assert_eq!(w.count, 5);
    }

    #[test]
    fn zero_unit_is_refused_instead_of_dividing_by_zero() {
        assert_eq!(
            check_view(0, 0, 0, 0, None),
            Err(ViewError::Unaligned)
        );
    }
}
