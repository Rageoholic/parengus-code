//! Iterator utilities.

/// Iterates over the indices of set bits in a `u32`, from lowest to highest.
///
/// ```
/// # use parengus_util::iter::SetBitIterU32;
/// let bits: Vec<u8> = SetBitIterU32::new(0b1010_1010).collect();
/// assert_eq!(bits, [1, 3, 5, 7]);
/// ```
#[derive(Copy, Clone)]
pub struct SetBitIterU32 {
    bits: u32,
    // Stored as u8 because trailing_zeros returning u32 when the
    // max useful value is 31 is annoying.
    current_bit: u8,
}

impl SetBitIterU32 {
    pub fn new(bits: u32) -> Self {
        Self {
            bits,
            current_bit: bits.trailing_zeros() as u8,
        }
    }
}

impl Iterator for SetBitIterU32 {
    type Item = u8;
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.current_bit >= 32 {
                break None;
            } else {
                let current_bit = self.current_bit;
                self.current_bit += 1;
                if self.bits & (1 << current_bit) != 0 {
                    break Some(current_bit);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::SetBitIterU32;

    #[test]
    fn zero_produces_nothing() {
        assert_eq!(SetBitIterU32::new(0).count(), 0);
    }

    #[test]
    fn all_bits_set() {
        let bits: Vec<u8> = SetBitIterU32::new(u32::MAX).collect();
        assert_eq!(bits.len(), 32);
        assert_eq!(bits[0], 0);
        assert_eq!(bits[31], 31);
    }

    #[test]
    fn single_bit() {
        let bits: Vec<u8> = SetBitIterU32::new(1 << 17).collect();
        assert_eq!(bits, [17]);
    }
}
