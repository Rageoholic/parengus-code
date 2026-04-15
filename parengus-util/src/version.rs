/// A semantic version number.
///
/// Compatibility rules:
/// - Pre-1.0 (`major == 0`): exact match required — every bump is
///   breaking.
/// - 1.0+: same `major` and `file_ver.minor <= loader_ver.minor`.
///   Patch bumps are always compatible in both directions.
///
/// Use [`PackedVersion`] for the wire representation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Version {
    pub major: u16,
    pub minor: u32,
    pub patch: u16,
}

/// Little-endian packed wire representation of a [`Version`].
///
/// Layout: 8 bytes, always little-endian:
/// `major(u16) << 48 | minor(u32) << 16 | patch(u16)`
///
/// The inner `[u8; 8]` can be written directly to / read directly
/// from a byte stream with no further byte-swapping.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct PackedVersion(pub [u8; 8]);

// SAFETY: PackedVersion is #[repr(transparent)] over [u8; 8], which
// is Pod and Zeroable. No padding, no invalid bit patterns.
unsafe impl bytemuck::Zeroable for PackedVersion {}
unsafe impl bytemuck::Pod for PackedVersion {}

impl PackedVersion {
    /// Construct from individual version components.
    #[inline]
    pub fn new(major: u16, minor: u32, patch: u16) -> Self {
        let raw: u64 =
            (major as u64) << 48 | (minor as u64) << 16 | patch as u64;
        Self(raw.to_le_bytes())
    }

    /// Write the 8 wire bytes to `w`.
    pub fn write_to(&self, w: &mut impl std::io::Write) -> std::io::Result<()> {
        w.write_all(&self.0)
    }

    /// Read 8 wire bytes from `r`.
    pub fn read_from(r: &mut impl std::io::Read) -> std::io::Result<Self> {
        let mut buf = [0u8; 8];
        r.read_exact(&mut buf)?;
        Ok(Self(buf))
    }
}

impl PackedVersion {
    #[inline]
    fn raw(&self) -> u64 {
        u64::from_le_bytes(self.0)
    }

    /// The major version component.
    #[inline]
    pub fn major(&self) -> u16 {
        (self.raw() >> 48) as u16
    }

    /// The minor version component.
    #[inline]
    pub fn minor(&self) -> u32 {
        (self.raw() >> 16) as u32
    }

    /// The patch version component.
    #[inline]
    pub fn patch(&self) -> u16 {
        self.raw() as u16
    }
}

impl From<Version> for PackedVersion {
    fn from(v: Version) -> Self {
        Self::new(v.major, v.minor, v.patch)
    }
}

impl From<PackedVersion> for Version {
    fn from(p: PackedVersion) -> Self {
        Self {
            major: p.major(),
            minor: p.minor(),
            patch: p.patch(),
        }
    }
}

impl Version {
    #[inline]
    pub const fn new(major: u16, minor: u32, patch: u16) -> Self {
        Self {
            major,
            minor,
            patch,
        }
    }

    /// Returns `true` if a file written at `file_ver` is loadable by
    /// code compiled against `self`.
    ///
    /// Pre-1.0: exact match only.
    /// 1.0+: same major, file minor ≤ loader minor.
    #[inline]
    pub fn is_compatible_with(self, file_ver: Version) -> bool {
        if self.major == 0 {
            self == file_ver
        } else {
            self.major == file_ver.major && file_ver.minor <= self.minor
        }
    }
}

impl std::fmt::Display for Version {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}.{}.{}", self.major, self.minor, self.patch)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_round_trip() {
        let v = Version::new(1, 2, 3);
        assert_eq!(Version::from(PackedVersion::from(v)), v);
    }

    #[test]
    fn pack_round_trip_zero() {
        let v = Version::new(0, 1, 0);
        assert_eq!(Version::from(PackedVersion::from(v)), v);
    }

    #[test]
    fn pack_round_trip_max_fields() {
        let v = Version::new(u16::MAX, u32::MAX, u16::MAX);
        assert_eq!(Version::from(PackedVersion::from(v)), v);
    }

    #[test]
    fn pre_1_0_exact_match() {
        let loader = Version::new(0, 1, 0);
        assert!(loader.is_compatible_with(Version::new(0, 1, 0)));
        assert!(!loader.is_compatible_with(Version::new(0, 2, 0)));
        assert!(!loader.is_compatible_with(Version::new(0, 1, 1)));
    }

    #[test]
    fn post_1_0_minor_compat() {
        let loader = Version::new(1, 3, 0);
        // Same or older minor: compatible.
        assert!(loader.is_compatible_with(Version::new(1, 3, 0)));
        assert!(loader.is_compatible_with(Version::new(1, 2, 0)));
        assert!(loader.is_compatible_with(Version::new(1, 0, 0)));
        // Newer minor: not compatible (file has features loader
        // doesn't know about).
        assert!(!loader.is_compatible_with(Version::new(1, 4, 0)));
    }

    #[test]
    fn post_1_0_major_mismatch() {
        let loader = Version::new(1, 0, 0);
        assert!(!loader.is_compatible_with(Version::new(2, 0, 0)));
        assert!(!loader.is_compatible_with(Version::new(0, 1, 0)));
    }

    #[test]
    fn display() {
        assert_eq!(Version::new(1, 2, 3).to_string(), "1.2.3");
    }

    #[test]
    fn packed_accessors() {
        let v = Version::new(1, 2, 3);
        let p = PackedVersion::from(v);
        assert_eq!(p.major(), 1);
        assert_eq!(p.minor(), 2);
        assert_eq!(p.patch(), 3);
    }

    #[test]
    fn packed_accessors_max_fields() {
        let v = Version::new(u16::MAX, u32::MAX, u16::MAX);
        let p = PackedVersion::from(v);
        assert_eq!(p.major(), u16::MAX);
        assert_eq!(p.minor(), u32::MAX);
        assert_eq!(p.patch(), u16::MAX);
    }

    #[test]
    fn packed_accessors_zero() {
        let v = Version::new(0, 0, 0);
        let p = PackedVersion::from(v);
        assert_eq!(p.major(), 0);
        assert_eq!(p.minor(), 0);
        assert_eq!(p.patch(), 0);
    }

    #[test]
    fn ordering() {
        // Major dominates.
        assert!(Version::new(2, 0, 0) > Version::new(1, 9, 9));
        // Minor dominates within same major.
        assert!(Version::new(1, 2, 0) > Version::new(1, 1, 9));
        // Patch is the tiebreaker.
        assert!(Version::new(1, 1, 2) > Version::new(1, 1, 1));
        // Equal versions compare equal.
        assert!(Version::new(1, 2, 3) == Version::new(1, 2, 3));
    }
}
