use std::io::{self, Read, Write};

pub const PMESH_MAGIC: u32 = u32::from_le_bytes(*b"PMSH");
pub const PTEX_MAGIC: u32 = u32::from_le_bytes(*b"PTEX");
pub const VERSION: u16 = 1;

// ── AssetKind ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssetKind {
    Mesh,
    Texture,
}

impl AssetKind {
    pub fn to_u16(self) -> u16 {
        match self {
            Self::Mesh => 0,
            Self::Texture => 1,
        }
    }

    pub fn from_u16(v: u16) -> Option<Self> {
        match v {
            0 => Some(Self::Mesh),
            1 => Some(Self::Texture),
            _ => None,
        }
    }
}

// ── SectionKind ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SectionKind {
    MeshPositions,
    MeshNormals,
    MeshTangents,
    MeshTexCoord0,
    MeshTexCoord1,
    MeshIndices16,
    MeshIndices32,
    MeshTexRef,
    TextureMip,
}

impl SectionKind {
    pub fn to_u32(self) -> u32 {
        match self {
            Self::MeshPositions => 0,
            Self::MeshNormals => 1,
            Self::MeshTangents => 2,
            Self::MeshTexCoord0 => 3,
            Self::MeshTexCoord1 => 4,
            Self::MeshIndices16 => 5,
            Self::MeshIndices32 => 6,
            Self::MeshTexRef => 7,
            Self::TextureMip => 100,
        }
    }

    /// Returns `None` for unknown kinds; callers should skip those
    /// sections rather than fail, to allow forward compatibility.
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::MeshPositions),
            1 => Some(Self::MeshNormals),
            2 => Some(Self::MeshTangents),
            3 => Some(Self::MeshTexCoord0),
            4 => Some(Self::MeshTexCoord1),
            5 => Some(Self::MeshIndices16),
            6 => Some(Self::MeshIndices32),
            7 => Some(Self::MeshTexRef),
            100 => Some(Self::TextureMip),
            _ => None,
        }
    }
}

// ── Compression ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compression {
    None,
    Lz4,
}

impl Compression {
    pub fn to_u32(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Lz4 => 1,
        }
    }

    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::None),
            1 => Some(Self::Lz4),
            _ => None,
        }
    }
}

// ── FileHeader ───────────────────────────────────────────────────────────────

/// Serialized size: 12 bytes.
pub struct FileHeader {
    pub magic: u32,
    pub version: u16,
    pub asset_kind: AssetKind,
    pub section_count: u32,
}

impl FileHeader {
    pub fn write_to(&self, w: &mut impl Write) -> io::Result<()> {
        w.write_all(&self.magic.to_le_bytes())?;
        w.write_all(&self.version.to_le_bytes())?;
        w.write_all(&self.asset_kind.to_u16().to_le_bytes())?;
        w.write_all(&self.section_count.to_le_bytes())?;
        Ok(())
    }

    pub fn read_from(r: &mut impl Read) -> io::Result<Self> {
        let mut buf = [0u8; 12];
        r.read_exact(&mut buf)?;
        let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        let version = u16::from_le_bytes(buf[4..6].try_into().unwrap());
        let kind_raw = u16::from_le_bytes(buf[6..8].try_into().unwrap());
        let section_count = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        let asset_kind = AssetKind::from_u16(kind_raw).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "unknown asset_kind")
        })?;
        Ok(Self {
            magic,
            version,
            asset_kind,
            section_count,
        })
    }
}

// ── SectionHeader ────────────────────────────────────────────────────────────

/// Serialized size: 28 bytes (includes 4-byte reserved field).
pub struct SectionHeader {
    pub kind: SectionKind,
    pub compression: Compression,
    pub byte_offset: u32,
    pub byte_len: u32,
    pub compressed_byte_len: u32,
    pub element_count: u32,
}

impl SectionHeader {
    pub fn write_to(&self, w: &mut impl Write) -> io::Result<()> {
        w.write_all(&self.kind.to_u32().to_le_bytes())?;
        w.write_all(&self.compression.to_u32().to_le_bytes())?;
        w.write_all(&0u32.to_le_bytes())?; // reserved
        w.write_all(&self.byte_offset.to_le_bytes())?;
        w.write_all(&self.byte_len.to_le_bytes())?;
        w.write_all(&self.compressed_byte_len.to_le_bytes())?;
        w.write_all(&self.element_count.to_le_bytes())?;
        Ok(())
    }

    /// Returns `Ok(None)` for unknown `kind`; caller should skip the
    /// section.
    pub fn read_from(r: &mut impl Read) -> io::Result<Option<Self>> {
        let mut buf = [0u8; 28];
        r.read_exact(&mut buf)?;
        let kind_raw = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        let comp_raw = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        // buf[8..12] is reserved — ignored
        let byte_offset = u32::from_le_bytes(buf[12..16].try_into().unwrap());
        let byte_len = u32::from_le_bytes(buf[16..20].try_into().unwrap());
        let compressed_byte_len =
            u32::from_le_bytes(buf[20..24].try_into().unwrap());
        let element_count = u32::from_le_bytes(buf[24..28].try_into().unwrap());

        let Some(kind) = SectionKind::from_u32(kind_raw) else {
            return Ok(None);
        };
        let compression = Compression::from_u32(comp_raw).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "unknown compression")
        })?;
        Ok(Some(Self {
            kind,
            compression,
            byte_offset,
            byte_len,
            compressed_byte_len,
            element_count,
        }))
    }
}

// ── Texture types ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TexFormat {
    Rgba8,
    Bc4,
    Bc5,
    Bc7,
}

impl TexFormat {
    pub fn to_u32(self) -> u32 {
        match self {
            Self::Rgba8 => 0,
            Self::Bc4 => 1,
            Self::Bc5 => 2,
            Self::Bc7 => 3,
        }
    }

    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Rgba8),
            1 => Some(Self::Bc4),
            2 => Some(Self::Bc5),
            3 => Some(Self::Bc7),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorSpace {
    Srgb,
    Linear,
}

impl ColorSpace {
    pub fn to_u32(self) -> u32 {
        match self {
            Self::Srgb => 0,
            Self::Linear => 1,
        }
    }

    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Srgb),
            1 => Some(Self::Linear),
            _ => None,
        }
    }
}
