use std::{
    io::{self, Read, Write},
    marker::PhantomData,
};

pub const PMESH_MAGIC: u32 = u32::from_le_bytes(*b"PMSH");
pub const PTEX_MAGIC: u32 = u32::from_le_bytes(*b"PTEX");
pub const VERSION: u16 = 1;

// ── AssetId ──────────────────────────────────────────────────────────────────

/// Phantom marker for mesh assets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Mesh;
/// Phantom marker for texture assets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Texture;
/// Phantom marker for shader assets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Shader;

/// FNV-1a 64-bit hash of an asset name, typed by phantom `T`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AssetId<T>(pub u64, PhantomData<fn() -> T>);

impl<T> AssetId<T> {
    /// Construct from a raw hash value (e.g. read from a binary
    /// asset file).
    #[inline]
    pub fn from_hash(hash: u64) -> Self {
        Self(hash, PhantomData)
    }
}

pub type MeshId = AssetId<Mesh>;
pub type TextureId = AssetId<Texture>;
pub type ShaderId = AssetId<Shader>;

const FNV_OFFSET: u64 = 14695981039346656037;
const FNV_PRIME: u64 = 1099511628211;

pub const fn fnv1a(s: &str) -> u64 {
    let bytes = s.as_bytes();
    let mut hash = FNV_OFFSET;
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
        i += 1;
    }
    hash
}

#[inline]
pub const fn mesh_id(name: &str) -> MeshId {
    AssetId(fnv1a(name), PhantomData)
}

#[inline]
pub const fn texture_id(name: &str) -> TextureId {
    AssetId(fnv1a(name), PhantomData)
}

#[inline]
pub const fn shader_id(name: &str) -> ShaderId {
    AssetId(fnv1a(name), PhantomData)
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
    TextureInfo,
}

impl SectionKind {
    #[inline]
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
            Self::TextureInfo => 200,
        }
    }

    /// Returns `None` for unknown kinds; callers should skip those
    /// sections rather than fail, to allow forward compatibility.
    #[inline]
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
            200 => Some(Self::TextureInfo),
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
    #[inline]
    pub fn to_u32(self) -> u32 {
        match self {
            Self::None => 0,
            Self::Lz4 => 1,
        }
    }

    #[inline]
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::None),
            1 => Some(Self::Lz4),
            _ => None,
        }
    }
}

// ── TexRole ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TexRole {
    Albedo,
    Normal,
    MetallicRoughness,
    Emissive,
    Occlusion,
}

impl TexRole {
    #[inline]
    pub fn to_u32(self) -> u32 {
        match self {
            Self::Albedo => 0,
            Self::Normal => 1,
            Self::MetallicRoughness => 2,
            Self::Emissive => 3,
            Self::Occlusion => 4,
        }
    }

    #[inline]
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Albedo),
            1 => Some(Self::Normal),
            2 => Some(Self::MetallicRoughness),
            3 => Some(Self::Emissive),
            4 => Some(Self::Occlusion),
            _ => None,
        }
    }
}

// ── FileHeader ───────────────────────────────────────────────────────────────

/// Serialized size: 10 bytes.
pub struct FileHeader {
    pub magic: u32,
    pub version: u16,
    pub section_count: u32,
}

impl FileHeader {
    pub fn write_to(&self, w: &mut impl Write) -> io::Result<()> {
        w.write_all(&self.magic.to_le_bytes())?;
        w.write_all(&self.version.to_le_bytes())?;
        w.write_all(&self.section_count.to_le_bytes())?;
        Ok(())
    }

    pub fn read_from(r: &mut impl Read) -> io::Result<Self> {
        let mut buf = [0u8; 10];
        r.read_exact(&mut buf)?;
        let magic = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        let version = u16::from_le_bytes(buf[4..6].try_into().unwrap());
        let section_count = u32::from_le_bytes(buf[6..10].try_into().unwrap());
        Ok(Self {
            magic,
            version,
            section_count,
        })
    }
}

// ── SectionHeader ────────────────────────────────────────────────────────────

/// Serialized size: 20 bytes (no padding or reserved fields).
#[derive(Clone)]
pub struct SectionHeader {
    pub kind: SectionKind,
    pub byte_offset: u32,
    pub byte_len: u32,
    pub compressed_byte_len: u32,
    pub element_count: u32,
}

impl SectionHeader {
    pub fn write_to(&self, w: &mut impl Write) -> io::Result<()> {
        w.write_all(&self.kind.to_u32().to_le_bytes())?;
        w.write_all(&self.byte_offset.to_le_bytes())?;
        w.write_all(&self.byte_len.to_le_bytes())?;
        w.write_all(&self.compressed_byte_len.to_le_bytes())?;
        w.write_all(&self.element_count.to_le_bytes())?;
        Ok(())
    }

    /// Returns `Ok(None)` for unknown `kind`; caller should skip the
    /// section.
    pub fn read_from(r: &mut impl Read) -> io::Result<Option<Self>> {
        let mut buf = [0u8; 20];
        r.read_exact(&mut buf)?;
        let kind_raw = u32::from_le_bytes(buf[0..4].try_into().unwrap());
        let byte_offset = u32::from_le_bytes(buf[4..8].try_into().unwrap());
        let byte_len = u32::from_le_bytes(buf[8..12].try_into().unwrap());
        let compressed_byte_len =
            u32::from_le_bytes(buf[12..16].try_into().unwrap());
        let element_count = u32::from_le_bytes(buf[16..20].try_into().unwrap());

        let Some(kind) = SectionKind::from_u32(kind_raw) else {
            return Ok(None);
        };
        Ok(Some(Self {
            kind,
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
    #[inline]
    pub fn to_u32(self) -> u32 {
        match self {
            Self::Rgba8 => 0,
            Self::Bc4 => 1,
            Self::Bc5 => 2,
            Self::Bc7 => 3,
        }
    }

    #[inline]
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
    #[inline]
    pub fn to_u32(self) -> u32 {
        match self {
            Self::Srgb => 0,
            Self::Linear => 1,
        }
    }

    #[inline]
    pub fn from_u32(v: u32) -> Option<Self> {
        match v {
            0 => Some(Self::Srgb),
            1 => Some(Self::Linear),
            _ => None,
        }
    }
}
