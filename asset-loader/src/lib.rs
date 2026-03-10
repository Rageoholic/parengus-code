use std::{
    collections::{BTreeMap, HashMap},
    fs,
    io::{self, Read, Seek, SeekFrom},
    path::{Path, PathBuf},
};

use asset_shared::{
    AssetId, ColorSpace, Compression, FileHeader, PMESH_MAGIC, PTEX_MAGIC,
    SectionHeader, SectionKind, TexFormat, TexRole, TextureId, VERSION, fnv1a,
};

// ── Error type ────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum LoadError {
    Io(io::Error),
    Format(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "io: {e}"),
            Self::Format(s) => write!(f, "format: {s}"),
        }
    }
}

impl From<io::Error> for LoadError {
    fn from(e: io::Error) -> Self {
        Self::Io(e)
    }
}

fn fmt_err(s: impl Into<String>) -> LoadError {
    LoadError::Format(s.into())
}

// ── AssetMap ──────────────────────────────────────────────────────────────────

/// Runtime asset map: name → compiled asset path.
/// Written by xtask; loaded once at app startup.
/// Provides O(1) lookup by `AssetId<T>` (FNV-1a hash).
#[derive(Debug)]
pub struct AssetMap {
    by_hash: HashMap<u64, PathBuf>,
}

#[derive(serde::Deserialize)]
struct RawAssetMap {
    map: BTreeMap<String, String>,
}

impl AssetMap {
    pub fn load(path: &Path) -> Result<Self, LoadError> {
        let text = fs::read_to_string(path)?;
        let raw: RawAssetMap = toml::from_str(&text)
            .map_err(|e| fmt_err(format!("parse asset map: {e}")))?;
        let by_hash = raw
            .map
            .into_iter()
            .map(|(name, p)| (fnv1a(&name), PathBuf::from(p)))
            .collect();
        Ok(Self { by_hash })
    }

    pub fn get<T>(&self, id: AssetId<T>) -> Option<&Path> {
        self.by_hash.get(&id.0).map(PathBuf::as_path)
    }
}

// ── Indices ───────────────────────────────────────────────────────────────────

pub enum Indices {
    U16(Vec<u16>),
    U32(Vec<u32>),
}

impl Indices {
    pub fn len(&self) -> usize {
        match self {
            Self::U16(v) => v.len(),
            Self::U32(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns an iterator of `u32` regardless of storage width.
    pub fn iter_u32(&self) -> impl Iterator<Item = u32> + '_ {
        let (u16s, u32s): (&[u16], &[u32]) = match self {
            Self::U16(v) => (v, &[]),
            Self::U32(v) => (&[], v),
        };
        u16s.iter().map(|&i| i as u32).chain(u32s.iter().copied())
    }
}

// ── MeshAsset ─────────────────────────────────────────────────────────────────

/// Opened mesh file. Section headers and tex-refs are loaded eagerly;
/// vertex/index data is read on demand via `File::try_clone()` +
/// seek, so multiple sections can be accessed without keeping the
/// entire file in memory at once.
pub struct MeshAsset {
    file: fs::File,
    sections: Vec<SectionHeader>,
    pub tex_refs: Vec<(TexRole, TextureId)>,
}

impl MeshAsset {
    pub fn open(path: &Path) -> Result<Self, LoadError> {
        let mut file = fs::File::open(path)?;

        let fhdr = FileHeader::read_from(&mut file)
            .map_err(|e| fmt_err(format!("file header: {e}")))?;

        if fhdr.magic != PMESH_MAGIC {
            return Err(fmt_err(format!("bad magic: {:#010x}", fhdr.magic)));
        }
        if fhdr.version != VERSION {
            return Err(fmt_err(format!(
                "unsupported version {}",
                fhdr.version
            )));
        }

        let mut sections = Vec::with_capacity(fhdr.section_count as usize);
        for _ in 0..fhdr.section_count {
            if let Some(h) = SectionHeader::read_from(&mut file)
                .map_err(|e| fmt_err(format!("section header: {e}")))?
            {
                sections.push(h);
            }
        }

        // Decode tex_refs eagerly — small, uncompressed
        let tex_refs = if let Some(hdr) =
            sections.iter().find(|h| h.kind == SectionKind::MeshTexRef)
        {
            let raw = read_section_data(&file, hdr)?;
            decode_tex_refs(&raw, hdr.element_count as usize)?
        } else {
            Vec::new()
        };

        Ok(Self {
            file,
            sections,
            tex_refs,
        })
    }

    pub fn positions(&self) -> Result<Vec<[f32; 3]>, LoadError> {
        let hdr = self.require(SectionKind::MeshPositions)?;
        let raw = read_section_data(&self.file, hdr)?;
        decode_f32s::<3>(raw, hdr.element_count as usize, "positions")
    }

    pub fn normals(&self) -> Result<Vec<[f32; 3]>, LoadError> {
        let hdr = self.require(SectionKind::MeshNormals)?;
        let raw = read_section_data(&self.file, hdr)?;
        decode_f32s::<3>(raw, hdr.element_count as usize, "normals")
    }

    pub fn tangents(&self) -> Result<Vec<[f32; 4]>, LoadError> {
        let hdr = self.require(SectionKind::MeshTangents)?;
        let raw = read_section_data(&self.file, hdr)?;
        decode_f32s::<4>(raw, hdr.element_count as usize, "tangents")
    }

    pub fn tex_coords0(&self) -> Result<Vec<[f32; 2]>, LoadError> {
        let hdr = self.require(SectionKind::MeshTexCoord0)?;
        let raw = read_section_data(&self.file, hdr)?;
        decode_f32s::<2>(raw, hdr.element_count as usize, "tex_coords0")
    }

    pub fn tex_coords1(&self) -> Result<Option<Vec<[f32; 2]>>, LoadError> {
        let Some(hdr) = self
            .sections
            .iter()
            .find(|h| h.kind == SectionKind::MeshTexCoord1)
        else {
            return Ok(None);
        };
        let raw = read_section_data(&self.file, hdr)?;
        decode_f32s::<2>(raw, hdr.element_count as usize, "tex_coords1")
            .map(Some)
    }

    pub fn indices(&self) -> Result<Indices, LoadError> {
        if let Some(hdr) = self
            .sections
            .iter()
            .find(|h| h.kind == SectionKind::MeshIndices16)
        {
            let raw = read_section_data(&self.file, hdr)?;
            return Ok(Indices::U16(decode_u16s(
                raw,
                hdr.element_count as usize,
            )?));
        }
        if let Some(hdr) = self
            .sections
            .iter()
            .find(|h| h.kind == SectionKind::MeshIndices32)
        {
            let raw = read_section_data(&self.file, hdr)?;
            return Ok(Indices::U32(decode_u32s(
                raw,
                hdr.element_count as usize,
            )?));
        }
        Err(fmt_err("missing indices section"))
    }

    /// Clone the underlying file handle for concurrent access.
    pub fn try_clone(&self) -> Result<Self, io::Error> {
        Ok(Self {
            file: self.file.try_clone()?,
            sections: self.sections.clone(),
            tex_refs: self.tex_refs.clone(),
        })
    }

    fn require(&self, kind: SectionKind) -> Result<&SectionHeader, LoadError> {
        self.sections
            .iter()
            .find(|h| h.kind == kind)
            .ok_or_else(|| fmt_err(format!("missing {kind:?} section")))
    }
}

// ── Section reading ───────────────────────────────────────────────────────────

/// Read a section by cloning the file handle and seeking to the
/// section's byte offset, so callers can read multiple sections
/// without interfering with each other's position.
fn read_section_data(
    file: &fs::File,
    hdr: &SectionHeader,
) -> Result<Vec<u8>, LoadError> {
    let mut f = file.try_clone()?;
    f.seek(SeekFrom::Start(hdr.byte_offset as u64))?;

    match hdr.compression {
        Compression::None => {
            let mut buf = vec![0u8; hdr.byte_len as usize];
            f.read_exact(&mut buf)?;
            Ok(buf)
        }
        Compression::Lz4 => {
            // Read compressed bytes into a buffer, then feed to
            // FrameDecoder via Cursor so it gets an exact-length
            // stream.
            let mut compressed = vec![0u8; hdr.compressed_byte_len as usize];
            f.read_exact(&mut compressed)?;
            let mut dec =
                lz4_flex::frame::FrameDecoder::new(io::Cursor::new(compressed));
            let mut out = Vec::with_capacity(hdr.byte_len as usize);
            dec.read_to_end(&mut out)
                .map_err(|e| fmt_err(format!("lz4 decompress: {e}")))?;
            Ok(out)
        }
    }
}

// ── Decode helpers ────────────────────────────────────────────────────────────

fn decode_f32s<const N: usize>(
    raw: Vec<u8>,
    elem_count: usize,
    label: &str,
) -> Result<Vec<[f32; N]>, LoadError> {
    let expected = elem_count * N * 4;
    if raw.len() != expected {
        return Err(fmt_err(format!(
            "{label}: expected {expected}B, got {}B",
            raw.len()
        )));
    }
    let mut out = Vec::with_capacity(elem_count);
    for chunk in raw.chunks_exact(N * 4) {
        let mut elem = [0f32; N];
        for (i, b) in chunk.chunks_exact(4).enumerate() {
            elem[i] = f32::from_le_bytes(b.try_into().unwrap());
        }
        out.push(elem);
    }
    Ok(out)
}

fn decode_u16s(raw: Vec<u8>, elem_count: usize) -> Result<Vec<u16>, LoadError> {
    let expected = elem_count * 2;
    if raw.len() != expected {
        return Err(fmt_err(format!(
            "indices16: expected {expected}B, got {}B",
            raw.len()
        )));
    }
    Ok(raw
        .chunks_exact(2)
        .map(|b| u16::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn decode_u32s(raw: Vec<u8>, elem_count: usize) -> Result<Vec<u32>, LoadError> {
    let expected = elem_count * 4;
    if raw.len() != expected {
        return Err(fmt_err(format!(
            "indices32: expected {expected}B, got {}B",
            raw.len()
        )));
    }
    Ok(raw
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect())
}

fn decode_tex_refs(
    raw: &[u8],
    count: usize,
) -> Result<Vec<(TexRole, TextureId)>, LoadError> {
    if raw.len() != count * 12 {
        return Err(fmt_err(format!(
            "tex_refs: expected {}B, got {}B",
            count * 12,
            raw.len()
        )));
    }
    let mut out = Vec::with_capacity(count);
    for chunk in raw.chunks_exact(12) {
        let role_raw = u32::from_le_bytes(chunk[0..4].try_into().unwrap());
        let id_raw = u64::from_le_bytes(chunk[4..12].try_into().unwrap());
        let role = TexRole::from_u32(role_raw)
            .ok_or_else(|| fmt_err(format!("unknown tex role {role_raw}")))?;
        out.push((role, AssetId::from_hash(id_raw)));
    }
    Ok(out)
}

// ── TextureInfo ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub struct TextureInfo {
    pub format: TexFormat,
    pub color_space: ColorSpace,
    pub width: u32,
    pub height: u32,
    pub mip_count: u32,
}

fn parse_texture_info(raw: &[u8]) -> Result<TextureInfo, LoadError> {
    if raw.len() < 20 {
        return Err(fmt_err(format!(
            "TextureInfo: expected 20B, got {}B",
            raw.len()
        )));
    }
    let fmt_raw = u32::from_le_bytes(raw[0..4].try_into().unwrap());
    let cs_raw = u32::from_le_bytes(raw[4..8].try_into().unwrap());
    let width = u32::from_le_bytes(raw[8..12].try_into().unwrap());
    let height = u32::from_le_bytes(raw[12..16].try_into().unwrap());
    let mip_count = u32::from_le_bytes(raw[16..20].try_into().unwrap());
    let format = TexFormat::from_u32(fmt_raw)
        .ok_or_else(|| fmt_err(format!("unknown TexFormat {fmt_raw}")))?;
    let color_space = ColorSpace::from_u32(cs_raw)
        .ok_or_else(|| fmt_err(format!("unknown ColorSpace {cs_raw}")))?;
    Ok(TextureInfo {
        format,
        color_space,
        width,
        height,
        mip_count,
    })
}

// ── TexAsset ──────────────────────────────────────────────────────────────────

/// Opened texture file.  `TextureInfo` and section headers are
/// loaded eagerly; mip pixel data is read on demand via
/// `File::try_clone()` + seek.
pub struct TexAsset {
    file: fs::File,
    sections: Vec<SectionHeader>,
    pub info: TextureInfo,
}

impl TexAsset {
    pub fn open(path: &Path) -> Result<Self, LoadError> {
        let mut file = fs::File::open(path)?;

        let fhdr = FileHeader::read_from(&mut file)
            .map_err(|e| fmt_err(format!("file header: {e}")))?;

        if fhdr.magic != PTEX_MAGIC {
            return Err(fmt_err(format!("bad magic: {:#010x}", fhdr.magic)));
        }
        if fhdr.version != VERSION {
            return Err(fmt_err(format!(
                "unsupported version {}",
                fhdr.version
            )));
        }

        let mut sections = Vec::with_capacity(fhdr.section_count as usize);
        for _ in 0..fhdr.section_count {
            if let Some(h) = SectionHeader::read_from(&mut file)
                .map_err(|e| fmt_err(format!("section header: {e}")))?
            {
                sections.push(h);
            }
        }

        // TextureInfo must be the first section
        let info_hdr = sections
            .first()
            .filter(|h| h.kind == SectionKind::TextureInfo)
            .ok_or_else(|| fmt_err("missing TextureInfo section"))?;
        let raw = read_section_data(&file, info_hdr)?;
        let info = parse_texture_info(&raw)?;

        Ok(Self {
            file,
            sections,
            info,
        })
    }

    /// Decompress and return the pixel/block data for mip `level`
    /// (0 = full resolution).
    pub fn mip(&self, level: usize) -> Result<Vec<u8>, LoadError> {
        let mip_hdrs: Vec<&SectionHeader> = self
            .sections
            .iter()
            .filter(|h| h.kind == SectionKind::TextureMip)
            .collect();

        let hdr = mip_hdrs.get(level).ok_or_else(|| {
            fmt_err(format!(
                "mip level {level} out of range ({})",
                mip_hdrs.len()
            ))
        })?;
        read_section_data(&self.file, hdr)
    }

    /// Clone the underlying file handle for concurrent access.
    pub fn try_clone(&self) -> Result<Self, io::Error> {
        Ok(Self {
            file: self.file.try_clone()?,
            sections: self.sections.clone(),
            info: self.info,
        })
    }
}
