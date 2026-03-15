use std::{
    collections::{BTreeMap, HashMap},
    fs,
    io::{self, Read},
    path::{Path, PathBuf},
};

#[cfg(unix)]
use std::os::unix::fs::FileExt as OsFileExtUnix;
#[cfg(windows)]
use std::os::windows::fs::FileExt as OsFileExtWindows;

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
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Self::U16(v) => v.len(),
            Self::U32(v) => v.len(),
        }
    }

    #[inline]
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

/// Iterator over fixed-size elements stored in a section. Parameterized
/// by a const `N` (bytes per element). Produces arrays `[u8; N]` so
/// callers can decode into typed values without allocations.
pub struct SectionElemIter<const N: usize> {
    file: fs::File,
    elements_remaining: usize,
    // positional offset into the raw stream (compressed or uncompressed)
    offset: u64,
    decoder: Option<lz4_flex::frame::FrameDecoder<PositionalReader>>,
    decomp_buf: Vec<u8>,
    decomp_pos: usize,
}

// Typed iterator wrappers for specific section element types.
pub struct PositionsIter(pub SectionElemIter<12>);
pub struct NormalsIter(pub SectionElemIter<12>);
pub struct TangentsIter(pub SectionElemIter<16>);
pub struct TexCoords2Iter(pub SectionElemIter<8>);

pub enum IndicesIter {
    U16(SectionElemIter<2>),
    U32(SectionElemIter<4>),
}

impl Iterator for PositionsIter {
    type Item = [f32; 3];
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|chunk| {
            let mut elem = [0f32; 3];
            for (i, b) in chunk.chunks_exact(4).enumerate() {
                elem[i] = f32::from_le_bytes(b.try_into().unwrap());
            }
            elem
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl Iterator for NormalsIter {
    type Item = [f32; 3];
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|chunk| {
            let mut elem = [0f32; 3];
            for (i, b) in chunk.chunks_exact(4).enumerate() {
                elem[i] = f32::from_le_bytes(b.try_into().unwrap());
            }
            elem
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl Iterator for TangentsIter {
    type Item = [f32; 4];
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|chunk| {
            let mut elem = [0f32; 4];
            for (i, b) in chunk.chunks_exact(4).enumerate() {
                elem[i] = f32::from_le_bytes(b.try_into().unwrap());
            }
            elem
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl Iterator for TexCoords2Iter {
    type Item = [f32; 2];
    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|chunk| {
            let mut elem = [0f32; 2];
            for (i, b) in chunk.chunks_exact(4).enumerate() {
                elem[i] = f32::from_le_bytes(b.try_into().unwrap());
            }
            elem
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.0.size_hint()
    }
}

impl Iterator for IndicesIter {
    type Item = u32;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            IndicesIter::U16(iter) => iter.next().map(|chunk| {
                let v = u16::from_le_bytes(chunk);
                v as u32
            }),
            IndicesIter::U32(iter) => iter.next().map(u32::from_le_bytes),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            IndicesIter::U16(iter) => iter.size_hint(),
            IndicesIter::U32(iter) => iter.size_hint(),
        }
    }
}

impl<const N: usize> Iterator for SectionElemIter<N> {
    type Item = [u8; N];

    fn next(&mut self) -> Option<Self::Item> {
        if self.elements_remaining == 0 {
            return None;
        }

        self.elements_remaining -= 1;

        if let Some(dec) = &mut self.decoder {
            while self.decomp_buf.len() - self.decomp_pos < N {
                let mut tmp = vec![0u8; 4096];
                match dec.read(&mut tmp) {
                    Ok(0) => break,
                    Ok(n) => self.decomp_buf.extend_from_slice(&tmp[..n]),
                    Err(_) => return None,
                }
            }
            let start = self.decomp_pos;
            let end = start + N;
            if end > self.decomp_buf.len() {
                return None;
            }
            let slice = &self.decomp_buf[start..end];
            let mut arr = [0u8; N];
            arr.copy_from_slice(slice);
            self.decomp_pos = end;
            return Some(arr);
        }

        // uncompressed: read exactly N bytes
        let mut buf = vec![0u8; N];
        if read_exact_at(&self.file, self.offset, &mut buf).is_err() {
            return None;
        }
        self.offset = self.offset.wrapping_add(N as u64);
        let mut arr = [0u8; N];
        arr.copy_from_slice(&buf);
        Some(arr)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.elements_remaining, Some(self.elements_remaining))
    }
}

impl<const N: usize> ExactSizeIterator for SectionElemIter<N> {
    #[inline]
    fn len(&self) -> usize {
        self.elements_remaining
    }
}

impl ExactSizeIterator for PositionsIter {
    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl ExactSizeIterator for NormalsIter {
    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl ExactSizeIterator for TangentsIter {
    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl ExactSizeIterator for TexCoords2Iter {
    #[inline]
    fn len(&self) -> usize {
        self.0.len()
    }
}

impl ExactSizeIterator for IndicesIter {
    #[inline]
    fn len(&self) -> usize {
        match self {
            IndicesIter::U16(iter) => iter.len(),
            IndicesIter::U32(iter) => iter.len(),
        }
    }
}

// A reader that reads from a file at a moving offset using positional
// reads. It tracks remaining compressed bytes and advances offset as
// bytes are read.
pub struct PositionalReader {
    file: fs::File,
    offset: u64,
    remaining: usize,
}

impl Read for PositionalReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if self.remaining == 0 {
            return Ok(0);
        }
        let to_read = std::cmp::min(buf.len(), self.remaining);
        let n = read_at_partial(&self.file, self.offset, &mut buf[..to_read])?;
        self.offset = self.offset.wrapping_add(n as u64);
        self.remaining -= n;
        Ok(n)
    }
}

/// Read up to `buf.len()` bytes from `file` at `offset`. Returns the
/// number of bytes read. Uses platform `read_at`/`seek_read`.
fn read_at_partial(
    file: &fs::File,
    offset: u64,
    buf: &mut [u8],
) -> io::Result<usize> {
    #[cfg(unix)]
    {
        <fs::File as OsFileExtUnix>::read_at(file, buf, offset)
    }
    #[cfg(windows)]
    {
        <fs::File as OsFileExtWindows>::seek_read(file, buf, offset)
    }
    #[cfg(not(any(unix, windows)))]
    {
        compile_error!(
            "read_at_partial is only implemented for unix and windows"
        );
    }
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
            let raw = read_section_data(&file, hdr, Compression::None)?;
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

    pub fn positions(&self) -> Result<PositionsIter, LoadError> {
        let iter = self.section_elems::<12>(SectionKind::MeshPositions)?;
        Ok(PositionsIter(iter))
    }

    pub fn normals(&self) -> Result<NormalsIter, LoadError> {
        let iter = self.section_elems::<12>(SectionKind::MeshNormals)?;
        Ok(NormalsIter(iter))
    }

    pub fn tangents(&self) -> Result<TangentsIter, LoadError> {
        let iter = self.section_elems::<16>(SectionKind::MeshTangents)?;
        Ok(TangentsIter(iter))
    }

    pub fn tex_coords0(&self) -> Result<TexCoords2Iter, LoadError> {
        let iter = self.section_elems::<8>(SectionKind::MeshTexCoord0)?;
        Ok(TexCoords2Iter(iter))
    }

    pub fn tex_coords1(&self) -> Result<Option<TexCoords2Iter>, LoadError> {
        let Some(_) = self
            .sections
            .iter()
            .find(|h| h.kind == SectionKind::MeshTexCoord1)
        else {
            return Ok(None);
        };
        let iter = self.section_elems::<8>(SectionKind::MeshTexCoord1)?;
        Ok(Some(TexCoords2Iter(iter)))
    }

    pub fn indices(&self) -> Result<IndicesIter, LoadError> {
        if self
            .sections
            .iter()
            .any(|h| h.kind == SectionKind::MeshIndices16)
        {
            let iter = self.section_elems::<2>(SectionKind::MeshIndices16)?;
            Ok(IndicesIter::U16(iter))
        } else if self
            .sections
            .iter()
            .any(|h| h.kind == SectionKind::MeshIndices32)
        {
            let iter = self.section_elems::<4>(SectionKind::MeshIndices32)?;
            Ok(IndicesIter::U32(iter))
        } else {
            Err(fmt_err("missing indices section"))
        }
    }

    /// Clone the underlying file handle for concurrent access.
    pub fn try_clone(&self) -> Result<Self, io::Error> {
        Ok(Self {
            file: self.file.try_clone()?,
            sections: self.sections.clone(),
            tex_refs: self.tex_refs.clone(),
        })
    }

    /// Return an iterator that yields raw element arrays for the
    /// specified `SectionKind`. Each item is exactly `N` bytes.
    pub fn section_elems<const N: usize>(
        &self,
        kind: SectionKind,
    ) -> Result<SectionElemIter<N>, LoadError> {
        let hdr = self.require(kind)?;
        let offset = hdr.byte_offset as u64;
        let file_for_iter = self.file.try_clone()?;

        match mesh_section_compression(kind) {
            Compression::None => Ok(SectionElemIter {
                file: file_for_iter,
                elements_remaining: hdr.element_count as usize,
                offset,
                decoder: None,
                decomp_buf: Vec::new(),
                decomp_pos: 0,
            }),
            Compression::Lz4 => {
                let file_for_reader = self.file.try_clone()?;
                let reader = PositionalReader {
                    file: file_for_reader,
                    offset,
                    remaining: hdr.compressed_byte_len as usize,
                };
                let dec = lz4_flex::frame::FrameDecoder::new(reader);
                Ok(SectionElemIter {
                    file: file_for_iter,
                    elements_remaining: hdr.element_count as usize,
                    offset,
                    decoder: Some(dec),
                    decomp_buf: Vec::new(),
                    decomp_pos: 0,
                })
            }
        }
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
    compression: Compression,
) -> Result<Vec<u8>, LoadError> {
    let offset = hdr.byte_offset as u64;

    match compression {
        Compression::None => {
            let mut buf = vec![0u8; hdr.byte_len as usize];
            read_exact_at(file, offset, &mut buf)?;
            Ok(buf)
        }
        Compression::Lz4 => {
            // Read compressed bytes into a buffer, then feed to
            // FrameDecoder via Cursor so it gets an exact-length
            // stream.
            let mut compressed = vec![0u8; hdr.compressed_byte_len as usize];
            read_exact_at(file, offset, &mut compressed)?;
            let mut dec =
                lz4_flex::frame::FrameDecoder::new(io::Cursor::new(compressed));
            let mut out = Vec::with_capacity(hdr.byte_len as usize);
            dec.read_to_end(&mut out)
                .map_err(|e| fmt_err(format!("lz4 decompress: {e}")))?;
            Ok(out)
        }
    }
}

/// Compression used for a given mesh section kind. All vertex/index
/// sections are Lz4; MeshTexRef is uncompressed.
fn mesh_section_compression(kind: SectionKind) -> Compression {
    match kind {
        SectionKind::MeshTexRef => Compression::None,
        _ => Compression::Lz4,
    }
}

/// Read exactly `buf.len()` bytes from `file` at `offset` without
/// changing the file cursor. Uses platform-specific `read_at` when
/// available and falls back to clone+seek on other platforms.
fn read_exact_at(
    file: &fs::File,
    mut offset: u64,
    mut buf: &mut [u8],
) -> io::Result<()> {
    #[cfg(unix)]
    {
        while !buf.is_empty() {
            let n = <fs::File as OsFileExtUnix>::read_at(file, buf, offset)?;
            if n == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "early EOF",
                ));
            }
            offset = offset.wrapping_add(n as u64);
            let tmp = buf;
            buf = &mut tmp[n..];
        }
        Ok(())
    }

    #[cfg(windows)]
    {
        while !buf.is_empty() {
            let n =
                <fs::File as OsFileExtWindows>::seek_read(file, buf, offset)?;
            if n == 0 {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "early EOF",
                ));
            }
            offset = offset.wrapping_add(n as u64);
            let tmp = buf;
            buf = &mut tmp[n..];
        }
        Ok(())
    }

    #[cfg(not(any(unix, windows)))]
    {
        compile_error!(
            "read_exact_at is only implemented for unix and windows; \
            implement platform-specific read_at/seek_read or add support"
        );
    }
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
    pub compression: Compression,
}

fn parse_texture_info(raw: &[u8]) -> Result<TextureInfo, LoadError> {
    if raw.len() < 24 {
        return Err(fmt_err(format!(
            "TextureInfo: expected 24B, got {}B",
            raw.len()
        )));
    }
    let fmt_raw = u32::from_le_bytes(raw[0..4].try_into().unwrap());
    let cs_raw = u32::from_le_bytes(raw[4..8].try_into().unwrap());
    let width = u32::from_le_bytes(raw[8..12].try_into().unwrap());
    let height = u32::from_le_bytes(raw[12..16].try_into().unwrap());
    let mip_count = u32::from_le_bytes(raw[16..20].try_into().unwrap());
    let comp_raw = u32::from_le_bytes(raw[20..24].try_into().unwrap());
    let format = TexFormat::from_u32(fmt_raw)
        .ok_or_else(|| fmt_err(format!("unknown TexFormat {fmt_raw}")))?;
    let color_space = ColorSpace::from_u32(cs_raw)
        .ok_or_else(|| fmt_err(format!("unknown ColorSpace {cs_raw}")))?;
    let compression = Compression::from_u32(comp_raw)
        .ok_or_else(|| fmt_err(format!("unknown Compression {comp_raw}")))?;
    Ok(TextureInfo {
        format,
        color_space,
        width,
        height,
        mip_count,
        compression,
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
        let raw = read_section_data(&file, info_hdr, Compression::None)?;
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
        read_section_data(&self.file, hdr, self.info.compression)
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
