use std::{
    fs,
    io::{self, Cursor},
    path::Path,
};

use asset_shared::{
    AssetKind, Compression, FileHeader, PMESH_MAGIC, SectionHeader, SectionKind,
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

pub struct MeshAsset {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub tangents: Vec<[f32; 4]>,
    pub tex_coords0: Vec<[f32; 2]>,
    pub tex_coords1: Option<Vec<[f32; 2]>>,
    pub indices: Indices,
    pub tex_ref: Option<String>,
}

// ── Decompression ─────────────────────────────────────────────────────────────

fn decompress(hdr: &SectionHeader, file: &[u8]) -> Result<Vec<u8>, LoadError> {
    let blob = file
        .get(
            hdr.byte_offset as usize
                ..hdr.byte_offset as usize + hdr.compressed_byte_len as usize,
        )
        .ok_or_else(|| fmt_err("section data out of bounds"))?;

    match hdr.compression {
        Compression::None => Ok(blob.to_vec()),
        Compression::Lz4 => lz4_flex::decompress_size_prepended(blob)
            .map_err(|e| fmt_err(format!("lz4 decompress: {e}"))),
    }
}

// ── Byte → typed array helpers ─────────────────────────────────────────────

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
        for (i, bytes) in chunk.chunks_exact(4).enumerate() {
            elem[i] = f32::from_le_bytes(bytes.try_into().unwrap());
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

// ── Public load function ───────────────────────────────────────────────────

pub fn load_mesh(path: &Path) -> Result<MeshAsset, LoadError> {
    let file = fs::read(path)?;
    let mut cursor = Cursor::new(file.as_slice());

    // ── File header ───────────────────────────────────────────────────────
    let fhdr = FileHeader::read_from(&mut cursor)
        .map_err(|e| fmt_err(format!("file header: {e}")))?;

    if fhdr.magic != PMESH_MAGIC {
        return Err(fmt_err(format!("bad magic: {:#010x}", fhdr.magic)));
    }
    if fhdr.asset_kind != AssetKind::Mesh {
        return Err(fmt_err("not a mesh asset"));
    }

    // ── Section headers ───────────────────────────────────────────────────
    let mut section_headers = Vec::with_capacity(fhdr.section_count as usize);
    for _ in 0..fhdr.section_count {
        // read_from consumes 28B regardless; returns None for
        // unknown kinds (forward-compat — skip those sections).
        if let Some(h) = SectionHeader::read_from(&mut cursor)
            .map_err(|e| fmt_err(format!("section header: {e}")))?
        {
            section_headers.push(h);
        }
    }

    // ── Decode sections ───────────────────────────────────────────────────
    let mut positions: Option<Vec<[f32; 3]>> = None;
    let mut normals: Option<Vec<[f32; 3]>> = None;
    let mut tangents: Option<Vec<[f32; 4]>> = None;
    let mut tex_coords0: Option<Vec<[f32; 2]>> = None;
    let mut tex_coords1: Option<Vec<[f32; 2]>> = None;
    let mut indices: Option<Indices> = None;
    let mut tex_ref: Option<String> = None;

    for hdr in &section_headers {
        let n = hdr.element_count as usize;
        let raw = decompress(hdr, &file)?;
        match hdr.kind {
            SectionKind::MeshPositions => {
                positions = Some(decode_f32s::<3>(raw, n, "positions")?);
            }
            SectionKind::MeshNormals => {
                normals = Some(decode_f32s::<3>(raw, n, "normals")?);
            }
            SectionKind::MeshTangents => {
                tangents = Some(decode_f32s::<4>(raw, n, "tangents")?);
            }
            SectionKind::MeshTexCoord0 => {
                tex_coords0 = Some(decode_f32s::<2>(raw, n, "tex_coords0")?);
            }
            SectionKind::MeshTexCoord1 => {
                tex_coords1 = Some(decode_f32s::<2>(raw, n, "tex_coords1")?);
            }
            SectionKind::MeshIndices16 => {
                indices = Some(Indices::U16(decode_u16s(raw, n)?));
            }
            SectionKind::MeshIndices32 => {
                indices = Some(Indices::U32(decode_u32s(raw, n)?));
            }
            SectionKind::MeshTexRef => {
                tex_ref = Some(
                    String::from_utf8(raw)
                        .map_err(|_| fmt_err("tex_ref is not valid UTF-8"))?,
                );
            }
            // Forward-compat: known-but-unhandled section kinds
            SectionKind::TextureMip => {}
        }
    }

    Ok(MeshAsset {
        positions: positions.ok_or_else(|| fmt_err("missing MeshPositions"))?,
        normals: normals.ok_or_else(|| fmt_err("missing MeshNormals"))?,
        tangents: tangents.ok_or_else(|| fmt_err("missing MeshTangents"))?,
        tex_coords0: tex_coords0
            .ok_or_else(|| fmt_err("missing MeshTexCoord0"))?,
        tex_coords1,
        indices: indices.ok_or_else(|| fmt_err("missing indices section"))?,
        tex_ref,
    })
}
