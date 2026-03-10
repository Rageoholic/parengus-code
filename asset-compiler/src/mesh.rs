use std::{
    fs::File,
    io::{BufWriter, Write as _},
    path::Path,
};

use asset_pipeline::{AssetType, Manifest};
use asset_shared::{
    Compression, FileHeader, PMESH_MAGIC, SectionHeader, SectionKind, TexRole,
    VERSION, texture_id,
};

// ── Vec3 helpers ──────────────────────────────────────────────────────────────

fn add3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn normalize(a: [f32; 3]) -> [f32; 3] {
    let l = dot(a, a).sqrt();
    if l < 1e-10 {
        [0.0, 0.0, 1.0]
    } else {
        [a[0] / l, a[1] / l, a[2] / l]
    }
}

// ── Coordinate transform ──────────────────────────────────────────────────────

/// glTF is Y-up; the engine is Z-up.
/// Rotation: +90° about X — (x, y, z) → (x, -z, y)
fn yup_to_zup(p: [f32; 3]) -> [f32; 3] {
    [p[0], -p[2], p[1]]
}

// ── Normal generation ─────────────────────────────────────────────────────────

fn gen_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut acc = vec![[0.0f32; 3]; positions.len()];
    for tri in indices.chunks_exact(3) {
        let [i0, i1, i2] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        let e1 = sub3(positions[i1], positions[i0]);
        let e2 = sub3(positions[i2], positions[i0]);
        let n = cross(e1, e2);
        acc[i0] = add3(acc[i0], n);
        acc[i1] = add3(acc[i1], n);
        acc[i2] = add3(acc[i2], n);
    }
    acc.iter().map(|&n| normalize(n)).collect()
}

// ── Tangent generation via bevy_mikktspace ────────────────────────────────────

struct TangentGen<'a> {
    positions: &'a [[f32; 3]],
    normals: &'a [[f32; 3]],
    tex_coords: &'a [[f32; 2]],
    indices: &'a [u32],
    tangents: Vec<[f32; 4]>,
}

impl<'a> TangentGen<'a> {
    fn new(
        positions: &'a [[f32; 3]],
        normals: &'a [[f32; 3]],
        tex_coords: &'a [[f32; 2]],
        indices: &'a [u32],
    ) -> Self {
        let tangents = vec![[0.0, 0.0, 1.0, 1.0]; positions.len()];
        Self {
            positions,
            normals,
            tex_coords,
            indices,
            tangents,
        }
    }

    fn vi(&self, face: usize, vert: usize) -> usize {
        self.indices[face * 3 + vert] as usize
    }
}

impl bevy_mikktspace::Geometry for TangentGen<'_> {
    fn num_faces(&self) -> usize {
        self.indices.len() / 3
    }

    fn num_vertices_of_face(&self, _face: usize) -> usize {
        3
    }

    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        self.positions[self.vi(face, vert)]
    }

    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        self.normals[self.vi(face, vert)]
    }

    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        self.tex_coords[self.vi(face, vert)]
    }

    fn set_tangent(
        &mut self,
        tangent_space: Option<bevy_mikktspace::TangentSpace>,
        face: usize,
        vert: usize,
    ) {
        let idx = self.vi(face, vert);
        let t = tangent_space
            .map(|ts| ts.tangent_encoded())
            .unwrap_or([0.0, 0.0, 1.0, 1.0]);
        self.tangents[idx] = t;
    }
}

fn gen_tangents(
    positions: &[[f32; 3]],
    normals: &[[f32; 3]],
    tex_coords: &[[f32; 2]],
    indices: &[u32],
) -> Vec<[f32; 4]> {
    let mut tgen = TangentGen::new(positions, normals, tex_coords, indices);
    if bevy_mikktspace::generate_tangents(&mut tgen).is_err() {
        tracing::warn!("mikktspace tangent generation failed, using defaults");
    }
    tgen.tangents
}

// ── Section helpers ───────────────────────────────────────────────────────────

pub struct Section {
    kind: SectionKind,
    element_count: u32,
    uncompressed_len: u32,
    compression: Compression,
    data: Vec<u8>,
}

fn lz4_section(kind: SectionKind, element_count: u32, raw: Vec<u8>) -> Section {
    let uncompressed_len = raw.len() as u32;
    let data = lz4_compress(&raw);
    Section {
        kind,
        element_count,
        uncompressed_len,
        compression: Compression::Lz4,
        data,
    }
}

fn raw_section(
    kind: SectionKind,
    element_count: u32,
    data: Vec<u8>,
) -> Section {
    let uncompressed_len = data.len() as u32;
    Section {
        kind,
        element_count,
        uncompressed_len,
        compression: Compression::None,
        data,
    }
}

fn lz4_compress(raw: &[u8]) -> Vec<u8> {
    let mut enc = lz4_flex::frame::FrameEncoder::new(Vec::new());
    enc.write_all(raw).expect("lz4 frame write");
    enc.finish().expect("lz4 frame finish")
}

// ── Attribute encoding ────────────────────────────────────────────────────────

fn encode_f32s<const N: usize>(vecs: &[[f32; N]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(vecs.len() * N * 4);
    for v in vecs {
        for f in v {
            out.extend_from_slice(&f.to_le_bytes());
        }
    }
    out
}

// ── TexRole helpers ───────────────────────────────────────────────────────────

fn role_from_str(s: &str) -> Result<TexRole, String> {
    match s {
        "albedo" => Ok(TexRole::Albedo),
        "normal" => Ok(TexRole::Normal),
        "metallic_roughness" | "metallic-roughness" => {
            Ok(TexRole::MetallicRoughness)
        }
        "emissive" => Ok(TexRole::Emissive),
        "occlusion" => Ok(TexRole::Occlusion),
        _ => Err(format!("unknown tex role '{s}'")),
    }
}

// move tests to bottom to satisfy clippy's items_after_test_module

// ── Main compile function ─────────────────────────────────────────────────────

// FileHeader: 10 bytes; SectionHeader: 24 bytes
const FILE_HEADER_SIZE: u32 = 10;
const SECTION_HEADER_SIZE: u32 = 24;

pub fn compile(
    src: &Path,
    dst: &Path,
    manifest: &Manifest,
    asset_name: &str,
) -> Result<(), String> {
    // Look up this asset in the manifest
    let entry = manifest
        .asset
        .iter()
        .find(|a| a.name == asset_name)
        .ok_or_else(|| format!("asset '{asset_name}' not found in manifest"))?;

    // 1-2. Import glTF, pick first mesh/primitive
    let (doc, buffers, _images) =
        gltf::import(src).map_err(|e| format!("{e}"))?;
    let prim = doc
        .meshes()
        .next()
        .ok_or("glTF has no meshes")?
        .primitives()
        .next()
        .ok_or("mesh has no primitives")?;
    let reader = prim.reader(|buf| Some(&buffers[buf.index()]));

    // 3. Positions — Y-up → Z-up
    let positions: Vec<[f32; 3]> = reader
        .read_positions()
        .ok_or("primitive has no POSITION")?
        .map(yup_to_zup)
        .collect();

    // 4. Normals (optional) — generate if absent
    let maybe_normals: Option<Vec<[f32; 3]>> =
        reader.read_normals().map(|it| it.map(yup_to_zup).collect());

    // 5. Tangents (optional) — generate if absent
    let maybe_tangents: Option<Vec<[f32; 4]>> =
        reader.read_tangents().map(|it| {
            it.map(|t| {
                let xyz = yup_to_zup([t[0], t[1], t[2]]);
                [xyz[0], xyz[1], xyz[2], t[3]]
            })
            .collect()
        });

    // 6. TEXCOORD_0
    let tex_coords: Vec<[f32; 2]> = reader
        .read_tex_coords(0)
        .ok_or("primitive has no TEXCOORD_0")?
        .into_f32()
        .collect();

    // 7. TEXCOORD_1 (optional)
    let tex_coords1: Option<Vec<[f32; 2]>> =
        reader.read_tex_coords(1).map(|it| it.into_f32().collect());

    // 8. Indices → u32 internally; choose u16/u32 for output
    let indices: Vec<u32> = reader
        .read_indices()
        .ok_or("primitive has no indices")?
        .into_u32()
        .collect();

    // Generate normals / tangents if absent
    let normals =
        maybe_normals.unwrap_or_else(|| gen_normals(&positions, &indices));
    let tangents = maybe_tangents.unwrap_or_else(|| {
        gen_tangents(&positions, &normals, &tex_coords, &indices)
    });

    // ── Build sections ────────────────────────────────────────────────────────

    let n_verts = positions.len() as u32;
    let n_idx = indices.len() as u32;

    let mut sections: Vec<Section> = vec![
        lz4_section(
            SectionKind::MeshPositions,
            n_verts,
            encode_f32s(&positions),
        ),
        lz4_section(SectionKind::MeshNormals, n_verts, encode_f32s(&normals)),
        lz4_section(SectionKind::MeshTangents, n_verts, encode_f32s(&tangents)),
        lz4_section(
            SectionKind::MeshTexCoord0,
            n_verts,
            encode_f32s(&tex_coords),
        ),
    ];

    if let Some(ref tc1) = tex_coords1 {
        sections.push(lz4_section(
            SectionKind::MeshTexCoord1,
            n_verts,
            encode_f32s(tc1),
        ));
    }

    // Index section — u16 if all indices fit, else u32
    let max_idx = indices.iter().copied().max().unwrap_or(0);
    if max_idx <= u16::MAX as u32 {
        let raw: Vec<u8> = indices
            .iter()
            .flat_map(|&i| (i as u16).to_le_bytes())
            .collect();
        sections.push(lz4_section(SectionKind::MeshIndices16, n_idx, raw));
    } else {
        let raw: Vec<u8> =
            indices.iter().flat_map(|&i| i.to_le_bytes()).collect();
        sections.push(lz4_section(SectionKind::MeshIndices32, n_idx, raw));
    }

    // TexRef section — (role: u32 LE, asset_id: u64 LE) per entry
    if !entry.tex_refs.is_empty() {
        let mut data = Vec::with_capacity(entry.tex_refs.len() * 12);
        for (role_str, tex_name) in &entry.tex_refs {
            // Validate the named asset exists as an image
            let valid = manifest.asset.iter().any(|a| {
                a.name == *tex_name && a.asset_type == AssetType::Image
            });
            if !valid {
                return Err(format!(
                    "tex_ref '{tex_name}': not found in manifest \
                     or not an image"
                ));
            }
            let role = role_from_str(role_str)?;
            let id = texture_id(tex_name).0;
            data.extend_from_slice(&role.to_u32().to_le_bytes());
            data.extend_from_slice(&id.to_le_bytes());
        }
        sections.push(raw_section(
            SectionKind::MeshTexRef,
            entry.tex_refs.len() as u32,
            data,
        ));
    }

    // ── Compute offsets and write ─────────────────────────────────────────────

    let section_count = sections.len() as u32;
    let data_base = FILE_HEADER_SIZE + SECTION_HEADER_SIZE * section_count;

    let mut offsets: Vec<u32> = Vec::with_capacity(sections.len());
    let mut cursor = data_base;
    for s in &sections {
        offsets.push(cursor);
        cursor += s.data.len() as u32;
    }

    // Open file and delegate to writer-based API
    let file = File::create(dst)
        .map_err(|e| format!("create {}: {e}", dst.display()))?;
    let mut w = BufWriter::new(file);
    compile_to_writer(&sections, &mut w)
}

pub fn compile_to_writer<W: std::io::Write>(
    sections: &[Section],
    w: &mut W,
) -> Result<(), String> {
    FileHeader {
        magic: PMESH_MAGIC,
        version: VERSION,
        section_count: sections.len() as u32,
    }
    .write_to(w)
    .map_err(|e| format!("write header: {e}"))?;

    // We need to recompute byte offsets for the provided sections
    let mut cursor =
        FILE_HEADER_SIZE + SECTION_HEADER_SIZE * (sections.len() as u32);
    let mut offsets: Vec<u32> = Vec::with_capacity(sections.len());
    for s in sections {
        offsets.push(cursor);
        cursor += s.data.len() as u32;
    }

    for (s, &byte_offset) in sections.iter().zip(&offsets) {
        SectionHeader {
            kind: s.kind,
            compression: s.compression,
            byte_offset,
            byte_len: s.uncompressed_len,
            compressed_byte_len: s.data.len() as u32,
            element_count: s.element_count,
        }
        .write_to(w)
        .map_err(|e| format!("write section header: {e}"))?;
    }

    for s in sections {
        w.write_all(&s.data)
            .map_err(|e| format!("write section data: {e}"))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_yup_to_zup_basic() {
        // glTF (x,y,z) = (1,2,3)
        // +90deg X rotation -> engine (x, -z, y)
        let in_p = [1.0f32, 2.0, 3.0];
        let out = yup_to_zup(in_p);
        assert_eq!(out, [1.0f32, -3.0, 2.0]);
    }

    #[test]
    fn test_yup_to_zup_zero() {
        let in_p = [0.0f32, 0.0, 0.0];
        let out = yup_to_zup(in_p);
        assert_eq!(out, [0.0f32, 0.0, 0.0]);
    }

    #[test]
    fn test_lz4_roundtrip() {
        use std::io::Read;

        // Create sample data with some entropy and repetition
        let mut raw: Vec<u8> = Vec::new();
        for i in 0..1024 {
            raw.push((i % 256) as u8);
        }
        // repeat pattern to allow good compression
        let raw = [raw.clone(), raw.clone()].concat();

        let comp = lz4_compress(&raw);
        let mut dec = lz4_flex::frame::FrameDecoder::new(&comp[..]);
        let mut out: Vec<u8> = Vec::new();
        dec.read_to_end(&mut out).expect("lz4 decompress");
        assert_eq!(out, raw);
    }

    #[test]
    fn test_fileheader_roundtrip() {
        use std::io::Cursor;

        let hdr = FileHeader {
            magic: PMESH_MAGIC,
            version: VERSION,
            section_count: 7,
        };
        let mut buf: Vec<u8> = Vec::new();
        hdr.write_to(&mut buf).expect("write header");
        let mut cur = Cursor::new(buf);
        let got = FileHeader::read_from(&mut cur).expect("read header");
        assert_eq!(got.magic, hdr.magic);
        assert_eq!(got.version, hdr.version);
        assert_eq!(got.section_count, hdr.section_count);
    }

    #[test]
    fn test_sectionheader_roundtrip() {
        use std::io::Cursor;

        let sh = SectionHeader {
            kind: SectionKind::MeshTexCoord0,
            compression: Compression::Lz4,
            byte_offset: 12345,
            byte_len: 54321,
            compressed_byte_len: 22222,
            element_count: 314,
        };
        let mut buf: Vec<u8> = Vec::new();
        sh.write_to(&mut buf).expect("write section header");
        let mut cur = Cursor::new(buf);
        let got_opt =
            SectionHeader::read_from(&mut cur).expect("read section header");
        let got = got_opt.expect("known kind");
        assert_eq!(got.kind, sh.kind);
        assert_eq!(got.compression, sh.compression);
        assert_eq!(got.byte_offset, sh.byte_offset);
        assert_eq!(got.byte_len, sh.byte_len);
        assert_eq!(got.compressed_byte_len, sh.compressed_byte_len);
        assert_eq!(got.element_count, sh.element_count);
    }
}
