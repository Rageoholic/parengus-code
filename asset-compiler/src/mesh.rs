use std::{
    collections::HashMap,
    fs::File,
    io::{BufWriter, Write as _},
    path::Path,
};

use asset_pipeline::{AssetType, Manifest};
use asset_shared::{
    FileHeader, PMESH_MAGIC, SectionHeader, SectionKind, TexRole, VERSION,
    texture_id,
};
use path_slash::PathExt as _;
use vek::{Mat4, Vec3};

// ── Coordinate transform ──────────────────────────────────────────────────────

/// glTF is Y-up; the engine is Z-up.
/// Rotation: +90° about X — (x, y, z) → (x, -z, y)
#[inline]
fn yup_to_zup(p: [f32; 3]) -> [f32; 3] {
    [p[0], -p[2], p[1]]
}

// ── glTF matrix conversion ────────────────────────────────────────────────────

/// Convert a glTF column-major `[[f32; 4]; 4]` matrix to `Mat4<f32>`.
fn mat4_from_gltf(m: [[f32; 4]; 4]) -> Mat4<f32> {
    // from_col_array takes a flat [T; 16] in column-major order.
    let [c0, c1, c2, c3] = m;
    Mat4::from_col_array([
        c0[0], c0[1], c0[2], c0[3], c1[0], c1[1], c1[2], c1[3], c2[0], c2[1],
        c2[2], c2[3], c3[0], c3[1], c3[2], c3[3],
    ])
}

// Convert a column-major 3×3 rotation matrix (r[col][row]) to
// a quaternion [x, y, z, w] using Shepperd's method.
// (vek 0.17 does not implement Quaternion::from(Mat3))
fn rot_mat_to_quat(r: [[f32; 3]; 3]) -> [f32; 4] {
    let (r00, r10, r20) = (r[0][0], r[0][1], r[0][2]);
    let (r01, r11, r21) = (r[1][0], r[1][1], r[1][2]);
    let (r02, r12, r22) = (r[2][0], r[2][1], r[2][2]);
    let trace = r00 + r11 + r22;
    let (x, y, z, w);
    if trace > 0.0 {
        let s = 0.5 / (trace + 1.0).sqrt();
        w = 0.25 / s;
        x = (r21 - r12) * s;
        y = (r02 - r20) * s;
        z = (r10 - r01) * s;
    } else if r00 > r11 && r00 > r22 {
        let s = 2.0 * (1.0 + r00 - r11 - r22).sqrt();
        w = (r21 - r12) / s;
        x = 0.25 * s;
        y = (r01 + r10) / s;
        z = (r02 + r20) / s;
    } else if r11 > r22 {
        let s = 2.0 * (1.0 + r11 - r00 - r22).sqrt();
        w = (r02 - r20) / s;
        x = (r01 + r10) / s;
        y = 0.25 * s;
        z = (r12 + r21) / s;
    } else {
        let s = 2.0 * (1.0 + r22 - r00 - r11).sqrt();
        w = (r10 - r01) / s;
        x = (r02 + r20) / s;
        y = (r12 + r21) / s;
        z = 0.25 * s;
    }
    [x, y, z, w]
}

// ── Normal generation ─────────────────────────────────────────────────────────

fn gen_normals(positions: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut acc = vec![Vec3::<f32>::broadcast(0.0); positions.len()];
    for tri in indices.chunks_exact(3) {
        let [i0, i1, i2] = [tri[0] as usize, tri[1] as usize, tri[2] as usize];
        let e1 = Vec3::from(positions[i1]) - Vec3::from(positions[i0]);
        let e2 = Vec3::from(positions[i2]) - Vec3::from(positions[i0]);
        let n = e1.cross(e2);
        acc[i0] += n;
        acc[i1] += n;
        acc[i2] += n;
    }
    acc.iter()
        .map(|&n| {
            let l = n.magnitude();
            if l < 1e-10 {
                [0.0, 0.0, 1.0]
            } else {
                [n.x / l, n.y / l, n.z / l]
            }
        })
        .collect()
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

    #[inline]
    fn vi(&self, face: usize, vert: usize) -> usize {
        self.indices[face * 3 + vert] as usize
    }
}

impl bevy_mikktspace::Geometry for TangentGen<'_> {
    #[inline]
    fn num_faces(&self) -> usize {
        self.indices.len() / 3
    }

    #[inline]
    fn num_vertices_of_face(&self, _face: usize) -> usize {
        3
    }

    #[inline]
    fn position(&self, face: usize, vert: usize) -> [f32; 3] {
        self.positions[self.vi(face, vert)]
    }

    #[inline]
    fn normal(&self, face: usize, vert: usize) -> [f32; 3] {
        self.normals[self.vi(face, vert)]
    }

    #[inline]
    fn tex_coord(&self, face: usize, vert: usize) -> [f32; 2] {
        self.tex_coords[self.vi(face, vert)]
    }

    #[inline]
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
    data: Vec<u8>,
}

fn lz4_section(kind: SectionKind, element_count: u32, raw: Vec<u8>) -> Section {
    let uncompressed_len = raw.len() as u32;
    let data = lz4_compress(&raw);
    Section {
        kind,
        element_count,
        uncompressed_len,
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

const FILE_HEADER_SIZE: u32 = FileHeader::SERIALIZED_SIZE;
const SECTION_HEADER_SIZE: u32 = 20;

pub fn compile(
    src: &Path,
    dst: &Path,
    manifest: &Manifest,
    asset_name: &str,
) -> Result<(), String> {
    // Look up this asset in the manifest
    let _entry = manifest
        .asset
        .iter()
        .find(|a| a.name == asset_name)
        .ok_or_else(|| format!("asset '{asset_name}' not found in manifest"))?;
    // 1-2. Import glTF and prepare to traverse scene nodes for all
    // meshes/primitives. We'll concatenate vertex/index data from
    // all primitives and emit per-submesh metadata sections.
    let (doc, buffers, _images) =
        gltf::import(src).map_err(|e| format!("{e}"))?;

    // Accumulators for concatenating vertex/index data
    let mut positions_all: Vec<[f32; 3]> = Vec::new();
    let mut normals_all: Vec<[f32; 3]> = Vec::new();
    let mut tangents_all: Vec<[f32; 4]> = Vec::new();
    let mut tex0_all: Vec<[f32; 2]> = Vec::new();
    let mut tex1_all: Vec<[f32; 2]> = Vec::new();
    let mut indices_all: Vec<u32> = Vec::new();

    // Sub-mesh table entries: (translation[3], rotation[4], scale[3],
    // index_base u32, index_count u32)
    let mut submeshes_raw: Vec<u8> = Vec::new();
    // Per-submesh albedo TextureId (u64 little-endian), parallel to
    // submeshes_raw.
    let mut sub_albedos_raw: Vec<u8> = Vec::new();

    // Material data array: 14 f32 = 56 bytes per material entry
    let mut material_data_raw: Vec<u8> = Vec::new();

    // Reverse map: image URI (forward-slash, relative to assets/) → asset name
    // Built from manifest image entries for URI → asset_name lookup.
    let img_uri_to_asset: HashMap<String, String> = manifest
        .asset
        .iter()
        .filter(|e| e.asset_type == AssetType::Image)
        .map(|e| (e.file.to_slash_lossy().into_owned(), e.name.clone()))
        .collect();

    // Global derived tex_refs (role -> asset name) collected across
    // all materials in the glTF. We dedupe by role: first occurrence wins.
    let mut derived_texrefs: HashMap<String, String> = HashMap::new();

    // Helper: write f32 slice into raw vec
    let push_f32 = |raw: &mut Vec<u8>, v: &[f32]| {
        for f in v {
            raw.extend_from_slice(&f.to_le_bytes());
        }
    };

    // Traverse nodes depth-first, composing world matrices and
    // extracting primitives.
    let scene = doc.scenes().next().ok_or("glTF has no scenes")?;

    // Stack: (node, world_matrix)
    let mut stack: Vec<(gltf::Node, Mat4<f32>)> = Vec::new();
    for root in scene.nodes() {
        let m = mat4_from_gltf(root.transform().matrix());
        stack.push((root, m));
    }

    while let Some((node, world_m)) = stack.pop() {
        // push children with composed matrix
        for child in node.children() {
            let cm = mat4_from_gltf(child.transform().matrix());
            stack.push((child, world_m * cm));
        }

        if let Some(mesh) = node.mesh() {
            for prim in mesh.primitives() {
                let reader = prim.reader(|buf| Some(&buffers[buf.index()]));

                // Positions — Y-up → Z-up (do not bake node transform)
                let prim_positions: Vec<[f32; 3]> = reader
                    .read_positions()
                    .ok_or("primitive has no POSITION")?
                    .map(yup_to_zup)
                    .collect();

                // Normals (optional) — generate if absent
                let maybe_normals: Option<Vec<[f32; 3]>> = reader
                    .read_normals()
                    .map(|it| it.map(yup_to_zup).collect());

                // Tangents (optional) — generate if absent
                let maybe_tangents: Option<Vec<[f32; 4]>> =
                    reader.read_tangents().map(|it| {
                        it.map(|t| {
                            let xyz = yup_to_zup([t[0], t[1], t[2]]);
                            [xyz[0], xyz[1], xyz[2], t[3]]
                        })
                        .collect()
                    });

                // TEXCOORD_0
                let prim_tex0: Vec<[f32; 2]> = reader
                    .read_tex_coords(0)
                    .ok_or("primitive has no TEXCOORD_0")?
                    .into_f32()
                    .collect();

                // TEXCOORD_1 (optional)
                let prim_tex1: Option<Vec<[f32; 2]>> =
                    reader.read_tex_coords(1).map(|it| it.into_f32().collect());

                // Indices → u32 internally
                let prim_indices: Vec<u32> = reader
                    .read_indices()
                    .ok_or("primitive has no indices")?
                    .into_u32()
                    .collect();

                // Generate missing normals/tangents per-primitive
                let prim_normals = maybe_normals.unwrap_or_else(|| {
                    gen_normals(&prim_positions, &prim_indices)
                });
                let prim_tangents = maybe_tangents.unwrap_or_else(|| {
                    gen_tangents(
                        &prim_positions,
                        &prim_normals,
                        &prim_tex0,
                        &prim_indices,
                    )
                });

                // Record base offsets
                let vertex_base = positions_all.len() as u32;
                let index_base = indices_all.len() as u32;

                // Append vertex attributes
                positions_all.extend_from_slice(&prim_positions);
                normals_all.extend_from_slice(&prim_normals);
                tangents_all.extend_from_slice(&prim_tangents);
                tex0_all.extend_from_slice(&prim_tex0);
                if let Some(t1) = prim_tex1 {
                    tex1_all.extend_from_slice(&t1);
                }

                // Append indices with offset
                indices_all
                    .extend(prim_indices.iter().map(|&i| i + vertex_base));

                // Extract translation from column 3 (column-major).
                let col3 = world_m.cols.w;
                let translation = yup_to_zup([col3.x, col3.y, col3.z]);

                // Scale = length of each upper-3×3 column in Y-up
                // space. Under Rx(+90°), scale_x stays with col0,
                // scale_y (Z-up Y) comes from col2, scale_z (Z-up Z)
                // comes from col1.
                let col_len =
                    |c: vek::Vec4<f32>| Vec3::new(c.x, c.y, c.z).magnitude();
                let sx = col_len(world_m.cols.x);
                let sy = col_len(world_m.cols.z);
                let sz = col_len(world_m.cols.y);

                // Build the Z-up column vectors (without scale) and
                // convert to a rotation quaternion.
                let safe_div = |v: f32, s: f32| {
                    if s > 1e-10 { v / s } else { 0.0f32 }
                };
                let mc = world_m.cols;
                let c0 = yup_to_zup([mc.x.x, mc.x.y, mc.x.z]);
                let c1 = yup_to_zup([-mc.z.x, -mc.z.y, -mc.z.z]);
                let c2 = yup_to_zup([mc.y.x, mc.y.y, mc.y.z]);
                let r3 = [
                    [
                        safe_div(c0[0], sx),
                        safe_div(c0[1], sx),
                        safe_div(c0[2], sx),
                    ],
                    [
                        safe_div(c1[0], sy),
                        safe_div(c1[1], sy),
                        safe_div(c1[2], sy),
                    ],
                    [
                        safe_div(c2[0], sz),
                        safe_div(c2[1], sz),
                        safe_div(c2[2], sz),
                    ],
                ];
                let rotation = rot_mat_to_quat(r3);
                let scale = [sx, sy, sz];

                push_f32(&mut submeshes_raw, &translation);
                push_f32(&mut submeshes_raw, &rotation);
                push_f32(&mut submeshes_raw, &scale);
                submeshes_raw.extend_from_slice(&index_base.to_le_bytes());
                let idx_count = (indices_all.len() as u32) - index_base;
                submeshes_raw.extend_from_slice(&idx_count.to_le_bytes());

                // Material constants (14 f32s)
                let mut mat_consts = [0.0f32; 14];
                let material = prim.material();
                let pbr = material.pbr_metallic_roughness();
                let base = pbr.base_color_factor();
                mat_consts[0] = base[0];
                mat_consts[1] = base[1];
                mat_consts[2] = base[2];
                mat_consts[3] = base[3];
                let em = material.emissive_factor();
                mat_consts[4] = em[0];
                mat_consts[5] = em[1];
                mat_consts[6] = em[2];
                mat_consts[8] = pbr.metallic_factor();
                mat_consts[9] = pbr.roughness_factor();
                push_f32(&mut material_data_raw, &mat_consts);

                // Derive tex_refs by matching glTF image URIs to manifest.
                let uri_of = |tex: gltf::Texture| -> Option<String> {
                    match tex.source().source() {
                        gltf::image::Source::Uri { uri, .. } => {
                            Some(uri.to_string())
                        }
                        gltf::image::Source::View { .. } => None,
                    }
                };
                let slots: &[(&str, Option<String>)] = &[
                    (
                        "albedo",
                        pbr.base_color_texture()
                            .and_then(|t| uri_of(t.texture())),
                    ),
                    (
                        "normal",
                        material
                            .normal_texture()
                            .and_then(|t| uri_of(t.texture())),
                    ),
                    (
                        "metallic-roughness",
                        pbr.metallic_roughness_texture()
                            .and_then(|t| uri_of(t.texture())),
                    ),
                    (
                        "emissive",
                        material
                            .emissive_texture()
                            .and_then(|t| uri_of(t.texture())),
                    ),
                    (
                        "occlusion",
                        material
                            .occlusion_texture()
                            .and_then(|t| uri_of(t.texture())),
                    ),
                ];
                for (role, uri_opt) in slots {
                    let Some(uri) = uri_opt else { continue };
                    let Some(name) = img_uri_to_asset.get(uri.as_str()) else {
                        continue;
                    };
                    derived_texrefs
                        .entry(role.to_string())
                        .or_insert_with(|| name.clone());
                }

                // Per-submesh albedo: look up the albedo URI and hash
                // it to a TextureId, or emit 0 if no albedo is present.
                let albedo_id: u64 = pbr
                    .base_color_texture()
                    .and_then(|t| uri_of(t.texture()))
                    .and_then(|uri| img_uri_to_asset.get(&uri))
                    .map(|name| texture_id(name).0)
                    .unwrap_or(0);
                sub_albedos_raw.extend_from_slice(&albedo_id.to_le_bytes());
            }
        }
    }

    // ── Build sections ────────────────────────────────────────────────────────

    let n_verts = positions_all.len() as u32;
    let n_idx = indices_all.len() as u32;

    let mut sections: Vec<Section> = vec![
        lz4_section(
            SectionKind::MeshPositions,
            n_verts,
            encode_f32s(&positions_all),
        ),
        lz4_section(
            SectionKind::MeshNormals,
            n_verts,
            encode_f32s(&normals_all),
        ),
        lz4_section(
            SectionKind::MeshTangents,
            n_verts,
            encode_f32s(&tangents_all),
        ),
        lz4_section(
            SectionKind::MeshTexCoord0,
            n_verts,
            encode_f32s(&tex0_all),
        ),
    ];

    if !tex1_all.is_empty() {
        sections.push(lz4_section(
            SectionKind::MeshTexCoord1,
            n_verts,
            encode_f32s(&tex1_all),
        ));
    }

    // Index section — u16 if all indices fit, else u32
    let max_idx = indices_all.iter().copied().max().unwrap_or(0);
    if max_idx <= u16::MAX as u32 {
        let raw: Vec<u8> = indices_all
            .iter()
            .flat_map(|&i| (i as u16).to_le_bytes())
            .collect();
        sections.push(lz4_section(SectionKind::MeshIndices16, n_idx, raw));
    } else {
        let raw: Vec<u8> =
            indices_all.iter().flat_map(|&i| i.to_le_bytes()).collect();
        sections.push(lz4_section(SectionKind::MeshIndices32, n_idx, raw));
    }
    // TexRef section — emit derived global texture refs (deduped)
    if !derived_texrefs.is_empty() {
        let mut data = Vec::with_capacity(derived_texrefs.len() * 12);
        for (role_str, tex_name) in &derived_texrefs {
            let role = role_from_str(role_str)?;
            let id = texture_id(tex_name).0;
            data.extend_from_slice(&role.to_u32().to_le_bytes());
            data.extend_from_slice(&id.to_le_bytes());
        }
        sections.push(raw_section(
            SectionKind::MeshTexRef,
            derived_texrefs.len() as u32,
            data,
        ));
    }

    // Emit SubMeshTable and MeshMaterialData sections if present
    if !submeshes_raw.is_empty() {
        sections.push(raw_section(
            SectionKind::MeshSubMeshTable,
            (submeshes_raw.len() / (3 * 4 + 4 * 4 + 3 * 4 + 4 + 4)) as u32,
            submeshes_raw,
        ));
    }
    if !sub_albedos_raw.is_empty() {
        sections.push(raw_section(
            SectionKind::MeshSubMeshAlbedo,
            (sub_albedos_raw.len() / 8) as u32,
            sub_albedos_raw,
        ));
    }
    if !material_data_raw.is_empty() {
        sections.push(raw_section(
            SectionKind::MeshMaterialData,
            (material_data_raw.len() / 56) as u32,
            material_data_raw,
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
        version: VERSION.into(),
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
            version: VERSION.into(),
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
        assert_eq!(got.byte_offset, sh.byte_offset);
        assert_eq!(got.byte_len, sh.byte_len);
        assert_eq!(got.compressed_byte_len, sh.compressed_byte_len);
        assert_eq!(got.element_count, sh.element_count);
    }
}
