use std::{
    fs::File,
    io::{BufWriter, Write as _},
    path::Path,
};

use asset_shared::{
    ColorSpace, Compression, FileHeader, PTEX_MAGIC, SectionHeader,
    SectionKind, TexFormat, VERSION,
};
use basis_universal::{
    BasisTextureFormat, ColorSpace as BuColorSpace, Compressor,
    CompressorParams, TranscodeParameters, Transcoder, TranscoderTextureFormat,
};

const FILE_HEADER_SIZE: u32 = FileHeader::SERIALIZED_SIZE;
const SECTION_HEADER_SIZE: u32 = 20;

pub fn compile(
    src: &Path,
    dst: &Path,
    format: TexFormat,
    color_space: ColorSpace,
    mips: bool,
    normal_map: bool,
) -> Result<(), String> {
    // Decode source image to RGBA8
    let img = image::open(src)
        .map_err(|e| format!("open {}: {e}", src.display()))?
        .into_rgba8();

    let base_w = img.width();
    let base_h = img.height();

    if !base_w.is_power_of_two() || !base_h.is_power_of_two() {
        return Err(format!(
            "{}: dimensions {base_w}×{base_h} are not \
             powers of two",
            src.display()
        ));
    }

    // Build mip chain (mip 0 = full size).
    // Capacity: floor(log2(max(w,h))) additional levels + 1 for mip 0.
    let mip_capacity = if mips {
        (base_w.max(base_h).ilog2() + 1) as usize
    } else {
        1
    };
    let mut mip_images: Vec<image::RgbaImage> =
        Vec::with_capacity(mip_capacity);
    mip_images.push(img);
    if mips {
        let mut w = base_w;
        let mut h = base_h;
        while w > 1 || h > 1 {
            w = (w / 2).max(1);
            h = (h / 2).max(1);
            let prev = mip_images.last().unwrap();
            let next = if normal_map {
                downsample_normal_mip(prev, w, h)
            } else {
                image::imageops::resize(
                    prev,
                    w,
                    h,
                    image::imageops::FilterType::Lanczos3,
                )
            };
            mip_images.push(next);
        }
    }

    let mip_count = mip_images.len() as u32;

    // Encode each mip; second element is uncompressed byte length
    let mip_data: Vec<(Vec<u8>, u32)> = mip_images
        .iter()
        .map(|img| encode_mip(img, format, color_space))
        .collect::<Result<_, _>>()?;

    // Derive canonical compression from format (same for every mip)
    let compression = compression_for_format(format);

    // Write TextureInfo body (24 bytes)
    let mut tex_info_body = Vec::with_capacity(24);
    tex_info_body.extend_from_slice(&format.to_u32().to_le_bytes());
    tex_info_body.extend_from_slice(&color_space.to_u32().to_le_bytes());
    tex_info_body.extend_from_slice(&base_w.to_le_bytes());
    tex_info_body.extend_from_slice(&base_h.to_le_bytes());
    tex_info_body.extend_from_slice(&mip_count.to_le_bytes());
    tex_info_body.extend_from_slice(&compression.to_u32().to_le_bytes());

    // Open file and delegate to writer-based API
    let file = File::create(dst)
        .map_err(|e| format!("create {}: {e}", dst.display()))?;
    let mut w = BufWriter::new(file);
    compile_to_writer(&tex_info_body, &mip_data, &mut w)
}

pub fn compile_to_writer<W: std::io::Write>(
    tex_info_body: &[u8],
    // (on-disk data, uncompressed byte length)
    mip_data: &[(Vec<u8>, u32)],
    w: &mut W,
) -> Result<(), String> {
    let mip_count = mip_data.len() as u32;
    let section_count = 1 + mip_count;

    FileHeader {
        magic: PTEX_MAGIC,
        version: VERSION.into(),
        section_count,
    }
    .write_to(w)
    .map_err(|e| format!("write header: {e}"))?;

    // Compute byte offsets for headers
    let mut offsets: Vec<u32> = Vec::new();
    let data_base = FILE_HEADER_SIZE + SECTION_HEADER_SIZE * section_count;
    let mut cursor = data_base;
    offsets.push(cursor);
    cursor += tex_info_body.len() as u32;
    for (data, _) in mip_data {
        offsets.push(cursor);
        cursor += data.len() as u32;
    }

    // TextureInfo section header
    let info_len = tex_info_body.len() as u32;
    SectionHeader {
        kind: SectionKind::TextureInfo,
        byte_offset: offsets[0],
        byte_len: info_len,
        compressed_byte_len: info_len,
        element_count: 1,
    }
    .write_to(w)
    .map_err(|e| format!("write TextureInfo header: {e}"))?;

    // TextureMip section headers
    for (i, (data, uncompressed_len)) in mip_data.iter().enumerate() {
        SectionHeader {
            kind: SectionKind::TextureMip,
            byte_offset: offsets[1 + i],
            byte_len: *uncompressed_len,
            compressed_byte_len: data.len() as u32,
            element_count: *uncompressed_len,
        }
        .write_to(w)
        .map_err(|e| format!("write mip {i} header: {e}"))?;
    }

    // Section data: TextureInfo body
    w.write_all(tex_info_body)
        .map_err(|e| format!("write TextureInfo data: {e}"))?;
    // Mip data
    for (i, (data, _)) in mip_data.iter().enumerate() {
        w.write_all(data)
            .map_err(|e| format!("write mip {i} data: {e}"))?;
    }

    Ok(())
}

#[inline]
fn compression_for_format(format: TexFormat) -> Compression {
    match format {
        TexFormat::Rgba8 => Compression::Lz4,
        TexFormat::Bc4 | TexFormat::Bc5 | TexFormat::Bc7 => Compression::None,
    }
}

/// Downsample a normal-map mip by box-filtering decoded XYZ vectors.
/// RG channels encode XY normals as unsigned-normalized bytes;
/// Z is reconstructed. Does NOT renormalize — the shader reconstructs
/// Z from stored XY, giving unit-length normals regardless.
fn downsample_normal_mip(
    src: &image::RgbaImage,
    dst_w: u32,
    dst_h: u32,
) -> image::RgbaImage {
    let scale_x = (src.width() / dst_w).max(1);
    let scale_y = (src.height() / dst_h).max(1);
    let count = (scale_x * scale_y) as f32;
    let mut buf = Vec::with_capacity((dst_w * dst_h * 4) as usize);
    for dy in 0..dst_h {
        for dx in 0..dst_w {
            let (mut sx, mut sy, mut sz) = (0.0f32, 0.0f32, 0.0f32);
            for ky in 0..scale_y {
                for kx in 0..scale_x {
                    let p = src.get_pixel(dx * scale_x + kx, dy * scale_y + ky);
                    let nx = p[0] as f32 / 255.0 * 2.0 - 1.0;
                    let ny = p[1] as f32 / 255.0 * 2.0 - 1.0;
                    let nz = (1.0 - nx * nx - ny * ny).max(0.0).sqrt();
                    sx += nx;
                    sy += ny;
                    sz += nz;
                }
            }
            let ax = sx / count;
            let ay = sy / count;
            let az = sz / count;
            // -1.0 → 0, 0.0 → 127.5 (rounds to 128), 1.0 → 255
            let enc = |v: f32| -> u8 { ((v + 1.0) * 127.5).round() as u8 };
            buf.extend_from_slice(&[enc(ax), enc(ay), enc(az), 255]);
        }
    }
    image::RgbaImage::from_raw(dst_w, dst_h, buf)
        .expect("buffer length exact by construction")
}

fn encode_mip(
    img: &image::RgbaImage,
    format: TexFormat,
    color_space: ColorSpace,
) -> Result<(Vec<u8>, u32), String> {
    let rgba = img.as_raw();
    let uncompressed_len = img.width() * img.height() * 4;

    match format {
        TexFormat::Rgba8 => {
            let mut enc = lz4_flex::frame::FrameEncoder::new(Vec::new());
            enc.write_all(rgba).map_err(|e| format!("lz4 write: {e}"))?;
            let compressed =
                enc.finish().map_err(|e| format!("lz4 finish: {e}"))?;
            Ok((compressed, uncompressed_len))
        }
        TexFormat::Bc7 => {
            let blocks =
                encode_bc7(rgba, img.width(), img.height(), color_space)?;
            let len = blocks.len() as u32;
            Ok((blocks, len))
        }
        TexFormat::Bc4 | TexFormat::Bc5 => {
            // Encode single-channel (BC4) or two-channel (BC5) targets by
            // producing a UASTC basis file and transcoding to the desired
            // block-compressed format.
            let mut params = CompressorParams::new();
            params.set_basis_format(BasisTextureFormat::UASTC4x4);
            let bu_cs = match color_space {
                ColorSpace::Srgb => BuColorSpace::Srgb,
                ColorSpace::Linear => BuColorSpace::Linear,
            };
            params.set_color_space(bu_cs);
            let w = img.width();
            let h = img.height();
            // basis BC5_RG uses channel 0 and channel 3 (alpha) from
            // the source. For BC5 we want R and G, so extract just
            // those two channels and pass channel_count=2, letting
            // basis treat them as ch0 and ch1.
            if format == TexFormat::Bc5 {
                let rg: Vec<u8> = rgba
                    .chunks_exact(4)
                    .flat_map(|px| [px[0], px[1]])
                    .collect();
                params.source_image_mut(0).init(&rg, w, h, 2);
            } else {
                params.source_image_mut(0).init(rgba, w, h, 4);
            }

            let mut compressor = Compressor::default();
            let ok = unsafe { compressor.init(&params) };
            if !ok {
                return Err("basis compressor init failed".to_string());
            }
            unsafe {
                compressor
                    .process()
                    .map_err(|e| format!("basis compress failed: {e:?}"))?;
            }

            let basis_data = compressor.basis_file().to_vec();

            // Transcode to the requested BC4/BC5 format
            let mut transcoder = Transcoder::new();
            transcoder
                .prepare_transcoding(&basis_data)
                .map_err(|_| "basis prepare_transcoding failed".to_string())?;

            let target = match format {
                TexFormat::Bc4 => TranscoderTextureFormat::BC4_R,
                TexFormat::Bc5 => TranscoderTextureFormat::BC5_RG,
                _ => unreachable!(),
            };

            let bc = transcoder
                .transcode_image_level(
                    &basis_data,
                    target,
                    TranscodeParameters {
                        image_index: 0,
                        level_index: 0,
                        ..Default::default()
                    },
                )
                .map_err(|e| format!("transcode failed: {e:?}"))?;

            transcoder.end_transcoding();
            let len = bc.len() as u32;
            Ok((bc, len))
        }
    }
}

fn encode_bc7(
    rgba: &[u8],
    w: u32,
    h: u32,
    color_space: ColorSpace,
) -> Result<Vec<u8>, String> {
    let bu_cs = match color_space {
        ColorSpace::Srgb => BuColorSpace::Srgb,
        ColorSpace::Linear => BuColorSpace::Linear,
    };

    // Encode to UASTC
    let mut params = CompressorParams::new();
    params.set_basis_format(BasisTextureFormat::UASTC4x4);
    params.set_color_space(bu_cs);
    params.source_image_mut(0).init(rgba, w, h, 4);

    let mut compressor = Compressor::default();
    let ok = unsafe { compressor.init(&params) };
    if !ok {
        return Err("basis compressor init failed".to_string());
    }
    unsafe {
        compressor
            .process()
            .map_err(|e| format!("basis compress failed: {e:?}"))?;
    }

    let basis_data = compressor.basis_file().to_vec();

    // Transcode UASTC → BC7
    let mut transcoder = Transcoder::new();
    transcoder
        .prepare_transcoding(&basis_data)
        .map_err(|_| "basis prepare_transcoding failed".to_string())?;

    let bc7 = transcoder
        .transcode_image_level(
            &basis_data,
            TranscoderTextureFormat::BC7_RGBA,
            TranscodeParameters {
                image_index: 0,
                level_index: 0,
                ..Default::default()
            },
        )
        .map_err(|e| format!("bc7 transcode failed: {e:?}"))?;

    transcoder.end_transcoding();
    Ok(bc7)
}
