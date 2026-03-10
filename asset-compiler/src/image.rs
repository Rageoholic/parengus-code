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

// FileHeader: 10 bytes; SectionHeader: 24 bytes
const FILE_HEADER_SIZE: u32 = 10;
const SECTION_HEADER_SIZE: u32 = 24;

pub fn compile(
    src: &Path,
    dst: &Path,
    format: TexFormat,
    color_space: ColorSpace,
    mips: bool,
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

    // Build mip chain (mip 0 = full size)
    let mut mip_images: Vec<image::RgbaImage> = vec![img];
    if mips {
        let mut w = base_w;
        let mut h = base_h;
        while w > 1 || h > 1 {
            w = (w / 2).max(1);
            h = (h / 2).max(1);
            let prev = mip_images.last().unwrap();
            let next = image::imageops::resize(
                prev,
                w,
                h,
                image::imageops::FilterType::Lanczos3,
            );
            mip_images.push(next);
        }
    }

    let mip_count = mip_images.len() as u32;

    // Encode each mip
    let mip_data: Vec<(Vec<u8>, Compression)> = mip_images
        .iter()
        .map(|img| encode_mip(img, format, color_space))
        .collect::<Result<_, _>>()?;

    // Write TextureInfo body (20 bytes)
    let mut tex_info_body = Vec::with_capacity(20);
    tex_info_body.extend_from_slice(&format.to_u32().to_le_bytes());
    tex_info_body.extend_from_slice(&color_space.to_u32().to_le_bytes());
    tex_info_body.extend_from_slice(&base_w.to_le_bytes());
    tex_info_body.extend_from_slice(&base_h.to_le_bytes());
    tex_info_body.extend_from_slice(&mip_count.to_le_bytes());

    // section_count = 1 (TextureInfo) + mip_count
    let section_count = 1 + mip_count;
    let data_base = FILE_HEADER_SIZE + SECTION_HEADER_SIZE * section_count;

    // Compute byte offsets
    let mut offsets: Vec<u32> = Vec::new();
    let mut cursor = data_base;
    offsets.push(cursor);
    cursor += tex_info_body.len() as u32;
    for (data, _) in &mip_data {
        offsets.push(cursor);
        cursor += data.len() as u32;
    }

    // Open file and delegate to writer-based API
    let file = File::create(dst)
        .map_err(|e| format!("create {}: {e}", dst.display()))?;
    let mut w = BufWriter::new(file);
    compile_to_writer(&tex_info_body, &mip_data, &mut w, format)
}

pub fn compile_to_writer<W: std::io::Write>(
    tex_info_body: &[u8],
    mip_data: &[(Vec<u8>, Compression)],
    w: &mut W,
    format: TexFormat,
) -> Result<(), String> {
    let mip_count = mip_data.len() as u32;
    let section_count = 1 + mip_count;

    FileHeader {
        magic: PTEX_MAGIC,
        version: VERSION,
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
        compression: Compression::None,
        byte_offset: offsets[0],
        byte_len: info_len,
        compressed_byte_len: info_len,
        element_count: 1,
    }
    .write_to(w)
    .map_err(|e| format!("write TextureInfo header: {e}"))?;

    // TextureMip section headers
    for (i, (data, compression)) in mip_data.iter().enumerate() {
        let mip_w = (0u32).saturating_add(0); // placeholder not used here
        let _ = mip_w;
        let byte_len = uncompressed_mip_len(1, 1, format); // dummy for element_count
        SectionHeader {
            kind: SectionKind::TextureMip,
            compression: *compression,
            byte_offset: offsets[1 + i],
            byte_len,
            compressed_byte_len: data.len() as u32,
            element_count: byte_len,
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

fn uncompressed_mip_len(w: u32, h: u32, format: TexFormat) -> u32 {
    match format {
        TexFormat::Rgba8 => w * h * 4,
        TexFormat::Bc4 => {
            let bw = w.div_ceil(4);
            let bh = h.div_ceil(4);
            bw * bh * 8
        }
        TexFormat::Bc5 => {
            let bw = w.div_ceil(4);
            let bh = h.div_ceil(4);
            bw * bh * 16
        }
        TexFormat::Bc7 => {
            let bw = w.div_ceil(4);
            let bh = h.div_ceil(4);
            bw * bh * 16
        }
    }
}

fn encode_mip(
    img: &image::RgbaImage,
    format: TexFormat,
    color_space: ColorSpace,
) -> Result<(Vec<u8>, Compression), String> {
    let rgba = img.as_raw();

    match format {
        TexFormat::Rgba8 => {
            let mut enc = lz4_flex::frame::FrameEncoder::new(Vec::new());
            enc.write_all(rgba).map_err(|e| format!("lz4 write: {e}"))?;
            let compressed =
                enc.finish().map_err(|e| format!("lz4 finish: {e}"))?;
            Ok((compressed, Compression::Lz4))
        }
        TexFormat::Bc7 => {
            let blocks =
                encode_bc7(rgba, img.width(), img.height(), color_space)?;
            Ok((blocks, Compression::None))
        }
        TexFormat::Bc4 | TexFormat::Bc5 => {
            Err(format!("format {format:?} not yet supported"))
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
