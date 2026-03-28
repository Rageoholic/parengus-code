use serde::{Deserialize, Serialize};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Serialize, Deserialize)]
struct CacheMeta {
    src_mtime: u64,
    format: String,
    color_space: String,
    mips: bool,
}

#[derive(Serialize, Deserialize)]
struct ShaderMeta {
    src_mtime: u64,
    compile_args: Vec<String>,
}

fn cache_root() -> PathBuf {
    PathBuf::from("cache/compiled")
}

pub fn ensure_cache_dir() -> std::io::Result<()> {
    let root = cache_root();
    if !root.exists() {
        fs::create_dir_all(&root)?;
    }
    Ok(())
}

fn meta_path(name: &str, ext: &str) -> PathBuf {
    let mut p = cache_root();
    p.push(format!("{name}.{ext}.meta.toml"));
    p
}

pub(crate) fn artifact_path(name: &str, ext: &str) -> PathBuf {
    let mut p = cache_root();
    p.push(format!("{name}.{ext}"));
    p
}

pub fn lookup_image(
    name: &str,
    src: &Path,
    format: &str,
    color_space: &str,
    mips: bool,
) -> Option<PathBuf> {
    let meta_p = meta_path(name, "ptex");
    let art_p = artifact_path(name, "ptex");
    if !meta_p.exists() || !art_p.exists() {
        return None;
    }
    let meta_txt = fs::read_to_string(&meta_p).ok()?;
    let meta: CacheMeta = toml::from_str(&meta_txt).ok()?;
    let src_mtime = fs::metadata(src).and_then(|m| m.modified()).ok()?;
    let src_mtime = src_mtime
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs();
    if meta.src_mtime == src_mtime
        && meta.format == format
        && meta.color_space == color_space
        && meta.mips == mips
    {
        Some(art_p)
    } else {
        None
    }
}

pub fn lookup_shader(
    name: &str,
    src: &Path,
    compile_args: &[String],
    artifact_ext: &str,
) -> Option<PathBuf> {
    let meta_p = meta_path(name, artifact_ext);
    let art_p = artifact_path(name, artifact_ext);
    if !meta_p.exists() || !art_p.exists() {
        return None;
    }
    let meta_txt = fs::read_to_string(&meta_p).ok()?;
    let meta: ShaderMeta = toml::from_str(&meta_txt).ok()?;
    let src_mtime = fs::metadata(src).and_then(|m| m.modified()).ok()?;
    let src_mtime = src_mtime
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs();
    if meta.src_mtime == src_mtime && meta.compile_args == compile_args {
        Some(art_p)
    } else {
        None
    }
}

/// Write the image cache metadata after compiling directly into the
/// cache artifact path. Does not copy the artifact.
pub(crate) fn write_image_meta(
    name: &str,
    src: &Path,
    format: &str,
    color_space: &str,
    mips: bool,
) -> std::io::Result<()> {
    let meta_p = meta_path(name, "ptex");
    let src_mtime = fs::metadata(src)?.modified()?;
    let src_mtime = src_mtime
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let meta = CacheMeta {
        src_mtime,
        format: format.to_string(),
        color_space: color_space.to_string(),
        mips,
    };
    let tom = toml::to_string(&meta).unwrap();
    let mut f = fs::File::create(&meta_p)?;
    f.write_all(tom.as_bytes())?;
    Ok(())
}

/// Write the shader cache metadata after compiling directly into the
/// cache artifact path. Does not copy the artifact.
pub(crate) fn write_shader_meta(
    name: &str,
    src: &Path,
    compile_args: &[String],
    ext: &str,
) -> std::io::Result<()> {
    let meta_p = meta_path(name, ext);
    let src_mtime = fs::metadata(src)?.modified()?;
    let src_mtime = src_mtime
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let meta = ShaderMeta {
        src_mtime,
        compile_args: compile_args.to_vec(),
    };
    let tom = toml::to_string(&meta).unwrap();
    let mut f = fs::File::create(&meta_p)?;
    f.write_all(tom.as_bytes())?;
    Ok(())
}
