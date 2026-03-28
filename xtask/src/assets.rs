use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::Path,
};

use asset_pipeline::{AppAssets, AssetType, Manifest, ManifestEntry};
use path_slash::PathBufExt as _;

use crate::cache;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Serialized to `asset_map.toml` so that `asset-loader::AssetMap`
/// can load it at runtime.
#[derive(serde::Serialize)]
struct AssetMapFile {
    map: BTreeMap<String, String>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    shader_debug: BTreeMap<String, String>,
}

/// Copy compiled assets from the shared cache into `dst_dir` for the
/// given app, then write `asset_map.toml`.
///
/// Compiled file naming convention (in cache and in dst_dir):
/// - Mesh   `{name}.pmesh`
/// - Image  `{name}.ptex`
/// - Shader uses `entry.file` extension (e.g. `.spv`)
pub(crate) fn copy_assets(
    manifest_path: &Path,
    app_assets_path: &Path,
    dst_dir: &Path,
) -> Result<()> {
    let manifest: Manifest =
        toml::from_str(&fs::read_to_string(manifest_path)?)?;
    let app_assets: AppAssets =
        toml::from_str(&fs::read_to_string(app_assets_path)?)?;

    let index: HashMap<&str, &ManifestEntry> = manifest
        .asset
        .iter()
        .map(|e| (e.name.as_str(), e))
        .collect();

    fs::create_dir_all(dst_dir)?;

    let mut map = BTreeMap::new();
    let mut shader_debug: BTreeMap<String, String> = BTreeMap::new();

    for req in &app_assets.asset {
        let entry = index.get(req.name.as_str()).ok_or_else(|| {
            format!("asset '{}' not found in manifest", req.name)
        })?;

        if entry.asset_type != req.asset_type {
            return Err(format!(
                "asset '{}': manifest type '{}' != app type '{}'",
                req.name, entry.asset_type, req.asset_type,
            )
            .into());
        }

        let (compiled_name, cache_ext) = match entry.asset_type {
            AssetType::Mesh => (format!("{}.pmesh", req.name), "pmesh"),
            AssetType::Image => (format!("{}.ptex", req.name), "ptex"),
            AssetType::Shader => {
                let ext = entry
                    .file
                    .extension()
                    .and_then(|s| s.to_str())
                    .unwrap_or("spv");
                (entry.file.to_string_lossy().into_owned(), ext)
            }
            _ => {
                return Err(format!(
                    "asset '{}': unsupported type '{}'",
                    req.name, entry.asset_type
                )
                .into());
            }
        };

        let src = cache::artifact_path(&req.name, cache_ext);
        let dst = dst_dir.join(&compiled_name);
        fs::copy(&src, &dst).map_err(|e| {
            format!("failed to copy '{}' from cache: {e}", req.name)
        })?;

        map.insert(req.name.clone(), compiled_name);

        // Copy debug shader variant if the manifest entry declares one.
        if entry.asset_type == AssetType::Shader
            && let Some(ref df) = entry.debug_file
        {
            let debug_ext =
                df.extension().and_then(|s| s.to_str()).unwrap_or("spv");
            let debug_compiled = df.to_string_lossy().into_owned();
            let debug_src = cache::artifact_path(&req.name, debug_ext);
            let debug_dst = dst_dir.join(&debug_compiled);
            fs::copy(&debug_src, &debug_dst).map_err(|e| {
                format!(
                    "failed to copy debug shader '{}' from \
                         cache: {e}",
                    req.name
                )
            })?;
            shader_debug.insert(
                req.name.clone(),
                std::path::PathBuf::from(debug_compiled)
                    .to_slash_lossy()
                    .into_owned(),
            );
        }
    }

    let asset_map = AssetMapFile {
        map: map
            .into_iter()
            .map(|(k, v)| {
                (k, std::path::PathBuf::from(v).to_slash_lossy().into_owned())
            })
            .collect(),
        shader_debug: shader_debug
            .into_iter()
            .map(|(k, v)| {
                (k, std::path::PathBuf::from(v).to_slash_lossy().into_owned())
            })
            .collect(),
    };
    fs::write(dst_dir.join("asset_map.toml"), toml::to_string(&asset_map)?)?;

    println!("Assets copied: {}", app_assets.asset.len());
    Ok(())
}
