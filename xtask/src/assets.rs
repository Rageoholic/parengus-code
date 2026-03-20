use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::Path,
};

use asset_pipeline::{AppAssets, AssetType, Manifest, ManifestEntry};
use path_slash::PathBufExt as _;

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

/// Serialized to `asset_map.toml` so that `asset-loader::AssetMap`
/// can load it at runtime.
#[derive(serde::Serialize)]
struct AssetMapFile {
    map: BTreeMap<String, String>,
}

/// Verify compiled assets exist in `dst_dir` for the given app,
/// then write `asset_map.toml`.
///
/// Compiled file naming convention:
/// - Mesh   `{name}.pmesh`
/// - Image  `{name}.ptex`
/// - Shader uses `entry.file` (already `.spv`)
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
    let mut skipped = 0u32;

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

        let compiled_name = match entry.asset_type {
            AssetType::Mesh => {
                format!("{}.pmesh", req.name)
            }
            AssetType::Image => {
                format!("{}.ptex", req.name)
            }
            AssetType::Shader => entry.file.to_string_lossy().into_owned(),
            _ => {
                return Err(format!(
                    "asset '{}': unsupported type '{}'",
                    req.name, entry.asset_type
                )
                .into());
            }
        };

        // The compiled asset is produced directly into `dst_dir` by the
        // xtask compile steps. Treat existing files in `dst_dir` as
        // present; we no longer copy from a separate compiled cache.
        let src = dst_dir.join(&compiled_name);
        if src.exists() {
            skipped += 1;
        } else {
            return Err(format!(
                "compiled asset missing: {} (expected in {})",
                compiled_name,
                dst_dir.display()
            )
            .into());
        }

        map.insert(req.name.clone(), compiled_name);
    }

    let asset_map = AssetMapFile {
        map: map
            .into_iter()
            .map(|(k, v)| {
                (k, std::path::PathBuf::from(v).to_slash_lossy().into_owned())
            })
            .collect(),
    };
    fs::write(dst_dir.join("asset_map.toml"), toml::to_string(&asset_map)?)?;

    println!("Assets present: {skipped}");
    Ok(())
}
