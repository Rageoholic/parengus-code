use std::{
    collections::{BTreeMap, HashMap},
    fs,
    path::Path,
};

use asset_pipeline::{AppAssets, AssetMap, Manifest, ManifestEntry};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub(crate) fn copy_assets(
    manifest_path: &Path,
    app_assets_path: &Path,
    assets_src_dir: &Path,
    dst_dir: &Path,
) -> Result<()> {
    let manifest: Manifest =
        toml::from_str(&fs::read_to_string(manifest_path)?)?;
    let app_assets: AppAssets =
        toml::from_str(&fs::read_to_string(app_assets_path)?)?;

    let index: HashMap<&str, &ManifestEntry> =
        manifest.asset.iter().map(|e| (e.name.as_str(), e)).collect();

    fs::create_dir_all(dst_dir)?;

    let mut map = BTreeMap::new();
    let mut copied = 0u32;
    let mut skipped = 0u32;

    for req in &app_assets.asset {
        let entry = index.get(req.name.as_str()).ok_or_else(|| {
            format!(
                "asset `{}` not found in manifest",
                req.name
            )
        })?;

        if entry.asset_type != req.asset_type {
            return Err(format!(
                "asset `{}`: manifest type `{}` != app type `{}`",
                req.name, entry.asset_type, req.asset_type,
            )
            .into());
        }

        let src = assets_src_dir.join(&entry.file);
        let dst = dst_dir.join(&entry.file);

        if is_up_to_date(&src, &dst) {
            skipped += 1;
        } else {
            fs::copy(&src, &dst)?;
            println!("Copied {}", entry.file.display());
            copied += 1;
        }

        map.insert(req.name.clone(), entry.file.clone());
    }

    let asset_map = AssetMap { map };
    fs::write(
        dst_dir.join("asset_map.toml"),
        toml::to_string(&asset_map)?,
    )?;

    println!("Assets: {copied} copied, {skipped} up-to-date");
    Ok(())
}

fn is_up_to_date(src: &Path, dst: &Path) -> bool {
    let Ok(src_meta) = src.metadata() else {
        return false;
    };
    let Ok(dst_meta) = dst.metadata() else {
        return false;
    };
    let Ok(src_mtime) = src_meta.modified() else {
        return false;
    };
    let Ok(dst_mtime) = dst_meta.modified() else {
        return false;
    };
    src_mtime <= dst_mtime
}
