use std::{collections::HashMap, fmt, path::PathBuf};

use serde::{Deserialize, Serialize};

// ---- Asset type ---------------------------------------------------------

#[non_exhaustive]
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone, Copy)]
#[serde(rename_all = "kebab-case")]
pub enum AssetType {
    Image,
    Mesh,
    Shader,
}

impl fmt::Display for AssetType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AssetType::Image => write!(f, "image"),
            AssetType::Mesh => write!(f, "mesh"),
            AssetType::Shader => write!(f, "shader"),
        }
    }
}

// ---- Workspace manifest (assets/manifest.toml) --------------------------

#[derive(Debug, Serialize, Deserialize)]
pub struct Manifest {
    pub asset: Vec<ManifestEntry>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ManifestEntry {
    pub name: String,
    #[serde(with = "path_serde")]
    pub file: PathBuf,
    #[serde(rename = "type")]
    pub asset_type: AssetType,
    /// For shader assets: source file to compile (relative to `assets/`).
    #[serde(
        default,
        skip_serializing_if = "Option::is_none",
        with = "opt_path_serde"
    )]
    pub source_file: Option<PathBuf>,
    /// For shader assets: extra arguments passed to `slangc`.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub compile_args: Vec<String>,
    /// For mesh assets: texture references — role string → asset name.
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub tex_refs: HashMap<String, String>,
    /// For image assets: target compressed format ("bc7", "rgba8", …).
    pub format: Option<String>,
    /// For image assets: colour space ("srgb", "linear").
    pub color_space: Option<String>,
    /// For image assets: whether to generate a full mip chain.
    pub mips: Option<bool>,
    pub source: Option<String>,
    pub author: Option<String>,
    pub license: Option<String>,
    pub notes: Option<String>,
}

// ---- App asset list (samp-app/src/assets.toml) --------------------------

#[derive(Debug, Serialize, Deserialize)]
pub struct AppAssets {
    pub asset: Vec<AppAssetRef>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AppAssetRef {
    pub name: String,
    #[serde(rename = "type")]
    pub asset_type: AssetType,
}

// ---- Serde helpers: forward-slash paths ---------------------------------

mod path_serde {
    use std::path::PathBuf;

    use path_slash::PathBufExt as _;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(
        path: &PathBuf,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        path.to_slash_lossy().serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<PathBuf, D::Error> {
        Ok(PathBuf::from_slash(String::deserialize(d)?))
    }
}

mod opt_path_serde {
    use std::path::PathBuf;

    use path_slash::PathBufExt as _;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(
        path: &Option<PathBuf>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        match path {
            Some(p) => p.to_slash_lossy().serialize(s),
            None => s.serialize_none(),
        }
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<Option<PathBuf>, D::Error> {
        Ok(Option::<String>::deserialize(d)?.map(PathBuf::from_slash))
    }
}
