use std::{collections::BTreeMap, fmt, path::PathBuf};

use serde::{Deserialize, Serialize};

// ---- Asset type ---------------------------------------------------------

#[non_exhaustive]
#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone, Copy)]
#[serde(rename_all = "kebab-case")]
pub enum AssetType {
    Image,
    Shader,
}

impl fmt::Display for AssetType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AssetType::Image => write!(f, "image"),
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

// ---- Generated map (out/.../assets/asset_map.toml) ----------------------

/// Maps logical asset names to filenames in the output assets directory.
/// Serialized by xtask during the build; deserialized by apps at runtime.
#[derive(Debug, Serialize, Deserialize)]
pub struct AssetMap {
    #[serde(with = "path_map_serde")]
    pub map: BTreeMap<String, PathBuf>,
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

mod path_map_serde {
    use std::{collections::BTreeMap, path::PathBuf};

    use path_slash::PathBufExt as _;
    use serde::{Deserialize, Deserializer, Serializer, ser::SerializeMap};

    pub fn serialize<S: Serializer>(
        map: &BTreeMap<String, PathBuf>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        let mut ser = s.serialize_map(Some(map.len()))?;
        for (k, v) in map {
            ser.serialize_entry(k, &*v.to_slash_lossy())?;
        }
        ser.end()
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<BTreeMap<String, PathBuf>, D::Error> {
        BTreeMap::<String, String>::deserialize(d).map(|m| {
            m.into_iter()
                .map(|(k, v)| (k, PathBuf::from_slash(v)))
                .collect()
        })
    }
}
