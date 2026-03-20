mod assets;

use std::{
    collections::HashMap,
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

use asset_compiler::{image, mesh};
use asset_pipeline::{AppAssets, AssetType, Manifest};
use asset_shared::fnv1a;
use clap::Parser;
use parengus_tracing::{TracingLogLevel, init_default};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

// ----------------------------------------------------------------
// Entry point
// ----------------------------------------------------------------

fn main() {
    if let Err(e) = try_main() {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn try_main() -> Result<()> {
    #[derive(Parser)]
    struct Cli {
        /// Task name to run (see `cargo xtask` for list)
        task: Option<String>,

        /// Root tracing level (Off, Error, Warn, Info, Debug, Trace)
        #[clap(long = "tracing-level", value_enum, default_value_t = TracingLogLevel::Off)]
        tracing_level: TracingLogLevel,

        /// Per-target tracing overrides, repeatable: e.g. --trace-target rgpu_vk=debug
        #[clap(long = "trace-target")]
        trace_target: Vec<String>,

        /// Write logs to file
        #[clap(long = "log-file")]
        log_file: Option<PathBuf>,

        /// Disable ANSI color in stdout logs
        #[clap(long = "no-color")]
        no_color: bool,
    }

    let cli = Cli::parse();

    // Build target-level map from `--trace-target` entries
    let mut target_levels: HashMap<String, TracingLogLevel> = HashMap::new();
    for t in &cli.trace_target {
        if let Some((k, v)) = t.split_once('=') {
            let lvl = match v.to_ascii_lowercase().as_str() {
                "off" => TracingLogLevel::Off,
                "error" => TracingLogLevel::Error,
                "warn" | "warning" => TracingLogLevel::Warn,
                "info" => TracingLogLevel::Info,
                "debug" => TracingLogLevel::Debug,
                "trace" => TracingLogLevel::Trace,
                _ => {
                    eprintln!("Invalid trace level '{}', ignoring", v);
                    continue;
                }
            };
            target_levels.insert(k.to_string(), lvl);
        } else {
            eprintln!(
                "Invalid --trace-target value '{}', expected target=level",
                t
            );
        }
    }

    // Initialize tracing according to CLI flags (no-op if all defaults)
    init_default(
        target_levels,
        cli.tracing_level,
        cli.log_file.clone(),
        cli.no_color,
    )
    .map_err(|e| format!("init tracing: {e}"))?;
    if let Some(task) = cli.task.as_deref() {
        execute_graph(task)
    } else {
        eprintln!("Usage: cargo xtask <task>\n");
        eprintln!("Tasks:");
        for task in &all_tasks() {
            eprintln!("  {}", task.name);
        }
        std::process::exit(1);
    }
}

// ----------------------------------------------------------------
// App registry
// ----------------------------------------------------------------

struct App {
    name: &'static str,
}

const APPS: &[App] = &[
    App { name: "samp-app" },
    App {
        name: "samp-app-noext",
    },
    App { name: "phoenix" },
];

// ----------------------------------------------------------------
// Task graph
// ----------------------------------------------------------------

struct Task {
    name: String,
    deps: Vec<String>,
    run: Box<dyn Fn() -> Result<()>>,
}

// Shared task names
const TASK_CHECK_COLLISIONS: &str = "check-collisions";

// Per-app task name prefixes
const TASK_COMPILE_SHADERS: &str = "compile-shaders";
const TASK_COMPILE_MESHES: &str = "compile-meshes";
const TASK_COMPILE_IMAGES: &str = "compile-images";
const TASK_CARGO_BUILD: &str = "cargo-build";
const TASK_COPY_EXE: &str = "copy-exe";
const TASK_COPY_ASSETS: &str = "copy-assets";
const TASK_BUILD: &str = "build";
const TASK_CLEAN: &str = "clean";

// Root / aggregate task names
const TASK_BUILD_ALL: &str = "build-all";

const TASKS_SHARED: usize = 1; // check-collisions
const TASKS_PER_APP: usize = 8; // compile-shaders, compile-meshes,
// compile-images, cargo-build,
// copy-exe, copy-assets, build-{app}, clean-{app}
const TASKS_ROOT: usize = 3; // build, build-all, clean
const TASK_COUNT: usize =
    TASKS_SHARED + TASKS_PER_APP * APPS.len() + TASKS_ROOT;

fn all_tasks() -> Vec<Task> {
    let mut tasks: Vec<Task> = Vec::with_capacity(TASK_COUNT);

    // ── Shared ───────────────────────────────────────────────────

    tasks.push(Task {
        name: TASK_CHECK_COLLISIONS.into(),
        deps: vec![],
        run: Box::new(check_collisions),
    });

    assert!(tasks.len() == TASKS_SHARED, "shared task count incorrect");

    // ── Per-app ──────────────────────────────────────────────────

    for app in APPS {
        let n = app.name;

        let compile_shaders = format!("{TASK_COMPILE_SHADERS}-{n}");
        let compile_meshes = format!("{TASK_COMPILE_MESHES}-{n}");
        let compile_images = format!("{TASK_COMPILE_IMAGES}-{n}");
        let cargo_build = format!("{TASK_CARGO_BUILD}-{n}");
        let copy_exe = format!("{TASK_COPY_EXE}-{n}");
        let copy_assets = format!("{TASK_COPY_ASSETS}-{n}");
        let build = format!("{TASK_BUILD}-{n}");
        let clean = format!("clean-{n}");

        let pre = tasks.len();

        tasks.push(Task {
            name: compile_shaders.clone(),
            deps: vec![],
            run: Box::new(move || compile_shaders_for(n)),
        });
        tasks.push(Task {
            name: compile_meshes.clone(),
            deps: vec![],
            run: Box::new(move || compile_meshes_for(n)),
        });
        tasks.push(Task {
            name: compile_images.clone(),
            deps: vec![],
            run: Box::new(move || compile_images_for(n)),
        });
        tasks.push(Task {
            name: cargo_build.clone(),
            deps: vec![],
            run: Box::new(move || cargo_build_pkg(n)),
        });
        tasks.push(Task {
            name: copy_exe.clone(),
            deps: vec![cargo_build.clone()],
            run: Box::new(move || copy_exe_for(n)),
        });
        tasks.push(Task {
            name: copy_assets.clone(),
            deps: vec![
                TASK_CHECK_COLLISIONS.into(),
                compile_shaders,
                compile_meshes,
                compile_images,
            ],
            run: Box::new(move || copy_assets_for(n)),
        });
        tasks.push(Task {
            name: clean.clone(),
            deps: vec![],
            run: Box::new(move || clean_for(n)),
        });
        tasks.push(Task {
            name: build,
            deps: vec![copy_exe, copy_assets],
            run: Box::new(|| Ok(())),
        });

        assert!(
            tasks.len() - pre == TASKS_PER_APP,
            "per-app task count incorrect"
        );
    }

    // ── Root / aggregate ─────────────────────────────────────────

    let build_deps: Vec<String> = APPS
        .iter()
        .map(|a| format!("{TASK_BUILD}-{}", a.name))
        .collect();

    let pre_root = tasks.len();

    tasks.push(Task {
        name: TASK_BUILD.into(),
        deps: build_deps,
        run: Box::new(|| Ok(())),
    });
    // Root clean: remove compiled cache and out/ directory.
    tasks.push(Task {
        name: TASK_CLEAN.into(),
        deps: vec![],
        run: Box::new(clean_root),
    });
    tasks.push(Task {
        name: TASK_BUILD_ALL.into(),
        deps: vec![TASK_BUILD.into()],
        run: Box::new(|| Ok(())),
    });

    assert!(
        tasks.len() - pre_root == TASKS_ROOT,
        "root task count incorrect"
    );

    assert!(tasks.len() == TASK_COUNT, "total task count incorrect");

    tasks
}

// ----------------------------------------------------------------
// Task graph execution
// ----------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Status {
    Succeeded,
    Failed,
    Skipped,
}

fn task_index(tasks: &[Task], name: &str) -> Result<usize> {
    tasks
        .iter()
        .position(|t| t.name == name)
        .ok_or_else(|| format!("unknown task: `{name}`").into())
}

fn collect_topo(
    tasks: &[Task],
    name: &str,
    visited: &mut Vec<bool>,
    order: &mut Vec<usize>,
) -> Result<()> {
    let idx = task_index(tasks, name)?;
    if visited[idx] {
        return Ok(());
    }
    visited[idx] = true;
    let deps: Vec<String> = tasks[idx].deps.clone();
    for dep in &deps {
        collect_topo(tasks, dep, visited, order)?;
    }
    order.push(idx);
    Ok(())
}

fn execute_graph(target: &str) -> Result<()> {
    let tasks = all_tasks();
    let mut visited = vec![false; tasks.len()];
    let mut order: Vec<usize> = Vec::new();
    collect_topo(&tasks, target, &mut visited, &mut order)?;

    let mut statuses: Vec<Option<Status>> = vec![None; tasks.len()];

    for &idx in &order {
        let task = &tasks[idx];
        let blocked = task.deps.iter().any(|dep| {
            let di = tasks.iter().position(|t| t.name == *dep).unwrap();
            matches!(statuses[di], Some(Status::Failed | Status::Skipped))
        });

        if blocked {
            eprintln!("skip: {}", task.name);
            statuses[idx] = Some(Status::Skipped);
            continue;
        }

        match (task.run)() {
            Ok(()) => statuses[idx] = Some(Status::Succeeded),
            Err(e) => {
                eprintln!("failed: {} — {e}", task.name);
                statuses[idx] = Some(Status::Failed);
            }
        }
    }

    let failed: Vec<&str> = order
        .iter()
        .filter(|&&i| statuses[i] == Some(Status::Failed))
        .map(|&i| tasks[i].name.as_str())
        .collect();

    if failed.is_empty() {
        Ok(())
    } else {
        Err(
            format!("{} task(s) failed: {}", failed.len(), failed.join(", "))
                .into(),
        )
    }
}

// ----------------------------------------------------------------
// Helpers
// ----------------------------------------------------------------

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask has a parent directory")
        .to_path_buf()
}

pub(crate) fn is_up_to_date(src: &Path, dst: &Path) -> bool {
    let Ok(sm) = src.metadata() else { return false };
    let Ok(dm) = dst.metadata() else { return false };
    let Ok(st) = sm.modified() else { return false };
    let Ok(dt) = dm.modified() else { return false };
    st <= dt
}

fn run(cmd: &mut Command) -> Result<()> {
    let status = cmd.status()?;
    if !status.success() {
        return Err(format!(
            "command {:?} failed with {}",
            cmd.get_program(),
            status
        )
        .into());
    }
    Ok(())
}

fn copy_if_changed(src: &Path, dst: &Path) -> Result<bool> {
    if is_up_to_date(src, dst) {
        return Ok(false);
    }
    fs::copy(src, dst)?;
    Ok(true)
}

fn manifest() -> Result<Manifest> {
    let root = workspace_root();
    let text = fs::read_to_string(root.join("assets").join("manifest.toml"))?;
    Ok(toml::from_str(&text)?)
}

fn app_assets(app_name: &str) -> Result<AppAssets> {
    let root = workspace_root();
    let text = fs::read_to_string(root.join(app_name).join("assets.toml"))?;
    Ok(toml::from_str(&text)?)
}

fn cargo() -> String {
    env::var("CARGO").unwrap_or_else(|_| "cargo".into())
}

// ----------------------------------------------------------------
// Task implementations
// ----------------------------------------------------------------

fn check_collisions() -> Result<()> {
    let manifest = manifest()?;

    let mut meshes: HashMap<u64, &str> = HashMap::new();
    let mut textures: HashMap<u64, &str> = HashMap::new();
    let mut shaders: HashMap<u64, &str> = HashMap::new();

    for entry in &manifest.asset {
        let hash = fnv1a(&entry.name);
        let bucket = match entry.asset_type {
            AssetType::Mesh => &mut meshes,
            AssetType::Image => &mut textures,
            AssetType::Shader => &mut shaders,
            _ => continue,
        };
        if let Some(prev) = bucket.insert(hash, entry.name.as_str()) {
            return Err(format!(
                "AssetId collision: '{}' and '{}' \
                 both hash to {hash:016x}",
                prev, entry.name
            )
            .into());
        }
    }

    println!(
        "No AssetId collisions ({} assets checked)",
        manifest.asset.len()
    );
    Ok(())
}

fn compile_shaders_for(app_name: &str) -> Result<()> {
    let root = workspace_root();
    let assets_dir = root.join("assets");
    let out_assets_dir =
        root.join("out").join(app_name).join("debug").join("assets");
    fs::create_dir_all(&out_assets_dir)?;

    let manifest = manifest()?;
    let app = app_assets(app_name)?;

    let index: HashMap<&str, _> = manifest
        .asset
        .iter()
        .map(|e| (e.name.as_str(), e))
        .collect();

    let mut compiled = 0u32;

    for req in &app.asset {
        if req.asset_type != AssetType::Shader {
            continue;
        }
        let entry = index
            .get(req.name.as_str())
            .ok_or_else(|| format!("shader '{}' not in manifest", req.name))?;
        let Some(source_file) = &entry.source_file else {
            continue;
        };

        let src = assets_dir.join(source_file);
        let dst = out_assets_dir.join(&entry.file);

        // Always recompile shaders to avoid caching issues.
        println!("Compiling shader {} → {}", entry.name, entry.file.display());
        run(Command::new("slangc")
            .arg(&src)
            .args(["-target", "spirv", "-o"])
            .arg(&dst)
            .args(&entry.compile_args))?;
        compiled += 1;
    }

    println!("{app_name} shaders: {compiled} compiled");
    Ok(())
}

fn compile_meshes_for(app_name: &str) -> Result<()> {
    let root = workspace_root();
    let assets_dir = root.join("assets");

    let out_assets_dir =
        root.join("out").join(app_name).join("debug").join("assets");
    fs::create_dir_all(&out_assets_dir)?;

    let manifest = manifest()?;
    let app = app_assets(app_name)?;

    let index: HashMap<&str, _> = manifest
        .asset
        .iter()
        .map(|e| (e.name.as_str(), e))
        .collect();

    let mut compiled = 0u32;

    for req in &app.asset {
        if req.asset_type != AssetType::Mesh {
            continue;
        }
        let entry = index
            .get(req.name.as_str())
            .ok_or_else(|| format!("mesh '{}' not in manifest", req.name))?;

        let src = assets_dir.join(&entry.file);
        let dst = out_assets_dir.join(format!("{}.pmesh", req.name));

        // Always recompile meshes.
        println!("Compiling mesh {} → {}.pmesh", req.name, req.name);
        mesh::compile(&src, &dst, &manifest, req.name.as_str()).map_err(
            |e| {
                Box::<dyn std::error::Error>::from(std::io::Error::other(
                    format!("mesh compile: {e}"),
                ))
            },
        )?;
        compiled += 1;
    }

    println!("{app_name} meshes: {compiled} compiled");
    Ok(())
}

fn compile_images_for(app_name: &str) -> Result<()> {
    let root = workspace_root();
    let assets_dir = root.join("assets");
    let out_assets_dir =
        root.join("out").join(app_name).join("debug").join("assets");
    fs::create_dir_all(&out_assets_dir)?;

    let manifest = manifest()?;
    let app = app_assets(app_name)?;

    let index: HashMap<&str, _> = manifest
        .asset
        .iter()
        .map(|e| (e.name.as_str(), e))
        .collect();

    let mut compiled = 0u32;

    for req in &app.asset {
        if req.asset_type != AssetType::Image {
            continue;
        }
        let entry = index
            .get(req.name.as_str())
            .ok_or_else(|| format!("image '{}' not in manifest", req.name))?;

        let src = assets_dir.join(&entry.file);
        let dst = out_assets_dir.join(format!("{}.ptex", req.name));

        let format = entry.format.as_deref().unwrap_or("rgba8");
        let color_space = entry.color_space.as_deref().unwrap_or("srgb");
        let mips = entry.mips.unwrap_or(false);

        // Always recompile images.
        println!("Compiling image {} → {}.ptex", req.name, req.name);
        let fmt = match format {
            "bc7" => asset_shared::TexFormat::Bc7,
            "rgba8" => asset_shared::TexFormat::Rgba8,
            other => return Err(format!("unknown format '{other}'").into()),
        };
        let cs = match color_space {
            "srgb" => asset_shared::ColorSpace::Srgb,
            "linear" => asset_shared::ColorSpace::Linear,
            other => {
                return Err(format!("unknown color-space '{other}'").into());
            }
        };
        image::compile(&src, &dst, fmt, cs, mips).map_err(|e| {
            Box::<dyn std::error::Error>::from(std::io::Error::other(format!(
                "image compile: {e}"
            )))
        })?;
        compiled += 1;
    }

    println!("{app_name} images: {compiled} compiled");
    Ok(())
}

fn cargo_build_pkg(pkg: &str) -> Result<()> {
    let root = workspace_root();
    run(Command::new(cargo())
        .args(["build", "-p", pkg])
        .current_dir(&root))
}

fn copy_exe_for(app_name: &str) -> Result<()> {
    let root = workspace_root();
    let out_dir = root.join("out").join(app_name).join("debug");
    fs::create_dir_all(&out_dir)?;

    let exe_suffix = env::consts::EXE_SUFFIX;
    let exe_name = format!("{app_name}{exe_suffix}");
    let src_exe = root.join("target").join("debug").join(&exe_name);
    let dst_exe = out_dir.join(&exe_name);

    if copy_if_changed(&src_exe, &dst_exe)? {
        println!("Copied {exe_name}");
    } else {
        println!("Up-to-date: {exe_name}");
    }

    #[cfg(windows)]
    {
        let pdb_name = format!("{}.pdb", app_name.replace('-', "_"));
        let src_pdb = root.join("target").join("debug").join(&pdb_name);
        let dst_pdb = out_dir.join(&pdb_name);
        copy_if_changed(&src_pdb, &dst_pdb)?;
    }

    Ok(())
}

fn copy_assets_for(app_name: &str) -> Result<()> {
    let root = workspace_root();
    assets::copy_assets(
        &root.join("assets").join("manifest.toml"),
        &root.join(app_name).join("assets.toml"),
        &root.join("out").join(app_name).join("debug").join("assets"),
    )
}

fn clean_for(app_name: &str) -> Result<()> {
    // Remove compiled assets for this app from the out assets directory
    // Load the app's asset list to determine compiled filenames
    let app = app_assets(app_name)?;
    let manifest = manifest()?;
    let index: HashMap<&str, _> = manifest
        .asset
        .iter()
        .map(|e| (e.name.as_str(), e))
        .collect();
    for req in &app.asset {
        let entry = index.get(req.name.as_str()).ok_or_else(|| {
            format!("asset '{}' not found in manifest", req.name)
        })?;
        let compiled_name = match entry.asset_type {
            AssetType::Mesh => format!("{}.pmesh", req.name),
            AssetType::Image => format!("{}.ptex", req.name),
            AssetType::Shader => entry.file.to_string_lossy().into_owned(),
            _ => continue,
        };
        let path = workspace_root()
            .join("out")
            .join(app_name)
            .join("debug")
            .join("assets")
            .join(&compiled_name);
        if path.exists() {
            fs::remove_file(&path)?;
            println!("Removed cached {}", compiled_name);
        }
    }

    // Run `cargo clean -p <pkg>` to remove build artifacts for the package
    cargo_clean_pkg(app_name)?;
    // Remove copied runtime outputs under `out/<app>`
    remove_out_for(app_name)?;
    Ok(())
}

fn cargo_clean_pkg(pkg: &str) -> Result<()> {
    let root = workspace_root();
    run(Command::new(cargo())
        .args(["clean", "-p", pkg])
        .current_dir(&root))
}

fn remove_out_for(app_name: &str) -> Result<()> {
    let root = workspace_root();
    let out_dir = root.join("out").join(app_name);
    if out_dir.exists() {
        fs::remove_dir_all(&out_dir)?;
        println!("Removed {}", out_dir.display());
    } else {
        println!("No out directory for {}", app_name);
    }
    Ok(())
}

fn clean_root() -> Result<()> {
    let root = workspace_root();
    let out_dir = root.join("out");
    if out_dir.exists() {
        fs::remove_dir_all(&out_dir)?;
        println!("Removed {}", out_dir.display());
    } else {
        println!("No out directory to remove");
    }

    Ok(())
}
