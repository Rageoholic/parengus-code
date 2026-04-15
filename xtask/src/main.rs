mod assets;

use rayon::ThreadPoolBuilder;
use rayon::prelude::*;
use std::{
    collections::{HashMap, HashSet},
    env, fs,
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
};

use asset_compiler::{image, mesh};
use asset_pipeline::{AppAssets, AssetType, Manifest};
use asset_shared::fnv1a;
use clap::Parser;
use parengus_tracing::{TracingLogLevel, init_default};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

mod cache;
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

        /// Per-target tracing overrides, repeatable:
        /// e.g. --trace-target rgpu_vk=debug
        #[clap(long = "trace-target")]
        trace_target: Vec<String>,

        /// Write logs to file
        #[clap(long = "log-file")]
        log_file: Option<PathBuf>,

        /// Disable ANSI color in stdout logs
        #[clap(long = "no-color")]
        no_color: bool,
        /// Number of worker threads for parallel compilation (0 = auto)
        #[clap(long = "threads", default_value_t = num_cpus::get())]
        threads: usize,

        /// Force recompile all assets even if cache is up-to-date
        #[clap(long = "force", short = 'f')]
        force: bool,

        /// Run BC4/BC5/BC7 texture encoding (default: true).
        /// Omit to let PARENGUS_NO_BC env var decide; set explicitly
        /// to override. e.g. --bc=false skips BC even on machines
        /// without the env var.
        #[clap(
            long = "bc",
            default_missing_value = "true",
            num_args = 0..=1
        )]
        bc: Option<bool>,
    }

    let cli = Cli::parse();

    // Configure Rayon global thread pool to the requested size.
    let threads = if cli.threads == 0 {
        num_cpus::get()
    } else {
        cli.threads
    };
    ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()
        .map_err(|e| format!("failed to configure thread pool: {e}"))?;

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
                "Invalid --trace-target value '{}', \
                 expected target=level",
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
    let no_bc = !cli
        .bc
        .unwrap_or_else(|| env::var("PARENGUS_NO_BC").is_err());
    if let Some(task) = cli.task.as_deref() {
        execute_graph(task, cli.force, no_bc)
    } else {
        eprintln!("Usage: cargo xtask <task>\n");
        eprintln!("Tasks:");
        for task in &all_tasks(false, false)? {
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
    run: Box<dyn Fn() -> Result<()> + Send + Sync>,
}

// Shared task names
const TASK_CHECK_COLLISIONS: &str = "check-collisions";

// Per-app task name prefixes
const TASK_CARGO_BUILD: &str = "cargo-build";
const TASK_COPY_EXE: &str = "copy-exe";
const TASK_COPY_ASSETS: &str = "copy-assets";
const TASK_BUILD: &str = "build";

// Root / aggregate task names
const TASK_BUILD_ALL: &str = "build-all";

fn all_tasks(force: bool, no_compress: bool) -> Result<Vec<Task>> {
    let mut tasks: Vec<Task> = Vec::new();

    let root = workspace_root();
    let manifest = Arc::new(manifest()?);
    let assets_dir = root.join("assets");

    // ── Shared ───────────────────────────────────────────────────

    tasks.push(Task {
        name: TASK_CHECK_COLLISIONS.into(),
        deps: vec![],
        run: Box::new(check_collisions),
    });

    // ── Per-asset compile tasks (deduped across apps) ─────────────
    //
    // Collect the union of asset names so we only compile assets
    // that are actually referenced by at least one app.
    let all_app_assets: Vec<AppAssets> = APPS
        .iter()
        .map(|a| app_assets(a.name))
        .collect::<Result<_>>()?;

    let needed: HashSet<String> = all_app_assets
        .iter()
        .flat_map(|a| a.asset.iter().map(|r| r.name.clone()))
        .collect();

    for entry in &manifest.asset {
        if !needed.contains(&entry.name) {
            continue;
        }

        let name = entry.name.clone();
        let run: Box<dyn Fn() -> Result<()> + Send + Sync> = match entry
            .asset_type
        {
            AssetType::Shader => {
                let Some(sf) = &entry.source_file else {
                    continue;
                };
                let src = assets_dir.join(sf);
                let ext = entry
                    .file
                    .extension()
                    .and_then(|s| s.to_str())
                    .unwrap_or("spv")
                    .to_string();
                let args = entry.compile_args.clone();
                let debug_ext = entry.debug_file.as_ref().and_then(|df| {
                    df.extension().and_then(|s| s.to_str()).map(str::to_string)
                });
                Box::new(move || {
                    compile_shader_asset(&src, &name, &ext, &args, force)?;
                    if let Some(ref dext) = debug_ext {
                        let mut dargs = args.clone();
                        dargs.push("-g".into());
                        compile_shader_asset(&src, &name, dext, &dargs, force)?;
                    }
                    Ok(())
                })
            }
            AssetType::Mesh => {
                let src = assets_dir.join(&entry.file);
                let mf = manifest.clone();
                Box::new(move || compile_mesh_asset(&src, &name, &mf, force))
            }
            AssetType::Image => {
                let src = assets_dir.join(&entry.file);
                let fmt = if no_compress {
                    "rgba8".into()
                } else {
                    entry.format.clone().unwrap_or_else(|| "rgba8".into())
                };
                let cs =
                    entry.color_space.clone().unwrap_or_else(|| "srgb".into());
                let mips = entry.mips.unwrap_or(false);
                let normal_map = entry.normal_map.unwrap_or(false);
                Box::new(move || {
                    compile_image_asset(
                        &src, &name, &fmt, &cs, mips, normal_map, force,
                    )
                })
            }
            _ => {
                return Err(format!(
                    "unsupported asset type for '{}'",
                    entry.name
                )
                .into());
            }
        };

        tasks.push(Task {
            name: format!("compile-asset-{}", entry.name),
            deps: vec![],
            run,
        });
    }

    // ── Per-app tasks ─────────────────────────────────────────────

    for (app, app_assets) in APPS.iter().zip(all_app_assets.iter()) {
        let n = app.name;

        let cargo_build = format!("{TASK_CARGO_BUILD}-{n}");
        let copy_exe = format!("{TASK_COPY_EXE}-{n}");
        let copy_assets_task = format!("{TASK_COPY_ASSETS}-{n}");
        let build = format!("{TASK_BUILD}-{n}");
        let clean = format!("clean-{n}");

        // copy-assets waits for collision check + every asset this
        // app needs
        let mut copy_deps = vec![TASK_CHECK_COLLISIONS.into()];
        for req in &app_assets.asset {
            copy_deps.push(format!("compile-asset-{}", req.name));
        }

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
            name: copy_assets_task.clone(),
            deps: copy_deps,
            run: Box::new(move || copy_assets_for(n)),
        });
        tasks.push(Task {
            name: clean,
            deps: vec![],
            run: Box::new(move || clean_for(n)),
        });
        tasks.push(Task {
            name: build,
            deps: vec![copy_exe, copy_assets_task],
            run: Box::new(|| Ok(())),
        });
    }

    // ── Root / aggregate ─────────────────────────────────────────

    let build_deps: Vec<String> = APPS
        .iter()
        .map(|a| format!("{TASK_BUILD}-{}", a.name))
        .collect();

    tasks.push(Task {
        name: TASK_BUILD.into(),
        deps: build_deps,
        run: Box::new(|| Ok(())),
    });
    tasks.push(Task {
        name: "clean".into(),
        deps: vec![],
        run: Box::new(clean_root),
    });
    tasks.push(Task {
        name: TASK_BUILD_ALL.into(),
        deps: vec![TASK_BUILD.into()],
        run: Box::new(|| Ok(())),
    });

    Ok(tasks)
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

fn execute_graph(target: &str, force: bool, no_compress: bool) -> Result<()> {
    let tasks = all_tasks(force, no_compress)?;
    let mut visited = vec![false; tasks.len()];
    let mut order: Vec<usize> = Vec::new();
    collect_topo(&tasks, target, &mut visited, &mut order)?;

    // Compute DAG depth: level 0 = no deps; level n = max dep level + 1.
    // Walking topo order guarantees all deps are resolved first.
    let mut level = vec![0usize; tasks.len()];
    for &idx in &order {
        for dep in &tasks[idx].deps {
            let di = task_index(&tasks, dep)?;
            if level[di] + 1 > level[idx] {
                level[idx] = level[di] + 1;
            }
        }
    }

    // Group topo-ordered indices by level.
    let max_level = order.iter().map(|&i| level[i]).max().unwrap_or(0);
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); max_level + 1];
    for &idx in &order {
        groups[level[idx]].push(idx);
    }

    let mut statuses: Vec<Option<Status>> = vec![None; tasks.len()];

    for group in &groups {
        // Partition blocked vs runnable before entering par_iter
        // (avoids aliasing `statuses` across threads).
        let (blocked, runnable): (Vec<usize>, Vec<usize>) =
            group.iter().copied().partition(|&idx| {
                tasks[idx].deps.iter().any(|dep| {
                    let di = tasks.iter().position(|t| t.name == *dep).unwrap();
                    matches!(
                        statuses[di],
                        Some(Status::Failed | Status::Skipped)
                    )
                })
            });

        for idx in blocked {
            eprintln!("skip: {}", tasks[idx].name);
            statuses[idx] = Some(Status::Skipped);
        }

        // Run all ready tasks at this level in parallel.
        let results: Vec<(usize, Status)> = runnable
            .par_iter()
            .map(|&idx| match (tasks[idx].run)() {
                Ok(()) => (idx, Status::Succeeded),
                Err(e) => {
                    eprintln!("failed: {} — {e}", tasks[idx].name);
                    (idx, Status::Failed)
                }
            })
            .collect();

        for (idx, st) in results {
            statuses[idx] = Some(st);
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

fn cargo() -> String {
    env::var("CARGO").unwrap_or_else(|_| "cargo".into())
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
                "AssetId collision: '{}' and '{}' both hash \
                 to {hash:016x}",
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

fn copy_if_changed(src: &Path, dst: &Path) -> Result<bool> {
    if is_up_to_date(src, dst) {
        return Ok(false);
    }
    fs::copy(src, dst)?;
    Ok(true)
}

// ----------------------------------------------------------------
// Single-asset compile functions (output directly to cache)
// ----------------------------------------------------------------

fn compile_shader_asset(
    src: &Path,
    name: &str,
    ext: &str,
    compile_args: &[String],
    force: bool,
) -> Result<()> {
    cache::ensure_cache_dir()?;
    if !force && cache::lookup_shader(name, src, compile_args, ext).is_some() {
        println!("Up-to-date: shader {name}");
        return Ok(());
    }
    let dst = cache::artifact_path(name, ext);
    println!("Compiling shader {name}");
    run(Command::new("slangc")
        .arg(src)
        .args(["-target", "spirv", "-o"])
        .arg(&dst)
        .args(compile_args))?;
    if let Err(e) = cache::write_shader_meta(name, src, compile_args, ext) {
        eprintln!("warning: shader meta for {name}: {e}");
    }
    Ok(())
}

fn compile_mesh_asset(
    src: &Path,
    name: &str,
    manifest: &Manifest,
    force: bool,
) -> Result<()> {
    cache::ensure_cache_dir()?;
    if !force && cache::lookup_mesh(name, src).is_some() {
        println!("Up-to-date: mesh {name}");
        return Ok(());
    }
    let dst = cache::artifact_path(name, "pmesh");
    println!("Compiling mesh {name}");
    let tmp = dst.with_extension("pmesh.tmp");
    mesh::compile(src, &tmp, manifest, name)
        .map_err(|e| format!("mesh compile '{name}': {e}"))?;
    if std::fs::rename(&tmp, &dst).is_err() {
        std::fs::copy(&tmp, &dst)?;
        let _ = std::fs::remove_file(&tmp);
    }
    if let Err(e) = cache::write_mesh_meta(name, src) {
        eprintln!("warning: mesh meta for {name}: {e}");
    }
    Ok(())
}

fn compile_image_asset(
    src: &Path,
    name: &str,
    format: &str,
    color_space: &str,
    mips: bool,
    normal_map: bool,
    force: bool,
) -> Result<()> {
    cache::ensure_cache_dir()?;
    if !force
        && cache::lookup_image(name, src, format, color_space, mips, normal_map)
            .is_some()
    {
        println!("Up-to-date: image {name}");
        return Ok(());
    }
    let dst = cache::artifact_path(name, "ptex");
    println!("Compiling image {name}");
    let fmt = match format {
        "bc7" => asset_shared::TexFormat::Bc7,
        "bc5" => asset_shared::TexFormat::Bc5,
        "rgba8" => asset_shared::TexFormat::Rgba8,
        other => {
            return Err(format!("unknown format '{other}'").into());
        }
    };
    let cs = match color_space {
        "srgb" => asset_shared::ColorSpace::Srgb,
        "linear" => asset_shared::ColorSpace::Linear,
        other => {
            return Err(format!("unknown color-space '{other}'").into());
        }
    };
    let tmp = dst.with_extension("ptex.tmp");
    image::compile(src, &tmp, fmt, cs, mips, normal_map)
        .map_err(|e| format!("image compile '{name}': {e}"))?;
    if std::fs::rename(&tmp, &dst).is_err() {
        std::fs::copy(&tmp, &dst)?;
        let _ = std::fs::remove_file(&tmp);
    }
    if let Err(e) = cache::write_image_meta(
        name,
        src,
        format,
        color_space,
        mips,
        normal_map,
    ) {
        eprintln!("warning: image meta for {name}: {e}");
    }
    Ok(())
}

// ----------------------------------------------------------------
// Per-app build steps
// ----------------------------------------------------------------

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
            println!("Removed {compiled_name}");
        }
    }

    cargo_clean_pkg(app_name)?;
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
