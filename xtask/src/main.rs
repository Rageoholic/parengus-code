mod assets;

use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

use asset_pipeline::{AssetType, Manifest};

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
    match env::args().nth(1).as_deref() {
        Some(task) => execute_graph(task),
        None => {
            eprintln!("Usage: cargo xtask <task>\n");
            eprintln!("Tasks:");
            for task in &all_tasks() {
                eprintln!("  {}", task.name);
            }
            std::process::exit(1);
        }
    }
}

// ----------------------------------------------------------------
// Task graph
// ----------------------------------------------------------------

struct Task {
    name: &'static str,
    deps: &'static [&'static str],
    run: fn() -> Result<()>,
}

fn noop() -> Result<()> {
    Ok(())
}

fn all_tasks() -> Vec<Task> {
    vec![
        Task {
            name: "cargo-build",
            deps: &[],
            run: cargo_build,
        },
        Task {
            name: "cargo-build-noext",
            deps: &[],
            run: cargo_build_noext,
        },
        Task {
            name: "compile-shaders",
            deps: &[],
            run: compile_shaders,
        },
        Task {
            name: "copy-exe",
            deps: &["cargo-build"],
            run: copy_exe,
        },
        Task {
            name: "copy-exe-noext",
            deps: &["cargo-build-noext"],
            run: copy_exe_noext,
        },
        Task {
            name: "copy-assets",
            deps: &["compile-shaders"],
            run: copy_assets,
        },
        Task {
            name: "copy-assets-noext",
            deps: &["compile-shaders"],
            run: copy_assets_noext,
        },
        Task {
            name: "build-samp-app",
            deps: &[
                "cargo-build",
                "compile-shaders",
                "copy-exe",
                "copy-assets",
            ],
            run: noop,
        },
        Task {
            name: "build-noext",
            deps: &[
                "cargo-build-noext",
                "compile-shaders",
                "copy-exe-noext",
                "copy-assets-noext",
            ],
            run: noop,
        },
        Task {
            name: "cargo-build-phoenix",
            deps: &[],
            run: cargo_build_phoenix,
        },
        Task {
            name: "copy-exe-phoenix",
            deps: &["cargo-build-phoenix"],
            run: copy_exe_phoenix,
        },
        Task {
            name: "copy-assets-phoenix",
            deps: &["compile-shaders"],
            run: copy_assets_phoenix,
        },
        Task {
            name: "build-phoenix",
            deps: &[
                "cargo-build-phoenix",
                "compile-shaders",
                "copy-exe-phoenix",
                "copy-assets-phoenix",
            ],
            run: noop,
        },
        Task {
            name: "build",
            deps: &["build-samp-app", "build-noext", "build-phoenix"],
            run: noop,
        },
        Task {
            name: "build-all",
            deps: &["build"],
            run: noop,
        },
    ]
}

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
    let deps = tasks[idx].deps;
    for &dep in deps {
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
        let blocked = task.deps.iter().any(|&dep| {
            let dep_idx = tasks.iter().position(|t| t.name == dep).unwrap();
            matches!(statuses[dep_idx], Some(Status::Failed | Status::Skipped))
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
        .map(|&i| tasks[i].name)
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
// Task implementations
// ----------------------------------------------------------------

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask has a parent directory")
        .to_path_buf()
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

// ---- Per-app helpers --------------------------------------------

fn cargo_build_pkg(pkg: &str) -> Result<()> {
    let root = workspace_root();
    let cargo = env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    run(Command::new(cargo)
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
        &root.join("assets"),
        &root.join("cache").join("shaders"),
        &root.join("out").join(app_name).join("debug").join("assets"),
    )
}

// ---- Task entry points ------------------------------------------

fn cargo_build() -> Result<()> {
    cargo_build_pkg("samp-app")
}

fn cargo_build_noext() -> Result<()> {
    cargo_build_pkg("samp-app-noext")
}

fn compile_shaders() -> Result<()> {
    let root = workspace_root();
    let assets_dir = root.join("assets");
    let cache_dir = root.join("cache").join("shaders");
    fs::create_dir_all(&cache_dir)?;

    let manifest: Manifest =
        toml::from_str(&fs::read_to_string(assets_dir.join("manifest.toml"))?)?;

    let mut compiled = 0u32;
    let mut skipped = 0u32;

    for entry in &manifest.asset {
        if entry.asset_type != AssetType::Shader {
            continue;
        }
        let Some(source_file) = &entry.source_file else {
            continue;
        };

        let src = assets_dir.join(source_file);
        let dst = cache_dir.join(&entry.file);

        if is_up_to_date(&src, &dst) {
            skipped += 1;
            continue;
        }

        println!(
            "Compiling {} -> {}",
            source_file.display(),
            entry.file.display(),
        );

        run(Command::new("slangc")
            .arg(&src)
            .args(["-target", "spirv", "-o"])
            .arg(&dst)
            .args(&entry.compile_args))?;

        compiled += 1;
    }

    println!("Shaders: {compiled} compiled, {skipped} up-to-date");
    Ok(())
}

fn copy_exe() -> Result<()> {
    copy_exe_for("samp-app")
}

fn copy_exe_noext() -> Result<()> {
    copy_exe_for("samp-app-noext")
}

fn copy_assets() -> Result<()> {
    copy_assets_for("samp-app")
}

fn copy_assets_noext() -> Result<()> {
    copy_assets_for("samp-app-noext")
}

fn cargo_build_phoenix() -> Result<()> {
    cargo_build_pkg("phoenix")
}

fn copy_exe_phoenix() -> Result<()> {
    copy_exe_for("phoenix")
}

fn copy_assets_phoenix() -> Result<()> {
    copy_assets_for("phoenix")
}
