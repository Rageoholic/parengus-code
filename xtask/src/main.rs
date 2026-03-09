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

// Task name prefixes for generated per-app tasks.
const TASK_CARGO_BUILD: &str = "cargo-build";
const TASK_COPY_EXE: &str = "copy-exe";
const TASK_COPY_ASSETS: &str = "copy-assets";
const TASK_BUILD: &str = "build";

// Names of shared / root tasks.
const TASK_COMPILE_SHADERS: &str = "compile-shaders";
const TASK_BUILD_ALL: &str = "build-all";

const TASKS_SHARED: usize = 1; // compile-shaders
const TASKS_PER_APP: usize = 4; // cargo-build, copy-exe, copy-assets, build-{app}
const TASKS_ROOT: usize = 2; // build, build-all
const TASK_COUNT: usize =
    TASKS_SHARED + TASKS_PER_APP * APPS.len() + TASKS_ROOT;

fn all_tasks() -> Vec<Task> {
    let mut tasks: Vec<Task> = Vec::with_capacity(TASK_COUNT);

    tasks.push(Task {
        name: TASK_COMPILE_SHADERS.into(),
        deps: vec![],
        run: Box::new(compile_shaders),
    });

    assert!(tasks.len() == TASKS_SHARED, "Shared task count incorrect");

    for app in APPS {
        let n = app.name;
        let cargo_build = format!("{TASK_CARGO_BUILD}-{n}");
        let copy_exe = format!("{TASK_COPY_EXE}-{n}");
        let copy_assets = format!("{TASK_COPY_ASSETS}-{n}");
        let build = format!("{TASK_BUILD}-{n}");

        let pre_app_task_count = tasks.len();

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
            deps: vec![TASK_COMPILE_SHADERS.into()],
            run: Box::new(move || copy_assets_for(n)),
        });
        tasks.push(Task {
            name: build,
            deps: vec![
                cargo_build,
                TASK_COMPILE_SHADERS.into(),
                copy_exe,
                copy_assets,
            ],
            run: Box::new(|| Ok(())),
        });

        assert!(
            tasks.len() - pre_app_task_count == TASKS_PER_APP,
            "per app task count incorrect"
        );
    }

    let build_deps: Vec<String> = APPS
        .iter()
        .map(|a| format!("{TASK_BUILD}-{}", a.name))
        .collect();

    let pre_root_task_count = tasks.len();

    tasks.push(Task {
        name: TASK_BUILD.into(),
        deps: build_deps,
        run: Box::new(|| Ok(())),
    });
    tasks.push(Task {
        name: TASK_BUILD_ALL.into(),
        deps: vec![TASK_BUILD.into()],
        run: Box::new(|| Ok(())),
    });

    assert!(
        tasks.len() - pre_root_task_count == TASKS_ROOT,
        "Root task count incorrect"
    );

    assert!(tasks.len() == TASK_COUNT, "Total task count incorrect");

    tasks
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
    // Collect deps into a temporary vec to avoid borrow conflicts.
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
            let dep_idx = tasks.iter().position(|t| t.name == *dep).unwrap();
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
