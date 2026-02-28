mod assets;

use std::{
    env,
    ffi::OsStr,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

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
            name: "copy-assets",
            deps: &[],
            run: copy_assets,
        },
        Task {
            name: "build",
            deps: &[
                "cargo-build",
                "compile-shaders",
                "copy-exe",
                "copy-assets",
            ],
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

fn cargo_build() -> Result<()> {
    let root = workspace_root();
    let cargo = env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    run(Command::new(cargo)
        .args(["build", "-p", "samp-app"])
        .current_dir(&root))
}

fn compile_shaders() -> Result<()> {
    let root = workspace_root();
    let src_dir = root.join("samp-app").join("shaders");
    let out_dir = root
        .join("out")
        .join("samp-app")
        .join("debug")
        .join("shaders");
    fs::create_dir_all(&out_dir)?;

    // (output suffix, extra slangc args)
    let variants: &[(&str, &[&str])] = &[("", &[]), (".debug", &["-g"])];

    let mut compiled = 0u32;
    let mut skipped = 0u32;

    for entry in fs::read_dir(&src_dir)? {
        let entry = entry?;
        let src = entry.path();
        if src.extension() != Some(OsStr::new("slang")) {
            continue;
        }
        let stem = src.file_stem().unwrap().to_string_lossy().into_owned();

        for &(suffix, extra_args) in variants {
            let dst = out_dir.join(format!("{stem}{suffix}.spv"));

            if is_up_to_date(&src, &dst) {
                skipped += 1;
                continue;
            }

            println!(
                "Compiling {} -> {}",
                src.file_name().unwrap().to_string_lossy(),
                dst.file_name().unwrap().to_string_lossy(),
            );

            run(Command::new("slangc")
                .arg(&src)
                .args(["-target", "spirv", "-o"])
                .arg(&dst)
                .args(extra_args))?;

            compiled += 1;
        }
    }

    println!("Shaders: {compiled} compiled, {skipped} up-to-date");
    Ok(())
}

fn copy_if_changed(src: &Path, dst: &Path) -> Result<bool> {
    if is_up_to_date(src, dst) {
        return Ok(false);
    }
    fs::copy(src, dst)?;
    Ok(true)
}

fn copy_exe() -> Result<()> {
    let root = workspace_root();
    let out_dir = root.join("out").join("samp-app").join("debug");
    fs::create_dir_all(&out_dir)?;

    let exe_suffix = env::consts::EXE_SUFFIX;
    let exe_name = format!("samp-app{exe_suffix}");
    let src_exe = root.join("target").join("debug").join(&exe_name);
    let dst_exe = out_dir.join(&exe_name);

    if copy_if_changed(&src_exe, &dst_exe)? {
        println!("Copied {exe_name}");
    } else {
        println!("Up-to-date: {exe_name}");
    }

    #[cfg(windows)]
    {
        let src_pdb = root.join("target").join("debug").join("samp_app.pdb");
        let dst_pdb = out_dir.join("samp_app.pdb");
        copy_if_changed(&src_pdb, &dst_pdb)?;
    }

    Ok(())
}

fn copy_assets() -> Result<()> {
    let root = workspace_root();
    assets::copy_assets(
        &root.join("assets").join("manifest.toml"),
        &root.join("samp-app").join("src").join("assets.toml"),
        &root.join("assets"),
        &root
            .join("out")
            .join("samp-app")
            .join("debug")
            .join("assets"),
    )
}
