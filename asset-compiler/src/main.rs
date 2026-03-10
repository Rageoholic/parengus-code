use std::collections::HashMap;
use std::path::PathBuf;

use asset_compiler::{image, mesh};
use clap::{Parser, Subcommand};
use parengus_tracing::{TracingLogLevel, init_default};

#[derive(Parser)]
#[command(name = "asset-compiler")]
struct Cli {
    #[command(subcommand)]
    command: Command,
    /// Enable tracing output. Defaults to `off`.
    #[arg(long, value_enum, default_value_t = TracingLogLevel::Off)]
    tracing_level: TracingLogLevel,
    /// Write tracing output to this file in addition to stdout when tracing is enabled.
    #[arg(long)]
    log_file: Option<PathBuf>,
    /// Disable ANSI colors in stdout tracing output.
    #[arg(long)]
    no_color: bool,
}

#[derive(Subcommand)]
enum Command {
    /// Compile a glTF mesh to .pmesh
    Mesh {
        input: PathBuf,
        output: PathBuf,
        /// Path to assets/manifest.toml
        #[arg(long)]
        manifest: PathBuf,
        /// Asset name (defaults to output file stem)
        #[arg(long)]
        name: Option<String>,
    },
    /// Compile an image to .ptex
    Image {
        input: PathBuf,
        output: PathBuf,
        /// Target format: bc7 or rgba8
        #[arg(long)]
        format: String,
        /// Colour space: srgb or linear
        #[arg(long)]
        color_space: String,
        /// Generate full mip chain
        #[arg(long)]
        mips: bool,
    },
}

fn main() {
    let cli = Cli::parse();

    // Build per-target map and call shared tracing init. If the user passed
    // only --log-file without enabling a level, use INFO as the default.
    let mut targets: HashMap<String, TracingLogLevel> = HashMap::new();
    if cli.tracing_level != TracingLogLevel::Off {
        targets.insert("asset-compiler".to_string(), cli.tracing_level);
    }
    let default_level = if cli.tracing_level == TracingLogLevel::Off
        && cli.log_file.is_some()
    {
        TracingLogLevel::Info
    } else {
        cli.tracing_level
    };

    if let Err(e) =
        init_default(targets, default_level, cli.log_file.clone(), cli.no_color)
    {
        eprintln!("failed to init tracing: {e}");
        std::process::exit(1);
    }

    let result = run(cli);
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn run(cli: Cli) -> Result<(), String> {
    match cli.command {
        Command::Mesh {
            input,
            output,
            manifest,
            name,
        } => {
            let asset_name = name.unwrap_or_else(|| {
                output
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("")
                    .to_owned()
            });
            let manifest_text =
                std::fs::read_to_string(&manifest).map_err(|e| {
                    format!("read manifest {}: {e}", manifest.display())
                })?;
            let manifest: asset_pipeline::Manifest =
                toml::from_str(&manifest_text)
                    .map_err(|e| format!("parse manifest: {e}"))?;
            mesh::compile(&input, &output, &manifest, &asset_name)
        }
        Command::Image {
            input,
            output,
            format,
            color_space,
            mips,
        } => {
            let fmt = match format.as_str() {
                "bc7" => asset_shared::TexFormat::Bc7,
                "rgba8" => asset_shared::TexFormat::Rgba8,
                _ => return Err(format!("unknown format '{format}'")),
            };
            let cs = match color_space.as_str() {
                "srgb" => asset_shared::ColorSpace::Srgb,
                "linear" => asset_shared::ColorSpace::Linear,
                _ => {
                    return Err(format!("unknown color-space '{color_space}'"));
                }
            };
            image::compile(&input, &output, fmt, cs, mips)
        }
    }
}
