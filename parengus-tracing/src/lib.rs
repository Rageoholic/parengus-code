use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;

use anyhow::Context;
use clap::ValueEnum;
use tracing_subscriber::filter::EnvFilter;
use tracing_subscriber::filter::LevelFilter;
use tracing_subscriber::fmt;
use tracing_subscriber::prelude::*;
use tracing_subscriber::registry::Registry;

#[derive(Clone, Copy, Debug, ValueEnum, PartialEq, Eq)]
pub enum TracingLogLevel {
    Off,
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

impl std::str::FromStr for TracingLogLevel {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "off" => Ok(TracingLogLevel::Off),
            "error" => Ok(TracingLogLevel::Error),
            "warn" | "warning" => Ok(TracingLogLevel::Warn),
            "info" => Ok(TracingLogLevel::Info),
            "debug" => Ok(TracingLogLevel::Debug),
            "trace" => Ok(TracingLogLevel::Trace),
            other => Err(format!("invalid tracing level '{other}'")),
        }
    }
}

impl From<TracingLogLevel> for LevelFilter {
    fn from(v: TracingLogLevel) -> Self {
        match v {
            TracingLogLevel::Off => LevelFilter::OFF,
            TracingLogLevel::Error => LevelFilter::ERROR,
            TracingLogLevel::Warn => LevelFilter::WARN,
            TracingLogLevel::Info => LevelFilter::INFO,
            TracingLogLevel::Debug => LevelFilter::DEBUG,
            TracingLogLevel::Trace => LevelFilter::TRACE,
        }
    }
}

/// Initialize tracing according to a per-target map. `target_levels` maps
/// tracing target prefixes (e.g. "rgpu_vk" or "parengus") to a
/// `TracingLogLevel`. `default_level` is used as the root level when no
/// per-target levels are provided (or when callers explicitly want a
/// non-`Off` default). If the map is empty and `default_level` is `Off` and
/// `log_file` is `None`, this function is a no-op (tracing remains disabled).
pub fn init_default(
    target_levels: HashMap<String, TracingLogLevel>,
    default_level: TracingLogLevel,
    log_file: Option<PathBuf>,
    no_color: bool,
) -> anyhow::Result<()> {
    // If nothing requested and no log file, leave tracing disabled.
    if target_levels.is_empty()
        && log_file.is_none()
        && default_level == TracingLogLevel::Off
    {
        return Ok(());
    }

    // Build EnvFilter directives from default_level + per-target overrides.
    let mut directives: Vec<String> = Vec::new();
    if default_level != TracingLogLevel::Off {
        let s = match default_level {
            TracingLogLevel::Error => "error",
            TracingLogLevel::Warn => "warn",
            TracingLogLevel::Info => "info",
            TracingLogLevel::Debug => "debug",
            TracingLogLevel::Trace => "trace",
            TracingLogLevel::Off => unreachable!(),
        };
        directives.push(s.to_string());
    }
    for (k, v) in target_levels.into_iter() {
        let s = match v {
            TracingLogLevel::Off => "off",
            TracingLogLevel::Error => "error",
            TracingLogLevel::Warn => "warn",
            TracingLogLevel::Info => "info",
            TracingLogLevel::Debug => "debug",
            TracingLogLevel::Trace => "trace",
        };
        directives.push(format!("{}={}", k, s));
    }

    if directives.is_empty() {
        // No directives but a log file was requested — enable at INFO by default.
        // This mirrors common expectations: requesting a log file should enable logging.
        directives.push("info".to_string());
    }

    let directive_str = directives.join(",");
    let env_filter = EnvFilter::try_new(directive_str)
        .with_context(|| "parse tracing directives")?;

    let stdout_layer = fmt::layer().with_ansi(!no_color);

    if let Some(path) = log_file {
        let f = File::create(&path)
            .with_context(|| format!("create log file {}", path.display()))?;
        let file_layer = fmt::layer().with_writer(f).with_ansi(false);
        Registry::default()
            .with(stdout_layer.with_filter(env_filter.clone()))
            .with(file_layer.with_filter(env_filter))
            .init();
    } else {
        Registry::default()
            .with(stdout_layer.with_filter(env_filter))
            .init();
    }

    Ok(())
}
