#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::undocumented_unsafe_blocks)]

use std::{
    fs::{self, File},
    sync::Arc,
};

use clap::Parser;
use rgpu::{
    instance::{Instance, InstanceExtensions},
    surface::Surface,
};
use strum_macros::EnumString;
use tracing_subscriber::{Layer, layer::SubscriberExt, util::SubscriberInitExt};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::ControlFlow,
    window::{Window as WinitWindow, WindowAttributes},
};

#[derive(Debug, EnumString, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Default)]
enum TracingLogLevel {
    Off,
    Trace,
    Info,
    Debug,
    Warn,
    #[default]
    Error,
}

impl From<TracingLogLevel> for tracing::Level {
    fn from(value: TracingLogLevel) -> Self {
        match value {
            //We clamp this to the lowest possible level but this shouldn't happen
            TracingLogLevel::Off => tracing::Level::TRACE,
            TracingLogLevel::Trace => tracing::Level::TRACE,
            TracingLogLevel::Info => tracing::Level::INFO,
            TracingLogLevel::Debug => tracing::Level::DEBUG,
            TracingLogLevel::Warn => tracing::Level::WARN,
            TracingLogLevel::Error => tracing::Level::ERROR,
        }
    }
}

#[derive(clap::Parser, Debug)]
struct CliArgs {
    #[arg(short, long)]
    tracing_log_level: TracingLogLevel,
    #[arg(short, long)]
    graphics_debug_level: Option<rgpu::log::VulkanLogLevel>,
}

fn main() -> eyre::Result<()> {
    let app_dirs = directories::ProjectDirs::from("", "parengus", "samp-app");

    let log_dir = app_dirs
        .as_ref()
        .and_then(|x| x.runtime_dir().or_else(|| Some(x.data_dir())))
        .map(|p| p.to_owned())
        .unwrap_or_else(|| std::env::current_dir().expect("failed to get current directory"));

    let cli_args = CliArgs::parse();

    if cli_args.tracing_log_level != TracingLogLevel::Off {
        fs::create_dir_all(&log_dir)?;

        let mut log_file_path = log_dir.clone();
        log_file_path.push("log-file");
        log_file_path.set_extension("txt");
        let log_file = File::create(&log_file_path)?;
        let file_log = tracing_subscriber::fmt::layer()
            .with_writer(log_file)
            .with_ansi(false);

        println!("log_file_path: {}", log_file_path.display());
        println!("cli_args: {:#?}", cli_args);

        let stdout_log = tracing_subscriber::fmt::layer().pretty();

        tracing_subscriber::registry()
            .with(
                stdout_log
                    .with_filter(tracing_subscriber::filter::LevelFilter::from_level(
                        cli_args.tracing_log_level.into(),
                    ))
                    .and_then(file_log),
            )
            .init();
    }

    let event_loop = winit::event_loop::EventLoop::builder().build()?;

    //SAFETY: Loads vulkan via libloading which is kinda unsafe but we're fine
    let instance = Arc::new(unsafe {
        rgpu::instance::Instance::new(
            "samp-app",
            cli_args.graphics_debug_level,
            Some(&event_loop),
            InstanceExtensions { surface: true },
        )
    }?);

    let mut app = AppRunner(Some(App::Initializing(InitializingState { instance })));

    tracing::trace!("Entering main event loop");
    Ok(event_loop.run_app(&mut app)?)
}

#[derive(Debug)]
struct AppRunner(Option<App>);

#[derive(Debug)]
enum App {
    Running(RunningState),
    Initializing(InitializingState),
    Suspended(SuspendedState),
    Exiting(ExitingState),
}

#[derive(Debug)]
struct InitializingState {
    instance: Arc<Instance>,
}
#[derive(Debug)]
struct RunningState {
    instance: Arc<Instance>,
    win: Arc<WinitWindow>,
    _surface: Surface<WinitWindow>,
}
#[derive(Debug)]
struct SuspendedState {
    instance: Arc<Instance>,
    win: Arc<WinitWindow>,
}
#[derive(Debug)]
struct ExitingState {}

impl ApplicationHandler for AppRunner {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        assert!(self.0.is_some());
        if let Some(initializing_state) = self.take_initializing() {
            event_loop.set_control_flow(ControlFlow::Poll);
            let win = Arc::new(
                match event_loop.create_window(WindowAttributes::default().with_inner_size(
                    LogicalSize {
                        width: 1600,
                        height: 900,
                    },
                )) {
                    Ok(w) => w,
                    Err(e) => {
                        tracing::error!("Error while creating window: {}", e);
                        tracing::debug!("State transition: Initializing -> Exiting");
                        self.set_exiting(ExitingState {});
                        event_loop.exit();
                        return;
                    }
                },
            );
            //SAFETY: We will drop surface when we enter into `suspend`
            let surface =
                match unsafe { Surface::new(&initializing_state.instance, Arc::clone(&win)) } {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::error!("Error while creating surface: {}", e);
                        tracing::debug!("State transition: Initializing -> Exiting");
                        self.set_exiting(ExitingState {});
                        event_loop.exit();
                        return;
                    }
                };
            tracing::debug!("State transition: Initializing -> Running");
            self.set_running(RunningState {
                instance: initializing_state.instance,
                win,
                _surface: surface,
            });
        } else if let Some(suspended_state) = self.take_suspended() {
            event_loop.set_control_flow(ControlFlow::Poll);
            //SAFETY: We will drop surface when we enter into `suspend`
            let surface = match unsafe {
                Surface::new(&suspended_state.instance, Arc::clone(&suspended_state.win))
            } {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!("Error while creating surface: {}", e);
                    tracing::debug!("State transition: Suspended -> Exiting");
                    self.set_exiting(ExitingState {});
                    event_loop.exit();
                    return;
                }
            };
            tracing::debug!("State transition: Suspended -> Running");
            self.set_running(RunningState {
                instance: suspended_state.instance,
                win: suspended_state.win,
                _surface: surface,
            });
        } else if self.is_exiting() {
            tracing::warn!("resumed() called while in Exiting state");
        }
    }
    fn suspended(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        assert!(self.0.is_some());
        if let Some(running_state) = self.take_running() {
            event_loop.set_control_flow(ControlFlow::Wait);
            tracing::debug!("State transition: Running -> Suspended");
            self.set_suspended(SuspendedState {
                instance: running_state.instance,
                win: running_state.win,
            });
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        window_event: winit::event::WindowEvent,
    ) {
        assert!(self.0.is_some());
        if let Some(running_state) = self.as_running() {
            match window_event {
                WindowEvent::CloseRequested if window_id == running_state.win.id() => {
                    tracing::trace!("Close window request received for window");
                    let _running_state = self.take_running().expect(
                        "Switching from a temporary borrow to a permanent \
                         take of the same type should always be good",
                    );
                    tracing::debug!("State transition: Running -> Exiting");
                    self.set_exiting(ExitingState {});
                    event_loop.exit();
                }
                _ => {}
            }
        }
    }
}

#[allow(dead_code, reason = "these functions exist for API completeness")]
impl AppRunner {
    fn is_initializing(&self) -> bool {
        assert!(self.0.is_some());
        matches!(self.0, Some(App::Initializing(_)))
    }

    fn take_initializing(&mut self) -> Option<InitializingState> {
        assert!(self.0.is_some());
        if matches!(self.0, Some(App::Initializing(_))) {
            match self.0.take() {
                Some(App::Initializing(s)) => Some(s),
                _ => unreachable!(),
            }
        } else {
            None
        }
    }

    fn as_initializing(&self) -> Option<&InitializingState> {
        assert!(self.0.is_some());
        match &self.0 {
            Some(App::Initializing(s)) => Some(s),
            _ => None,
        }
    }

    fn as_initializing_mut(&mut self) -> Option<&mut InitializingState> {
        assert!(self.0.is_some());
        match &mut self.0 {
            Some(App::Initializing(s)) => Some(s),
            _ => None,
        }
    }

    fn set_initializing(&mut self, state: InitializingState) {
        assert!(self.0.is_none());
        self.0 = Some(App::Initializing(state));
    }

    fn is_running(&self) -> bool {
        assert!(self.0.is_some());
        matches!(self.0, Some(App::Running(_)))
    }

    fn take_running(&mut self) -> Option<RunningState> {
        assert!(self.0.is_some());
        if matches!(self.0, Some(App::Running(_))) {
            match self.0.take() {
                Some(App::Running(s)) => Some(s),
                _ => unreachable!(),
            }
        } else {
            None
        }
    }

    fn as_running(&self) -> Option<&RunningState> {
        assert!(self.0.is_some());
        match &self.0 {
            Some(App::Running(s)) => Some(s),
            _ => None,
        }
    }

    fn as_running_mut(&mut self) -> Option<&mut RunningState> {
        assert!(self.0.is_some());
        match &mut self.0 {
            Some(App::Running(s)) => Some(s),
            _ => None,
        }
    }

    fn set_running(&mut self, state: RunningState) {
        assert!(self.0.is_none());
        self.0 = Some(App::Running(state));
    }

    fn is_suspended(&self) -> bool {
        assert!(self.0.is_some());
        matches!(self.0, Some(App::Suspended(_)))
    }

    fn take_suspended(&mut self) -> Option<SuspendedState> {
        assert!(self.0.is_some());
        if matches!(self.0, Some(App::Suspended(_))) {
            match self.0.take() {
                Some(App::Suspended(s)) => Some(s),
                _ => unreachable!(),
            }
        } else {
            None
        }
    }

    fn as_suspended(&self) -> Option<&SuspendedState> {
        assert!(self.0.is_some());
        match &self.0 {
            Some(App::Suspended(s)) => Some(s),
            _ => None,
        }
    }

    fn as_suspended_mut(&mut self) -> Option<&mut SuspendedState> {
        assert!(self.0.is_some());
        match &mut self.0 {
            Some(App::Suspended(s)) => Some(s),
            _ => None,
        }
    }

    fn set_suspended(&mut self, state: SuspendedState) {
        assert!(self.0.is_none());
        self.0 = Some(App::Suspended(state));
    }

    fn is_exiting(&self) -> bool {
        assert!(self.0.is_some());
        matches!(self.0, Some(App::Exiting(_)))
    }

    fn take_exiting(&mut self) -> Option<ExitingState> {
        assert!(self.0.is_some());
        if matches!(self.0, Some(App::Exiting(_))) {
            match self.0.take() {
                Some(App::Exiting(s)) => Some(s),
                _ => unreachable!(),
            }
        } else {
            None
        }
    }

    fn as_exiting(&self) -> Option<&ExitingState> {
        assert!(self.0.is_some());
        match &self.0 {
            Some(App::Exiting(s)) => Some(s),
            _ => None,
        }
    }

    fn as_exiting_mut(&mut self) -> Option<&mut ExitingState> {
        assert!(self.0.is_some());
        match &mut self.0 {
            Some(App::Exiting(s)) => Some(s),
            _ => None,
        }
    }

    fn set_exiting(&mut self, state: ExitingState) {
        assert!(self.0.is_none());
        self.0 = Some(App::Exiting(state));
    }
}
