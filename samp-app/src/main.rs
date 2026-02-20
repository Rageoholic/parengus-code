#![deny(unsafe_op_in_unsafe_fn)]
#![warn(clippy::undocumented_unsafe_blocks)]

use std::{
    fs::{self, File},
    sync::Arc,
};

use clap::Parser;
use rgpu::{
    device::{Device, DeviceConfig, QueueMode},
    instance::{Instance, InstanceExtensions},
    surface::Surface,
    swapchain::Swapchain,
};
use tracing_subscriber::{Layer, layer::SubscriberExt, util::SubscriberInitExt};
use winit::{
    application::ApplicationHandler,
    dpi::LogicalSize,
    event::WindowEvent,
    event_loop::ControlFlow,
    window::{Window as WinitWindow, WindowAttributes},
};

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Default, clap::ValueEnum)]
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
    #[arg(short, long, default_value = "error")]
    tracing_log_level: TracingLogLevel,
    #[arg(short, long)]
    graphics_debug_level: Option<CliVulkanLogLevel>,
    #[arg(long, default_value = "auto")]
    queue_mode: CliQueueMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, clap::ValueEnum)]
enum CliQueueMode {
    #[default]
    Auto,
    Unified,
    Single,
}

impl From<CliQueueMode> for QueueMode {
    fn from(value: CliQueueMode) -> Self {
        match value {
            CliQueueMode::Auto => QueueMode::Auto,
            CliQueueMode::Unified => QueueMode::Unified,
            CliQueueMode::Single => QueueMode::Single,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum CliVulkanLogLevel {
    Verbose,
    Info,
    Warning,
    Error,
}

impl From<CliVulkanLogLevel> for rgpu::log::VulkanLogLevel {
    fn from(value: CliVulkanLogLevel) -> Self {
        match value {
            CliVulkanLogLevel::Verbose => rgpu::log::VulkanLogLevel::Verbose,
            CliVulkanLogLevel::Info => rgpu::log::VulkanLogLevel::Info,
            CliVulkanLogLevel::Warning => rgpu::log::VulkanLogLevel::Warning,
            CliVulkanLogLevel::Error => rgpu::log::VulkanLogLevel::Error,
        }
    }
}

fn main() -> eyre::Result<()> {
    let app_dirs = directories::ProjectDirs::from("", "parengus", "samp-app");

    let log_dir = match app_dirs
        .as_ref()
        .and_then(|x| x.runtime_dir().or_else(|| Some(x.data_dir())))
        .map(|p| p.to_owned())
    {
        Some(path) => path,
        None => std::env::current_dir()?,
    };

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
            cli_args.graphics_debug_level.map(Into::into),
            Some(&event_loop),
            InstanceExtensions { surface: true },
        )
    }?);

    let device_config = DeviceConfig {
        swapchain: true,
        queue_mode: cli_args.queue_mode.into(),
    };

    let mut app = AppRunner(Some(App::Initializing(InitializingState {
        instance,
        device_config,
    })));

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
    device_config: DeviceConfig,
}
#[derive(Debug)]
struct RunningState {
    instance: Arc<Instance>,
    win: Arc<WinitWindow>,
    device: Arc<Device>,
    _surface: Arc<Surface<WinitWindow>>,
    // `None` means the window/surface is currently zero-sized. We stay in
    // Running and recreate on the next non-zero resize/scale event.
    swapchain: Option<Swapchain<WinitWindow>>,
}
#[derive(Debug)]
struct SuspendedState {
    instance: Arc<Instance>,
    win: Arc<WinitWindow>,
    device: Arc<Device>,
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
                        self.transition_to_exiting("Initializing", event_loop);
                        return;
                    }
                },
            );
            //SAFETY: We will drop surface when we enter into `suspend`
            let surface = Arc::new(
                match unsafe { Surface::new(&initializing_state.instance, Arc::clone(&win)) } {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::error!("Error while creating surface: {}", e);
                        self.transition_to_exiting("Initializing", event_loop);
                        return;
                    }
                },
            );

            let device = match Device::create_compatible(
                &initializing_state.instance,
                &surface,
                initializing_state.device_config,
            ) {
                Ok(d) => Arc::new(d),
                Err(e) => {
                    tracing::error!("Error while creating device: {}", e);
                    self.transition_to_exiting("Initializing", event_loop);
                    return;
                }
            };
            let win_size = win.inner_size();
            let swapchain = if win_size.width == 0 || win_size.height == 0 {
                tracing::trace!(
                    "Skipping initial swapchain create because window extent is zero: {}x{}",
                    win_size.width,
                    win_size.height
                );
                None
            } else {
                let requested_extent = rgpu::ash::vk::Extent2D {
                    width: win_size.width,
                    height: win_size.height,
                };
                let swapchain_create_span = tracing::trace_span!(
                    "initial_swapchain_create",
                    requested_width = requested_extent.width,
                    requested_height = requested_extent.height
                )
                .entered();
                let swapchain = match Swapchain::new(&device, &surface, requested_extent) {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::error!("Error while creating swapchain: {}", e);
                        self.transition_to_exiting("Initializing", event_loop);
                        return;
                    }
                };
                drop(swapchain_create_span);
                Some(swapchain)
            };
            tracing::debug!("State transition: Initializing -> Running");
            self.set_running(RunningState {
                instance: initializing_state.instance,
                win,
                device,
                _surface: surface,
                swapchain,
            });
        } else if let Some(suspended_state) = self.take_suspended() {
            event_loop.set_control_flow(ControlFlow::Poll);
            //SAFETY: We will drop surface when we enter into `suspend`
            let surface = Arc::new(
                match unsafe {
                    Surface::new(&suspended_state.instance, Arc::clone(&suspended_state.win))
                } {
                    Ok(s) => s,
                    Err(e) => {
                        tracing::error!("Error while creating surface: {}", e);
                        self.transition_to_exiting("Suspended", event_loop);
                        return;
                    }
                },
            );

            let win_size = suspended_state.win.inner_size();
            let swapchain = if win_size.width == 0 || win_size.height == 0 {
                tracing::trace!(
                    "Skipping swapchain create on resume because window extent is zero: {}x{}",
                    win_size.width,
                    win_size.height
                );
                None
            } else {
                match Swapchain::new(
                    &suspended_state.device,
                    &surface,
                    rgpu::ash::vk::Extent2D {
                        width: win_size.width,
                        height: win_size.height,
                    },
                ) {
                    Ok(s) => Some(s),
                    Err(e) => {
                        tracing::error!("Error while creating swapchain: {}", e);
                        self.transition_to_exiting("Suspended", event_loop);
                        return;
                    }
                }
            };
            tracing::debug!("State transition: Suspended -> Running");
            self.set_running(RunningState {
                instance: suspended_state.instance,
                win: suspended_state.win,
                device: suspended_state.device,
                _surface: surface,
                swapchain,
            });
        } else if self.is_exiting() {
            tracing::warn!("resumed() called while in Exiting state");
        }
    }
    fn suspended(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        assert!(self.0.is_some());
        if let Some(running_state) = self.take_running() {
            event_loop.set_control_flow(ControlFlow::Wait);
            let RunningState {
                instance,
                win,
                device,
                _surface: _,
                swapchain: _,
            } = running_state;

            if let Err(e) = device.wait_idle() {
                tracing::error!("Error while waiting for device idle during suspend: {}", e);
                self.transition_to_exiting("Running", event_loop);
                return;
            }

            tracing::debug!("State transition: Running -> Suspended");
            self.set_suspended(SuspendedState {
                instance,
                win,
                device,
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
        if !self.is_running_window(window_id) {
            return;
        }

        if matches!(&window_event, WindowEvent::CloseRequested) {
            tracing::trace!("Close window request received for window");
            self.exit_from_running(event_loop);
            return;
        }

        match &window_event {
            WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. } => {
                let desired_extent = {
                    if let Some(running_state) = self.as_running()
                        && let Some(extent) =
                            Self::desired_extent_for_event(running_state, &window_event)
                    {
                        extent
                    } else {
                        return;
                    }
                };

                let should_keep_running = {
                    let running_state = match self.as_running_mut() {
                        Some(running_state) => running_state,
                        None => return,
                    };

                    Self::recreate_swapchain_if_needed(running_state, desired_extent)
                };

                if !should_keep_running {
                    self.exit_from_running(event_loop);
                }
            }
            _ => {}
        }
    }
}

#[allow(dead_code, reason = "these functions exist for API completeness")]
impl AppRunner {
    fn transition_to_exiting(
        &mut self,
        from_state: &'static str,
        event_loop: &winit::event_loop::ActiveEventLoop,
    ) {
        tracing::debug!("State transition: {} -> Exiting", from_state);
        self.set_exiting(ExitingState {});
        event_loop.exit();
    }

    fn exit_from_running(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.take_running().is_some() {
            self.transition_to_exiting("Running", event_loop);
        } else {
            tracing::warn!("Requested Running -> Exiting transition while not in Running state");
            event_loop.exit();
        }
    }

    fn desired_extent_for_event(
        running_state: &RunningState,
        window_event: &WindowEvent,
    ) -> Option<rgpu::ash::vk::Extent2D> {
        match window_event {
            WindowEvent::Resized(size) => Some(rgpu::ash::vk::Extent2D {
                width: size.width,
                height: size.height,
            }),
            WindowEvent::ScaleFactorChanged { .. } => {
                let size = running_state.win.inner_size();
                Some(rgpu::ash::vk::Extent2D {
                    width: size.width,
                    height: size.height,
                })
            }
            _ => None,
        }
    }

    fn is_running_window(&self, window_id: winit::window::WindowId) -> bool {
        if let Some(running_state) = self.as_running()
            && window_id == running_state.win.id()
        {
            true
        } else {
            false
        }
    }

    fn recreate_swapchain_if_needed(
        running_state: &mut RunningState,
        desired_extent: rgpu::ash::vk::Extent2D,
    ) -> bool {
        if desired_extent.width == 0 || desired_extent.height == 0 {
            tracing::trace!(
                "Received zero extent ({}x{}); tearing down swapchain and waiting for idle",
                desired_extent.width,
                desired_extent.height
            );
            if let Err(e) = running_state.device.wait_idle() {
                tracing::error!("Error while waiting for device idle on zero extent: {}", e);
                return false;
            }
            running_state.swapchain = None;
            return true;
        }

        if let Some(existing_swapchain) = running_state.swapchain.as_ref()
            && existing_swapchain.extent() == desired_extent
        {
            tracing::trace!(
                "Skipping swapchain recreate because extent is unchanged: {}x{}",
                desired_extent.width,
                desired_extent.height
            );
            return true;
        }

        tracing::trace!(
            "Recreating swapchain for new extent: {}x{}",
            desired_extent.width,
            desired_extent.height
        );

        match Swapchain::new_with_old(
            &running_state.device,
            &running_state._surface,
            desired_extent,
            running_state.swapchain.as_ref(),
        ) {
            Ok(swapchain) => {
                tracing::trace!(
                    "Swapchain recreation succeeded for extent: {}x{}",
                    desired_extent.width,
                    desired_extent.height
                );
                running_state.swapchain = Some(swapchain);
                true
            }
            Err(e) => {
                tracing::error!("Error while recreating swapchain: {}", e);
                false
            }
        }
    }

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
