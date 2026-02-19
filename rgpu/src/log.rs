#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub enum VulkanLogLevel {
    Verbose,
    Info,
    Warning,
    Error,
}
