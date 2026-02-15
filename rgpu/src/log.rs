use strum_macros::EnumString;

#[derive(Debug, EnumString, PartialEq, Eq, PartialOrd, Ord, Clone, Copy)]
pub enum VulkanLogLevel {
    Verbose,
    Info,
    Warning,
    Error,
}
