//! Streaming framework for job events.
//!
//! This module provides a framework for streaming events from running jobs:
//! - Stream types and messages
//! - Configurable filtering and deduplication
//! - Endpoint configuration for different job types
//! - Handlers for processing stream messages
//! - Job streamer for polling and dispatching

pub mod config;
pub mod endpoints;
pub mod handler;
pub mod streamer;
pub mod types;

pub use config::StreamConfig;
pub use endpoints::StreamEndpoints;
pub use handler::{
    BufferedHandler, CallbackHandler, FilteredHandler, JsonHandler, MultiHandler, StreamHandler,
};
pub use streamer::JobStreamer;
pub use types::{StreamMessage, StreamType};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_module_exports() {
        // Verify all types are accessible
        let _ = StreamType::Events;
        let _ = StreamConfig::default();
        let _ = StreamEndpoints::learning("test");
        let _ = BufferedHandler::new();
    }
}
