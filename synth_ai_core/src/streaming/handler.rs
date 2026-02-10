//! Stream handler trait and built-in handlers.
//!
//! Handlers process stream messages and can filter, transform, or output them.

use super::types::StreamMessage;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;

/// Trait for handling stream messages.
pub trait StreamHandler: Send + Sync {
    /// Process a stream message.
    fn handle(&self, message: &StreamMessage);

    /// Filter predicate - return false to skip handling this message.
    fn should_handle(&self, _message: &StreamMessage) -> bool {
        true
    }

    /// Flush any buffered output.
    fn flush(&self) {}

    /// Called when streaming starts.
    fn on_start(&self, _job_id: &str) {}

    /// Called when streaming ends.
    fn on_end(&self, _job_id: &str, _final_status: Option<&str>) {}
}

/// A handler that calls a callback function.
pub struct CallbackHandler<F>
where
    F: Fn(&StreamMessage) + Send + Sync,
{
    callback: F,
}

impl<F> CallbackHandler<F>
where
    F: Fn(&StreamMessage) + Send + Sync,
{
    /// Create a new callback handler.
    pub fn new(callback: F) -> Self {
        Self { callback }
    }
}

impl<F> StreamHandler for CallbackHandler<F>
where
    F: Fn(&StreamMessage) + Send + Sync,
{
    fn handle(&self, message: &StreamMessage) {
        (self.callback)(message);
    }
}

/// A handler that outputs JSON lines.
pub struct JsonHandler {
    output_path: Option<PathBuf>,
    file: Mutex<Option<std::fs::File>>,
    pretty: bool,
}

impl JsonHandler {
    /// Create a handler that writes to stdout.
    pub fn stdout() -> Self {
        Self {
            output_path: None,
            file: Mutex::new(None),
            pretty: false,
        }
    }

    /// Create a handler that writes to a file.
    pub fn file(path: impl Into<PathBuf>) -> Self {
        Self {
            output_path: Some(path.into()),
            file: Mutex::new(None),
            pretty: false,
        }
    }

    /// Enable pretty-printing.
    pub fn pretty(mut self) -> Self {
        self.pretty = true;
        self
    }

    fn ensure_file(&self) -> Option<std::io::Result<()>> {
        if let Some(ref path) = self.output_path {
            let mut guard = self.file.lock().unwrap();
            if guard.is_none() {
                match std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(path)
                {
                    Ok(f) => *guard = Some(f),
                    Err(e) => return Some(Err(e)),
                }
            }
        }
        None
    }
}

impl StreamHandler for JsonHandler {
    fn handle(&self, message: &StreamMessage) {
        let json = if self.pretty {
            serde_json::to_string_pretty(message).unwrap_or_default()
        } else {
            serde_json::to_string(message).unwrap_or_default()
        };

        if let Some(ref _path) = self.output_path {
            self.ensure_file();
            let mut guard = self.file.lock().unwrap();
            if let Some(ref mut file) = *guard {
                let _ = writeln!(file, "{}", json);
            }
        } else {
            println!("{}", json);
        }
    }

    fn flush(&self) {
        if let Some(ref _path) = self.output_path {
            let mut guard = self.file.lock().unwrap();
            if let Some(ref mut file) = *guard {
                let _ = file.flush();
            }
        }
    }
}

/// A handler that buffers messages in memory.
pub struct BufferedHandler {
    messages: Mutex<Vec<StreamMessage>>,
    max_size: Option<usize>,
}

impl BufferedHandler {
    /// Create a new buffered handler.
    pub fn new() -> Self {
        Self {
            messages: Mutex::new(Vec::new()),
            max_size: None,
        }
    }

    /// Create a handler with a maximum buffer size.
    pub fn with_max_size(max_size: usize) -> Self {
        Self {
            messages: Mutex::new(Vec::with_capacity(max_size.min(1000))),
            max_size: Some(max_size),
        }
    }

    /// Get all buffered messages.
    pub fn messages(&self) -> Vec<StreamMessage> {
        self.messages.lock().unwrap().clone()
    }

    /// Clear the buffer.
    pub fn clear(&self) {
        self.messages.lock().unwrap().clear();
    }

    /// Get the number of buffered messages.
    pub fn len(&self) -> usize {
        self.messages.lock().unwrap().len()
    }

    /// Check if the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for BufferedHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamHandler for BufferedHandler {
    fn handle(&self, message: &StreamMessage) {
        let mut messages = self.messages.lock().unwrap();

        // Drop oldest if at max size
        if let Some(max) = self.max_size {
            if messages.len() >= max {
                messages.remove(0);
            }
        }

        messages.push(message.clone());
    }
}

/// A handler that filters messages before passing to another handler.
pub struct FilteredHandler<H: StreamHandler, F: Fn(&StreamMessage) -> bool + Send + Sync> {
    inner: H,
    filter: F,
}

impl<H: StreamHandler, F: Fn(&StreamMessage) -> bool + Send + Sync> FilteredHandler<H, F> {
    /// Create a new filtered handler.
    pub fn new(inner: H, filter: F) -> Self {
        Self { inner, filter }
    }
}

impl<H: StreamHandler, F: Fn(&StreamMessage) -> bool + Send + Sync> StreamHandler
    for FilteredHandler<H, F>
{
    fn handle(&self, message: &StreamMessage) {
        if (self.filter)(message) {
            self.inner.handle(message);
        }
    }

    fn should_handle(&self, message: &StreamMessage) -> bool {
        (self.filter)(message) && self.inner.should_handle(message)
    }

    fn flush(&self) {
        self.inner.flush();
    }

    fn on_start(&self, job_id: &str) {
        self.inner.on_start(job_id);
    }

    fn on_end(&self, job_id: &str, final_status: Option<&str>) {
        self.inner.on_end(job_id, final_status);
    }
}

/// A handler that dispatches to multiple handlers.
pub struct MultiHandler {
    handlers: Vec<Box<dyn StreamHandler>>,
}

impl MultiHandler {
    /// Create a new multi-handler.
    pub fn new() -> Self {
        Self {
            handlers: Vec::new(),
        }
    }

    /// Add a handler.
    pub fn add<H: StreamHandler + 'static>(mut self, handler: H) -> Self {
        self.handlers.push(Box::new(handler));
        self
    }

    /// Add a boxed handler.
    pub fn add_boxed(mut self, handler: Box<dyn StreamHandler>) -> Self {
        self.handlers.push(handler);
        self
    }
}

impl Default for MultiHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl StreamHandler for MultiHandler {
    fn handle(&self, message: &StreamMessage) {
        for handler in &self.handlers {
            if handler.should_handle(message) {
                handler.handle(message);
            }
        }
    }

    fn flush(&self) {
        for handler in &self.handlers {
            handler.flush();
        }
    }

    fn on_start(&self, job_id: &str) {
        for handler in &self.handlers {
            handler.on_start(job_id);
        }
    }

    fn on_end(&self, job_id: &str, final_status: Option<&str>) {
        for handler in &self.handlers {
            handler.on_end(job_id, final_status);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::streaming::types::StreamType;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    #[test]
    fn test_callback_handler() {
        let count = Arc::new(AtomicUsize::new(0));
        let count_clone = count.clone();

        let handler = CallbackHandler::new(move |_| {
            count_clone.fetch_add(1, Ordering::SeqCst);
        });

        let msg = StreamMessage::new(StreamType::Events, "job-1", serde_json::json!({}));
        handler.handle(&msg);
        handler.handle(&msg);

        assert_eq!(count.load(Ordering::SeqCst), 2);
    }

    #[test]
    fn test_buffered_handler() {
        let handler = BufferedHandler::new();

        let msg1 = StreamMessage::new(StreamType::Events, "job-1", serde_json::json!({"seq": 1}));
        let msg2 = StreamMessage::new(StreamType::Events, "job-1", serde_json::json!({"seq": 2}));

        handler.handle(&msg1);
        handler.handle(&msg2);

        assert_eq!(handler.len(), 2);
        assert_eq!(handler.messages().len(), 2);

        handler.clear();
        assert!(handler.is_empty());
    }

    #[test]
    fn test_buffered_handler_max_size() {
        let handler = BufferedHandler::with_max_size(2);

        for i in 0..5 {
            let msg =
                StreamMessage::new(StreamType::Events, "job-1", serde_json::json!({"seq": i}));
            handler.handle(&msg);
        }

        // Should only have the last 2 messages
        assert_eq!(handler.len(), 2);
        let messages = handler.messages();
        assert_eq!(messages[0].get_i64("seq"), Some(3));
        assert_eq!(messages[1].get_i64("seq"), Some(4));
    }

    #[test]
    fn test_filtered_handler() {
        let buffer = Arc::new(BufferedHandler::new());
        let buffer_ref = Arc::clone(&buffer);

        // Create a simple handler wrapper that uses the Arc
        struct ArcBufferHandler(Arc<BufferedHandler>);
        impl StreamHandler for ArcBufferHandler {
            fn handle(&self, message: &StreamMessage) {
                self.0.handle(message);
            }
        }

        let filtered = FilteredHandler::new(ArcBufferHandler(buffer_ref), |msg| {
            msg.get_i64("value").unwrap_or(0) > 5
        });

        filtered.handle(&StreamMessage::new(
            StreamType::Events,
            "job",
            serde_json::json!({"value": 3}),
        ));
        filtered.handle(&StreamMessage::new(
            StreamType::Events,
            "job",
            serde_json::json!({"value": 10}),
        ));

        assert_eq!(buffer.len(), 1);
    }

    #[test]
    fn test_multi_handler() {
        let buffer1 = Arc::new(BufferedHandler::new());
        let buffer2 = Arc::new(BufferedHandler::new());

        struct ArcBufferHandler(Arc<BufferedHandler>);
        impl StreamHandler for ArcBufferHandler {
            fn handle(&self, message: &StreamMessage) {
                self.0.handle(message);
            }
        }

        let multi = MultiHandler::new()
            .add(ArcBufferHandler(Arc::clone(&buffer1)))
            .add(ArcBufferHandler(Arc::clone(&buffer2)));

        let msg = StreamMessage::new(StreamType::Events, "job", serde_json::json!({}));
        multi.handle(&msg);

        assert_eq!(buffer1.len(), 1);
        assert_eq!(buffer2.len(), 1);
    }
}
