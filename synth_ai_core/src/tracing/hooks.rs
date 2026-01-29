//! Hook system for tracing events.
//!
//! Allows registering callbacks that are triggered at various points
//! during trace recording.

use std::collections::HashMap;
use std::sync::Arc;

use super::models::{MarkovBlanketMessage, TracingEvent};

/// Events that can trigger hooks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HookEvent {
    /// Session started
    SessionStart,
    /// Session ended
    SessionEnd,
    /// Timestep started
    TimestepStart,
    /// Timestep ended
    TimestepEnd,
    /// Event recorded
    EventRecorded,
    /// Message recorded
    MessageRecorded,
    /// Before saving to storage
    BeforeSave,
    /// After saving to storage
    AfterSave,
}

impl std::fmt::Display for HookEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            HookEvent::SessionStart => write!(f, "session_start"),
            HookEvent::SessionEnd => write!(f, "session_end"),
            HookEvent::TimestepStart => write!(f, "timestep_start"),
            HookEvent::TimestepEnd => write!(f, "timestep_end"),
            HookEvent::EventRecorded => write!(f, "event_recorded"),
            HookEvent::MessageRecorded => write!(f, "message_recorded"),
            HookEvent::BeforeSave => write!(f, "before_save"),
            HookEvent::AfterSave => write!(f, "after_save"),
        }
    }
}

/// Context passed to hook callbacks.
#[derive(Debug, Clone, Default)]
pub struct HookContext {
    /// Current session ID (if any)
    pub session_id: Option<String>,
    /// Current step ID (if any)
    pub step_id: Option<String>,
    /// Event that triggered the hook (for EventRecorded)
    pub event: Option<TracingEvent>,
    /// Message that triggered the hook (for MessageRecorded)
    pub message: Option<MarkovBlanketMessage>,
}

impl HookContext {
    /// Create a new empty context.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the session ID.
    pub fn with_session(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Set the step ID.
    pub fn with_step(mut self, step_id: impl Into<String>) -> Self {
        self.step_id = Some(step_id.into());
        self
    }

    /// Set the event.
    pub fn with_event(mut self, event: TracingEvent) -> Self {
        self.event = Some(event);
        self
    }

    /// Set the message.
    pub fn with_message(mut self, message: MarkovBlanketMessage) -> Self {
        self.message = Some(message);
        self
    }
}

/// Type alias for hook callbacks.
///
/// Callbacks receive a reference to the hook context and should not block.
pub type HookCallback = Arc<dyn Fn(&HookContext) + Send + Sync>;

/// A registered hook with its callback and priority.
struct HookRegistration {
    callback: HookCallback,
    priority: i32,
}

/// Manages hook registrations and triggering.
pub struct HookManager {
    hooks: HashMap<HookEvent, Vec<HookRegistration>>,
}

impl Default for HookManager {
    fn default() -> Self {
        Self::new()
    }
}

impl HookManager {
    /// Create a new hook manager.
    pub fn new() -> Self {
        Self {
            hooks: HashMap::new(),
        }
    }

    /// Register a hook callback for an event.
    ///
    /// Higher priority callbacks are executed first.
    pub fn register(&mut self, event: HookEvent, callback: HookCallback, priority: i32) {
        let registrations = self.hooks.entry(event).or_insert_with(Vec::new);

        registrations.push(HookRegistration { callback, priority });

        // Sort by priority descending (higher priority first)
        registrations.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    /// Register a hook with default priority (0).
    pub fn on(&mut self, event: HookEvent, callback: HookCallback) {
        self.register(event, callback, 0);
    }

    /// Trigger all hooks for an event.
    ///
    /// Hooks are called in priority order. Failures in hooks do not
    /// propagate - they are logged but don't stop other hooks.
    pub fn trigger(&self, event: HookEvent, context: &HookContext) {
        if let Some(registrations) = self.hooks.get(&event) {
            for reg in registrations {
                // Call the hook - we don't propagate panics
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    (reg.callback)(context);
                }))
                .ok();
            }
        }
    }

    /// Check if any hooks are registered for an event.
    pub fn has_hooks(&self, event: HookEvent) -> bool {
        self.hooks
            .get(&event)
            .map(|v| !v.is_empty())
            .unwrap_or(false)
    }

    /// Get the number of hooks registered for an event.
    pub fn hook_count(&self, event: HookEvent) -> usize {
        self.hooks.get(&event).map(|v| v.len()).unwrap_or(0)
    }

    /// Clear all hooks for an event.
    pub fn clear(&mut self, event: HookEvent) {
        self.hooks.remove(&event);
    }

    /// Clear all hooks.
    pub fn clear_all(&mut self) {
        self.hooks.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicI32, Ordering};

    #[test]
    fn test_hook_registration() {
        let mut manager = HookManager::new();

        assert!(!manager.has_hooks(HookEvent::SessionStart));

        manager.on(HookEvent::SessionStart, Arc::new(|_| {}));

        assert!(manager.has_hooks(HookEvent::SessionStart));
        assert_eq!(manager.hook_count(HookEvent::SessionStart), 1);
    }

    #[test]
    fn test_hook_trigger() {
        let mut manager = HookManager::new();
        let counter = Arc::new(AtomicI32::new(0));
        let counter_clone = counter.clone();

        manager.on(
            HookEvent::SessionStart,
            Arc::new(move |_| {
                counter_clone.fetch_add(1, Ordering::SeqCst);
            }),
        );

        let context = HookContext::new().with_session("test-session");
        manager.trigger(HookEvent::SessionStart, &context);

        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_hook_priority() {
        let mut manager = HookManager::new();
        let order = Arc::new(std::sync::Mutex::new(Vec::new()));

        let order1 = order.clone();
        manager.register(
            HookEvent::SessionStart,
            Arc::new(move |_| {
                order1.lock().unwrap().push(1);
            }),
            10,
        );

        let order2 = order.clone();
        manager.register(
            HookEvent::SessionStart,
            Arc::new(move |_| {
                order2.lock().unwrap().push(2);
            }),
            20, // Higher priority, should run first
        );

        let order3 = order.clone();
        manager.register(
            HookEvent::SessionStart,
            Arc::new(move |_| {
                order3.lock().unwrap().push(3);
            }),
            5,
        );

        manager.trigger(HookEvent::SessionStart, &HookContext::new());

        let result = order.lock().unwrap();
        assert_eq!(*result, vec![2, 1, 3]); // 20, 10, 5 priority order
    }

    #[test]
    fn test_hook_context() {
        let context = HookContext::new()
            .with_session("session-1")
            .with_step("step-1");

        assert_eq!(context.session_id, Some("session-1".to_string()));
        assert_eq!(context.step_id, Some("step-1".to_string()));
    }

    #[test]
    fn test_hook_clear() {
        let mut manager = HookManager::new();

        manager.on(HookEvent::SessionStart, Arc::new(|_| {}));
        manager.on(HookEvent::SessionEnd, Arc::new(|_| {}));

        assert!(manager.has_hooks(HookEvent::SessionStart));

        manager.clear(HookEvent::SessionStart);
        assert!(!manager.has_hooks(HookEvent::SessionStart));
        assert!(manager.has_hooks(HookEvent::SessionEnd));

        manager.clear_all();
        assert!(!manager.has_hooks(HookEvent::SessionEnd));
    }
}
