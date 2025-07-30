---
title: 'Session Tracer'
description: 'Session-based tracing system'
---

# Session Tracer

The session tracer provides comprehensive tracing capabilities for AI applications.

## Classes

### SessionTracer

Main tracer class for managing tracing sessions.

**Signature:** `SessionTracer(hooks: List = None)`

**Methods:**
- `start_session(session_id: str)` - Start a new tracing session
- `end_session(save: bool = True)` - End current session
- `record_message(message: SessionEventMessage)` - Record a message event
- `record_event(event: RuntimeEvent)` - Record a runtime event

### SessionEventMessage

Represents a message event in a tracing session.

**Signature:** `SessionEventMessage(content: str, message_type: str, time_record: TimeRecord)`

**Attributes:**
- `content` - Message content
- `message_type` - Type of message
- `time_record` - Timing information

### RuntimeEvent

Represents a runtime event in a tracing session.

**Signature:** `RuntimeEvent(system_instance_id: str, time_record: TimeRecord, actions: List, metadata: dict)`

**Attributes:**
- `system_instance_id` - ID of the system instance
- `time_record` - Timing information
- `actions` - List of actions performed
- `metadata` - Additional metadata

