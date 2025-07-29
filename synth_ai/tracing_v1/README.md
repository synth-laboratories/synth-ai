# Synth SDK Primitives

This document explains the core abstractions used in the Synth SDK tracing system.

## Basic Input/Output Types

### Message I/O
- `MessageInputs`: Represents chat messages sent to a model, structured as a list of role-content pairs (e.g., user/assistant messages)
- `MessageOutputs`: Represents responses received from a model, following the same structure

### Arbitrary I/O
- `ArbitraryInputs`: Holds any key-value pairs for general inputs that aren't messages
- `ArbitraryOutputs`: Holds any key-value pairs for general outputs that aren't messages

## Compute Steps

### ComputeStep
The base unit of computation tracking, containing:
- Event ordering information
- Timing data (when computation started/ended)
- Input/output data
- Serialization logic for data persistence

### AgentComputeStep
Extends ComputeStep for AI model interactions:
- Tracks model-specific information (name, parameters)
- Handles both message-based and arbitrary I/O
- Includes learning flags for training purposes
- Primarily used for LLM/model computations

### EnvironmentComputeStep
Extends ComputeStep for external system interactions:
- Focused on arbitrary I/O (non-message based)
- Used for tracking interactions with databases, APIs, or other external services
- Helps understand system context and side effects

## Events and Organization

### Event
Represents a complete interaction unit:
- Contains one agent compute step (primary computation)
- Can have multiple environment compute steps (side effects/context)
- Tracks timing (opened/closed)
- Includes system identification and metadata
- Events can occur simultaneously within a partition

### EventPartitionElement
Organizes simultaneous events:
- Groups events that can happen in parallel
- Maintains ordering through partition index
- Ensures proper sequencing of dependent operations

### SystemTrace
Top-level organization structure:
- Contains ordered partitions of events
- Maintains system-level identification
- Holds metadata about the trace
- Tracks current partition for ongoing operations

## Metadata and Context

### System-Level Metadata
- Stored in SystemTrace
- Contains information about the system configuration
- Useful for understanding the broader context of traces

### Event Metadata
- Specific to individual events
- Can contain event-specific annotations or context
- Useful for debugging or additional analysis

### Instance Metadata
- Related to specific runs or instances
- Helpful for tracking deployment-specific information
- Can contain runtime configuration details

## Training and Evaluation

### TrainingQuestion
Represents a task or query:
- Contains the intended goal
- Includes success criteria
- Used for evaluating model performance

### RewardSignal
Tracks performance metrics:
- Links questions to system instances
- Contains numerical or boolean rewards
- Can include annotations for context

### Dataset
Organizes training data:
- Groups related questions
- Collects corresponding reward signals
- Facilitates model evaluation and training

## Common Usage Patterns

1. **Model Interaction Tracking**:
   ```
   Event
   └── AgentComputeStep
       ├── MessageInputs (prompt)
       └── MessageOutputs (completion)
   ```

2. **Complex Interaction Flow**:
   ```
   Event
   ├── AgentComputeStep (model computation)
   └── EnvironmentComputeSteps
       ├── Database query
       └── API call
   ```

3. **Parallel Operations**:
   ```
   EventPartitionElement
   └── Events (simultaneous)
       ├── Event A
       └── Event B
   ```

This structure allows for detailed tracking of both simple and complex AI system behaviors, while maintaining clear relationships between different components and operations. 