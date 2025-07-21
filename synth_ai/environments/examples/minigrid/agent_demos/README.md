# MiniGrid Agent Demos

This directory contains evaluation scripts, agents, and tools for testing AI agents on MiniGrid environments - a collection of grid-based navigation and puzzle-solving tasks.

## 🎯 Quick Start

**For quick evaluation with minimal setup:**
```bash
python minigrid_quick_evaluation.py
```

**For comprehensive evaluation with detailed metrics:**
```bash
python minigrid_evaluation_framework.py
```

**For trace generation compatible with the viewer:**
```bash
python minigrid_trace_evaluation.py
```

## 📁 Files Overview

### Core Agent Implementation
- **`minigrid_react_agent.py`** (20KB, 541 lines) - **Main ReAct agent implementation**
  - Complete ReAct (Reasoning + Acting) agent for MiniGrid environments
  - Handles grid-based navigation, object interaction, and goal-seeking
  - Includes comprehensive observation formatting and action selection
  - **Use for:** Understanding agent behavior, debugging navigation logic, extending agent capabilities

### Evaluation Systems

#### Comprehensive Evaluation Framework
- **`minigrid_evaluation_framework.py`** (45KB, 1055 lines) - **Advanced evaluation system**
  - Detailed metrics: success rates, navigation efficiency, exploration coverage
  - Achievement tracking system with 18 different achievements
  - Comprehensive scoring: composite scores, navigation scores, efficiency ratios
  - Multi-task evaluation across different MiniGrid environments
  - **Use for:** Research, detailed performance analysis, agent comparison

#### Quick Evaluation
- **`minigrid_quick_evaluation.py`** (1.5KB, 48 lines) - **Simple evaluation script**
  - Fast setup for basic agent testing
  - Minimal dependencies and configuration
  - **Use for:** Development, quick performance checks, debugging

#### Trace Generation
- **`minigrid_trace_evaluation.py`** (6.3KB, 205 lines) - **Viewer-compatible trace generation**
  - Generates traces compatible with the trace viewer system
  - Includes image observations and step-by-step action recording
  - Creates evaluation summaries with metadata
  - **Use for:** Creating viewer-ready traces, visual analysis, debugging agent behavior

## 🚀 Usage Examples

### 1. Quick Agent Testing

```bash
# Run quick evaluation with default settings
python minigrid_quick_evaluation.py

# Test specific model
python minigrid_quick_evaluation.py --model gpt-4o-mini
```

### 2. Comprehensive Evaluation

```bash
# Run comprehensive evaluation framework
python -c "
import asyncio
from minigrid_evaluation_framework import run_minigrid_eval

async def main():
    report = await run_minigrid_eval(
        model_names=['gpt-4o-mini'],
        difficulties=['easy', 'medium'],
        num_trajectories=5
    )
    
asyncio.run(main())
"
```

### 3. Generate Traces for Viewer

```bash
# Generate traces with image observations
python minigrid_trace_evaluation.py
```

**Expected Output:**
```
🎮 Running MiniGrid Evaluation
   Environment: MiniGrid-Empty-6x6-v0
   Episodes: 3
   Max steps: 50

📍 Episode 1/3
   ✅ Success! Reached goal in 12 steps
   💾 Saved trace: minigrid_trace_abc123.json

✅ Evaluation complete!
   Success rate: 100.0%
   Average steps: 12.0
   Output directory: src/evals/minigrid/run_1234567890
```

## 🎮 MiniGrid Environment Details

MiniGrid is a collection of grid-based environments featuring:

### Environment Types
- **Empty**: Navigate to goal in empty rooms
- **DoorKey**: Find keys to unlock doors blocking the goal
- **MultiRoom**: Navigate through connected rooms
- **FourRooms**: Classic four-room navigation challenge
- **UnlockPickup**: Combine unlocking and object pickup
- **LavaCrossing**: Avoid lava while reaching the goal

### Key Features
- **Grid-based navigation**: 2D discrete action space
- **Partial observability**: Limited vision around the agent
- **Object interaction**: Keys, doors, balls, boxes
- **Visual observations**: RGB image rendering
- **Scalable difficulty**: Various room sizes and complexity levels

### Action Space
```python
actions = {
    0: "left",      # Turn left (90 degrees)
    1: "right",     # Turn right (90 degrees)  
    2: "forward",   # Move forward one cell
    3: "pickup",    # Pick up object in front
    4: "drop",      # Drop carried object
    5: "toggle",    # Open/close doors
    6: "done"       # Signal task completion
}
```

## 📊 Evaluation Metrics

### Core Performance Metrics
- **Success Rate**: Percentage of episodes reaching the goal
- **Steps to Goal**: Number of actions taken to complete task
- **Efficiency Ratio**: Optimal steps / actual steps taken
- **Exploration Coverage**: Percentage of accessible area explored

### Navigation Analysis
- **Wall Collision Count**: Number of invalid movement attempts
- **Backtrack Count**: Number of revisited positions
- **Rooms Visited**: Count of different rooms explored
- **Final Position**: Agent's position at episode end

### Achievement System
**Basic Achievements** (6 total):
- `reach_goal` - Complete any goal-reaching task
- `first_pickup` - Pick up first object
- `first_door_open` - Open first door
- `first_key_use` - Use key to unlock door
- `navigate_empty_room` - Complete Empty room tasks
- `complete_5_tasks` - Complete 5 different tasks

**Intermediate Achievements** (6 total):
- `door_key_master` - Complete DoorKey tasks consistently
- `multi_room_navigator` - Complete MultiRoom tasks
- `unlock_pickup_combo` - Complete UnlockPickup tasks
- `four_rooms_explorer` - Complete FourRooms tasks
- `complete_20_tasks` - Complete 20 different tasks
- `efficiency_expert` - Complete task in <50% of max steps

**Advanced Achievements** (6 total):
- `lava_crosser` - Complete LavaCrossing tasks
- `large_room_master` - Complete 16x16+ room tasks
- `complex_multi_room` - Complete N6+ MultiRoom tasks
- `speed_runner` - Complete task in <25% of max steps
- `complete_50_tasks` - Complete 50 different tasks
- `perfect_navigator` - 90%+ success rate across all task types

### Scoring Systems

#### Composite Score
```python
composite_score = (
    achievement_score * 0.30 +    # Achievement unlocking
    completion_score * 0.40 +     # Task completion rate
    efficiency_score * 0.20 +     # Movement efficiency
    exploration_score * 0.10      # Exploration coverage
)
```

#### Navigation Score
```python
navigation_score = (
    success_rate * 0.70 +         # Success rate
    efficiency_ratio * 0.30       # Path efficiency
) - collision_penalty             # Wall collision penalty
```

## 🔧 Environment Interface

### Task Instance Creation
```python
from synth_env.examples.minigrid.taskset import create_minigrid_taskset

taskset = await create_minigrid_taskset()
task_instance = taskset.instances[0]  # Get first task
```

### Environment Usage
```python
from synth_env.examples.minigrid.environment import MiniGridEnvironment

env = MiniGridEnvironment(task_instance)
obs = await env.initialize()

# Take action
obs = await env.step({"action": "forward"})

# Check results
if obs.get("terminated"):
    print(f"Episode complete! Reward: {obs.get('total_reward', 0)}")
```

### Agent Integration
```python
from minigrid_react_agent import MiniGridReActAgent
from synth_ai.zyk import LM

# Create agent
llm = LM(model_name="gpt-4o-mini")
agent = MiniGridReActAgent(llm, max_turns=30, verbose=True)

# Run episode
result = await agent.run_episode(env)
print(f"Success: {result['success']}, Steps: {result['total_steps']}")
```

## 🧪 Development Workflow

### 1. Agent Development
1. Modify `minigrid_react_agent.py` to test new navigation strategies
2. Run `minigrid_quick_evaluation.py` for rapid iteration
3. Use `minigrid_evaluation_framework.py` for detailed analysis

### 2. Evaluation Setup
1. Configure evaluation parameters in the framework
2. Run comprehensive evaluations across multiple environments
3. Generate traces with `minigrid_trace_evaluation.py` for visual analysis

### 3. Analysis and Debugging
1. Use achievement tracking to identify specific capabilities
2. Analyze navigation patterns and efficiency metrics
3. Review traces in the viewer to understand agent behavior

## 🔍 Common Challenges and Solutions

### Navigation Issues
**Problem**: Agent gets stuck in loops or hits walls repeatedly
**Solution**: 
- Improve observation formatting to highlight blocked directions
- Add memory of recent positions to avoid backtracking
- Implement wall collision detection and avoidance

### Exploration Efficiency
**Problem**: Agent doesn't explore systematically
**Solution**:
- Add frontier-based exploration strategies
- Track visited cells and prioritize unexplored areas
- Implement room-aware navigation for multi-room tasks

### Object Interaction
**Problem**: Agent fails to pick up keys or open doors
**Solution**:
- Enhance object detection in observation formatting
- Add explicit tool calls for pickup/toggle actions
- Implement state tracking for carried objects

### Performance Optimization
**Problem**: Evaluation takes too long
**Solution**:
- Reduce `max_turns` for faster episodes
- Use smaller grid sizes for development
- Implement parallel evaluation for multiple episodes

## 📂 Output Structure

Evaluations create the following structure:
```
src/evals/minigrid/run_TIMESTAMP/
├── evaluation_summary.json     # Aggregate results and metadata
├── traces/                     # Individual episode traces
│   ├── minigrid_trace_1.json  # Trace for episode 1
│   ├── minigrid_trace_2.json  # Trace for episode 2
│   └── ...
└── detailed_report.json       # Comprehensive metrics and analysis
```

## 🎯 Performance Benchmarks

### Expected Performance Ranges

**Easy Tasks** (Empty-5x5, Empty-6x6):
- Success Rate: 70-95%
- Average Steps: 8-15
- Efficiency Ratio: 0.6-0.9

**Medium Tasks** (DoorKey-5x5, DoorKey-6x6):
- Success Rate: 40-80%
- Average Steps: 15-30
- Efficiency Ratio: 0.4-0.7

**Hard Tasks** (MultiRoom, LavaCrossing):
- Success Rate: 20-60%
- Average Steps: 25-50
- Efficiency Ratio: 0.3-0.6

### Model Comparisons
Different models show varying strengths:
- **GPT-4o**: Strong reasoning, good exploration
- **GPT-4o-mini**: Fast, decent navigation, cost-effective
- **Claude-3.5**: Excellent spatial reasoning
- **Gemini-1.5**: Good at object interaction tasks

## 📚 Related Documentation

- [MiniGrid Environment Guide](../README.md) - Environment setup and usage
- [Trace Viewer Documentation](../../../../viewer/README.md) - Viewing and analyzing traces
- [Agent Development Guide](../../../docs/agent_development.md) - Building navigation agents
- [MiniGrid Official Docs](https://minigrid.farama.org/) - Environment specifications

## 🤝 Contributing

When extending the MiniGrid demos:

1. **Follow naming conventions**: Use descriptive names like `minigrid_*_evaluation.py`
2. **Update this README**: Document new files and their purposes
3. **Include comprehensive metrics**: Track navigation efficiency and exploration
4. **Generate traces**: Ensure compatibility with the trace viewer system
5. **Add error handling**: Graceful handling of navigation failures
6. **Test thoroughly**: Verify with different environments and difficulty levels

## 📈 Future Enhancements

Planned improvements for the evaluation system:

- **Multi-agent evaluation**: Collaborative navigation tasks
- **Curriculum learning**: Progressive difficulty evaluation
- **Transfer learning**: Cross-environment performance analysis
- **Real-time visualization**: Live agent behavior monitoring
- **Advanced metrics**: Path optimality analysis, decision quality scoring

## 🎯 Learning Objectives

These demos help you understand:

- **Grid-based navigation**: Spatial reasoning and pathfinding
- **Partial observability**: Handling limited vision and exploration
- **Object interaction**: Key-door mechanics and tool usage
- **Multi-step reasoning**: Planning sequences of actions
- **Performance evaluation**: Navigation efficiency and success metrics

Run the demos to see these concepts in action! 🚀 