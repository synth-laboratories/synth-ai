# NetHack Environment for Synth-Env

This is a NetHack environment implementation following the synth-env framework patterns. It provides both a mock implementation for testing and full integration with NLE (NetHack Learning Environment) for actual gameplay.

## Overview

NetHack is one of the most complex and challenging games ever created, featuring:
- Procedurally generated dungeons
- Hundreds of items, monsters, and interactions
- Deep strategic gameplay
- ASCII-based interface
- Permadeath and high difficulty

This environment wraps NetHack for use with AI agents, providing structured observations and action interfaces.

## Installation

```bash
# Install NLE (NetHack Learning Environment)
pip install nle

# Or with uv
uv add nle
```

## Understanding States

### Public State (What the Agent Sees)

The `NetHackPublicState` contains all information visible to the agent:

```python
@dataclass
class NetHackPublicState:
    # Current dungeon level (1 = first floor, higher = deeper)
    dungeon_level: int = 1
    
    # Character statistics visible to player
    character_stats: Dict[str, Any] = field(default_factory=dict)
    # Includes: hp, max_hp, strength, dexterity, constitution, 
    #          intelligence, wisdom, charisma, ac (armor class),
    #          gold, experience_level, etc.
    
    # Items the player is carrying
    inventory: List[Dict[str, Any]] = field(default_factory=list)
    # Each item: {'letter': 'a', 'description': 'a +1 long sword'}
    
    # Player position on the map (x, y)
    position: Tuple[int, int] = (0, 0)
    
    # The game map as ASCII art (80x24 characters)
    ascii_map: str = ""
    
    # Latest game message (e.g., "You hit the goblin!")
    message: str = ""
    
    # Cursor position (for menus/targeting)
    cursor_position: Tuple[int, int] = (0, 0)
    
    # Game progress
    turn_count: int = 0
    max_turns: int = 10000
    last_action: str = ""
    terminated: bool = False
    
    # Menu/interaction state
    in_menu: bool = False
    menu_items: List[str] = field(default_factory=list)
```

### Private State (Hidden Game Information)

The `NetHackPrivateState` contains information not visible to the agent but used for evaluation:

```python
@dataclass
class NetHackPrivateState:
    # Reward information
    reward_last: float = 0.0
    total_reward: float = 0.0
    
    # Game termination status
    terminated: bool = False
    truncated: bool = False  # Time limit reached
    
    # Performance metrics
    score: int = 0
    depth_reached: int = 1
    experience_level: int = 1
    monsters_killed: int = 0
    items_collected: int = 0
```

## Using the Environment

### Basic Usage with Mock Engine

```python
from synth_env.examples.nethack.taskset import NetHackTaskSet
from synth_env.examples.nethack.environment import NetHackEnvironment

# Create a task
taskset = NetHackTaskSet()
task_instance = await taskset.sample_task_instance(
    task_id="explore_dungeon",
    config={
        "character_role": "knight",
        "difficulty": "novice",
        "time_limit": 1000
    }
)

# Create environment
env = NetHackEnvironment()
public_state, private_state = await env.start(task_instance=task_instance)

# Take actions
from synth_env.environment.tools import EnvToolCall

tool_call = EnvToolCall(
    tool="interact",
    args={"action": "north"}
)

observation, reward, done, info = await env.process_action([tool_call])
```

### Using with Real NLE Backend

```python
from synth_env.examples.nethack.helpers.nle_wrapper import NLEWrapper

# Create NLE wrapper
wrapper = NLEWrapper(character_role="wizard")

# Reset game
obs = wrapper.reset()

# Take actions
obs, reward, done, info = wrapper.step("north")
obs, reward, done, info = wrapper.step("search")
obs, reward, done, info = wrapper.step("inventory")
```

## Understanding Observations

### ASCII Map Characters

The `ascii_map` field contains the dungeon using these characters:

- `@` - You (the player)
- `.` - Floor/walkable ground
- `#` - Wall or corridor
- `+` - Closed door
- `-` - Open door
- `<` - Stairs up
- `>` - Stairs down
- `{` - Fountain
- `^` - Trap
- `$` - Gold
- `%` - Food
- `!` - Potion
- `?` - Scroll
- `/` - Wand
- `)` - Weapon
- `[` - Armor
- `d` - Dog (pet)
- `f` - Cat (pet)
- Letters - Various monsters (g=goblin, o=orc, etc.)

### Player Stats

The `character_stats` dictionary includes:

```python
{
    'x': 10,  # X position
    'y': 5,   # Y position
    'hp': 12,  # Current hit points
    'max_hp': 16,  # Maximum hit points
    'strength': 18,  # STR attribute (18/** is exceptional)
    'dexterity': 15,  # DEX attribute
    'constitution': 14,  # CON attribute
    'intelligence': 10,  # INT attribute
    'wisdom': 8,  # WIS attribute
    'charisma': 12,  # CHA attribute
    'ac': 6,  # Armor class (lower is better)
    'gold': 100,  # Gold pieces carried
    'experience_level': 3,  # Character level
    'experience_points': 245,  # Total XP
    'depth': 2,  # Dungeon level (1=top)
    'hunger_state': 1,  # 0=satiated, higher=hungrier
    'score': 532  # Game score
}
```

## Available Actions

The environment supports 80+ NetHack actions, including:

### Movement
- `north`, `south`, `east`, `west`
- `northeast`, `northwest`, `southeast`, `southwest`
- `up` (climb stairs up), `down` (climb stairs down)

### Basic Actions
- `wait` - Do nothing for one turn
- `search` - Search for secret doors/traps
- `look` - Examine surroundings
- `pickup` - Pick up items
- `drop` - Drop items
- `inventory` - View inventory

### Item Usage
- `eat` - Eat food
- `drink`/`quaff` - Drink potions
- `read` - Read scrolls/spellbooks
- `wear` - Put on armor
- `wield` - Ready a weapon
- `apply` - Use tools

### Combat
- Movement into monsters attacks them
- `fire` - Fire ranged weapon
- `throw` - Throw items
- `zap` - Use wands

### Special
- `pray` - Pray to your deity
- `offer` - Make sacrifices
- `engrave` - Write on the floor
- `#quit` - Quit game

## Task Configuration

Tasks can be configured with:

```python
config = {
    "character_role": "wizard",  # Role to play as
    "difficulty": "expert",      # novice/intermediate/expert/master
    "target_depth": 10,          # Goal depth to reach
    "time_limit": 5000,          # Max turns allowed
    "target_item": "amulet"      # Specific item to find
}
```

### Character Roles

- `tourist` - Starts with camera and money
- `knight` - Strong warrior with good equipment
- `wizard` - Magic user with spells
- `rogue` - Stealthy with lockpicking abilities
- `barbarian` - Strong fighter, primitive equipment
- `priest` - Divine magic and healing
- `monk` - Martial arts, no armor
- `valkyrie` - Nordic warrior
- `ranger` - Archer with nature abilities
- `samurai` - Honorable warrior
- `archaeologist` - Academic with tools
- `caveman` - Primitive but strong
- `healer` - Medical knowledge

## Quick Reference Guide

### Getting Information from Observations

```python
# After taking an action and getting observation:

# Player position
x, y = observation['position']

# Health status  
current_hp = observation['character_stats']['hp']
max_hp = observation['character_stats']['max_hp']
is_hurt = current_hp < max_hp

# Location info
dungeon_level = observation['dungeon_level']  # 1 = first floor
turn = observation['turn_count']

# Game status
is_done = observation['terminated']
last_message = observation['message']

# Inventory check
has_items = len(observation['inventory']) > 0
for item in observation['inventory']:
    print(f"{item['letter']}: {item['description']}")

# Map access
map_text = observation['ascii_map']
map_lines = map_text.split('\n')
```

### Common State Checks

```python
# Am I in danger?
hp_ratio = observation['character_stats']['hp'] / observation['character_stats']['max_hp']
in_danger = hp_ratio < 0.3

# Do I have gold?
gold = observation['character_stats'].get('gold', 0)

# What's my armor class?
ac = observation['character_stats'].get('ac', 10)  # Lower is better

# Am I hungry?
hunger = observation['character_stats'].get('hunger_state', 0)
need_food = hunger > 3  # Getting hungry

# What level am I?
char_level = observation['character_stats'].get('experience_level', 1)
```

### Safe Exploration Pattern

```python
# Search thoroughly before moving
for _ in range(10):
    obs = await env.step({"action": "search"})

# Then move carefully
for direction in ["north", "east", "south", "west"]:
    obs = await env.step({"action": direction})
    if obs.get('reward_last', 0) > 0:
        print(f"Got reward: {obs['reward_last']}")
```

### Health Management

```python
hp = obs['character_stats']['hp']
max_hp = obs['character_stats']['max_hp']

if hp < max_hp * 0.5:
    # Try to heal
    # 1. Pray (if not used recently)
    await env.step({"action": "pray"})
    # 2. Quaff healing potion
    await env.step({"action": "quaff"})
    # 3. Rest with "wait"
    await env.step({"action": "wait"})
```

### Message Parsing

```python
msg = observation['message'].lower()

# Combat messages
if "hit" in msg or "miss" in msg:
    print("In combat!")

# Important events  
if "hear" in msg:
    print("Something nearby")
    
if "see here" in msg:
    print("Items on ground")
    
if "hungry" in msg:
    print("Need food!")

if "dies" in msg or "destroyed" in msg:
    print("Killed something")
```

### Debug Tips

```python
# Print full state
print(f"Turn {obs['turn_count']}:")
print(f"  Position: {obs['position']}")
print(f"  HP: {obs['character_stats']['hp']}/{obs['character_stats']['max_hp']}")
print(f"  Level: D:{obs['dungeon_level']} L:{obs['character_stats']['experience_level']}")
print(f"  Message: {obs['message']}")
print(f"  Inventory: {len(obs['inventory'])} items")

# Show nearby map
x, y = obs['position']
map_lines = obs['ascii_map'].split('\n')
for dy in range(-3, 4):
    if 0 <= y+dy < len(map_lines):
        print(map_lines[y+dy][max(0,x-10):x+11])
```

## Reward Events

Rewards happen when the game score increases:
- Pick up gold: +1 per gold piece
- Kill monster: +XP value of monster  
- Go down stairs: +50 × depth
- Eat when hungry: +1
- Identify items: varies

## BALROG Scoring

NetHack uses the BALROG benchmark scoring system:

### BALROG Rewards (Training)
Per-step shaped rewards for analysis:
- Score deltas: +score_change/100
- Gold collection: +gold_gained/1000  
- Experience: +exp_gained/100
- Depth progression: +depth_change*10 (big rewards!)
- Level up: +level_change*5
- Death penalty: -100

### BALROG Score (Evaluation)
Official leaderboard metric (0-100%) based on milestone achievements:
- **Dungeon progression**: Reaching deeper levels (dlvl5=1%, dlvl10=3%, castle=50%, etc.)
- **Experience progression**: Character levels (lvl2=1%, lvl5=9%, lvl10=24%, etc.)
- **Final score**: max(dungeon_progression, experience_progression)
- **SOTA**: ~1-2% (NetHack is extremely challenging!)

Achievement milestones are defined in `helpers/achievements.json`.
- `knight` - Mounted warrior with lance
- `wizard` - Magic user with spells
- `barbarian` - Strong fighter
- `monk` - Martial artist
- `priest` - Divine magic user
- `rogue` - Stealthy thief
- `ranger` - Archer and tracker
- `samurai` - Honorable warrior
- `valkyrie` - Female warrior
- `archeologist` - Explorer with tools
- `healer` - Medical expert
- `caveman` - Primitive fighter

## Rewards

NetHack uses a score-based reward system:

- **Reward = Score(t) - Score(t-1)**
- Positive rewards for:
  - Picking up gold (1 point per gold)
  - Killing monsters (varies by type)
  - Descending dungeon levels
  - Eating when hungry
  - Identifying items
  - Various achievements

Note: Rewards can be sparse, especially early in the game.

## Recording and Visualization

The environment supports trajectory recording and replay:

```python
from synth_env.examples.nethack.helpers.recording_wrapper import RecordingNetHackEnvironment

# Create recording environment
env = RecordingNetHackEnvironment(
    save_dir="nethack_trajectories",
    auto_record=True
)

# Play game...
# Trajectories are automatically saved

# Replay a trajectory
from synth_env.examples.nethack.helpers.visualization.replay_viewer import ReplayViewer
viewer = ReplayViewer("trajectory_file.gz")
viewer.interactive_replay()
```

## Tips for Agent Development

1. **Start Simple**: Begin with basic movement and exploration before complex strategies
2. **Handle Sparsity**: Rewards are rare - consider auxiliary rewards or curiosity
3. **Parse Messages**: The message field contains crucial information
4. **Memory**: NetHack requires long-term memory (visited locations, identified items)
5. **Safety**: Many actions can be fatal - careful exploration is key

## Common Pitfalls

1. **Attacking Peaceful Creatures**: Attacking shopkeepers or priests is usually fatal
2. **Starvation**: Monitor hunger state and eat regularly
3. **Cursed Items**: Some items are harmful when used
4. **Overencumbrance**: Carrying too much slows movement
5. **Prayer Timeout**: Can only pray every 300-800 turns

## Debugging

Enable verbose output:
```python
import logging
logging.getLogger("synth_env.examples.nethack").setLevel(logging.DEBUG)
```

## Examples

See the `agent_demos/` directory for example agents:
- `test_synth_react.py` - ReAct agent using LLMs
- `dev/test_nle_interactive.py` - Interactive human play
- `dev/test_nethack_recording.py` - Recording and replay demo

## References

- [NetHack Wiki](https://nethackwiki.com/) - Comprehensive game information
- [NLE Paper](https://arxiv.org/abs/2006.13760) - NetHack Learning Environment
- [NetHack Guidebook](https://www.nethack.org/v366/Guidebook.html) - Official guide