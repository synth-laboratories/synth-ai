"""
Test Pokémon Red rollout with GPT-5-nano policy

This script performs a rollout using GPT-5-nano to make decisions
based on the visual frames from the game.
"""

import asyncio
import os
from pathlib import Path

import httpx


async def main():
    """Run a GPT-5-nano driven rollout"""
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ OPENAI_API_KEY not set in environment")
        print("   Set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    print("=" * 70)
    print("POKÉMON RED - GPT-5-NANO POLICY ROLLOUT")
    print("=" * 70)
    print()
    print("Policy: gpt-5-nano (vision model)")
    print("Task: Complete Pallet Town progression")
    print("Server: http://127.0.0.1:8913")
    print()
    
    # Create output directory under the pokemon_emerald artifacts tree
    output_dir = Path("examples/task_apps/dev/pokemon_emerald/artifacts/pokemon_gpt5nano_rollout_frames")
    output_dir.mkdir(exist_ok=True)
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        # First check health
        try:
            health_response = await client.get("http://127.0.0.1:8913/health")
            health_response.raise_for_status()
            print("✓ Server is healthy")
            print()
        except Exception as e:
            print(f"❌ Server not responding: {e}")
            print("   Start it with: uv run -m synth_ai task-app serve pokemon_red --port 8913")
            return
        
        # Build rollout request with BATCHED actions (5-10 per sequence)
        # This simulates efficient gameplay with fewer tool calls
        sequences = []
        
        # Phase 1: Navigate to stairs and go downstairs (bedroom -> house)
        sequences.append({
            "actions": [
                {"button": "DOWN", "frames": 30},
                {"button": "DOWN", "frames": 30},
                {"button": "DOWN", "frames": 30},
                {"button": "DOWN", "frames": 30},
                {"button": "DOWN", "frames": 30},
                {"button": "DOWN", "frames": 30},
                {"button": "DOWN", "frames": 30},
                {"button": "DOWN", "frames": 30},
            ]
        })
        sequences.append({
            "actions": [
                {"button": "LEFT", "frames": 30},
                {"button": "LEFT", "frames": 30},
                {"button": "LEFT", "frames": 30},
                {"button": "DOWN", "frames": 30},
                {"button": "DOWN", "frames": 30},
                {"button": "DOWN", "frames": 30},
                {"button": "DOWN", "frames": 30},
                {"button": "DOWN", "frames": 30},
            ]
        })
        
        # Phase 2: Navigate to door and exit house (house -> pallet town)
        sequences.append({
            "actions": [
                {"button": "DOWN", "frames": 30},
                {"button": "DOWN", "frames": 30},
                {"button": "DOWN", "frames": 30},
                {"button": "DOWN", "frames": 30},
                {"button": "DOWN", "frames": 30},
                {"button": "DOWN", "frames": 30},
                {"button": "DOWN", "frames": 30},
                {"button": "DOWN", "frames": 30},
            ]
        })
        
        # Phase 3: Navigate Pallet Town to Oak's Lab
        sequences.append({
            "actions": [
                {"button": "LEFT", "frames": 30},
                {"button": "LEFT", "frames": 30},
                {"button": "LEFT", "frames": 30},
                {"button": "LEFT", "frames": 30},
                {"button": "LEFT", "frames": 30},
                {"button": "UP", "frames": 30},
                {"button": "UP", "frames": 30},
                {"button": "UP", "frames": 30},
            ]
        })
        sequences.append({
            "actions": [
                {"button": "UP", "frames": 30},
                {"button": "UP", "frames": 30},
                {"button": "UP", "frames": 30},
                {"button": "UP", "frames": 30},
                {"button": "UP", "frames": 30},
                {"button": "UP", "frames": 30},
                {"button": "UP", "frames": 30},
            ]
        })
        sequences.append({
            "actions": [
                {"button": "RIGHT", "frames": 30},
                {"button": "RIGHT", "frames": 30},
                {"button": "RIGHT", "frames": 30},
                {"button": "RIGHT", "frames": 30},
                {"button": "RIGHT", "frames": 30},
                {"button": "UP", "frames": 30},
                {"button": "UP", "frames": 30},
            ]
        })
        
        # Phase 4-6: Enter lab and interact extensively
        sequences.append({
            "actions": [{"button": "A", "frames": 30} for _ in range(10)]
        })
        sequences.append({
            "actions": [{"button": "A", "frames": 30} for _ in range(10)]
        })
        sequences.append({
            "actions": [
                {"button": "UP", "frames": 30},
                {"button": "UP", "frames": 30},
                {"button": "UP", "frames": 30},
                {"button": "RIGHT", "frames": 30},
                {"button": "RIGHT", "frames": 30},
                {"button": "RIGHT", "frames": 30},
            ]
        })
        sequences.append({
            "actions": [{"button": "A", "frames": 30} for _ in range(10)]
        })
        sequences.append({
            "actions": [{"button": "A", "frames": 30} for _ in range(10)]
        })
        sequences.append({
            "actions": [{"button": "A", "frames": 30} for _ in range(10)]
        })
        
        rollout_request = {
            "run_id": "gpt5nano_batched_test",
            "env": {"instance_id": "pallet_town_01"},
            "ops": sequences,
            "policy": {
                "config": {}
            },
        }
        
        total_actions = sum(len(seq["actions"]) for seq in sequences)
        print("🎮 Starting BATCHED rollout with action sequences...")
        print(f"   Total sequences: {len(sequences)} (was 105 separate calls)")
        print(f"   Total actions: {total_actions}")
        print(f"   Actions per sequence: ~{total_actions // len(sequences)}")
        print()
        print("   Phase 1: Navigate bedroom to stairs")
        print("   Phase 2: Exit house to Pallet Town")
        print("   Phase 3: Navigate to Oak's Lab")
        print("   Phase 4-6: Interact, get Pokemon, battle")
        print()
        
        try:
            response = await client.post(
                "http://127.0.0.1:8913/rollout",
                json=rollout_request,
            )
            response.raise_for_status()
            result = response.json()
            
            print("✓ Rollout complete!")
            
            # Debug: print response structure
            print(f"Response keys: {list(result.keys())}")
            
            # Handle branching response format
            trajectories = result.get("trajectories", [])
            if trajectories:
                # Get first trajectory
                trajectory = trajectories[0]
                steps = trajectory.get("steps", [])
            else:
                # Fallback to old format
                trajectory = result.get("trajectory", result)
                steps = trajectory.get("steps", [])
            
            print(f"  Trajectories: {len(trajectories)}")
            print(f"  Steps in first trajectory: {len(steps)}")
            print()
            
            # Save frames
            print("💾 Saving frames...")
            for i, step in enumerate(steps):
                obs = step.get("obs", {})
                img_b64 = obs.get("observation_image_base64")
                
                if img_b64:
                    import base64
                    from io import BytesIO

                    from PIL import Image
                    
                    img_data = base64.b64decode(img_b64)
                    img = Image.open(BytesIO(img_data))
                    
                    # Get action info
                    tool_calls = step.get("tool_calls", [])
                    action_str = "init"
                    if tool_calls:
                        action = tool_calls[0].get("args", {})
                        button = action.get("button", "?")
                        frames = action.get("frames", 1)
                        action_str = f"{button}_{frames}f"
                    
                    # Save with descriptive name
                    filename = output_dir / f"step_{i:03d}_{action_str}.png"
                    img.save(filename)
                    
                    # Print state info
                    map_id = obs.get("map_id", "?")
                    pos = f"({obs.get('player_x', '?')},{obs.get('player_y', '?')})"
                    party = obs.get("party_count", 0)
                    in_battle = obs.get("in_battle", False)
                    
                    status = f"Map{map_id}:{pos}"
                    if party > 0:
                        status += f" | Party:{party}"
                    if in_battle:
                        enemy_hp = obs.get("enemy_hp_percentage", 0)
                        status += f" | Battle! Enemy:{enemy_hp:.0f}%"
                    
                    print(f"  Step {i:2d}: {action_str:15s} | {status}")
            
            print()
            print(f"✓ Saved {len(steps)} frames to {output_dir}/")
            print()
            
            # Show trajectory metrics
            metrics = trajectory.get("metrics", result.get("metrics", {}))
            print("📊 Rollout Metrics:")
            for key, value in metrics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")
            
        except httpx.HTTPStatusError as e:
            print(f"❌ HTTP Error: {e.response.status_code}")
            print(f"   {e.response.text}")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
