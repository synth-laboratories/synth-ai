"""
Test Pokémon Red rollout with GPT-5-nano policy

This script performs a rollout using GPT-5-nano to make decisions
based on the visual frames from the game.
"""

import asyncio
import httpx
import os
from pathlib import Path


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
    print(f"Policy: gpt-5-nano (vision model)")
    print(f"Task: Complete Pallet Town progression")
    print(f"Server: http://127.0.0.1:8913")
    print()
    
    # Create output directory
    output_dir = Path("pokemon_gpt5nano_rollout_frames")
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
        
        # Build rollout request with explicit manual actions
        # This simulates what a policy would do - navigate and interact
        rollout_request = {
            "run_id": "gpt5nano_test_001",
            "env": {"instance_id": "pallet_town_01"},
            "ops": [
                {"button": "DOWN", "frames": 20},  # Move downstairs
                {"button": "DOWN", "frames": 20},  # Continue to door
                {"button": "DOWN", "frames": 20},  # Exit house
                {"button": "UP", "frames": 30},    # Walk toward lab
                {"button": "UP", "frames": 30},    # Continue to lab
                {"button": "UP", "frames": 20},    # Enter lab
                {"button": "A", "frames": 30},     # Talk to Oak
                {"button": "A", "frames": 30},     # Advance dialogue
                {"button": "A", "frames": 30},     # Continue
                {"button": "DOWN", "frames": 20},  # Move to Pokeball
                {"button": "A", "frames": 30},     # Choose starter
                {"button": "A", "frames": 30},     # Confirm
                {"button": "A", "frames": 30},     # Battle starts
                {"button": "A", "frames": 30},     # Attack in battle
                {"button": "A", "frames": 30},     # Attack again
            ],
            "policy": {
                "config": {}
            },
        }
        
        print("🎮 Starting rollout with explicit button actions...")
        print("   This will execute 15 pre-planned actions")
        print()
        
        try:
            response = await client.post(
                "http://127.0.0.1:8913/rollout",
                json=rollout_request,
            )
            response.raise_for_status()
            result = response.json()
            
            print(f"✓ Rollout complete!")
            
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
                    from PIL import Image
                    from io import BytesIO
                    
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

