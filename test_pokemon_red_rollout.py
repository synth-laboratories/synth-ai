#!/usr/bin/env python3
"""Test Pokémon Red rollout with random actions and save frame PNGs."""
import asyncio
import base64
import json
import random
from pathlib import Path

import httpx

# Game Boy buttons
BUTTONS = ["A", "B", "UP", "DOWN", "LEFT", "RIGHT", "START", "SELECT"]

async def main():
    output_dir = Path("pokemon_red_rollout_frames")
    output_dir.mkdir(exist_ok=True)
    
    base_url = "http://127.0.0.1:8913"
    
    # Generate 10 random button presses
    random_actions = [
        {"action": {"button": random.choice(BUTTONS), "frames": 10}}
        for _ in range(10)
    ]
    
    print("Running rollout with random actions...")
    for i, act in enumerate(random_actions, 1):
        print(f"  Step {i}: {act['action']['button']}")
    
    async with httpx.AsyncClient(timeout=120.0) as client:
        rollout_req = {
            "run_id": "red-random-test",
            "env": {"env_name": "pokemon_red", "seed": 42, "config": {}},
            "policy": {"policy_name": "scripted", "config": {}},
            "ops": random_actions,
            "on_done": "terminate",
        }
        
        resp = await client.post(f"{base_url}/rollout", json=rollout_req)
        data = resp.json()
        
        if "trajectories" not in data or not data["trajectories"]:
            print("Error: No trajectories in response")
            print(json.dumps(data, indent=2))
            return
        
        traj = data["trajectories"][0]
        steps = traj["steps"]
        
        print(f"\n✓ Received {len(steps)} steps")
        
        # Save each step's frame
        for idx, step in enumerate(steps):
            obs = step["obs"]
            tool_calls = step.get("tool_calls", [])
            
            # Determine button label
            if idx == 0:
                label = "init"
            elif tool_calls:
                tc = tool_calls[0]
                args = tc.get("args", {})
                label = args.get("button", "unknown")
            else:
                label = "unknown"
            
            if "observation_image_base64" in obs:
                img_data = base64.b64decode(obs["observation_image_base64"])
                frame_path = output_dir / f"step_{idx:02d}_{label}.png"
                frame_path.write_bytes(img_data)
                print(f"Saved: {frame_path}")
                print(f"  Position: {obs.get('position')}, In battle: {obs.get('in_battle')}, Reward: {obs.get('reward_last_step')}")
            else:
                print(f"Step {idx}: No image available")
        
        print(f"\n✓ Saved {len(steps)} frames to {output_dir}/")
        print(f"  View them: open {output_dir}/")

if __name__ == "__main__":
    asyncio.run(main())

