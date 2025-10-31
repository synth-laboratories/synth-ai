#!/usr/bin/env python3
"""Run pokemon_vl eval with Qwen3-VL and extract images from trajectory response.

This script runs a qwen eval and extracts images directly from the trajectory steps
in the rollout response, similar to run_eval_extract_images.py but for Qwen models.
"""

import argparse
import asyncio
import base64
import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()


async def run_qwen_eval_and_extract_images(
    task_app_url: str,
    output_dir: Path,
    seed: int = 10,
    max_turns: int = 10,
    model: str = "Qwen/Qwen3-VL-30B-A3B-Thinking",
):
    """Run qwen eval and extract images from trajectory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    async with httpx.AsyncClient(timeout=600.0) as client:  # Longer timeout for qwen
        # Build rollout request matching eval_qwen3_vl.toml config
        rollout_request = {
            "run_id": f"qwen_eval_seed_{seed}",
            "env": {
                "env_name": "pokemon_red",
                "seed": seed,
                "config": {
                    "split": "train",
                    "index": seed,
                    "env_params": {"max_steps_per_episode": 100},
                },
            },
            "policy": {
                "policy_name": "pokemon_vl_qwen3_vl",
                "config": {
                    "model": model,
                    "provider": "synth",
                    "inference_url": "https://synth-laboratories-dev--learning-v2-service-fastapi-app.modal.run/chat/completions",
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "max_tokens": 2048,
                    "use_vision": True,
                    "image_only_mode": False,
                    "max_llm_calls": max_turns,
                    "thinking_mode": "think",
                    "thinking_budget": 3072,
                },
            },
            "ops": ["policy"] * max_turns,
            "mode": "eval",
            "record": {
                "return_trace": True,
                "trace_format": "full",
            },
        }
        
        print(f"Running eval with {model} (seed={seed})...")
        print(f"This may take a while as Qwen models load...")
        response = await client.post(f"{task_app_url}/rollout", json=rollout_request)
        response.raise_for_status()
        result = response.json()
        
        # Extract trajectory
        trajectories = result.get("trajectories", [])
        if not trajectories:
            print("Error: No trajectories in response")
            return
        
        trajectory = trajectories[0]
        steps = trajectory.get("steps", [])
        
        print(f"✓ Received {len(steps)} steps")
        print(f"Extracting images (filtering intermediate text box frames)...")
        
        # First pass: collect all images with their state
        image_data = []
        for idx, step in enumerate(steps):
            obs = step.get("obs", {})
            img_b64 = obs.get("observation_image_base64")
            
            if not img_b64:
                continue
            
            try:
                img_data = base64.b64decode(img_b64)
                map_id = obs.get("map_id", "?")
                player_x = obs.get("player_x", "?")
                player_y = obs.get("player_y", "?")
                text_box_active = obs.get("text_box_active", False)
                
                image_data.append({
                    "idx": idx,
                    "img_data": img_data,
                    "map_id": map_id,
                    "player_x": player_x,
                    "player_y": player_y,
                    "text_box_active": text_box_active,
                })
            except Exception as e:
                print(f"  Error decoding step {idx}: {e}")
                continue
        
        # Second pass: filter out intermediate text box frames
        # Keep: text_box_active=False OR the last frame of a text box sequence
        filtered_images = []
        for i, img_info in enumerate(image_data):
            text_box_active = img_info["text_box_active"]
            prev_text_box_active = image_data[i - 1]["text_box_active"] if i > 0 else False
            next_text_box_active = image_data[i + 1]["text_box_active"] if i + 1 < len(image_data) else False
            
            # Keep if:
            # 1. Not in a text box (text_box_active=False)
            # 2. Last frame of text box sequence (text_box_active=True and next is False)
            # 3. Last frame overall and in text box (no next frame)
            if not text_box_active:
                # Always keep non-text-box frames
                filtered_images.append(img_info)
            elif text_box_active and (not next_text_box_active or i + 1 >= len(image_data)):
                # Keep final frame of text box sequence (transition out or end of trajectory)
                filtered_images.append(img_info)
            # Otherwise skip intermediate text box loading frames
        
        # Save filtered images
        image_count = 0
        for img_info in filtered_images:
            try:
                map_id = img_info["map_id"]
                player_x = img_info["player_x"]
                player_y = img_info["player_y"]
                text_box_active = img_info["text_box_active"]
                idx = img_info["idx"]
                
                pos_str = f"Map{map_id}_{player_x},{player_y}"
                textbox_str = "True" if text_box_active else "False"
                filename = f"step_{idx:03d}_pos_{pos_str}_textbox_{textbox_str}_seed{seed}.png"
                
                filepath = output_dir / filename
                filepath.write_bytes(img_info["img_data"])
                
                print(f"  Saved: {filename}")
                image_count += 1
            except Exception as e:
                print(f"  Error saving step {img_info['idx']}: {e}")
                continue
        
        print(f"\n  Filtered: {len(image_data)} -> {len(filtered_images)} images (removed {len(image_data) - len(filtered_images)} intermediate text box frames)")
        
        print(f"\n✓ Extracted {image_count} images to {output_dir}/")
        
        # Also save metrics
        metrics = result.get("metrics", {})
        if metrics:
            metrics_file = output_dir / "metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(metrics, f, indent=2)
            print(f"✓ Saved metrics to {metrics_file}")


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task-app-url",
        default="http://127.0.0.1:8914",
        help="Task app URL",
    )
    parser.add_argument(
        "--output-dir",
        default="examples/blog_posts/pokemon_vl/images_qwen",
        help="Output directory for images",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=10,
        help="Random seed (default matches eval_qwen3_vl.toml)",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=10,
        help="Maximum turns",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen3-VL-30B-A3B-Thinking",
        help="Qwen model name",
    )
    args = parser.parse_args()
    
    await run_qwen_eval_and_extract_images(
        args.task_app_url,
        Path(args.output_dir),
        args.seed,
        args.max_turns,
        args.model,
    )


if __name__ == "__main__":
    asyncio.run(main())

