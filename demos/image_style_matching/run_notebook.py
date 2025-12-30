#!/usr/bin/env python3
"""
Run the image style matching notebook end-to-end and save results.

This script executes the notebook logic and saves example images from the optimized graph.
"""

import os
import sys
import json
import base64
import uuid
from pathlib import Path
from io import BytesIO
from typing import Any, Optional

import httpx
from PIL import Image, ImageDraw

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from synth_ai.sdk.api.train.graphgen import GraphGenJob, load_graphgen_taskset
from dotenv import load_dotenv

load_dotenv()


def _create_placeholder_image(color: tuple[int, int, int]) -> str:
    """Create a simple placeholder image as base64 data URL."""
    img = Image.new("RGB", (256, 256), color)
    draw = ImageDraw.Draw(img)
    
    center = 128
    radius = 80
    draw.ellipse(
        [center - radius, center - radius, center + radius, center + radius],
        fill=(255, 255, 255),
        outline=(0, 0, 0),
        width=3
    )
    
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{img_b64}"


def save_image_from_data_url(data_url: str, output_path: Path) -> None:
    """Save an image from a base64 data URL to a file."""
    if not data_url.startswith("data:image"):
        print(f"Warning: Not a valid image data URL: {data_url[:50]}...")
        return
    
    # Extract base64 data
    header, encoded = data_url.split(",", 1)
    img_data = base64.b64decode(encoded)
    
    # Determine format from header
    if "png" in header.lower():
        ext = "png"
    elif "jpeg" in header.lower() or "jpg" in header.lower():
        ext = "jpg"
    else:
        ext = "png"  # Default
    
    # Save image
    output_path = output_path.with_suffix(f".{ext}")
    with open(output_path, "wb") as f:
        f.write(img_data)
    print(f"  Saved: {output_path}")


def main():
    """Run notebook end-to-end and save results."""
    # Step 1: Setup - Use local backend if LOCAL_BACKEND is set, otherwise try production
    # If LOCAL_BACKEND is set, use localhost:8000
    if os.environ.get('LOCAL_BACKEND', '').lower() in ('1', 'true', 'yes'):
        SYNTH_API_BASE = 'http://127.0.0.1:8000/api'
        backend_type = 'LOCAL'
    else:
        # Try production backend (gemini-2.5-flash-image is allowed in prod)
        # If DEV_BACKEND_URL is set, use it; otherwise use production
        SYNTH_API_BASE = os.environ.get('DEV_BACKEND_URL') or os.environ.get('BACKEND_BASE_URL') or 'https://api.usesynth.ai'
        if not SYNTH_API_BASE.startswith('http'):
            SYNTH_API_BASE = f'https://{SYNTH_API_BASE}'
        # Ensure /api suffix
        if not SYNTH_API_BASE.endswith('/api'):
            SYNTH_API_BASE = SYNTH_API_BASE.rstrip('/') + '/api'
        backend_type = 'DEV' if 'dev' in SYNTH_API_BASE.lower() else 'PROD'
    
    print(f'Backend: {SYNTH_API_BASE} ({backend_type} - for gemini-2.5-flash-image)')
    
    # Check backend health
    r = httpx.get(f'{SYNTH_API_BASE}/health', timeout=30)
    if r.status_code == 200:
        print(f'Backend health: {r.json()}')
    else:
        print(f'WARNING: Backend returned status {r.status_code}')
        raise RuntimeError(f'Backend not healthy: status {r.status_code}')
    
    # Step 2: Authentication
    API_KEY = os.environ.get('SYNTH_API_KEY', '')
    if not API_KEY:
        print('No SYNTH_API_KEY found, minting demo key...')
        resp = httpx.post(f'{SYNTH_API_BASE}/api/demo/keys', json={'ttl_hours': 4}, timeout=30)
        resp.raise_for_status()
        API_KEY = resp.json()['api_key']
        print(f'Demo API Key: {API_KEY[:25]}...')
    else:
        print(f'Using SYNTH_API_KEY: {API_KEY[:20]}...')
    
    # Step 3: Create Dataset
    print('\n' + '=' * 80)
    print('Creating GraphGen Dataset')
    print('=' * 80)
    
    tasks = [
        {
            "id": "pokemon_dragon",
            "input": {
                "subject": "dragon",
                "style": "pokemon",
                "description": "A dragon creature in Pokemon art style"
            }
        },
        {
            "id": "pokemon_cat",
            "input": {
                "subject": "cat",
                "style": "pokemon", 
                "description": "A cat creature in Pokemon art style"
            }
        },
        {
            "id": "pokemon_bird",
            "input": {
                "subject": "bird",
                "style": "pokemon",
                "description": "A bird creature in Pokemon art style"
            }
        },
    ]
    
    gold_outputs = []
    colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255)]
    
    for i, task in enumerate(tasks):
        gold_outputs.append({
            "task_id": task["id"],
            "output": {
                "image_url": _create_placeholder_image(colors[i % len(colors)]),
                "note": f"Reference Pokemon image for {task['input']['subject']}"
            }
        })
    
    gold_outputs.append({
        "output": {
            "image_url": _create_placeholder_image((255, 255, 100)),
            "note": "Standalone style reference"
        }
    })
    
    input_schema = {
        "type": "object",
        "properties": {
            "subject": {"type": "string"},
            "style": {"type": "string"},
            "description": {"type": "string"}
        },
        "required": ["subject", "style", "description"]
    }
    
    output_schema = {
        "type": "object",
        "properties": {
            "image_url": {
                "type": "string",
                "description": "Base64-encoded image data URL (data:image/png;base64,...)"
            }
        },
        "required": ["image_url"]
    }
    
    unique_id = uuid.uuid4().hex[:8]
    dataset = {
        "version": "1.0",
        "metadata": {
            "name": f"pokemon-style-matching-{unique_id}",
            "description": "Match Pokemon art style using Gemini with contrastive VLM judge (gpt-4.1-nano).",
            "input_schema": input_schema,
            "output_schema": output_schema,
        },
        "initial_prompt": "Generate an image.",
        "tasks": tasks,
        "gold_outputs": gold_outputs,
        "input_schema": input_schema,
        "output_schema": output_schema,
        "default_rubric": {
            "outcome": {
                "criteria": [
                    {
                        "name": "pokemon_style_match",
                        "description": "The image should match Pokemon art style: anime-inspired, colorful, cute creature design with expressive features. Score 1.0 for authentic Pokemon style, 0.5 for anime-adjacent style, 0.0 for photorealistic or non-anime style.",
                        "weight": 1.0
                    },
                    {
                        "name": "subject_recognition",
                        "description": "The creature should be clearly recognizable as the requested subject (dragon, cat, bird, etc.) while maintaining Pokemon aesthetic. Score 1.0 if subject is clear, 0.5 if somewhat recognizable, 0.0 if unclear.",
                        "weight": 0.8
                    },
                    {
                        "name": "visual_quality",
                        "description": "The image should be high quality: clean lines, vibrant colors, proper composition. Score 1.0 for high quality, 0.5 for acceptable with minor issues, 0.0 for poor quality.",
                        "weight": 0.5
                    }
                ]
            }
        },
        "judge_config": {
            "mode": "contrastive",
            "model": "gpt-4.1-nano",
            "provider": "openai"
        }
    }
    
    dataset_path = Path(__file__).parent / "image_style_matching_dataset.json"
    with open(dataset_path, "w") as f:
        json.dump(dataset, f, indent=2)
    
    print(f'Created dataset: {dataset["metadata"]["name"]}')
    print(f'  Tasks: {len(tasks)}')
    print(f'  Gold outputs: {len(gold_outputs)}')
    print(f'  Judge mode: {dataset["judge_config"]["mode"]}')
    print(f'  Judge model: {dataset["judge_config"]["model"]}')
    print(f'  Saved to: {dataset_path}')
    
    # Step 4: Run GraphGen Optimization
    print('\n' + '=' * 80)
    print('Running GraphGen Optimization')
    print('=' * 80)
    
    dataset_obj = load_graphgen_taskset(dataset_path)
    
    problem_spec = (
        "Generate images that match Pokemon art style. "
        "The workflow should use Gemini for image generation and a VLM judge (gpt-4.1-nano) "
        "to compare generated images against gold reference images for style matching."
    )
    
    print(f'Creating GraphGen job...')
    print(f'  NOTE: gemini-2.5-flash-image requires backend code update.')
    print(f'  The deployed backend may not support this model yet.')
    print(f'  Backend code shows it should be allowed - deployment may be needed.')
    
    job = GraphGenJob.from_dataset(
        dataset=dataset_obj,
        policy_model="gemini-2.5-flash-image",  # Image generation model
        rollout_budget=10,
        proposer_effort="medium",
        population_size=2,
        num_generations=1,
        problem_spec=problem_spec,
        backend_url=SYNTH_API_BASE,
        api_key=API_KEY,
        auto_start=True,
    )
    
    print(f'  Policy model: {job.config.policy_model} (image generation)')
    print(f'  Rollout budget: {job.config.rollout_budget}')
    
    result = job.submit()
    print(f'\nJob submitted: {result.graphgen_job_id}')
    
    final_status = job.poll_until_complete(timeout=600.0, interval=5.0, progress=True)
    
    print(f'\nFINAL STATUS: {final_status.get("status")}')
    if final_status.get("status") == "succeeded":
        print(f'BEST SCORE: {final_status.get("best_score")}')
    elif final_status.get("status") == "failed":
        print(f'ERROR: {final_status.get("error")}')
        return
    
    # Step 5: Download Graph and Run Inference
    print('\n' + '=' * 80)
    print('Downloading Optimized Graph')
    print('=' * 80)
    
    try:
        graph_txt = job.download_graph_txt()
        print(graph_txt[:500] + "..." if len(graph_txt) > 500 else graph_txt)
    except Exception as e:
        print(f'Could not download graph: {e}')
        import traceback
        traceback.print_exc()
        return
    
    # Step 6: Run Inference and Save Results
    print('\n' + '=' * 80)
    print('Running Inference and Saving Results')
    print('=' * 80)
    
    # Create results directory
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Save graph
    graph_path = results_dir / "optimized_graph.txt"
    with open(graph_path, "w") as f:
        f.write(graph_txt)
    print(f'Saved graph to: {graph_path}')
    
    # Test inputs for inference
    test_inputs = [
        {
            "subject": "wolf",
            "style": "pokemon",
            "description": "A wolf creature in Pokemon art style"
        },
        {
            "subject": "fox",
            "style": "pokemon",
            "description": "A fox creature in Pokemon art style"
        },
        {
            "subject": "rabbit",
            "style": "pokemon",
            "description": "A rabbit creature in Pokemon art style"
        },
    ]
    
    print(f'\nRunning inference on {len(test_inputs)} test inputs...')
    print(f'NOTE: If images are empty, the graph may have been generated before the backend fix.')
    print(f'      Check usage models - should be gemini-2.5-flash-image for image generation.\n')
    
    for i, test_input in enumerate(test_inputs):
        print(f'\nTest {i+1}: {test_input["subject"]}')
        try:
            # Use longer timeout for image generation (can take 2-3 minutes)
            output = job.run_inference(test_input, timeout=180.0)
            print(f'  Output type: {type(output)}')
            
            # Always save output as JSON first
            output_path = results_dir / f"test_{i+1}_{test_input['subject']}_output.json"
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)
            print(f'  Saved output to: {output_path}')
            
            if isinstance(output, dict):
                # Check for image_url in various possible locations
                # Priority: nested node outputs (full data) > top-level (may be truncated)
                image_url = None
                
                # First, check nested node outputs (these contain the full image data)
                nested_output = output.get("output", {})
                if isinstance(nested_output, dict):
                    # Check all {node_name}_output keys for image_url
                    for key, value in nested_output.items():
                        if key.endswith("_output") and isinstance(value, dict):
                            if "image_url" in value:
                                candidate = value["image_url"]
                                if isinstance(candidate, str) and len(candidate) > 100:  # Must be substantial
                                    image_url = candidate
                                    print(f'  Found image in {key}.image_url (length: {len(image_url)})')
                                    break
                    
                    # Also check direct image_url in nested output
                    if not image_url:
                        candidate = nested_output.get("image_url", "")
                        if isinstance(candidate, str) and len(candidate) > 100:
                            image_url = candidate
                            print(f'  Found image in output.image_url (length: {len(image_url)})')
                
                # Fallback to top-level (may be truncated by parse_output)
                if not image_url:
                    candidate = output.get("image_url", "")
                    if isinstance(candidate, str) and len(candidate) > 100:
                        image_url = candidate
                        print(f'  Found image in top-level image_url (length: {len(image_url)})')
                
                # Also check usage to see what model was used
                usage = output.get("usage", [])
                models_used = [u.get("model") for u in usage if isinstance(u, dict)]
                print(f'  Models used: {models_used}')
                
                if image_url and isinstance(image_url, str) and len(image_url) > 0:
                    if image_url.startswith("data:image"):
                        print(f'  Found image URL! Length: {len(image_url)}, prefix: {image_url[:80]}...')
                        # Save image
                        image_path = results_dir / f"test_{i+1}_{test_input['subject']}.png"
                        try:
                            save_image_from_data_url(image_url, image_path)
                            print(f'  ✅ Saved image to: {image_path}')
                        except Exception as e:
                            print(f'  ❌ Failed to save image: {e}')
                            import traceback
                            traceback.print_exc()
                    else:
                        print(f'  image_url exists but is not a data URL (length: {len(image_url)}, starts with: {image_url[:50]})')
                else:
                    print(f'  ⚠️  No valid image_url found (empty or missing)')
                    print(f'  Output keys: {list(output.keys())}')
                    if "output" in output:
                        print(f'  Nested output keys: {list(output["output"].keys())}')
            else:
                print(f'  Output type: {type(output)}')
        except Exception as e:
            print(f'  Inference failed: {e}')
            import traceback
            traceback.print_exc()
    
    # Save job metadata
    metadata = {
        "job_id": result.graphgen_job_id,
        "status": final_status.get("status"),
        "best_score": final_status.get("best_score"),
        "dataset_name": dataset["metadata"]["name"],
        "test_inputs": test_inputs,
    }
    
    metadata_path = results_dir / "job_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f'\nSaved job metadata to: {metadata_path}')
    
    print('\n' + '=' * 80)
    print('Done!')
    print('=' * 80)
    print(f'Results saved to: {results_dir}')


if __name__ == "__main__":
    main()

