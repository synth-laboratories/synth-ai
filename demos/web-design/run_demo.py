#!/usr/bin/env python3
"""Run Web Design Style Prompt Optimization with GEPA.

This demo optimizes a style system prompt that guides Gemini 2.5 Flash Image
to generate visually accurate webpage screenshots from functional descriptions.

Usage:
    uv run python demos/web-design/run_demo.py --local   # Local mode (fast iteration)
    uv run python demos/web-design/run_demo.py           # Production mode (with tunnels)
"""

import argparse
import os
import sys
import asyncio
import json
import time
from typing import Any
from pathlib import Path

# Parse args early
parser = argparse.ArgumentParser(description="Run Web Design GEPA demo")
parser.add_argument("--local", action="store_true", help="Local mode (localhost, no tunnels)")
parser.add_argument("--local-host", type=str, default="localhost", help="Local host for APIs")
args = parser.parse_args()

LOCAL_MODE = args.local
LOCAL_HOST = args.local_host

# Setup paths
demo_dir = Path(__file__).parent
repo_root = demo_dir.parent.parent
sys.path.insert(0, str(repo_root))

# Load .env (optional)
try:
    from dotenv import load_dotenv
    env_file = repo_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)
        print(f"Loaded {env_file}")
except ImportError:
    print("python-dotenv not installed, using existing environment variables")

# Imports
import httpx
from datasets import load_from_disk
from PIL import Image
import io

from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob
from synth_ai.sdk.learning.prompt_learning_client import PromptLearningClient
from synth_ai.sdk.localapi.auth import ensure_localapi_auth
from synth_ai.sdk.localapi import LocalAPIConfig, create_local_api
from synth_ai.sdk.task import run_server_background
from synth_ai.sdk.task.contracts import RolloutMetrics, RolloutRequest, RolloutResponse, TaskInfo
from synth_ai.sdk.tunnels import TunnelBackend, TunneledLocalAPI, acquire_port, PortConflictBehavior
from synth_ai.core.env import PROD_BASE_URL, mint_demo_api_key


# Backend config
if LOCAL_MODE:
    SYNTH_API_BASE = "http://localhost:8000"
    TUNNEL_BACKEND = TunnelBackend.Localhost
    LOCAL_API_PORT = 8103
    print("=" * 80)
    print("RUNNING IN LOCAL MODE")
    print("=" * 80)
else:
    SYNTH_API_BASE = PROD_BASE_URL
    TUNNEL_BACKEND = TunnelBackend.CloudflareManagedTunnel
    LOCAL_API_PORT = 8001

print(f"Backend: {SYNTH_API_BASE}")
print(f"Local API Port: {LOCAL_API_PORT}")

# Check backend health
r = httpx.get(f"{SYNTH_API_BASE}/health", timeout=30)
if r.status_code == 200:
    print(f"Backend health: {r.json()}")
else:
    raise RuntimeError(f"Backend not healthy: status {r.status_code}")

# Get API key
API_KEY = os.environ.get("SYNTH_API_KEY", "")

if not API_KEY:
    print("No SYNTH_API_KEY, minting demo key...")
    API_KEY = mint_demo_api_key()
    print(f"Demo API Key: {API_KEY[:25]}...")
else:
    print(f"Using SYNTH_API_KEY: {API_KEY[:20]}...")

os.environ["SYNTH_API_KEY"] = API_KEY

# Ensure environment key
ENVIRONMENT_API_KEY = ensure_localapi_auth(
    backend_base=SYNTH_API_BASE,
    synth_api_key=API_KEY,
)
print(f"Env key ready: {ENVIRONMENT_API_KEY[:12]}...{ENVIRONMENT_API_KEY[-4:]}")


# ==============================================================================
# IMAGE VERIFICATION FUNCTION
# ==============================================================================

async def verify_generated_image(
    original_image: Image.Image,
    generated_image_bytes: bytes,
    inference_url: str,
    api_key: str,
) -> dict:
    """Verify generated image against original using vision model via inference_url."""
    import base64

    if not inference_url:
        raise ValueError("inference_url is required for verification")

    generated_image = Image.open(io.BytesIO(generated_image_bytes))

    prompt = """You are evaluating how well a GENERATED webpage matches the ORIGINAL webpage visually.

Score on these criteria (0-10):
1. COLOR SCHEME & BRANDING
2. TYPOGRAPHY & STYLING
3. LAYOUT & SPACING
4. VISUAL ELEMENTS
5. OVERALL VISUAL FIDELITY

Return ONLY valid JSON (no markdown):
{
  "color_scheme_branding": <0-10>,
  "typography_styling": <0-10>,
  "layout_spacing": <0-10>,
  "visual_elements": <0-10>,
  "overall_visual_fidelity": <0-10>,
  "average_score": <average>
}"""

    # Convert PIL images to base64 data URLs
    original_bytes_io = io.BytesIO()
    original_image.save(original_bytes_io, format="PNG")
    original_bytes_io.seek(0)
    original_b64 = base64.b64encode(original_bytes_io.read()).decode('utf-8')
    original_data_url = f"data:image/png;base64,{original_b64}"

    generated_bytes_io = io.BytesIO(generated_image_bytes)
    generated_image_pil = Image.open(generated_bytes_io)
    generated_bytes_io_2 = io.BytesIO()
    generated_image_pil.save(generated_bytes_io_2, format="PNG")
    generated_bytes_io_2.seek(0)
    generated_b64 = base64.b64encode(generated_bytes_io_2.read()).decode('utf-8')
    generated_data_url = f"data:image/png;base64,{generated_b64}"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"{prompt}\n\nORIGINAL image first, then GENERATED image. Return JSON:"},
                {"type": "image_url", "image_url": {"url": original_data_url}},
                {"type": "image_url", "image_url": {"url": generated_data_url}},
            ]
        }
    ]

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{inference_url.rstrip('/')}/chat/completions",
            json={
                "model": "gemini-2.5-flash-image",
                "messages": messages,
                "temperature": 0.0,
            },
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        response.raise_for_status()
        llm_response = response.json()

    text = llm_response["choices"][0]["message"]["content"].strip()

    # Parse JSON response
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]

    result = json.loads(text.strip())

    # Ensure average_score
    if "average_score" not in result:
        scores = [
            result.get("color_scheme_branding", 0),
            result.get("typography_styling", 0),
            result.get("layout_spacing", 0),
            result.get("visual_elements", 0),
            result.get("overall_visual_fidelity", 0)
        ]
        result["average_score"] = sum(scores) / len(scores)

    return result


# ==============================================================================
# DATASET
# ==============================================================================

class WebDesignDataset:
    """Astral website pages for style optimization."""

    def __init__(self):
        self._examples = None
        self._images_dir = demo_dir / "task_images"
        self._images_dir.mkdir(exist_ok=True)

    def _load(self):
        if self._examples is not None:
            return

        dataset_path = demo_dir / "hf_dataset"
        dataset = load_from_disk(str(dataset_path))

        # Filter Astral pages
        astral = [
            ex for ex in dataset
            if ex.get("site_name") == "astral"
            and ex.get("functional_description")
            and len(ex["functional_description"]) > 100
            and isinstance(ex.get("image"), Image.Image)
        ]

        print(f"Loaded {len(astral)} Astral pages")

        # Save images
        for i, ex in enumerate(astral[:15]):  # Limit to 15 for quick iteration
            page_name = ex["page_name"].replace("/", "_").replace(" ", "_")
            image_path = self._images_dir / f"astral_{page_name}_{i:03d}.png"

            if not image_path.exists():
                ex["image"].save(image_path)

            ex["image_path"] = str(image_path)

        self._examples = astral[:15]

    def size(self) -> int:
        self._load()
        return len(self._examples)

    def sample(self, index: int) -> dict:
        self._load()
        idx = index % len(self._examples)
        ex = self._examples[idx]
        return {
            "index": idx,
            "functional_description": ex["functional_description"],
            "image_path": ex["image_path"],
            "site_name": ex["site_name"],
            "page_name": ex["page_name"],
        }


# ==============================================================================
# LOCAL API
# ==============================================================================

APP_ID = "web_design_generator"
APP_NAME = "Web Design Style Optimization"


def create_web_design_local_api(style_prompt: str):
    """Create the local API for web design generation."""
    dataset = WebDesignDataset()

    async def run_rollout(request: RolloutRequest, fastapi_request: Any) -> RolloutResponse:
        """Run a single rollout: generate image using policy model and verify."""
        import httpx
        import base64
        from synth_ai.sdk.task.trace_correlation_helpers import extract_trace_correlation_id

        seed = request.env.seed
        sample = dataset.sample(seed)

        # Load original image
        original_image = Image.open(sample["image_path"])

        try:
            # Get inference URL from policy config (points to inference interceptor)
            inference_url = request.policy.config.get("inference_url", "")
            if not inference_url:
                raise ValueError("inference_url is required in policy config")

            model = request.policy.config.get("model", "gemini-2.5-flash-image")

            # Extract policy config for trace correlation (exclude trace fields)
            policy_cfg_for_trace = {
                key: value
                for key, value in (request.policy.config or {}).items()
                if key not in {"trace_correlation_id", "trace"}
            }
            
            # Build prompt: Gemini doesn't support system messages, so combine everything into user message
            # GEPA will match and optimize the style_prompt portion at the beginning
            full_prompt = f"""{style_prompt}

Generate a webpage screenshot based on this functional description:

{sample["functional_description"]}

Apply the visual style guidelines to match the original design."""

            # Call policy model through inference interceptor
            # The interceptor will handle image generation models automatically
            messages = [
                {"role": "user", "content": full_prompt}
            ]
            
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    f"{inference_url.rstrip('/')}/chat/completions",
                    json={
                        "model": model,
                        "messages": messages,
                        "temperature": 0.7,
                    },
                    headers={
                        "Authorization": f"Bearer {ENVIRONMENT_API_KEY}",
                        "Content-Type": "application/json",
                    },
                )
                response.raise_for_status()
                llm_response = response.json()
            
            # Extract image from response (multipart content format)
            generated_bytes = None
            choices = llm_response.get("choices", [])
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "")
                
                if isinstance(content, list):
                    # Multipart content - find image_url part
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "image_url":
                            image_url = part.get("image_url", {}).get("url", "")
                            if image_url.startswith("data:image/"):
                                # Extract base64 data from data URL
                                header, data = image_url.split(",", 1)
                                generated_bytes = base64.b64decode(data)
                                break
                elif isinstance(content, str) and content.startswith("data:image/"):
                    # Direct data URL
                    header, data = content.split(",", 1)
                    generated_bytes = base64.b64decode(data)
            
            if not generated_bytes:
                raise ValueError("No image data in policy model response")

            # Extract trace correlation ID from generation call (before verification)
            trace_correlation_id = extract_trace_correlation_id(
                policy_config=policy_cfg_for_trace,
                inference_url=str(inference_url or ""),
                mode=request.mode,
            )

            # Verify against original using inference_url
            verification = await verify_generated_image(
                original_image,
                generated_bytes,
                inference_url=inference_url,
                api_key=ENVIRONMENT_API_KEY,
            )

            # Average score is our reward (0-10 scale, convert to 0-1)
            reward = verification["average_score"] / 10.0

        except Exception as e:
            print(f"Rollout error: {e}")
            import traceback
            traceback.print_exc()
            reward = 0.0
            # Keep trace_correlation_id if it was extracted before error
            if 'trace_correlation_id' not in locals():
                trace_correlation_id = None

        return RolloutResponse(
            run_id=request.run_id,
            metrics=RolloutMetrics(outcome_reward=reward),
            trace=None,
            trace_correlation_id=trace_correlation_id,
            inference_url=str(inference_url or ""),
        )

    def provide_taskset_description():
        return {
            "splits": ["train"],
            "sizes": {"train": 15},  # Hardcoded to avoid loading dataset on health check
        }

    def provide_task_instances(seeds):
        for seed in seeds:
            sample = dataset.sample(seed)
            yield TaskInfo(
                task={"id": APP_ID, "name": APP_NAME},
                dataset={"id": APP_ID, "split": "train", "index": sample["index"]},
                inference={},
                limits={"max_turns": 1},
                task_metadata={
                    "page": f"{sample['site_name']}/{sample['page_name']}",
                    "description_length": len(sample["functional_description"]),
                },
            )

    return create_local_api(LocalAPIConfig(
        app_id=APP_ID,
        name=APP_NAME,
        description="Web design style prompt optimization local API",
        provide_taskset_description=provide_taskset_description,
        provide_task_instances=provide_task_instances,
        rollout=run_rollout,
        cors_origins=["*"],
    ))


print("Web design local API defined")


# ==============================================================================
# MAIN
# ==============================================================================

async def main():
    BASELINE_STYLE_PROMPT = """You are generating a professional startup website screenshot.

VISUAL STYLE GUIDELINES:
- Use a clean, modern, minimalist design aesthetic
- Color Scheme: Light backgrounds with high contrast dark text
- Typography: Large, bold headings with clear hierarchy
- Layout: Spacious with generous padding and margins
- Branding: Professional, tech-forward visual identity

Create a webpage that feels polished, modern, and trustworthy."""

    def format_duration(seconds: float) -> str:
        if seconds < 60:
            return f"{seconds:.1f}s"
        mins, secs = divmod(int(seconds), 60)
        return f"{mins}m {secs}s"

    timings = {}
    total_start = time.time()

    # Start local API
    print("\n" + "=" * 80)
    print("STARTING LOCAL API")
    print("=" * 80)

    app = create_web_design_local_api(BASELINE_STYLE_PROMPT)
    port = acquire_port(LOCAL_API_PORT, on_conflict=PortConflictBehavior.FIND_NEW)

    run_server_background(app, port)
    print(f"Local API running on port {port}")

    # Wait for health check
    await asyncio.sleep(3)

    # Get local API URL (with tunnel if production)
    if LOCAL_MODE:
        LOCAL_API_URL = f"http://{LOCAL_HOST}:{port}"
        tunnel = None
    else:
        print("\nProvisioning Cloudflare tunnel...")
        tunnel_start = time.time()
        tunnel = await TunneledLocalAPI.create(
            local_port=port,
            backend=TUNNEL_BACKEND,
            progress=True,
        )
        LOCAL_API_URL = tunnel.url
        timings["tunnel"] = time.time() - tunnel_start

    print(f"Local API URL: {LOCAL_API_URL}")

    # Run GEPA optimization
    print("\n" + "=" * 80)
    print("RUNNING GEPA OPTIMIZATION")
    print("=" * 80)

    gepa_config_path = demo_dir / "gepa_config.toml"

    # Create prompt learning job (synchronous submission)
    import toml
    from synth_ai.sdk.api.train.prompt_learning import PromptLearningJobConfig

    # Read TOML and override task_app_url with tunnel URL
    with open(gepa_config_path) as f:
        config_dict = toml.load(f)

    config_dict["prompt_learning"]["task_app_url"] = LOCAL_API_URL
    print(f"Using task_app_url: {LOCAL_API_URL}")

    job_config = PromptLearningJobConfig(
        config_dict=config_dict,
        backend_url=SYNTH_API_BASE,
        api_key=API_KEY,
    )

    job = PromptLearningJob(config=job_config)

    print("Submitting GEPA job...")
    opt_start = time.time()
    job_id = job.submit()
    print(f"Job ID: {job_id}")

    print("\nPolling for completion...")
    result = job.poll_until_complete(timeout=3600.0, interval=10.0, progress=True)
    timings["optimization"] = time.time() - opt_start

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    if result.succeeded:
        print(f"Status: SUCCESS")
        print(f"Best Score: {result.best_score:.3f}/1.0 ({result.best_score * 10:.1f}/10)")

        if result.best_prompt:
            print("\n" + "=" * 80)
            print("OPTIMIZED STYLE PROMPT")
            print("=" * 80)
            print(result.best_prompt)

            # Save
            output_dir = demo_dir / "optimization_results"
            output_dir.mkdir(exist_ok=True)

            prompt_file = output_dir / "optimized_style_prompt.txt"
            prompt_file.write_text(result.best_prompt)
            print(f"\n✓ Saved to: {prompt_file}")

            result_file = output_dir / "gepa_results.json"
            with open(result_file, "w") as f:
                json.dump(result.raw, f, indent=2, default=str)
            print(f"✓ Results: {result_file}")

    else:
        print(f"Status: FAILED")
        print(f"Error: {result.error}")

    # Cleanup
    if tunnel:
        print("\nCleaning up tunnel...")
        from synth_ai.sdk.tunnels import cleanup_all
        cleanup_all()

    # Summary
    total_time = time.time() - total_start
    print("\n" + "=" * 80)
    print("DEMO COMPLETE")
    print("=" * 80)
    print(f"Total time: {format_duration(total_time)}")
    print(f"Optimization time: {format_duration(timings.get('optimization', 0))}")
    if result.succeeded:
        print(f"Final score: {result.best_score * 10:.1f}/10")
    print()


if __name__ == "__main__":
    asyncio.run(main())
