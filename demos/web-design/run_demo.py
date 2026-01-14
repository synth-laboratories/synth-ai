#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Run Web Design Style Prompt Optimization with GEPA.

This demo optimizes a style system prompt that guides Gemini 2.5 Flash Image
to generate visually accurate webpage screenshots from functional descriptions.

Usage:
    uv run python demos/web-design/run_demo.py --local   # Local mode (fast iteration)
    uv run python demos/web-design/run_demo.py           # Production mode (with tunnels)
"""

import argparse
import asyncio
import base64
import io
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any

import httpx
import toml
from datasets import Image as HFImage
from datasets import load_dataset, load_from_disk
from PIL import Image

try:
    from synth_ai.core.env import get_backend_url, mint_demo_api_key
    from synth_ai.core.urls import BACKEND_URL_BASE
    from synth_ai.sdk.api.train.prompt_learning import PromptLearningJob, PromptLearningJobConfig
    from synth_ai.sdk.localapi import LocalAPIConfig, create_local_api
    from synth_ai.sdk.localapi.auth import ensure_localapi_auth
    from synth_ai.sdk.task.server import RubricBundle
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "Failed to import `synth_ai`.\n"
        "This demo expects the `synth-ai` package to be installed in your environment.\n\n"
        "If you're running from a repo checkout, install it first (from the repo root):\n"
        "  - uv:  `uv sync`  (then run: `uv run python demos/web-design/run_demo.py`)\n"
        "  - pip: `python -m pip install -e .`\n"
    ) from e
try:
    # Preferred: stable locations (avoids relying on sdk.task re-exports).
    from synth_ai.sdk.task.server import run_server_background
except ImportError:  # pragma: no cover
    # Back-compat with older packaging.
    from synth_ai.sdk.task import run_server_background

from synth_ai.sdk.task.contracts import RolloutMetrics, RolloutRequest, RolloutResponse, TaskInfo
from synth_ai.sdk.task.rubrics import Criterion, Rubric
from synth_ai.sdk.task.trace_correlation_helpers import extract_trace_correlation_id
from synth_ai.sdk.tunnels import (
    PortConflictBehavior,
    TunnelBackend,
    TunneledLocalAPI,
    acquire_port,
    cleanup_all,
)

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Suppress verbose HTTP logs from httpx and google-genai
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("google").setLevel(logging.WARNING)
logging.getLogger("google.auth").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)

# Parse args early
parser = argparse.ArgumentParser(description="Run Web Design GEPA demo")
parser.add_argument("--local", action="store_true", help="Local mode (localhost, no tunnels)")
parser.add_argument("--local-host", type=str, default="127.0.0.1", help="Local host for APIs")
args = parser.parse_args()

LOCAL_MODE = args.local
LOCAL_HOST = args.local_host

WEB_DESIGN_RUBRICS = RubricBundle(
    outcome=Rubric(
        version="1.0",
        goal_text="Evaluate how well the generated webpage matches the original visually",
        criteria=[
            Criterion(
                id="visual_fidelity",
                description=(
                    "How well does the generated webpage match the original webpage visually? "
                    "Evaluate color scheme, typography, layout, spacing, and overall visual fidelity."
                ),
                weight=1.0,
                required=True,
            )
        ],
        aggregation="weighted_sum",
    )
)

# Setup paths
demo_dir = Path(__file__).parent
repo_root = demo_dir.parent.parent


# Print a quick diagnostic if we're accidentally importing synth_ai from elsewhere.
def _maybe_warn_on_synth_ai_mismatch() -> None:
    try:
        import synth_ai as _synth_ai
    except Exception:
        return

    synth_path = Path(getattr(_synth_ai, "__file__", "") or "").resolve()
    if synth_path and repo_root not in synth_path.parents:
        version_str = "unknown"
        try:
            from importlib import metadata

            version_str = metadata.version("synth-ai")
        except Exception:
            pass

        print(
            "WARNING: You are not importing `synth_ai` from this repo checkout.\n"
            f"- Repo root: {repo_root}\n"
            f"- Imported synth_ai from: {synth_path}\n"
            f"- Installed distribution version (if any): {version_str}\n"
            "This can cause import errors if your installed package is older than the demo.\n"
            "Fix: run from the repo venv, or uninstall the old `synth-ai` wheel, or ensure this repo is first on PYTHONPATH."
        )


_maybe_warn_on_synth_ai_mismatch()

# Backend config - respect SYNTH_BACKEND_URL env var, fall back to --local flag behavior
SYNTH_API_BASE = get_backend_url() if os.environ.get("SYNTH_BACKEND_URL") else ("http://127.0.0.1:8000" if LOCAL_MODE else BACKEND_URL_BASE)
if LOCAL_MODE:
    TUNNEL_BACKEND = TunnelBackend.Localhost
    LOCAL_API_PORT = 8103
    print("=" * 80)
    print("RUNNING IN LOCAL MODE")
    print("=" * 80)
else:
    TUNNEL_BACKEND = TunnelBackend.CloudflareManagedTunnel
    LOCAL_API_PORT = 8001

print(f"Backend: {SYNTH_API_BASE}")
print(f"Local API Port: {LOCAL_API_PORT}")

# Check backend health
r = httpx.get(f"{SYNTH_API_BASE}/health", timeout=60)
if r.status_code == 200:
    print(f"Backend health: {r.json()}")
else:
    raise RuntimeError(f"Backend not healthy: status {r.status_code}")

# Get API key
API_KEY = os.environ.get("SYNTH_API_KEY", "")

if not API_KEY:
    print("No SYNTH_API_KEY, minting demo key...")
    API_KEY = mint_demo_api_key(backend_url=SYNTH_API_BASE)
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

# No manual verification - GEPA's backend verifier will handle it


# ==============================================================================
# DATASET
# ==============================================================================

HF_DATASET_ID_ENV = "SYNTH_WEB_DESIGN_DATASET"
HF_DATASET_REVISION_ENV = "SYNTH_WEB_DESIGN_DATASET_REVISION"
HF_DATASET_SPLIT = "train"

DEFAULT_HF_DATASET_ID = "JoshPurtell/web-design-screenshots"

DEFAULT_SITE_FILTER = "astral"
DEFAULT_MAX_EXAMPLES = int(os.environ.get("SYNTH_WEB_DESIGN_MAX_EXAMPLES", "8"))
DEFAULT_MAX_IMAGE_PIXELS = int(
    os.environ.get("SYNTH_WEB_DESIGN_MAX_IMAGE_PIXELS", "12000000")
)  # 12MP


class WebDesignDataset:
    """Astral website pages for style optimization."""

    def __init__(
        self,
        resize_size: int = 384,
        site_filter: str = DEFAULT_SITE_FILTER,
        max_examples: int = DEFAULT_MAX_EXAMPLES,
    ):
        self._examples = None
        self._site_filter = site_filter
        self._max_examples = max_examples

        cache_dir = os.environ.get("SYNTH_WEB_DESIGN_CACHE_DIR", "").strip()
        if cache_dir:
            self._cache_dir = Path(cache_dir).expanduser()
        else:
            self._cache_dir = Path.home() / ".cache" / "synth_ai" / "web_design"

        self._images_dir = self._cache_dir / "task_images"
        self._images_dir.mkdir(parents=True, exist_ok=True)

        self._resized_dir = self._images_dir / f"resized_{resize_size}"
        self._resized_dir.mkdir(parents=True, exist_ok=True)
        self._resize_size = resize_size
        self._max_image_pixels = DEFAULT_MAX_IMAGE_PIXELS

    def _load(self):
        if self._examples is not None:
            return

        dataset_id_env = (os.environ.get(HF_DATASET_ID_ENV) or "").strip()
        # Allow an explicit "local only" escape hatch for offline/CI.
        if dataset_id_env.lower() in {"local", "disk"}:
            dataset_id = ""
        else:
            dataset_id = dataset_id_env or DEFAULT_HF_DATASET_ID
        dataset_revision = os.environ.get(HF_DATASET_REVISION_ENV, "").strip() or None

        if dataset_id:
            logger.info(
                f"Loading web-design dataset from Hub: {dataset_id} (split={HF_DATASET_SPLIT})"
            )
            dataset = load_dataset(dataset_id, split=HF_DATASET_SPLIT, revision=dataset_revision)
        else:
            dataset_path = demo_dir / "hf_dataset"
            if not dataset_path.exists():
                raise RuntimeError(
                    f"No dataset configured.\n\n"
                    f"- Set {HF_DATASET_ID_ENV} to your public Hugging Face dataset (e.g. org/web-design-screenshots)\n"
                    f"- OR create a local dataset at {dataset_path} via create_hf_dataset.py\n"
                )
            logger.info(f"Loading web-design dataset from disk: {dataset_path}")
            dataset = load_from_disk(str(dataset_path))

        # CRITICAL: Avoid decoding images during filtering; some screenshots are extremely large and can
        # trigger PIL DecompressionBombWarning (and/or use huge memory) even if we later downsample.
        # We only filter on metadata columns here.
        if "image" in dataset.column_names:
            dataset = dataset.cast_column("image", HFImage(decode=False))

        dataset = dataset.filter(
            lambda site_name, functional_description: (
                site_name == self._site_filter
                and bool(functional_description)
                and len(functional_description) > 100
            ),
            input_columns=["site_name", "functional_description"],
        )

        logger.info(f"Loaded {len(dataset)} '{self._site_filter}' pages")

        selected: list[dict] = []
        cursor = 0
        max_to_scan = min(len(dataset), max(self._max_examples * 10, self._max_examples))

        while len(selected) < self._max_examples and cursor < max_to_scan:
            ex = dict(dataset[cursor])
            cursor += 1

            site_name = (ex.get("site_name") or "site").replace("/", "_").replace(" ", "_")
            page_name = (
                (ex.get("page_name") or f"page_{cursor:03d}").replace("/", "_").replace(" ", "_")
            )

            # We only cache a resized image to avoid storing/encoding huge screenshots.
            resized_path = self._resized_dir / f"{site_name}_{page_name}_{len(selected):03d}.png"

            if resized_path.exists():
                # Validate cached file is actually small; if not, overwrite it.
                try:
                    with Image.open(resized_path) as cached:
                        if (cached.size[0] * cached.size[1]) > self._max_image_pixels or max(
                            cached.size
                        ) > self._resize_size:
                            resized_path.unlink(missing_ok=True)
                except Exception:
                    # If unreadable/corrupt, overwrite.
                    resized_path.unlink(missing_ok=True)

            if not resized_path.exists():
                img_obj = ex.get("image")
                if isinstance(img_obj, Image.Image):
                    img = img_obj
                elif isinstance(img_obj, dict):
                    # When datasets.Image(decode=False), rows look like {"bytes": <bytes|None>, "path": <str|None>}
                    raw_bytes = img_obj.get("bytes")
                    raw_path = img_obj.get("path")
                    if isinstance(raw_bytes, (bytes, bytearray)) and raw_bytes:
                        img = Image.open(io.BytesIO(raw_bytes))
                    elif isinstance(raw_path, str) and raw_path:
                        img = Image.open(raw_path)
                    else:
                        raise RuntimeError(
                            f"Dataset row {cursor - 1} had an undecodable image dict (no bytes/path). "
                            f"keys={list(img_obj.keys())}"
                        )
                else:
                    raise RuntimeError(
                        f"Dataset row {cursor - 1} did not decode 'image' to a PIL Image and had no usable bytes/path. "
                        f"Got type={type(img_obj)}"
                    )

                # Size check (cheap; reads header for most formats). Skip pathological images.
                width, height = img.size
                pixels = int(width) * int(height)
                if pixels > self._max_image_pixels:
                    print(
                        f"Skipping oversized image: {site_name}/{page_name} "
                        f"({width}x{height}={pixels:,} px; limit={self._max_image_pixels:,})"
                    )
                    continue

                # Downscale before writing to disk.
                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.thumbnail((self._resize_size, self._resize_size), Image.Resampling.LANCZOS)
                img.save(resized_path, format="PNG", optimize=True)

            ex["image_path"] = str(resized_path)
            ex["resized_image_path"] = str(resized_path)
            selected.append(ex)

        self._examples = selected

        # Images are already written in resized form (and capped by pixel limit).

    def _ensure_resized_images(self):
        """Resize all images to cached versions if they don't exist."""
        if self._examples is None:
            return

        for ex in self._examples:
            original_path = Path(ex["image_path"])
            resized_path = self._resized_dir / original_path.name

            # Only resize if resized version doesn't exist or is older than original
            if (
                not resized_path.exists()
                or resized_path.stat().st_mtime < original_path.stat().st_mtime
            ):
                try:
                    with Image.open(original_path) as img:
                        # Convert to RGB if needed (handles RGBA, P, etc.)
                        if img.mode != "RGB":
                            img = img.convert("RGB")

                        # Resize maintaining aspect ratio
                        img.thumbnail(
                            (self._resize_size, self._resize_size), Image.Resampling.LANCZOS
                        )

                        # Save resized version
                        img.save(resized_path, format="PNG", optimize=True)
                        logger.debug(
                            f"Resized {original_path.name} -> {resized_path.name} ({img.size[0]}x{img.size[1]})"
                        )
                    ex["resized_image_path"] = str(resized_path)
                except Exception as e:
                    logger.warning(f"Failed to resize {original_path.name}: {e}")
                    # Fallback: use original path if resize fails
                    ex["resized_image_path"] = str(original_path)
            else:
                ex["resized_image_path"] = str(resized_path)

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
            "resized_image_path": ex.get("resized_image_path", ex["image_path"]),
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

    # Pre-load dataset and resize images BEFORE server starts
    logger.info("Pre-loading dataset and resizing images...")
    dataset._load()  # This triggers _ensure_resized_images()
    logger.info(f"Dataset ready with {dataset.size()} examples")

    async def run_rollout(request: RolloutRequest, fastapi_request: Any) -> RolloutResponse:
        """Run a single rollout: generate image using policy model and verify."""
        seed = request.env.seed
        sample = dataset.sample(seed)

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
            messages = [{"role": "user", "content": full_prompt}]

            # Build URL - inference_url may already include /chat/completions
            url = inference_url.rstrip('/')
            if '/chat/completions' not in url:
                url = f"{url}/chat/completions"

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(
                    url,
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

            # Save generated image for review
            output_dir = Path(__file__).parent / "generated_images"
            output_dir.mkdir(exist_ok=True)
            run_id_short = (
                request.trace_correlation_id[:8] if request.trace_correlation_id else "unknown"
            )
            img_path = output_dir / f"seed_{seed}_run_{run_id_short}.png"
            with open(img_path, "wb") as f:
                f.write(generated_bytes)
            logger.info(f"Saved generated image to {img_path}")

            # Extract trace correlation ID from generation call (before verification)
            trace_correlation_id = extract_trace_correlation_id(
                policy_config=policy_cfg_for_trace,
                inference_url=str(inference_url or ""),
            )

            # Backend verifier will score this - return dummy reward
            # The verifier config in gepa_config.toml will provide the real reward
            reward = 0.0

        except Exception as e:
            logger.error(f"Rollout error: {e}")
            import traceback

            traceback.print_exc()
            reward = 0.0
            # Keep trace_correlation_id if it was extracted before error
            if "trace_correlation_id" not in locals():
                trace_correlation_id = None

        return RolloutResponse(
            run_id=request.trace_correlation_id or "unknown",
            reward_info=RolloutMetrics(outcome_reward=reward),
            trace=None,
            trace_correlation_id=trace_correlation_id or request.trace_correlation_id or "unknown",
            inference_url=str(inference_url or ""),
        )

    def provide_taskset_description():
        return {
            "splits": ["train"],
            "sizes": {"train": dataset.size()},
        }

    def provide_task_instances(seeds):
        for seed in seeds:
            sample = dataset.sample(seed)

            # Load resized image and encode to base64
            resized_image_path = Path(sample["resized_image_path"])
            try:
                with Image.open(resized_image_path) as img:
                    # Ensure RGB mode
                    if img.mode != "RGB":
                        img = img.convert("RGB")

                    # Encode to base64
                    buffer = io.BytesIO()
                    img.save(buffer, format="PNG", optimize=True)
                    img_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
                    original_data_url = f"data:image/png;base64,{img_b64}"
            except Exception as e:
                print(f"Warning: Failed to load image {resized_image_path}: {e}")
                original_data_url = None

            yield TaskInfo(
                task={"id": APP_ID, "name": APP_NAME},
                dataset={"id": APP_ID, "split": "train", "index": sample["index"]},
                environment="web_design",  # Must match gepa_config.toml prompt_learning.gepa.env_name
                inference={},
                limits={"max_turns": 1},
                task_metadata={
                    "page": f"{sample['site_name']}/{sample['page_name']}",
                    "description_length": len(sample["functional_description"]),
                    "original_image_url": original_data_url,
                },
            )

    return create_local_api(
        LocalAPIConfig(
            app_id=APP_ID,
            name=APP_NAME,
            description="Web design style prompt optimization local API",
            provide_taskset_description=provide_taskset_description,
            provide_task_instances=provide_task_instances,
            rollout=run_rollout,
            rubrics=WEB_DESIGN_RUBRICS,
            cors_origins=["*"],
        )
    )


print("Web design local API defined")


# ==============================================================================
# MAIN
# ==============================================================================


async def main():
    baseline_style_prompt = """You are generating a professional startup website screenshot.

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

    app = create_web_design_local_api(baseline_style_prompt)
    port = acquire_port(LOCAL_API_PORT, on_conflict=PortConflictBehavior.FIND_NEW)

    run_server_background(app, port)
    print(f"Local API running on port {port}")

    # Wait for health check
    await asyncio.sleep(3)

    # Get local API URL (with tunnel if production)
    if LOCAL_MODE:
        local_api_url = f"http://{LOCAL_HOST}:{port}"
        tunnel = None
    else:
        print("\nProvisioning Cloudflare tunnel...")
        tunnel_start = time.time()
        tunnel = await TunneledLocalAPI.create(
            local_port=port,
            backend=TUNNEL_BACKEND,
            progress=True,
        )
        local_api_url = tunnel.url
        timings["tunnel"] = time.time() - tunnel_start

        # Wait a bit longer for DNS propagation to ensure backend can reach it
        print("Waiting 10s for full DNS propagation...")
        await asyncio.sleep(10)

    print(f"Local API URL: {local_api_url}")

    # Run GEPA optimization
    print("\n" + "=" * 80)
    print("RUNNING GEPA OPTIMIZATION")
    print("=" * 80)

    gepa_config_path = demo_dir / "gepa_config.toml"

    # Read TOML and override task_app_url and task_app_api_key
    with open(gepa_config_path) as f:
        config_dict = toml.load(f)

    config_dict["prompt_learning"]["task_app_url"] = local_api_url
    config_dict["prompt_learning"]["task_app_api_key"] = ENVIRONMENT_API_KEY
    print(f"Using task_app_url: {local_api_url}")
    print(f"Using task_app_api_key: {ENVIRONMENT_API_KEY[:12]}...{ENVIRONMENT_API_KEY[-4:]}")

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

    # Banking77-style event polling: run in a background thread so it keeps printing even if
    # status polling is slow/blocked, and so we can print progress even when backend doesn't
    # attach a message field.
    def _format_event_line(event_type: str, message: str, data: dict) -> str | None:
        if message:
            return message

        if event_type == "prompt.learning.phase.changed":
            prev = data.get("from") or data.get("prev") or data.get("old")
            nxt = data.get("to") or data.get("next") or data.get("new")
            if prev or nxt:
                return f"Phase: {prev} → {nxt}"
            return "Phase changed"

        if event_type in {
            "prompt.learning.progress",
            "prompt.learning.gepa.rollouts_limit_progress",
        }:
            step = data.get("step") or data.get("iteration") or data.get("iter")
            total = data.get("total") or data.get("max")
            best = data.get("best_score") or data.get("best_reward")
            parts: list[str] = []
            if step is not None and total is not None:
                parts.append(f"{step}/{total}")
            elif step is not None:
                parts.append(str(step))
            if best is not None:
                parts.append(f"best={best}")
            return "Progress: " + " ".join(parts) if parts else "Progress"

        if event_type in {
            "prompt.learning.gepa.new_best",
            "prompt.learning.gepa.candidate.evaluated",
        }:
            score = data.get("score") or data.get("best_score") or data.get("best_reward")
            if score is not None:
                return (
                    f"✨ New best: {score}"
                    if event_type.endswith("new_best")
                    else f"Candidate scored: {score}"
                )
            return "GEPA update"

        return None

    stop_events = threading.Event()

    def _event_poller() -> None:
        # The backend uses `seq > since_seq` (strict gt), so we keep `since_seq` as the last seen seq.
        last_seq = 0
        url = f"{SYNTH_API_BASE}/api/prompt-learning/online/jobs/{job_id}/events"
        headers = {"Authorization": f"Bearer {API_KEY}"}
        while not stop_events.is_set():
            try:
                resp = httpx.get(
                    url, params={"since_seq": last_seq, "limit": 200}, headers=headers, timeout=30.0
                )
                if resp.status_code == 200 and resp.headers.get("content-type", "").startswith(
                    "application/json"
                ):
                    payload = resp.json() or {}
                    events = payload.get("events") or []
                    next_seq = payload.get("next_seq")
                    if isinstance(next_seq, int) and next_seq >= 0:
                        last_seq = max(last_seq, next_seq - 1)

                    for ev in events:
                        if not isinstance(ev, dict):
                            continue
                        event_type = str(ev.get("type") or "")
                        message = str(ev.get("message") or "")
                        data = ev.get("data") or {}
                        if not isinstance(data, dict):
                            data = {}
                        line = _format_event_line(event_type, message, data)
                        if line:
                            print(f"\n  {line}", flush=True)

                        ev_seq = ev.get("seq")
                        if isinstance(ev_seq, int) and ev_seq > last_seq:
                            last_seq = ev_seq
            except Exception:
                pass
            time.sleep(2.0)

    event_thread = threading.Thread(target=_event_poller, daemon=True)
    event_thread.start()

    result = job.poll_until_complete(
        timeout=3600.0,
        interval=3.0,
        progress=False,  # we print events instead
        request_timeout=300.0,  # 5 min timeout for vision model generation + verification
    )
    stop_events.set()
    timings["optimization"] = time.time() - opt_start

    # Display results
    print("\n" + "=" * 80)
    print("RESULTS")
    print("=" * 80)

    if result.succeeded:
        print("Status: SUCCESS")
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
        print("Status: FAILED")
        print(f"Error: {result.error}")

    # Cleanup
    if tunnel:
        print("\nCleaning up tunnel...")
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
