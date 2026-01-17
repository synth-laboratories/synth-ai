"""Web Design demo task app (import-safe).

This module intentionally has **no side effects** at import time:
- no backend health checks
- no key minting
- no printing

It exists so scripts (demo, eval jobs, tests) can reuse the dataset + Local API without
accidentally hitting production.
"""

import base64
import io
import logging
import os
from pathlib import Path
from typing import Any, Iterable

import httpx
from datasets import Image as HFImage
from datasets import load_dataset, load_from_disk
from PIL import Image
from synth_ai.core.urls import join_url
from synth_ai.sdk.localapi import LocalAPIConfig, create_local_api
from synth_ai.sdk.task.contracts import (
    RolloutMetrics,
    RolloutRequest,
    RolloutResponse,
    TaskInfo,
)
from synth_ai.sdk.task.rubrics import Criterion, Rubric
from synth_ai.sdk.task.server import RubricBundle
from synth_ai.sdk.task.trace_correlation_helpers import (
    build_trace_payload,
    extract_trace_correlation_id,
)

logger = logging.getLogger(__name__)

demo_dir = Path(__file__).resolve().parent

APP_ID = "web_design_generator"
APP_NAME = "Web Design Style Optimization"

HF_DATASET_ID_ENV = "SYNTH_WEB_DESIGN_DATASET"
HF_DATASET_REVISION_ENV = "SYNTH_WEB_DESIGN_DATASET_REVISION"
HF_DATASET_SPLIT = "train"

DEFAULT_HF_DATASET_ID = "JoshPurtell/web-design-screenshots"
DEFAULT_SITE_FILTER = "astral"
DEFAULT_MAX_EXAMPLES = int(os.environ.get("SYNTH_WEB_DESIGN_MAX_EXAMPLES", "8"))
DEFAULT_MAX_IMAGE_PIXELS = int(
    os.environ.get("SYNTH_WEB_DESIGN_MAX_IMAGE_PIXELS", "12000000")
)  # 12MP

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


class WebDesignDataset:
    """Subset of website pages used for style optimization."""

    def __init__(
        self,
        resize_size: int = 384,
        site_filter: str = DEFAULT_SITE_FILTER,
        max_examples: int = DEFAULT_MAX_EXAMPLES,
    ) -> None:
        self._examples: list[dict[str, Any]] | None = None
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

    def _load(self) -> None:
        if self._examples is not None:
            return

        dataset_id_env = (os.environ.get(HF_DATASET_ID_ENV) or "").strip()
        if dataset_id_env.lower() in {"local", "disk"}:
            dataset_id = ""
        else:
            dataset_id = dataset_id_env or DEFAULT_HF_DATASET_ID
        dataset_revision = os.environ.get(HF_DATASET_REVISION_ENV, "").strip() or None

        if dataset_id:
            logger.info(
                "Loading web-design dataset from Hub: %s (split=%s)", dataset_id, HF_DATASET_SPLIT
            )
            dataset = load_dataset(dataset_id, split=HF_DATASET_SPLIT, revision=dataset_revision)
        else:
            dataset_path = demo_dir / "hf_dataset"
            if not dataset_path.exists():
                raise RuntimeError(
                    "No dataset configured.\n\n"
                    f"- Set {HF_DATASET_ID_ENV} to a Hugging Face dataset (e.g. org/web-design-screenshots)\n"
                    f"- OR create a local dataset at {dataset_path} via create_hf_dataset.py\n"
                )
            logger.info("Loading web-design dataset from disk: %s", dataset_path)
            dataset = load_from_disk(str(dataset_path))

        # Avoid decoding images during filtering (can trigger DecompressionBombWarning)
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

        logger.info("Loaded %d '%s' pages", len(dataset), self._site_filter)

        selected: list[dict[str, Any]] = []
        cursor = 0
        max_to_scan = min(len(dataset), max(self._max_examples * 10, self._max_examples))

        while len(selected) < self._max_examples and cursor < max_to_scan:
            ex = dict(dataset[cursor])
            cursor += 1

            site_name = (ex.get("site_name") or "site").replace("/", "_").replace(" ", "_")
            page_name = (
                (ex.get("page_name") or f"page_{cursor:03d}").replace("/", "_").replace(" ", "_")
            )

            resized_path = self._resized_dir / f"{site_name}_{page_name}_{len(selected):03d}.png"

            if resized_path.exists():
                try:
                    with Image.open(resized_path) as cached:
                        if (cached.size[0] * cached.size[1]) > self._max_image_pixels or max(
                            cached.size
                        ) > self._resize_size:
                            resized_path.unlink(missing_ok=True)
                except Exception:
                    resized_path.unlink(missing_ok=True)

            if not resized_path.exists():
                img_obj = ex.get("image")
                if isinstance(img_obj, Image.Image):
                    img = img_obj
                elif isinstance(img_obj, dict):
                    raw_bytes = img_obj.get("bytes")
                    raw_path = img_obj.get("path")
                    if isinstance(raw_bytes, (bytes, bytearray)) and raw_bytes:
                        img = Image.open(io.BytesIO(raw_bytes))
                    elif isinstance(raw_path, str) and raw_path:
                        img = Image.open(raw_path)
                    else:
                        continue
                else:
                    continue

                # Safety: skip enormous images
                w, h = img.size
                if (w * h) > self._max_image_pixels:
                    continue

                if img.mode != "RGB":
                    img = img.convert("RGB")
                img.thumbnail((self._resize_size, self._resize_size), Image.Resampling.LANCZOS)
                img.save(resized_path, format="PNG", optimize=True)

            ex_out = {
                "index": len(selected),
                "functional_description": ex.get("functional_description") or "",
                "image_path": str(resized_path),
                "resized_image_path": str(resized_path),
                "site_name": ex.get("site_name") or "",
                "page_name": ex.get("page_name") or "",
            }
            selected.append(ex_out)

        if not selected:
            raise RuntimeError("No usable examples found after filtering/safety checks.")

        self._examples = selected

    def size(self) -> int:
        self._load()
        assert self._examples is not None
        return len(self._examples)

    def sample(self, seed: int) -> dict[str, Any]:
        self._load()
        assert self._examples is not None
        idx = seed % len(self._examples)
        return dict(self._examples[idx])


def create_web_design_local_api(style_prompt: str) -> Any:
    """Create a Local API task app for the web design demo."""

    dataset = WebDesignDataset()
    dataset._load()

    def provide_taskset_description() -> dict[str, Any]:
        return {"splits": ["train"], "sizes": {"train": dataset.size()}}

    def provide_task_instances(seeds: Iterable[int]):
        for seed in seeds:
            sample = dataset.sample(int(seed))

            # Encode resized image to data URL for the verifier payload
            resized_image_path = Path(sample["resized_image_path"])
            original_data_url: str | None = None
            try:
                with Image.open(resized_image_path) as img:
                    if img.mode != "RGB":
                        img = img.convert("RGB")
                    buffer = io.BytesIO()
                    img.save(buffer, format="PNG", optimize=True)
                    img_b64 = base64.b64encode(buffer.getvalue()).decode("ascii")
                    original_data_url = f"data:image/png;base64,{img_b64}"
            except Exception:
                original_data_url = None

            yield TaskInfo(
                task={"id": APP_ID, "name": APP_NAME},
                dataset={"id": APP_ID, "split": "train", "index": sample["index"]},
                environment="web_design",
                inference={},
                limits={"max_turns": 1},
                task_metadata={
                    "page": f"{sample['site_name']}/{sample['page_name']}",
                    "description_length": len(sample["functional_description"]),
                    "functional_description": sample["functional_description"],
                    "original_image_url": original_data_url,
                    "original_image_path": sample["resized_image_path"],
                },
            )

    async def run_rollout(request: RolloutRequest, fastapi_request: Any) -> RolloutResponse:
        # The backend verifier provides reward. Task app performs the generation through the interceptor.
        seed = int(request.env.seed)
        sample = dataset.sample(seed)

        policy_cfg = dict(request.policy.config or {})
        inference_url = str(policy_cfg.get("inference_url") or "")
        if not inference_url:
            raise ValueError("inference_url is required in policy config")

        env_api_key = (os.environ.get("ENVIRONMENT_API_KEY") or "").strip()
        if not env_api_key:
            raise RuntimeError(
                "ENVIRONMENT_API_KEY is required in environment for calling inference interceptor"
            )

        model = str(policy_cfg.get("model") or "gemini-2.5-flash-image")

        full_prompt = (
            f"{style_prompt}\n\n"
            "Generate a webpage screenshot based on this functional description:\n\n"
            f"{sample['functional_description']}\n\n"
            "Apply the visual style guidelines to match the original design."
        )

        messages = [{"role": "user", "content": full_prompt}]

        # Call policy model through inference interceptor (OpenAI-compatible endpoint)
        llm_response: dict[str, Any] = {}
        async with httpx.AsyncClient(timeout=180.0) as client:
            resp = await client.post(
                join_url(inference_url, "/chat/completions"),
                json={
                    "model": model,
                    "messages": messages,
                    "temperature": 0.7,
                },
                headers={
                    "Authorization": f"Bearer {env_api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            llm_response = resp.json()

        # Ensure an image was produced (multipart content)
        generated_bytes: bytes | None = None
        choices = llm_response.get("choices", [])
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        image_url = part.get("image_url", {}).get("url", "")
                        if isinstance(image_url, str) and image_url.startswith("data:image/"):
                            _, data = image_url.split(",", 1)
                            generated_bytes = base64.b64decode(data)
                            break
            elif isinstance(content, str) and content.startswith("data:image/"):
                _, data = content.split(",", 1)
                generated_bytes = base64.b64decode(data)

        if not generated_bytes:
            raise ValueError("No image data in policy model response")

        # Save generated image to disk for inspection
        output_dir = Path(__file__).parent / "generated_images"
        output_dir.mkdir(exist_ok=True)
        run_id_short = (
            request.trace_correlation_id[:8] if request.trace_correlation_id else "unknown"
        )
        img_path = output_dir / f"seed_{seed}_run_{run_id_short}.png"
        with open(img_path, "wb") as f:
            f.write(generated_bytes)
        logger.info(f"Saved generated image to {img_path}")

        trace_correlation_id = extract_trace_correlation_id(
            policy_config=policy_cfg,
            inference_url=inference_url,
        )

        trace_payload = build_trace_payload(
            messages=messages,
            response=llm_response,
            correlation_id=trace_correlation_id,
            metadata={"run_id": request.trace_correlation_id, "seed": seed},
        )

        return RolloutResponse(
            run_id=request.trace_correlation_id,
            reward_info=RolloutMetrics(outcome_reward=0.0),
            trace=trace_payload,
            trace_correlation_id=trace_correlation_id,
            inference_url=inference_url,
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
