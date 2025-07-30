```python
#!/usr/bin/env python3
"""
Vertex AI Fine‑Tuning Script (Gemini Flash)
==========================================
Uploads a JSONL file to GCS, starts a Gemini‑Flash tuning job, and polls until completion.
"""

import os
import sys
import time
import argparse
import json
import random
from pathlib import Path
from typing import Optional

# --- Lazy‑install required packages -----------------------------------------
def _lazy_import(pkg: str, pip_name: Optional[str] = None):
    try:
        return __import__(pkg)
    except ImportError:                               # pragma: no cover
        print(f"📦 Installing {pkg}…")
        os.system(f"pip install {pip_name or pkg}")
        return __import__(pkg)

storage   = _lazy_import("google.cloud.storage")
aiplatform = _lazy_import("google.cloud.aiplatform")
vertexai   = _lazy_import("vertexai")
generative_models = _lazy_import("vertexai.preview.generative_models")

tiktoken  = _lazy_import("tiktoken")

# --- Helpers ----------------------------------------------------------------
def encoding_for(model: str = "cl100k_base"):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding("cl100k_base")

def analyze_jsonl_tokens(file_path: Path, model: str) -> tuple[int, int, float]:
    enc = encoding_for(model)
    inp_tok = out_tok = lines = 0
    with open(file_path, "r") as fh:
        for ln, raw in enumerate(fh, 1):
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            msgs = data.get("messages", [])
            input_text = " ".join(m["content"] for m in msgs[:-1] if m.get("content"))
            output_text = msgs[-1]["content"] if msgs else ""
            inp_tok += len(enc.encode(input_text))
            out_tok += len(enc.encode(output_text))
            lines += 1
    avg = (inp_tok + out_tok) / lines if lines else 0
    print(f"🔍 {lines:,} lines ‑ {inp_tok+out_tok:,} tokens (avg ≈ {avg:.1f}/line)")
    return lines, inp_tok + out_tok, avg

def create_subset_file(src: Path, n: int) -> Path:
    dst = src.with_name(f"{src.stem}_subset_{n}.jsonl")
    lines = [l.strip() for l in src.open() if l.strip()]
    if n < len(lines):
        lines = random.sample(lines, n)
    with dst.open("w") as fh:
        fh.write("\n".join(lines) + "\n")
    return dst

def upload_to_gcs(bucket: str, file_path: Path) -> str:
    client = storage.Client()
    bkt = client.bucket(bucket)
    blob = bkt.blob(file_path.name)
    blob.upload_from_filename(file_path)
    uri = f"gs://{bucket}/{file_path.name}"
    print(f"📤 Uploaded to {uri}")
    return uri

def start_tuning_job(project: str, region: str, gcs_uri: str,
                     base_model: str, display_name: str,
                     epochs: int, lr_mult: float):
    vertexai.init(project=project, location=region)
    model = generative_models.GenerativeModel(base_model)
    job = model.tune_model(
        training_data=generative_models.FileData(
            path=gcs_uri, mime_type="jsonl"),
        tuned_model_display_name=display_name,
        hyperparameters={
            "epochs": epochs,
            "learning_rate_multiplier": lr_mult,
        },
    )
    return job

def wait(job, poll: int):
    print("⏳ Waiting for tuning to finish…")
    while True:
        job.refresh()
        state = job.state
        print(f"   Status: {state}")
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            break
        time.sleep(poll)
    return state

# --- Main -------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Vertex AI Gemini‑Flash Fine‑Tuning")
    ap.add_argument("jsonl_file", type=Path, help="Training data (JSONL)")
    ap.add_argument("--project", required=True)
    ap.add_argument("--region", default="us‑central1")
    ap.add_argument("--bucket", required=True, help="GCS bucket for upload")
    ap.add_argument("--model", default="gemini‑1.0‑flash")
    ap.add_argument("--display-name", default="gemini-flash-tuned")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr-mult", type=float, default=0.05)
    ap.add_argument("--poll-interval", type=int, default=60)
    args = ap.parse_args()

    if not args.jsonl_file.exists():
        sys.exit("❌ Training file not found.")

    lines, total_tok, _ = analyze_jsonl_tokens(args.jsonl_file, args.model)

    subset = input("Use subset? (y/N): ").strip().lower()
    train_file = args.jsonl_file
    if subset == "y":
        n = int(input(f"Lines (1‑{lines}): "))
        train_file = create_subset_file(args.jsonl_file, n)

    gcs_uri = upload_to_gcs(args.bucket, train_file)
    job = start_tuning_job(args.project, args.region, gcs_uri,
                           args.model, args.display_name,
                           args.epochs, args.lr_mult)
    final_state = wait(job, args.poll_interval)

    if final_state == "SUCCEEDED":
        print(f"🎉 Tuned model: {job.resource_name}")
        print("\n📝 Usage example:")
        print(f"""
from vertexai.preview import generative_models
vertexai.init(project="{args.project}", location="{args.region}")
model = generative_models.GenerativeModel("{job.resource_name}")
resp = model.generate_content("Hello!")
print(resp.text)
""")
    else:
        sys.exit("❌ Tuning did not succeed.")

if __name__ == "__main__":
    main()
