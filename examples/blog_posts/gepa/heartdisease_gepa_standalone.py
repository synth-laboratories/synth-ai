#!/usr/bin/env python3
"""
Standalone GEPA Script for Heart Disease Classification

This script demonstrates a complete GEPA workflow without using the CLI:
1. Start a local task app server for heart disease
2. Submit a GEPA optimization job via the API
3. Poll until the job completes
4. Extract the best prompt candidate
5. Evaluate the optimized prompt on a held-out test set
6. Compare baseline vs optimized performance

Usage:
    python heartdisease_gepa_standalone.py

Requirements:
    - GROQ_API_KEY environment variable set
    - synth-ai package installed
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# Add parent directories to path for imports
script_dir = Path(__file__).resolve().parent
examples_dir = script_dir.parent.parent
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

from datasets import load_dataset
from dotenv import load_dotenv

from synth_ai.api.train.prompt_learning import PromptLearningJob
from synth_ai.baseline import TaskResult

# Load environment variables
load_dotenv()


class HeartDiseaseEvaluator:
    """Evaluator for heart disease classification with a specific prompt."""

    def __init__(self, model: str = "llama-3.1-8b-instant", provider: str = "groq"):
        self.model = model
        self.provider = provider
        self.dataset = load_dataset("buio/heart-disease", split="train")

    def _extract_features(self, example: Dict[str, Any]) -> tuple[str, str]:
        """Extract features and label from a dataset example."""
        features = {}
        label = None

        for key, value in example.items():
            if key in ("target", "label", "class", "disease"):
                label = str(int(value)) if isinstance(value, (int, float)) else str(value)
            else:
                features[key] = value

        feature_text = "\n".join([f"{key}: {value}" for key, value in features.items()])
        return feature_text, label or "0"

    async def evaluate_prompt(
        self,
        system_prompt: str,
        user_prompt_template: str,
        test_seeds: list[int],
    ) -> Dict[str, Any]:
        """
        Evaluate a prompt on test seeds.

        Args:
            system_prompt: System prompt text
            user_prompt_template: User prompt template (with {features} placeholder)
            test_seeds: List of test seed indices

        Returns:
            Dictionary with accuracy, predictions, and timing info
        """
        import httpx

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY environment variable is required")

        base_url = "https://api.groq.com/openai/v1"
        model_name = self.model

        # Tool definition
        tool = {
            "type": "function",
            "function": {
                "name": "heart_disease_classify",
                "description": "Classify whether a patient has heart disease",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "classification": {
                            "type": "string",
                            "enum": ["0", "1"],
                            "description": "The classification: '1' for heart disease, '0' for no heart disease",
                        }
                    },
                    "required": ["classification"],
                },
            },
        }

        correct = 0
        predictions = []
        start_time = time.time()

        async with httpx.AsyncClient(timeout=30.0) as http_client:
            for seed in test_seeds:
                example = self.dataset[seed]
                feature_text, expected_label = self._extract_features(example)

                # Build messages
                user_prompt = user_prompt_template.replace("{features}", feature_text)
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]

                # Call API
                try:
                    resp = await http_client.post(
                        f"{base_url}/chat/completions",
                        json={
                            "model": model_name,
                            "messages": messages,
                            "tools": [tool],
                            "tool_choice": {"type": "function", "function": {"name": "heart_disease_classify"}},
                            "temperature": 0.0,
                            "max_tokens": 128,
                        },
                        headers={"Authorization": f"Bearer {api_key}"},
                    )
                    response = resp.json()

                    # Extract prediction
                    predicted_label = ""
                    if "choices" in response and len(response["choices"]) > 0:
                        message = response["choices"][0].get("message", {})
                        tool_calls = message.get("tool_calls", [])
                        if tool_calls:
                            args = tool_calls[0]["function"].get("arguments", "")
                            if isinstance(args, str):
                                args = json.loads(args)
                            predicted_label = args.get("classification", "") if isinstance(args, dict) else ""

                    # Normalize prediction
                    predicted_label = predicted_label.strip()
                    if predicted_label not in ("0", "1"):
                        for char in predicted_label:
                            if char in ("0", "1"):
                                predicted_label = char
                                break

                    is_correct = predicted_label == expected_label
                    if is_correct:
                        correct += 1

                    predictions.append({
                        "seed": seed,
                        "expected": expected_label,
                        "predicted": predicted_label,
                        "correct": is_correct,
                    })

                except Exception as e:
                    print(f"Error evaluating seed {seed}: {e}")
                    predictions.append({
                        "seed": seed,
                        "expected": expected_label,
                        "predicted": "",
                        "correct": False,
                        "error": str(e),
                    })

        elapsed_time = time.time() - start_time
        accuracy = correct / len(test_seeds) if test_seeds else 0.0

        return {
            "accuracy": accuracy,
            "correct": correct,
            "total": len(test_seeds),
            "predictions": predictions,
            "elapsed_time": elapsed_time,
        }


def start_task_app_server(port: int = 8114) -> subprocess.Popen:
    """
    Start a local task app server for heart disease.

    Returns:
        subprocess.Popen object for the server process
    """
    print(f"\n{'='*80}")
    print("Starting Task App Server")
    print(f"{'='*80}\n")

    baseline_file = Path(__file__).parent / "heartdisease_baseline.py"
    if not baseline_file.exists():
        raise FileNotFoundError(f"Baseline file not found: {baseline_file}")

    # Start server
    cmd = [
        "uvx",
        "synth-ai",
        "serve",
        str(baseline_file),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]

    print(f"Command: {' '.join(cmd)}")
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    # Wait for server to be ready (check for health endpoint)
    import httpx
    max_wait = 30
    wait_interval = 1
    for i in range(max_wait):
        try:
            with httpx.Client() as client:
                resp = client.get(f"http://127.0.0.1:{port}/health", timeout=2.0)
                if resp.status_code == 200:
                    print(f"✓ Task app server ready at http://127.0.0.1:{port}")
                    return process
        except Exception:
            pass
        time.sleep(wait_interval)
        print(f"Waiting for server to start... ({i+1}/{max_wait})")

    # If we get here, server didn't start
    process.terminate()
    raise RuntimeError("Task app server failed to start")


def parse_prompt_from_job_results(results: Dict[str, Any]) -> tuple[str, str]:
    """
    Parse system prompt and user prompt template from job results.

    Args:
        results: Results from PromptLearningJob.get_results()

    Returns:
        Tuple of (system_prompt, user_prompt_template)
    """
    # Get the best prompt
    best_prompt = results.get("best_prompt")
    if not best_prompt:
        raise ValueError("No best_prompt found in results")

    # Extract messages
    messages = best_prompt.get("messages", [])
    if not messages:
        raise ValueError("No messages found in best_prompt")

    # Find system and user messages
    system_prompt = ""
    user_prompt_template = ""

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("pattern") or msg.get("content", "")

        if role == "system":
            system_prompt = content
        elif role == "user":
            user_prompt_template = content

    return system_prompt, user_prompt_template


async def main():
    """Main execution flow."""
    print("\n" + "="*80)
    print("Heart Disease GEPA Standalone Demo")
    print("="*80 + "\n")

    # Check environment
    if not os.getenv("GROQ_API_KEY"):
        print("Error: GROQ_API_KEY environment variable is required")
        sys.exit(1)

    # Configuration
    config_path = Path(__file__).parent / "configs" / "heartdisease_gepa_local.toml"
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    task_app_port = 8114
    test_seeds = list(range(80, 100))  # Hold-out test set (20 examples)

    # Step 1: Start task app server
    print("\n" + "="*80)
    print("Step 1: Starting Task App Server")
    print("="*80 + "\n")

    server_process = None
    try:
        server_process = start_task_app_server(port=task_app_port)

        # Step 2: Submit GEPA job
        print("\n" + "="*80)
        print("Step 2: Submitting GEPA Optimization Job")
        print("="*80 + "\n")

        job = PromptLearningJob.from_config(
            config_path=config_path,
            backend_url=os.getenv("BACKEND_BASE_URL", "http://localhost:8000"),
            api_key=os.getenv("SYNTH_API_KEY", "test"),
            task_app_api_key=os.getenv("ENVIRONMENT_API_KEY", "test"),
        )

        job_id = job.submit()
        print(f"✓ Job submitted: {job_id}")

        # Step 3: Poll until complete
        print("\n" + "="*80)
        print("Step 3: Polling Job Status")
        print("="*80 + "\n")

        result = job.poll_until_complete(
            timeout=1800.0,  # 30 minutes
            interval=5.0,
            on_status=lambda status: print(f"Status: {status.get('status', 'unknown')}"),
        )

        print(f"\n✓ Job complete! Final status: {result.get('status')}")

        # Step 4: Extract best prompt
        print("\n" + "="*80)
        print("Step 4: Extracting Best Prompt")
        print("="*80 + "\n")

        results = job.get_results()
        best_score = results.get("best_score", 0.0)
        print(f"Best validation score: {best_score:.4f} ({best_score*100:.2f}%)")

        system_prompt, user_prompt_template = parse_prompt_from_job_results(results)

        print("\nOptimized System Prompt:")
        print("-" * 80)
        print(system_prompt)
        print("-" * 80)

        print("\nOptimized User Prompt Template:")
        print("-" * 80)
        print(user_prompt_template)
        print("-" * 80)

        # Step 5: Evaluate on test set
        print("\n" + "="*80)
        print("Step 5: Evaluating on Test Set")
        print("="*80 + "\n")

        evaluator = HeartDiseaseEvaluator()

        # Baseline evaluation
        print("Evaluating baseline prompt...")
        baseline_system = """You are a medical classification assistant. Based on the patient's features, classify whether they have heart disease. Respond with '1' for heart disease or '0' for no heart disease.

You have access to the function `heart_disease_classify` which accepts your predicted classification. Call this tool with your classification when you're ready to submit your answer."""
        baseline_user = "Patient Features:\n{features}\n\nClassify: Does this patient have heart disease? Respond with '1' for yes or '0' for no."

        baseline_results = await evaluator.evaluate_prompt(
            system_prompt=baseline_system,
            user_prompt_template=baseline_user,
            test_seeds=test_seeds,
        )

        # Optimized evaluation
        print("Evaluating optimized prompt...")
        optimized_results = await evaluator.evaluate_prompt(
            system_prompt=system_prompt,
            user_prompt_template=user_prompt_template,
            test_seeds=test_seeds,
        )

        # Step 6: Report results
        print("\n" + "="*80)
        print("Final Results")
        print("="*80 + "\n")

        print(f"Test Set Size: {len(test_seeds)} examples")
        print()
        print(f"Baseline Accuracy:  {baseline_results['accuracy']:.4f} ({baseline_results['accuracy']*100:.2f}%)")
        print(f"Optimized Accuracy: {optimized_results['accuracy']:.4f} ({optimized_results['accuracy']*100:.2f}%)")
        print()
        improvement = optimized_results['accuracy'] - baseline_results['accuracy']
        print(f"Improvement: {improvement:+.4f} ({improvement*100:+.2f}%)")
        print()

        # Save results to file
        output_dir = Path(__file__).parent / "results" / "heartdisease_gepa_standalone"
        output_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_dir / "results.json"
        with open(output_file, "w") as f:
            json.dump({
                "job_id": job_id,
                "best_validation_score": best_score,
                "baseline_test_accuracy": baseline_results['accuracy'],
                "optimized_test_accuracy": optimized_results['accuracy'],
                "improvement": improvement,
                "test_set_size": len(test_seeds),
                "optimized_prompts": {
                    "system": system_prompt,
                    "user": user_prompt_template,
                },
                "baseline_predictions": baseline_results['predictions'],
                "optimized_predictions": optimized_results['predictions'],
            }, f, indent=2)

        print(f"✓ Results saved to: {output_file}")
        print("\n" + "="*80)
        print("Demo Complete!")
        print("="*80 + "\n")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup: Stop task app server
        if server_process:
            print("\nStopping task app server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server_process.kill()
            print("✓ Task app server stopped")


if __name__ == "__main__":
    asyncio.run(main())
