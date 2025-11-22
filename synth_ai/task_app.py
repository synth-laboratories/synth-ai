import tomllib
from dotenv import load_dotenv
from pathlib import Path
from synth_ai.task import InProcessTaskApp
from synth_ai.utils import require_keys
from synth_ai.utils.paths import print_paths_formatted
from synth_ai.utils.train_cfgs import find_train_cfgs_in_cwd, validate_train_cfg
from starlette.types import ASGIApp
import sys
import asyncio
import time


async def run_task_app(
    task_app: ASGIApp,
    *,
    train_config: Path | None,
    dotenv_path: Path | None
) -> None:
    load_dotenv(dotenv_path)
    try:
        require_keys(
            "SYNTH_API_KEY",
            "ENVIRONMENT_API_KEY"
        )
    except Exception as exc:
        print(str(exc))
        print("Run `uvx synth-ai setup` to load required environment variables.")
        sys.exit(1)
    try:
        async with InProcessTaskApp(app=task_app) as ta:
            if not train_config:
                available_cfgs = find_train_cfgs_in_cwd()
                if len(available_cfgs) == 1:
                    train_type, cfg_path_str, _ = available_cfgs[0]
                    train_config = Path(cfg_path_str)
                    print(f"Automatically selected {train_type} training config at", train_config)
                else:
                    if len(available_cfgs) == 0:
                        print("No training config found in cwd.")
                        print("Validate your training config: synth-ai train-cfg check [CFG_PATH]")
                    else:
                        print("Multiple training configs found. Please specify which one to use:")
                        print_paths_formatted(available_cfgs)
                    return None
            train_type = validate_train_cfg(train_config)
            try:
                match train_type:
                    case "prompt":
                        from synth_ai.api.train.prompt_learning import PromptLearningJob
                        from synth_ai.learning.prompt_learning_client import PromptLearningClient

                        job = PromptLearningJob.from_config(config_path=train_config)
                        job_id = job.submit()
                        start_time = time.time()
                        last_status = None
                        def on_status(status):
                            nonlocal last_status
                            elapsed = time.time() - start_time
                            state = status.get("status", "unknown")

                            # Only print if status changed or every 10 seconds
                            if state != last_status or int(elapsed) % 10 == 0:
                                timestamp = time.strftime("%H:%M:%S")
                                progress = status.get("progress", {})
                                best_score = status.get("best_score")
                                
                                if progress:
                                    completed = progress.get("completed", 0)
                                    total = progress.get("total", 0)
                                    if total > 0:
                                        pct = (completed / total) * 100
                                        score_str = f" | Best: {best_score:.3f}" if best_score is not None else ""
                                        print(
                                            f"[{timestamp}] {elapsed:6.1f}s  Status: {state} ({completed}/{total} = {pct:.1f}%){score_str}"
                                        )
                                    else:
                                        score_str = f" | Best: {best_score:.3f}" if best_score is not None else ""
                                        print(f"[{timestamp}] {elapsed:6.1f}s  Status: {state}{score_str}")
                                else:
                                    score_str = f" | Best: {best_score:.3f}" if best_score is not None else ""
                                    print(f"[{timestamp}] {elapsed:6.1f}s  Status: {state}{score_str}")
                                last_status = state

                        await asyncio.get_event_loop().run_in_executor(
                            None,
                            lambda: job.poll_until_complete(
                                timeout=3600.0,
                                interval=5.0,
                                on_status=on_status,
                            ),
                        )
                        total_time = time.time() - start_time
                        print(f"Optimization complete in {total_time:.1f}s")
                        client = PromptLearningClient(synth_key=require_keys("SYNTH_API_KEY")["SYNTH_API_KEY"])
                        prompt_results = await client.get_prompts(job_id)
                        print("=" * 80)
                        print("Results")
                        print("=" * 80 + "\n")

                        if prompt_results.best_score is not None:
                            print(f"Best score: {prompt_results.best_score:.2%}")
                        else:
                            print("Best score: N/A (job may have failed)")

                        # Parse and display candidates info
                        if prompt_results.attempted_candidates is not None:
                            candidates = prompt_results.attempted_candidates
                            if isinstance(candidates, list):
                                if len(candidates) > 0:
                                    # Extract useful stats from candidates
                                    accuracies = [c.get("accuracy", 0.0) for c in candidates if isinstance(c, dict)]
                                    avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
                                    max_accuracy = max(accuracies) if accuracies else 0.0
                                    min_accuracy = min(accuracies) if accuracies else 0.0
                                    
                                    print(f"Total candidates: {len(candidates)}")
                                    if accuracies:
                                        print(f"  Accuracy range: {min_accuracy:.2%} - {max_accuracy:.2%} (avg: {avg_accuracy:.2%})")
                                else:
                                    print("Total candidates: 0 (no candidates evaluated)")
                            else:
                                print(f"Total candidates: {candidates}")
                        else:
                            print("Total candidates: N/A")
                        print()

                        if prompt_results.best_prompt:
                            print("Best prompt:")
                            print("-" * 80)
                            # Extract prompt text
                            if "prompt_sections" in prompt_results.best_prompt:
                                sections = prompt_results.best_prompt["prompt_sections"]
                                prompt_text = "\n\n".join(
                                    [s.get("content", "") for s in sections if s.get("content")]
                                )
                                print(prompt_text[:500])
                                if len(prompt_text) > 500:
                                    print("\n... [truncated]")
                            print()
                    case "rl":
                        return None
                    case "sft":
                        return None
            except Exception as exc:
                raise exc
    except KeyboardInterrupt:
        sys.exit(1)
    except Exception as exc:
        print(f"Error: {exc}")
        sys.exit(1)
