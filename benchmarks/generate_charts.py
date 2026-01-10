#!/usr/bin/env python3
"""Generate charts from benchmark results comparing baseline vs optimized performance."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

# Baseline scores from benchmarks/README.md
BASELINE_SCORES = {
    "banking77": {
        "gpt-4.1-nano": 0.70,
        "gpt-4o-mini": 0.44,
        "gpt-5-nano": 0.54,
    },
    "iris": {
        "gpt-4.1-nano": 0.933,
        "gpt-4o-mini": 0.917,
        "gpt-5-nano": 0.783,
    },
    "hotpotqa": {
        "gpt-4.1-nano": 0.46,
        "gpt-4o-mini": 0.54,
        "gpt-5-nano": 0.58,
    },
    "hover": {
        "gpt-4.1-nano": 0.68,
        "gpt-4o-mini": 0.78,
        "gpt-5-nano": 0.78,
    },
}


def normalize_model_name(model: str) -> str:
    """Normalize model names to match baseline keys."""
    model = model.lower()
    if "gpt-4.1" in model or "gpt41" in model:
        return "gpt-4.1-nano"
    elif "gpt-4o" in model or "gpt4o" in model:
        return "gpt-4o-mini"
    elif "gpt-5" in model or "gpt5" in model:
        return "gpt-5-nano"
    return model


def load_benchmark_results(benchmarks_dir: Path) -> Dict[str, Dict[str, List[float]]]:
    """Load all benchmark results and aggregate by benchmark and model."""
    results: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))

    for benchmark_dir in benchmarks_dir.iterdir():
        if not benchmark_dir.is_dir() or benchmark_dir.name == "__pycache__":
            continue

        benchmark_name = benchmark_dir.name
        results_dir = benchmark_dir / "results"

        if not results_dir.exists():
            continue

        # Load all result JSON files
        for result_file in results_dir.glob("*_result.json"):
            if "summary" in result_file.name:
                continue

            try:
                with open(result_file) as f:
                    data = json.load(f)

                if data.get("status") != "succeeded":
                    continue

                # Try to get score from result file
                score = data.get("best_score")

                # If score is None, try loading from prompt file (for iris)
                if score is None and "prompt_file" in data:
                    prompt_file = Path(data["prompt_file"])
                    if prompt_file.exists():
                        with open(prompt_file) as pf:
                            prompt_data = json.load(pf)
                            score = prompt_data.get("best_score") or prompt_data.get(
                                "train_accuracy"
                            )

                # If still None, try finding corresponding prompt file
                if score is None:
                    prompt_file = result_file.parent / result_file.name.replace(
                        "_result.json", "_prompt.json"
                    )
                    if prompt_file.exists():
                        with open(prompt_file) as pf:
                            prompt_data = json.load(pf)
                            score = prompt_data.get("best_score") or prompt_data.get(
                                "train_accuracy"
                            )

                if score is None:
                    continue

                model = normalize_model_name(data["model"])
                results[benchmark_name][model].append(score)

            except Exception as e:
                print(f"Error loading {result_file}: {e}")
                continue

    return dict(results)


def calculate_averages(results: Dict[str, Dict[str, List[float]]]) -> Dict[str, Dict[str, float]]:
    """Calculate average scores for each benchmark/model combination."""
    averages = {}
    for benchmark, models in results.items():
        averages[benchmark] = {}
        for model, scores in models.items():
            if scores:
                averages[benchmark][model] = sum(scores) / len(scores)
    return averages


def generate_chart(
    results: Dict[str, Dict[str, List[float]]],
    averages: Dict[str, Dict[str, float]],
    output_path: Path,
    dark_mode: bool = False,
):
    """Generate a bar chart comparing baseline vs optimized performance."""
    # Aggregate data across all benchmarks
    benchmark_names = ["banking77", "iris", "hotpotqa", "hover"]
    model_names = ["gpt-4.1-nano", "gpt-4o-mini", "gpt-5-nano"]
    model_labels = ["GPT-4.1 Nano", "GPT-4o Mini", "GPT-5 Nano"]

    # Calculate overall averages (weighted by number of runs)
    baseline_scores = []
    optimized_scores = []
    optimized_stds = []

    for model in model_names:
        baseline_total = 0
        baseline_count = 0
        optimized_total = 0
        optimized_count = 0
        optimized_scores_list = []

        for benchmark in benchmark_names:
            if benchmark in BASELINE_SCORES and model in BASELINE_SCORES[benchmark]:
                baseline_total += BASELINE_SCORES[benchmark][model]
                baseline_count += 1

            if benchmark in results and model in results[benchmark]:
                scores = results[benchmark][model]
                if scores:
                    optimized_total += sum(scores)
                    optimized_count += len(scores)
                    optimized_scores_list.extend(scores)

        baseline_avg = baseline_total / baseline_count if baseline_count > 0 else 0
        optimized_avg = optimized_total / optimized_count if optimized_count > 0 else baseline_avg
        optimized_std = np.std(optimized_scores_list) if optimized_scores_list else 0

        baseline_scores.append(baseline_avg)
        optimized_scores.append(optimized_avg)
        optimized_stds.append(optimized_std)

    # Set up the plot style
    if dark_mode:
        plt.style.use("dark_background")
        bg_color = "#1e1e1e"
        text_color = "#ffffff"
        bar_color_baseline = "#4a5568"
        bar_color_optimized = "#48bb78"
    else:
        plt.style.use("default")
        bg_color = "#ffffff"
        text_color = "#000000"
        bar_color_baseline = "#cbd5e0"
        bar_color_optimized = "#38a169"

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)

    x = np.arange(len(model_labels))
    width = 0.35

    # Create bars
    bars1 = ax.bar(
        x - width / 2,
        [s * 100 for s in baseline_scores],
        width,
        label="Baseline",
        color=bar_color_baseline,
        alpha=0.8,
    )
    bars2 = ax.bar(
        x + width / 2,
        [s * 100 for s in optimized_scores],
        width,
        label="GEPA Optimized",
        color=bar_color_optimized,
        alpha=0.8,
    )

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                color=text_color,
                fontsize=10,
                fontweight="bold",
            )

    # Customize the plot
    ax.set_ylabel("Accuracy (%)", color=text_color, fontsize=12, fontweight="bold")
    ax.set_xlabel("Model", color=text_color, fontsize=12, fontweight="bold")
    ax.set_title(
        "Average Accuracy on LangProBe Prompt Optimization Benchmarks",
        color=text_color,
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, color=text_color, fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, linestyle="--", color=text_color)
    ax.legend(
        loc="upper left",
        frameon=True,
        facecolor=bg_color,
        edgecolor=text_color,
        labelcolor=text_color,
        fontsize=11,
    )

    # Set tick colors
    ax.tick_params(colors=text_color)
    for spine in ax.spines.values():
        spine.set_color(text_color)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=bg_color)
    plt.close()


def main():
    """Main function to generate charts."""
    benchmarks_dir = Path(__file__).parent
    assets_dir = benchmarks_dir.parent / "assets"

    # Ensure assets directory exists
    assets_dir.mkdir(exist_ok=True)

    # Load results
    print("Loading benchmark results...")
    results = load_benchmark_results(benchmarks_dir)
    averages = calculate_averages(results)

    # Print summary
    print("\nBenchmark Results Summary:")
    for benchmark, models in sorted(results.items()):
        print(f"\n{benchmark}:")
        for model, scores in sorted(models.items()):
            avg = sum(scores) / len(scores) if scores else 0
            baseline = BASELINE_SCORES.get(benchmark, {}).get(model, 0)
            improvement = (avg - baseline) * 100
            print(
                f"  {model}: {avg * 100:.1f}% (baseline: {baseline * 100:.1f}%, +{improvement:.1f}%)"
            )

    # Generate charts
    print("\nGenerating charts...")
    generate_chart(
        results,
        averages,
        assets_dir / "langprobe_v2_light.png",
        dark_mode=False,
    )
    generate_chart(
        results,
        averages,
        assets_dir / "langprobe_v2_dark.png",
        dark_mode=True,
    )
    print(f"\nCharts saved to {assets_dir}/")


if __name__ == "__main__":
    main()
