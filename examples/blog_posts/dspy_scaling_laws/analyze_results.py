"""Analyze and visualize DSPy scaling experiment results."""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Any

sns.set_style("whitegrid")
sns.set_palette("husl")


def load_all_results(results_dir: Path) -> pd.DataFrame:
    """Load all experiment results into a DataFrame."""

    results = []

    # Look for results.json files
    for results_file in results_dir.rglob("results.json"):
        try:
            with open(results_file) as f:
                data = json.load(f)
                results.append(data)
        except Exception as e:
            print(f"Warning: Could not load {results_file}: {e}")

    if not results:
        print(f"No results found in {results_dir}")
        return pd.DataFrame()

    df = pd.DataFrame(results)
    return df


def create_scaling_curves(df: pd.DataFrame, output_dir: Path):
    """Create scaling curves showing performance vs pipeline complexity."""

    output_dir.mkdir(parents=True, exist_ok=True)

    # Overall scaling curve (all benchmarks)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for optimizer_idx, optimizer in enumerate(["gepa", "mipro"]):
        ax = axes[optimizer_idx]
        opt_data = df[df["optimizer"] == optimizer]

        for benchmark in df["benchmark"].unique():
            bench_data = opt_data[opt_data["benchmark"] == benchmark].sort_values("num_steps")

            ax.plot(
                bench_data["num_steps"],
                bench_data["final_score"],
                marker='o',
                label=benchmark.capitalize(),
                linewidth=2,
                markersize=8,
            )

        ax.set_xlabel("Pipeline Steps", fontsize=12)
        ax.set_ylabel("Final Score (Accuracy)", fontsize=12)
        ax.set_title(f"{optimizer.upper()} Optimizer", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks([1, 2, 3, 5])

    plt.tight_layout()
    plt.savefig(output_dir / "scaling_curves_overall.png", dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_dir / 'scaling_curves_overall.png'}")
    plt.close()

    # Per-benchmark comparison
    for benchmark in df["benchmark"].unique():
        fig, ax = plt.subplots(figsize=(8, 6))
        bench_data = df[df["benchmark"] == benchmark]

        for optimizer in ["gepa", "mipro"]:
            opt_data = bench_data[bench_data["optimizer"] == optimizer].sort_values("num_steps")

            ax.plot(
                opt_data["num_steps"],
                opt_data["final_score"],
                marker='o',
                label=optimizer.upper(),
                linewidth=2,
                markersize=8,
            )

        ax.set_xlabel("Pipeline Steps", fontsize=12)
        ax.set_ylabel("Final Score (Accuracy)", fontsize=12)
        ax.set_title(f"{benchmark.capitalize()} Scaling", fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks([1, 2, 3, 5])

        plt.tight_layout()
        plt.savefig(output_dir / f"scaling_curve_{benchmark}.png", dpi=300, bbox_inches='tight')
        print(f"📊 Saved: {output_dir / f'scaling_curve_{benchmark}.png'}")
        plt.close()


def create_improvement_heatmap(df: pd.DataFrame, output_dir: Path):
    """Create heatmap of improvements over baseline."""

    # Calculate improvement percentage
    df["improvement_pct"] = (df["improvement"] / df["baseline_score"] * 100).round(1)

    # Create pivot table for heatmap
    for optimizer in ["gepa", "mipro"]:
        opt_data = df[df["optimizer"] == optimizer]

        pivot = opt_data.pivot_table(
            values="improvement_pct",
            index="benchmark",
            columns="num_steps",
            aggfunc="mean"
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(
            pivot,
            annot=True,
            fmt=".1f",
            cmap="RdYlGn",
            center=0,
            cbar_kws={"label": "Improvement %"},
            ax=ax,
        )

        ax.set_title(f"{optimizer.upper()}: Improvement over Baseline (%)", fontsize=14, fontweight='bold')
        ax.set_xlabel("Pipeline Steps", fontsize=12)
        ax.set_ylabel("Benchmark", fontsize=12)

        plt.tight_layout()
        plt.savefig(output_dir / f"improvement_heatmap_{optimizer}.png", dpi=300, bbox_inches='tight')
        print(f"📊 Saved: {output_dir / f'improvement_heatmap_{optimizer}.png'}")
        plt.close()


def create_optimizer_comparison(df: pd.DataFrame, output_dir: Path):
    """Create comparison between optimizers."""

    fig, axes = plt.subplots(1, len(df["num_steps"].unique()), figsize=(16, 5))

    for step_idx, steps in enumerate(sorted(df["num_steps"].unique())):
        ax = axes[step_idx] if len(df["num_steps"].unique()) > 1 else axes
        step_data = df[df["num_steps"] == steps]

        # Create grouped bar chart
        benchmarks = step_data["benchmark"].unique()
        x = range(len(benchmarks))
        width = 0.35

        gepa_scores = [step_data[(step_data["benchmark"] == b) & (step_data["optimizer"] == "gepa")]["final_score"].values[0]
                      if len(step_data[(step_data["benchmark"] == b) & (step_data["optimizer"] == "gepa")]) > 0 else 0
                      for b in benchmarks]
        mipro_scores = [step_data[(step_data["benchmark"] == b) & (step_data["optimizer"] == "mipro")]["final_score"].values[0]
                       if len(step_data[(step_data["benchmark"] == b) & (step_data["optimizer"] == "mipro")]) > 0 else 0
                       for b in benchmarks]

        ax.bar([i - width/2 for i in x], gepa_scores, width, label='GEPA', alpha=0.8)
        ax.bar([i + width/2 for i in x], mipro_scores, width, label='MIPRO', alpha=0.8)

        ax.set_xlabel("Benchmark", fontsize=10)
        ax.set_ylabel("Final Score", fontsize=10)
        ax.set_title(f"{steps}-Step Pipeline", fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([b.capitalize() for b in benchmarks], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / "optimizer_comparison.png", dpi=300, bbox_inches='tight')
    print(f"📊 Saved: {output_dir / 'optimizer_comparison.png'}")
    plt.close()


def generate_summary_table(df: pd.DataFrame, output_dir: Path):
    """Generate summary table of all results."""

    # Create summary DataFrame
    summary = df.copy()
    summary["improvement_pct"] = (summary["improvement"] / summary["baseline_score"] * 100).round(1)

    # Reorder columns
    summary = summary[[
        "benchmark", "num_steps", "optimizer",
        "baseline_score", "final_score", "improvement", "improvement_pct",
        "train_n", "val_n"
    ]]

    # Round numeric columns
    for col in ["baseline_score", "final_score", "improvement"]:
        summary[col] = summary[col].round(4)

    # Sort by benchmark, steps, optimizer
    summary = summary.sort_values(["benchmark", "num_steps", "optimizer"])

    # Save as CSV
    csv_path = output_dir / "summary_results.csv"
    summary.to_csv(csv_path, index=False)
    print(f"📄 Saved: {csv_path}")

    # Save as markdown table
    md_path = output_dir / "summary_results.md"
    with open(md_path, "w") as f:
        f.write("# DSPy Scaling Experiment Results\n\n")
        f.write(summary.to_markdown(index=False))
        f.write("\n")

    print(f"📄 Saved: {md_path}")

    # Print summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    for benchmark in summary["benchmark"].unique():
        bench_data = summary[summary["benchmark"] == benchmark]
        print(f"\n{benchmark.upper()}:")
        print(f"  Best improvement: {bench_data['improvement_pct'].max():.1f}% ({bench_data.loc[bench_data['improvement_pct'].idxmax(), 'num_steps']}-step, {bench_data.loc[bench_data['improvement_pct'].idxmax(), 'optimizer'].upper()})")
        print(f"  Avg improvement: {bench_data['improvement_pct'].mean():.1f}%")

    print("\nOVERALL:")
    print(f"  Best improvement: {summary['improvement_pct'].max():.1f}%")
    print(f"  Avg improvement: {summary['improvement_pct'].mean():.1f}%")

    print("="*80 + "\n")


def main():
    """Main analysis pipeline."""

    base_dir = Path(__file__).parent
    results_dir = base_dir / "results"
    viz_dir = base_dir / "visualizations"

    print("📊 Loading results...")
    df = load_all_results(results_dir)

    if df.empty:
        print("❌ No results to analyze")
        return

    print(f"✅ Loaded {len(df)} experiments\n")

    # Generate visualizations
    print("📊 Generating visualizations...")
    create_scaling_curves(df, viz_dir)
    create_improvement_heatmap(df, viz_dir)
    create_optimizer_comparison(df, viz_dir)

    # Generate summary table
    print("\n📄 Generating summary tables...")
    generate_summary_table(df, base_dir)

    print("\n✅ Analysis complete!")
    print(f"📁 Visualizations: {viz_dir}")
    print(f"📁 Summary: {base_dir / 'summary_results.csv'}")


if __name__ == "__main__":
    main()
