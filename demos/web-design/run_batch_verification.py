"""
Run verification on multiple examples and collect aggregate statistics.
"""

import json
import os
from datetime import datetime
from pathlib import Path

from datasets import load_from_disk
from PIL import Image
from verify_generation import generate_image_from_description, verify_with_vision_model


def run_batch_verification(num_examples: int = 10, api_key: str = None):
    """Run verification on multiple examples and collect statistics."""

    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")

    # Load dataset
    dataset_path = Path(__file__).parent / "hf_dataset"
    dataset = load_from_disk(str(dataset_path))

    # Filter examples with functional descriptions
    valid_examples = [
        ex
        for ex in dataset
        if ex.get("functional_description") and len(ex["functional_description"]) > 100
    ]

    print(f"Found {len(valid_examples)} valid examples")
    print(f"Running verification on {min(num_examples, len(valid_examples))} examples\n")

    # Create output directory
    output_dir = Path(__file__).parent / "verification_results"
    output_dir.mkdir(exist_ok=True)

    results = []

    for i, example in enumerate(valid_examples[:num_examples]):
        print(f"\n{'=' * 80}")
        print(f"Example {i + 1}/{min(num_examples, len(valid_examples))}")
        print(f"Site: {example['site_name']} - Page: {example['page_name']}")
        print(f"{'=' * 80}")

        try:
            # Generate image
            print("Generating image...")
            generated_image_bytes = generate_image_from_description(
                example["functional_description"], api_key
            )

            # Save generated image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_file = (
                output_dir
                / f"{example['site_name']}_{example['page_name']}_{timestamp}_generated.png"
            )
            with open(image_file, "wb") as f:
                f.write(generated_image_bytes)

            print(f"✓ Saved to: {image_file.name}")

            # Get original image
            if not isinstance(example["image"], Image.Image):
                print("⚠ Skipping verification - unexpected image format")
                continue

            original_image = example["image"]
            generated_image = Image.open(image_file)

            # Run verification
            print("Running verification...")
            verification_result = verify_with_vision_model(
                original_image, generated_image, example["functional_description"], api_key
            )

            # Save verification result
            result_file = (
                output_dir
                / f"{example['site_name']}_{example['page_name']}_{timestamp}_verification.json"
            )
            with open(result_file, "w") as f:
                json.dump(verification_result, f, indent=2)

            # Add metadata
            verification_result["site_name"] = example["site_name"]
            verification_result["page_name"] = example["page_name"]
            verification_result["url"] = example["url"]
            verification_result["image_file"] = image_file.name
            verification_result["result_file"] = result_file.name

            results.append(verification_result)

            # Print scores
            print("✓ Scores:")
            print(f"  Color Scheme: {verification_result['color_scheme_branding']}/10")
            print(f"  Typography: {verification_result['typography_styling']}/10")
            print(f"  Layout: {verification_result['layout_spacing']}/10")
            print(f"  Visual Elements: {verification_result['visual_elements']}/10")
            print(f"  Overall Fidelity: {verification_result['overall_visual_fidelity']}/10")
            print(f"  AVERAGE: {verification_result['average_score']}/10")
            print(f"  Critiques: {len(verification_result.get('visual_critiques', []))} items")

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback

            traceback.print_exc()
            continue

    # Generate aggregate statistics
    print(f"\n{'=' * 80}")
    print("AGGREGATE STATISTICS")
    print(f"{'=' * 80}")

    if results:
        avg_color = sum(r["color_scheme_branding"] for r in results) / len(results)
        avg_typography = sum(r["typography_styling"] for r in results) / len(results)
        avg_layout = sum(r["layout_spacing"] for r in results) / len(results)
        avg_visual = sum(r["visual_elements"] for r in results) / len(results)
        avg_overall = sum(r["overall_visual_fidelity"] for r in results) / len(results)
        avg_total = sum(r["average_score"] for r in results) / len(results)

        print(f"Verified {len(results)} examples:")
        print(f"  Color Scheme & Branding:  {avg_color:.2f}/10")
        print(f"  Typography & Styling:     {avg_typography:.2f}/10")
        print(f"  Layout & Spacing:         {avg_layout:.2f}/10")
        print(f"  Visual Elements:          {avg_visual:.2f}/10")
        print(f"  Overall Visual Fidelity:  {avg_overall:.2f}/10")
        print(f"  AVERAGE ACROSS ALL:       {avg_total:.2f}/10")

        # Find best and worst
        best = max(results, key=lambda r: r["average_score"])
        worst = min(results, key=lambda r: r["average_score"])

        print(
            f"\nBest result: {best['site_name']}/{best['page_name']} ({best['average_score']:.1f}/10)"
        )
        print(
            f"Worst result: {worst['site_name']}/{worst['page_name']} ({worst['average_score']:.1f}/10)"
        )

        # Save aggregate results
        aggregate_file = (
            output_dir / f"batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(aggregate_file, "w") as f:
            json.dump(
                {
                    "num_examples": len(results),
                    "aggregate_scores": {
                        "color_scheme_branding": avg_color,
                        "typography_styling": avg_typography,
                        "layout_spacing": avg_layout,
                        "visual_elements": avg_visual,
                        "overall_visual_fidelity": avg_overall,
                        "average_total": avg_total,
                    },
                    "best": {
                        "site": best["site_name"],
                        "page": best["page_name"],
                        "score": best["average_score"],
                    },
                    "worst": {
                        "site": worst["site_name"],
                        "page": worst["page_name"],
                        "score": worst["average_score"],
                    },
                    "all_results": results,
                },
                f,
                indent=2,
            )

        print(f"\n✓ Aggregate results saved to: {aggregate_file.name}")
    else:
        print("No successful verifications")

    return results


if __name__ == "__main__":
    import sys

    num_examples = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_batch_verification(num_examples)
