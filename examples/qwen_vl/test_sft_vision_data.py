"""Generate test vision SFT dataset for Qwen3-VL-2B."""

import base64
import json
from pathlib import Path
from io import BytesIO

try:
    from PIL import Image
except ImportError:
    print("❌ PIL not available")
    exit(1)

BASE_DIR = Path(__file__).resolve().parent

def create_test_image(color: str) -> str:
    """Create a 64x64 colored square and return base64 data URL."""
    colors = {
        "red": (255, 0, 0),
        "blue": (0, 0, 255),
        "green": (0, 255, 0),
        "yellow": (255, 255, 0),
        "purple": (128, 0, 128),
    }
    
    img = Image.new('RGB', (64, 64), color=colors[color])
    buffer = BytesIO()
    img.save(buffer, format='PNG')
    b64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    return f"data:image/png;base64,{b64}"


def main():
    output_dir = BASE_DIR / "test_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "vision_sft_test.jsonl"
    
    # Create 10 training examples with different colored images
    examples = []
    colors = ["red", "blue", "green", "yellow", "purple"]
    
    for i, color in enumerate(colors):
        # Simple color identification
        examples.append({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What color is this image? Answer in one word."},
                        {"type": "image_url", "image_url": {"url": create_test_image(color)}},
                    ],
                },
                {
                    "role": "assistant",
                    "content": color.capitalize(),
                },
            ],
            "metadata": {"example_id": f"color_{i}", "type": "color_id"},
        })
        
        # Describe the image
        examples.append({
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image briefly."},
                        {"type": "image_url", "image_url": {"url": create_test_image(color)}},
                    ],
                },
                {
                    "role": "assistant",
                    "content": f"This is a {color} colored square image.",
                },
            ],
            "metadata": {"example_id": f"describe_{i}", "type": "description"},
        })
    
    # Write JSONL
    with output_file.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")
    
    print(f"✅ Created {len(examples)} vision SFT examples")
    print(f"   Output: {output_file}")
    print(f"   Size: {output_file.stat().st_size / 1024:.1f} KB")
    
    # Validate with SDK
    try:
        from synth_ai.learning.sft.data import load_jsonl, validate_vision_example
        
        loaded = load_jsonl(output_file, min_messages=1)
        print(f"   Loaded: {len(loaded)} examples")
        
        valid_count = 0
        for ex in loaded:
            is_valid, error = validate_vision_example(ex, require_images=True)
            if is_valid:
                valid_count += 1
            else:
                print(f"   ⚠️  Invalid example: {error}")
        
        print(f"   Valid: {valid_count}/{len(loaded)}")
    except ImportError:
        print("   (SDK validation skipped - synth_ai not available)")


if __name__ == "__main__":
    main()
