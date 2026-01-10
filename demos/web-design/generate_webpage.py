"""
Generate HTML/CSS webpage from functional description using Gemini 2.5.
"""

import json
import os
from pathlib import Path

import google.generativeai as genai

WEBPAGE_GENERATION_PROMPT = """You are a web designer tasked with creating a modern, visually appealing webpage based on a functional description.

You will receive a FUNCTIONAL description that tells you WHAT content is on the page, but NOT how it should look. Your job is to:

1. Create complete, production-ready HTML and CSS
2. Design a modern, clean, visually appealing layout
3. Choose appropriate colors, fonts, spacing, and visual hierarchy
4. Implement responsive design
5. Add appropriate visual styling (shadows, gradients, borders, etc.)
6. Make design decisions that fit modern web design trends

IMPORTANT RULES:
- Output ONLY the complete HTML file with inline CSS in a <style> tag
- Do NOT include any explanations or markdown code blocks
- Do NOT use external dependencies or frameworks
- Include ALL CSS inline in the <style> tag
- Make the page look professional and modern
- Use your judgment for all visual design decisions

The functional description will tell you what content to include. You decide how to make it look beautiful.

Now, generate the complete HTML for this functional description:

"""


def generate_webpage(functional_description: str, api_key: str) -> str:
    """Generate HTML/CSS from functional description using Gemini."""
    # Configure the API
    genai.configure(api_key=api_key)

    # Create model
    model = genai.GenerativeModel("gemini-2.0-flash-exp")

    # Generate the webpage
    response = model.generate_content(WEBPAGE_GENERATION_PROMPT + functional_description)

    return response.text


def load_functional_description(json_file: str, index: int = 0) -> dict:
    """Load a functional description from the JSON file."""
    with open(json_file) as f:
        data = json.load(f)
    return data[index]


def clean_html_response(html: str) -> str:
    """Clean the HTML response by removing markdown code blocks if present."""
    # Remove markdown code blocks
    if html.startswith("```html"):
        html = html[7:]
    elif html.startswith("```"):
        html = html[3:]

    if html.endswith("```"):
        html = html[:-3]

    return html.strip()


if __name__ == "__main__":
    # Get API key from environment
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")

    if not api_key:
        print("Error: GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set")
        exit(1)

    # Load functional descriptions
    descriptions_file = Path(__file__).parent / "sample_functional_descriptions.json"

    # Let user choose which description to use
    with open(descriptions_file) as f:
        data = json.load(f)

    print("Available functional descriptions:")
    for i, item in enumerate(data):
        image_path = Path(item["image_path"])
        print(f"  [{i}] {image_path.parent.name}/{image_path.name}")

    choice = input("\nSelect index to generate (default 0): ").strip()
    index = int(choice) if choice else 0

    # Load the selected description
    item = load_functional_description(descriptions_file, index)
    functional_description = item["functional_description"]
    image_path = Path(item["image_path"])

    print(f"\n{'=' * 80}")
    print(f"Generating webpage from: {image_path.parent.name}/{image_path.name}")
    print(f"{'=' * 80}\n")

    print("Functional Description:")
    print(
        functional_description[:500] + "..."
        if len(functional_description) > 500
        else functional_description
    )
    print(f"\n{'=' * 80}")
    print("Generating HTML/CSS with Gemini 2.5...")
    print(f"{'=' * 80}\n")

    # Generate the webpage
    html = generate_webpage(functional_description, api_key)
    html = clean_html_response(html)

    # Save the generated HTML
    output_dir = Path(__file__).parent / "generated_pages"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"{image_path.parent.name}_{image_path.stem}.html"
    with open(output_file, "w") as f:
        f.write(html)

    print(f"✓ Generated webpage saved to: {output_file}")
    print(f"\nTo view the generated page, open: {output_file}")
    print(f"Compare with original screenshot: {image_path}")
