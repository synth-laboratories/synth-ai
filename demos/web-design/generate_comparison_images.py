#!/usr/bin/env python3
"""Generate comparison images using baseline vs mutated prompts directly."""

import os
import sys
from pathlib import Path

from google import genai

# Get API key
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("ERROR: Set GOOGLE_API_KEY environment variable")
    sys.exit(1)

client = genai.Client(api_key=api_key)

BASELINE_PROMPT = """You are generating a professional startup website screenshot.

VISUAL STYLE GUIDELINES:
- Use a clean, modern, minimalist design aesthetic
- Color Scheme: Light backgrounds with high contrast dark text
- Typography: Large, bold headings with clear hierarchy
- Layout: Spacious with generous padding and margins
- Branding: Professional, tech-forward visual identity

Create a webpage that feels polished, modern, and trustworthy."""

MUTATED_PROMPT = """1. Input Description:
- You will receive a detailed functional description and visual style guidelines for creating a startup company homepage screenshot.
- The input includes sections, content details, interactive elements, and branding style directions.

2. Core Task Description:
- Your task is to generate a professional, polished, and modern screenshot of a startup company's homepage (named Astral) according to the provided functional specifications and visual style guidelines.
- The screenshot should visually represent all described sections, content, and interactive elements, reflecting the company's tech-forward and trustworthy branding.

3. Premises:
- The homepage is centered on Astral, a company focused on improving Python development.
- The page includes multiple defined sections: Header, Hero, Mission, Beliefs, Founder narrative, Team growth, Announcements, Call to Action, and Footer.
- The visual style must be clean, modern, minimalist, with light backgrounds, high contrast dark text, bold typography, spacious layout, and professional branding.

4. Context:
- The screenshot is a static, visual representation of the webpage as it would appear to a user.
- Interactive elements (navigation links, buttons, and links) must be depicted clearly, showing their presence and labels.
- The company identity, key messaging (headline, mission, beliefs), and latest announcements are prominently displayed.

5. Task Priority:
- Ensure accurate depiction of each section with correct text content and layout as described.
- Reflect the brand's modern and professional aesthetic consistently throughout.
- Demonstrate a clear visual hierarchy in typography and layout.
- Include all specified graphics, logos, and images where indicated.

6. Heuristics:
- Use spacious margins and padding to create an airy, uncluttered look.
- Employ large, bold headings to establish clear section hierarchy.
- Use light background colors with high contrast dark text for readability.
- Represent logos (Accel, Caffeinated, Guillermo Rauch) and founder's headshot visually distinct.
- Visually differentiate interactive elements such as buttons and links with styling cues (color, shape).
- Organize footer links into clear labeled groups for easy navigation.

7. Constraints:
- Do not add or omit any content sections, text details, or interactive elements beyond what is specified.
- Avoid complex or ornate design styles inconsistent with the minimalist, modern guidelines.
- Do not produce an actual webpage code, only a screenshot-style visual representation.

8. Rules:
- Do not alter or paraphrase the textual content specified; reproduce it faithfully.
- Do not include any extra elements unrelated to the provided sections.
- Do not include textual explanations or descriptions outside the visual representation.
- Present the screenshot as a single cohesive image conveying the entire homepage.

9. Output Description:
- Produce a single, comprehensive screenshot image of the Astral homepage.
- The image must visually convey all sections with correct content, layout, typography, colors, images, and interactive elements as specified.
- The style must closely follow the clean, modern, minimalist aesthetic with professional branding."""

# Sample functional description (from Astral dataset)
FUNCTIONAL_DESCRIPTION = """1. Page Overview:
Homepage for a company named Astral, focused on improving Python development. The page introduces the company's mission, beliefs, team, and relevant announcements.

2. Sections:
*   Header: Contains company logo, navigation links for Products, Docs, Blog, and Company, and a call-to-action "Get Started".
*   Hero Section: Contains the headline "Changing the Way Python Work Gets Done," a lightning bolt graphic, and a chevron graphic.
*   Astral's Mission Section: A numbered point stating Astral's mission to build high-performance developer tools for the Python ecosystem.
*   Astral's Beliefs Section: A numbered point stating Astral's belief that a great tool can multiply the effectiveness of individual developers, teams, and entire organizations.
*   My Path to Astral Section: A short narrative by Charlie Marsh, Founder of Astral.
*   We're growing the team Section: Introductory text about the company's team.
*   Announcements Section: Contains a list of announcements with dates and titles.
*   Call to Action Section: Headline "Supercharge your Python tooling" with links to "Get Started" and "Browse Docs".
*   Footer: Contains multiple lists of links and Astral's logo.

3. Interactive Elements:
*   Header Navigation: Links to "Products", "Docs", "Blog", and "Company".
*   "Get Started" button: In the header and call to action section.
*   Footer Links: Links in lists under "RUFF", "GITHUB", "COMPANY", etc.

4. Content Details:
*   Headline: "Changing the Way Python Work Gets Done"
*   Astral's Mission: "We build high-performance developer tools for the Python ecosystem."
*   Call to Action: "Supercharge your Python tooling"
"""


def generate_image(style_prompt: str, functional_desc: str) -> bytes:
    """Generate an image using Gemini 2.5 Flash Image."""
    full_prompt = f"""{style_prompt}

Generate a webpage screenshot based on this functional description:

{functional_desc}

Apply the visual style guidelines to match the original design."""

    response = client.models.generate_content(model="gemini-2.5-flash-image", contents=full_prompt)

    # Extract image bytes from response
    if hasattr(response, "candidates") and len(response.candidates) > 0:
        candidate = response.candidates[0]
        if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
            for part in candidate.content.parts:
                if hasattr(part, "inline_data") and part.inline_data is not None:
                    return part.inline_data.data

    # Try response.parts directly
    if hasattr(response, "parts"):
        for part in response.parts:
            if hasattr(part, "inline_data") and part.inline_data is not None:
                return part.inline_data.data

    raise ValueError(f"No image data in response: {response}")


def main():
    output_dir = Path(__file__).parent / "comparison_images"
    output_dir.mkdir(exist_ok=True)

    print("=" * 60)
    print("GENERATING COMPARISON IMAGES")
    print("=" * 60)
    print("Using model: gemini-2.5-flash-image")

    # Generate with baseline prompt
    print("\n[1/2] Generating with BASELINE prompt...")
    try:
        baseline_bytes = generate_image(BASELINE_PROMPT, FUNCTIONAL_DESCRIPTION)
        baseline_path = output_dir / "baseline_generated.png"
        with open(baseline_path, "wb") as f:
            f.write(baseline_bytes)
        print(f"    Saved: {baseline_path}")
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback

        traceback.print_exc()
        baseline_path = None

    # Generate with mutated prompt
    print("\n[2/2] Generating with MUTATED prompt...")
    try:
        mutated_bytes = generate_image(MUTATED_PROMPT, FUNCTIONAL_DESCRIPTION)
        mutated_path = output_dir / "mutated_generated.png"
        with open(mutated_path, "wb") as f:
            f.write(mutated_bytes)
        print(f"    Saved: {mutated_path}")
    except Exception as e:
        print(f"    ERROR: {e}")
        import traceback

        traceback.print_exc()
        mutated_path = None

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nCompare the images in: {output_dir}")
    if baseline_path:
        print(f"  open {baseline_path}")
    if mutated_path:
        print(f"  open {mutated_path}")


if __name__ == "__main__":
    main()
