"""
Screen analysis functions for Pokemon Red to provide rich textual descriptions
of what's actually visible on the game screen.
"""

import hashlib
from typing import Any, Dict, List, Tuple

import numpy as np

# Define some common Pokemon Red screen colors (RGB values)
POKEMON_RED_COLORS = {
    (255, 255, 255): "WHITE",  # White/light backgrounds
    (192, 192, 192): "LIGHT_GRAY",  # Light gray
    (128, 128, 128): "GRAY",  # Gray
    (64, 64, 64): "DARK_GRAY",  # Dark gray
    (0, 0, 0): "BLACK",  # Black/dark
    (248, 248, 248): "OFF_WHITE",  # Slightly off white
    (200, 200, 200): "SILVER",  # Silver-ish
}


def analyze_screen_buffer(screen_buffer: np.ndarray) -> Dict[str, Any]:
    """
    Analyze the Pokemon Red screen buffer to extract meaningful information
    about what's currently visible on screen.

    Args:
        screen_buffer: numpy array of shape (144, 160, 3) or (144, 160, 4) representing the screen

    Returns:
        Dictionary containing analysis of the screen content
    """
    if screen_buffer is None:
        return {"error": "No screen buffer provided"}

    # Convert RGBA to RGB if needed
    if screen_buffer.shape[2] == 4:
        rgb_buffer = screen_buffer[:, :, :3]
    else:
        rgb_buffer = screen_buffer

    analysis = {
        "screen_hash": hashlib.md5(screen_buffer.tobytes()).hexdigest()[:8],
        "screen_type": detect_screen_type(rgb_buffer),
        "text_content": extract_text_content(rgb_buffer),
        "entities": detect_entities(rgb_buffer),
        "ui_elements": detect_ui_elements(rgb_buffer),
        "color_analysis": analyze_colors(rgb_buffer),
        "ascii_representation": create_ascii_representation(rgb_buffer),
    }

    return analysis


def detect_screen_type(rgb_buffer: np.ndarray) -> str:
    """
    Determine what type of screen is being displayed based on visual patterns.
    """
    height, width = rgb_buffer.shape[:2]

    # Check for common screen types based on color patterns and layout

    # Check if mostly one color (like a menu background)
    unique_colors = len(np.unique(rgb_buffer.reshape(-1, 3), axis=0))

    if unique_colors < 8:
        # Very few colors - could be simple UI but be more conservative about "MENU"
        if is_text_box_screen(rgb_buffer):
            return "TEXT_BOX"
        else:
            return "SIMPLE_UI"  # Don't assume MENU, use neutral term

    elif unique_colors > 50:
        # Many colors - likely overworld or complex scene
        return "OVERWORLD"

    else:
        # Medium complexity
        if is_battle_screen(rgb_buffer):
            return "BATTLE"
        else:
            return "GAME_SCREEN"  # Use neutral term instead of assuming menu


def is_menu_screen(rgb_buffer: np.ndarray) -> bool:
    """Check if this looks like a menu screen - made more conservative."""
    # This function is now unused but keeping for potential future use
    # The logic was too aggressive in detecting menus
    height, width = rgb_buffer.shape[:2]

    # Much more strict criteria for what constitutes a menu
    # Look for very specific menu patterns like uniform background with text boxes
    horizontal_consistency = 0
    for row in range(height):
        row_colors = rgb_buffer[row, :, :]
        unique_in_row = len(np.unique(row_colors.reshape(-1, 3), axis=0))
        if unique_in_row < 3:  # Very consistent row color (stricter than before)
            horizontal_consistency += 1

    # Much higher threshold - only classify as menu if almost the entire screen is uniform
    return horizontal_consistency > height * 0.8


def is_text_box_screen(rgb_buffer: np.ndarray) -> bool:
    """Check if this looks like a text box is displayed."""
    # Look for typical text box patterns - usually bottom portion has different colors
    height, width = rgb_buffer.shape[:2]
    bottom_quarter = rgb_buffer[height * 3 // 4 :, :, :]
    top_quarter = rgb_buffer[: height // 4, :, :]

    bottom_colors = len(np.unique(bottom_quarter.reshape(-1, 3), axis=0))
    top_colors = len(np.unique(top_quarter.reshape(-1, 3), axis=0))

    return bottom_colors > top_colors * 1.5


def is_battle_screen(rgb_buffer: np.ndarray) -> bool:
    """Check if this looks like a battle screen."""
    # Battle screens typically have more color variation
    height, width = rgb_buffer.shape[:2]
    unique_colors = len(np.unique(rgb_buffer.reshape(-1, 3), axis=0))
    return unique_colors > 30


def extract_text_content(rgb_buffer: np.ndarray) -> List[str]:
    """
    Attempt to extract readable text from the screen.
    This is simplified - real OCR would be more complex.
    """
    # For now, just analyze patterns that might indicate text
    text_lines = []

    height, width = rgb_buffer.shape[:2]

    # Look for text-like patterns in bottom area (common location for text boxes)
    text_area = rgb_buffer[height * 2 // 3 :, :, :]

    # Simple heuristic: if there are alternating light/dark patterns, might be text
    for row_idx in range(text_area.shape[0]):
        row = text_area[row_idx, :, :]
        # Check for patterns that might indicate text
        brightness = np.mean(row, axis=1)
        changes = np.diff(brightness > np.mean(brightness))
        if np.sum(changes) > width // 10:  # Lots of brightness changes might be text
            text_lines.append(f"Text detected at row {height * 2 // 3 + row_idx}")

    return text_lines


def detect_entities(rgb_buffer: np.ndarray) -> List[Dict[str, Any]]:
    """
    Detect entities like player character, NPCs, objects on screen.
    """
    entities = []

    # This is a simplified approach - real entity detection would be more sophisticated
    height, width = rgb_buffer.shape[:2]

    # Look for small moving/distinct colored regions that might be characters
    # Scan in a grid pattern
    block_size = 16  # Pokemon Red typically uses 16x16 sprites

    for y in range(0, height - block_size, block_size):
        for x in range(0, width - block_size, block_size):
            block = rgb_buffer[y : y + block_size, x : x + block_size, :]

            # Check if this block has interesting characteristics
            unique_colors_in_block = len(np.unique(block.reshape(-1, 3), axis=0))

            if 3 <= unique_colors_in_block <= 8:  # Sprite-like color count
                avg_color = np.mean(block, axis=(0, 1))
                entities.append(
                    {
                        "type": "SPRITE_LIKE",
                        "position": (x, y),
                        "avg_color": tuple(avg_color.astype(int)),
                        "color_count": unique_colors_in_block,
                    }
                )

    return entities


def detect_ui_elements(rgb_buffer: np.ndarray) -> Dict[str, Any]:
    """
    Detect UI elements like borders, windows, buttons.
    """
    height, width = rgb_buffer.shape[:2]

    ui_elements = {"has_border": False, "windows": [], "button_like_areas": []}

    # Check for borders (consistent colors around edges)
    edges = [
        rgb_buffer[0, :, :],  # top edge
        rgb_buffer[-1, :, :],  # bottom edge
        rgb_buffer[:, 0, :],  # left edge
        rgb_buffer[:, -1, :],  # right edge
    ]

    for edge in edges:
        unique_colors = len(np.unique(edge.reshape(-1, 3), axis=0))
        if unique_colors < 3:  # Very consistent edge color
            ui_elements["has_border"] = True
            break

    return ui_elements


def analyze_colors(rgb_buffer: np.ndarray) -> Dict[str, Any]:
    """
    Analyze the color composition of the screen.
    """
    unique_colors, counts = np.unique(rgb_buffer.reshape(-1, 3), axis=0, return_counts=True)

    # Sort by frequency
    sorted_indices = np.argsort(counts)[::-1]
    top_colors = unique_colors[sorted_indices[:10]]  # Top 10 colors
    top_counts = counts[sorted_indices[:10]]

    total_pixels = rgb_buffer.shape[0] * rgb_buffer.shape[1]

    color_analysis = {
        "total_unique_colors": len(unique_colors),
        "dominant_colors": [],
        "color_complexity": "HIGH"
        if len(unique_colors) > 50
        else "MEDIUM"
        if len(unique_colors) > 20
        else "LOW",
    }

    for i, (color, count) in enumerate(zip(top_colors, top_counts)):
        percentage = (count / total_pixels) * 100
        color_analysis["dominant_colors"].append(
            {
                "rgb": tuple(color),
                "percentage": round(percentage, 1),
                "name": get_color_name(tuple(color)),
            }
        )

    return color_analysis


def get_color_name(rgb_tuple: Tuple[int, int, int]) -> str:
    """
    Get a human-readable name for an RGB color.
    """
    # Find closest match in our color dictionary
    min_distance = float("inf")
    closest_name = "UNKNOWN"

    for known_rgb, name in POKEMON_RED_COLORS.items():
        distance = sum((a - b) ** 2 for a, b in zip(rgb_tuple, known_rgb))
        if distance < min_distance:
            min_distance = distance
            closest_name = name

    # If no close match, generate a descriptive name
    if min_distance > 10000:  # Threshold for "close enough"
        r, g, b = rgb_tuple
        if r > 200 and g > 200 and b > 200:
            return "LIGHT"
        elif r < 50 and g < 50 and b < 50:
            return "DARK"
        else:
            return "MEDIUM"

    return closest_name


def create_ascii_representation(rgb_buffer: np.ndarray) -> str:
    """
    Create a simplified ASCII representation of the screen.
    """
    height, width = rgb_buffer.shape[:2]

    # Downsample to a reasonable ASCII size (e.g., 40x20)
    ascii_height = 20
    ascii_width = 40

    ascii_chars = []

    for row in range(ascii_height):
        ascii_row = ""
        for col in range(ascii_width):
            # Map to original coordinates
            orig_row = int((row / ascii_height) * height)
            orig_col = int((col / ascii_width) * width)

            # Get average brightness of this region
            region = rgb_buffer[
                orig_row : orig_row + height // ascii_height,
                orig_col : orig_col + width // ascii_width,
                :,
            ]

            brightness = np.mean(region)

            # Convert brightness to ASCII character
            if brightness > 200:
                ascii_row += " "  # Bright = space
            elif brightness > 150:
                ascii_row += "."  # Medium-bright = dot
            elif brightness > 100:
                ascii_row += ":"  # Medium = colon
            elif brightness > 50:
                ascii_row += "x"  # Dark = x
            else:
                ascii_row += "#"  # Very dark = hash

        ascii_chars.append(ascii_row)

    return "\n".join(ascii_chars)


def create_detailed_screen_description(screen_analysis: Dict[str, Any]) -> str:
    """
    Create a detailed text description of what's on screen based on the analysis.
    """
    description_lines = []

    # Screen type and basic info
    screen_type = screen_analysis.get("screen_type", "UNKNOWN")
    description_lines.append(f"SCREEN TYPE: {screen_type}")

    # Color analysis
    color_info = screen_analysis.get("color_analysis", {})
    complexity = color_info.get("color_complexity", "UNKNOWN")
    description_lines.append(f"VISUAL COMPLEXITY: {complexity}")

    # Dominant colors
    if "dominant_colors" in color_info:
        color_desc = "DOMINANT COLORS: "
        top_3_colors = color_info["dominant_colors"][:3]
        color_names = [f"{c['name']}({c['percentage']:.0f}%)" for c in top_3_colors]
        color_desc += ", ".join(color_names)
        description_lines.append(color_desc)

    # Entities
    entities = screen_analysis.get("entities", [])
    if entities:
        description_lines.append(f"DETECTED ENTITIES: {len(entities)} sprite-like objects")
        # Describe a few entities
        for i, entity in enumerate(entities[:3]):
            pos = entity["position"]
            description_lines.append(f"  Entity {i + 1}: {entity['type']} at ({pos[0]}, {pos[1]})")

    # Text content
    text_content = screen_analysis.get("text_content", [])
    if text_content:
        description_lines.append("TEXT DETECTED:")
        for text_line in text_content[:3]:  # Show first 3 text detections
            description_lines.append(f"  {text_line}")

    # UI elements
    ui_elements = screen_analysis.get("ui_elements", {})
    if ui_elements.get("has_border"):
        description_lines.append("UI: Window/border detected")

    # ASCII representation
    ascii_repr = screen_analysis.get("ascii_representation", "")
    if ascii_repr:
        description_lines.append("\nASCII REPRESENTATION:")
        description_lines.append(ascii_repr)

    return "\n".join(description_lines)
