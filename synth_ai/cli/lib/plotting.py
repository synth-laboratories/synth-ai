"""Optimization curve plotting utilities."""

from __future__ import annotations


def plot_optimization_curve(
    trial_counts: list[int],
    best_scores: list[float],
    *,
    width: int = 80,
    height: int = 15,
    title: str = "Optimization Curve: Best Score vs Trial Count",
) -> str:
    """Generates an ASCII plot of the optimization curve.
    
    Args:
        trial_counts: List of trial counts (x-axis)
        best_scores: List of best scores at each trial count (y-axis)
        width: Width of plot in characters
        height: Height of plot in characters
        title: Plot title
    
    Returns:
        Multi-line string containing the ASCII plot
    """
    if not trial_counts or not best_scores:
        return f"{title}\n(No data to plot)\n"
    
    if len(trial_counts) != len(best_scores):
        return f"{title}\n(Data mismatch: {len(trial_counts)} trials vs {len(best_scores)} scores)\n"
    
    # Find min/max for scaling
    min_trial = min(trial_counts)
    max_trial = max(trial_counts)
    min_score = min(best_scores)
    max_score = max(best_scores)
    
    # Handle edge case where all scores are the same
    score_range = 1.0 if max_score == min_score else max_score - min_score
    
    # Create grid (height x width)
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Plot points
    for i, (trial, score) in enumerate(zip(trial_counts, best_scores, strict=True)):
        # Map trial to x position
        if max_trial == min_trial:
            x = width - 1
        else:
            x = int((trial - min_trial) / (max_trial - min_trial) * (width - 1))
        
        # Map score to y position (invert y-axis: top is max_score)
        if score_range == 0:
            y = height - 1
        else:
            y = int((max_score - score) / score_range * (height - 1))
        
        # Clamp to grid bounds
        x = max(0, min(width - 1, x))
        y = max(0, min(height - 1, y))
        
        # Use different characters for different points
        if i == 0:
            grid[y][x] = '●'  # First point
        elif i == len(trial_counts) - 1:
            grid[y][x] = '★'  # Last point
        else:
            grid[y][x] = '·'  # Intermediate points
    
    # Draw connecting lines (simple line drawing)
    for i in range(len(trial_counts) - 1):
        x1 = int((trial_counts[i] - min_trial) / (max_trial - min_trial) * (width - 1)) if max_trial != min_trial else width - 1
        y1 = int((max_score - best_scores[i]) / score_range * (height - 1)) if score_range != 0 else height - 1
        x2 = int((trial_counts[i + 1] - min_trial) / (max_trial - min_trial) * (width - 1)) if max_trial != min_trial else width - 1
        y2 = int((max_score - best_scores[i + 1]) / score_range * (height - 1)) if score_range != 0 else height - 1
        
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        
        # Draw line between points (simple Bresenham-like)
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        x, y = x1, y1
        while True:
            if grid[y][x] == ' ':
                grid[y][x] = '─' if dx > dy else '│'
            if x == x2 and y == y2:
                break
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
    
    # Build output string
    lines = []
    lines.append(title)
    lines.append("=" * max(len(title), width + 10))
    
    # Y-axis labels (score values)
    for i in range(height):
        score_val = max_score - (i / (height - 1) * score_range) if height > 1 else max_score
        label = f"{score_val:.3f} │"
        lines.append(f"{label:>10} {''.join(grid[i])}")
    
    # X-axis separator
    lines.append(" " * 10 + "─" * width)
    
    # X-axis label (trial counts)
    x_label = f"Trial Count ({min_trial} to {max_trial})"
    lines.append(" " * 10 + x_label.center(width))
    
    # Legend
    lines.append("")
    lines.append("Legend: ● = First trial, · = Intermediate, ★ = Final trial")
    
    return "\n".join(lines)

