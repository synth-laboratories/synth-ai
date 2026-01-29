import random
from typing import Sequence


def stratified_seed_sample(df, label_col: str, total: int, rng: random.Random | None = None) -> list[int]:
    """Return up to `total` row indices, roughly balanced across label groups."""
    if total <= 0:
        return []
    rng = rng or random.Random()
    frame = df.reset_index(drop=True)
    groups = frame.groupby(label_col).indices
    labels = list(groups.keys())
    if not labels:
        return []
    rng.shuffle(labels)

    base = total // len(labels)
    remainder = total % len(labels)
    seeds: list[int] = []
    for idx, label in enumerate(labels):
        target = base + (1 if idx < remainder else 0)
        indices = list(groups[label])
        if target >= len(indices):
            seeds.extend(indices)
        else:
            seeds.extend(rng.sample(indices, target))

    if len(seeds) < total:
        all_indices = list(range(len(frame)))
        remaining = list(set(all_indices) - set(seeds))
        needed = total - len(seeds)
        if needed <= len(remaining):
            seeds.extend(rng.sample(remaining, needed))
        else:
            seeds.extend(remaining)
            seeds.extend(rng.choices(all_indices, k=needed - len(remaining)))

    rng.shuffle(seeds)
    return seeds[:total]


def split_seed_slices(seeds: Sequence[int], slices: int) -> list[list[int]]:
    if slices <= 1:
        return [list(seeds)]
    base = len(seeds) // slices
    remainder = len(seeds) % slices
    sizes = [base + (1 if i < remainder else 0) for i in range(slices)]
    seed_slices: list[list[int]] = []
    cursor = 0
    for size in sizes:
        seed_slices.append(list(seeds[cursor:cursor + size]))
        cursor += size
    return seed_slices
