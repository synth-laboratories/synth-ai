def confidence_band(mean_reward: float, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return 0.0
    return z * ((mean_reward * (1.0 - mean_reward)) / n) ** 0.5
