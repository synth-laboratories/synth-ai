from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Any


class RewardComponent(ABC):
    """Interface for a component contributing to stepwise reward."""

    weight: float = 1.0

    @abstractmethod
    async def score(self, state: Any, action: Any) -> float:
        """Compute the component's reward given current state and action."""
        ...


class RewardStack:
    """Aggregates multiple RewardComponent instances to compute a total reward."""

    def __init__(self, components: List[RewardComponent]):
        self.components = components

    async def step_reward(self, state: Any, action: Any) -> float:
        """Compute the sum of weighted component rewards for a single step."""
        total = 0.0
        for comp in self.components:
            total += comp.weight * await comp.score(state, action)
        return total
