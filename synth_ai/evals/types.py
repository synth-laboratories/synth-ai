from __future__ import annotations

from typing import Literal, TypedDict

Track = Literal["process", "reasoning", "progress", "outcome"]


class Judgement(TypedDict, total=False):
	key: str
	title: str
	description: str
	score: float
	reason: str
	confidence: float
	scale: Literal["binary", "bounded", "count", "custom"]
	source: dict


class RewardJudgement(TypedDict, total=False):
	judgement: Judgement
	scope: Literal["step", "event", "outcome"]
	turn: int | None
	episode_id: str | None
	reward_value: float | None
	links: dict


class TrackAggregate(TypedDict, total=False):
	mean: float
	median: float
	std: float
	n: int


class RewardMetadata(TypedDict, total=False):
	per_window: list[RewardJudgement]
	aggregates: dict[Track, TrackAggregate]
	overall: dict[str, float]  # {"final_outcome_score": float}
	rubric: dict               # {"ids": {...}, "hash": "..."}
	model_info: dict           # {"model": "...", ...}


