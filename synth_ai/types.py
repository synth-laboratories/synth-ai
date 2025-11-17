import typing
from typing import Literal

TrainType = Literal[
    "prompt",
    "rl",
    "sft"
]
TRAIN_TYPES = list(typing.get_args(TrainType))

ModelName = Literal[
    "synth-small",
    "synth-medium"
]
MODEL_NAMES = list(typing.get_args(ModelName))
