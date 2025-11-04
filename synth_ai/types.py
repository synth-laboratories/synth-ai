import typing
from typing import Literal

ModelName = Literal[
    "synth-small",
    "synth-medium"
]
MODEL_NAMES = list(typing.get_args(ModelName))
