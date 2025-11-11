"""LangProBe benchmark task apps."""

# Import modules for side effects (task app registration) when package is imported.
from . import gsm8k_task_app  # noqa: F401
from . import heartdisease_task_app  # noqa: F401
from . import iris_task_app  # noqa: F401
# MATH, HumanEval, and AlfWorld descoped - datasets not available or build issues
# from . import math_task_app  # noqa: F401
# from . import humaneval_task_app  # noqa: F401
# from . import alfworld_task_app  # noqa: F401

