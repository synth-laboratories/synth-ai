"""A bounded future response discriminant, preserving its actual wire value."""

from typing import Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import CoreSchema, core_schema


class UnknownValue(str):
    """An observation the installed SDK does not yet interpret as a known state."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: GetCoreSchemaHandler) -> CoreSchema:
        return core_schema.no_info_after_validator_function(
            cls, core_schema.str_schema(strict=True, min_length=1, max_length=128)
        )
