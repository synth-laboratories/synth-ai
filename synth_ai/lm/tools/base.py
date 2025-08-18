"""
Base class for LM tools.

This module provides the base class for defining tools that can be used with language models.
"""

from typing import Any

from pydantic import BaseModel


class BaseTool(BaseModel):
    """
    Base class for defining tools that can be used with language models.

    Attributes:
        name: The name of the tool
        arguments: Pydantic model defining the tool's arguments
        description: Human-readable description of what the tool does
        strict: Whether to enforce strict schema validation (default True)
    """

    name: str
    arguments: type[BaseModel]
    description: str = ""
    strict: bool = True

    def to_openai_tool(self) -> dict[str, Any]:
        """
        Convert the tool to OpenAI's tool format.

        Returns:
            dict: Tool definition in OpenAI's expected format

        Note:
            - Ensures additionalProperties is False for strict validation
            - Fixes array items that lack explicit types
            - Handles nested schemas in anyOf constructs
        """
        schema = self.arguments.model_json_schema()
        schema["additionalProperties"] = False

        if "properties" in schema:
            for _prop_name, prop_schema in schema["properties"].items():
                if prop_schema.get("type") == "array":
                    items_schema = prop_schema.get("items", {})
                    if not isinstance(items_schema, dict) or not items_schema.get("type"):
                        prop_schema["items"] = {"type": "string"}

                elif "anyOf" in prop_schema:
                    for sub_schema in prop_schema["anyOf"]:
                        if isinstance(sub_schema, dict) and sub_schema.get("type") == "array":
                            items_schema = sub_schema.get("items", {})
                            if not isinstance(items_schema, dict) or not items_schema.get("type"):
                                sub_schema["items"] = {"type": "string"}

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
                "strict": self.strict,
            },
        }

    def to_anthropic_tool(self) -> dict[str, Any]:
        """
        Convert the tool to Anthropic's tool format.

        Returns:
            dict: Tool definition in Anthropic's expected format

        Note:
            Anthropic uses a different format with input_schema instead of parameters.
        """
        schema = self.arguments.model_json_schema()
        schema["additionalProperties"] = False

        return {
            "name": self.name,
            "description": self.description,
            "input_schema": {
                "type": "object",
                "properties": schema["properties"],
                "required": schema.get("required", []),
            },
        }

    def to_mistral_tool(self) -> dict[str, Any]:
        """
        Convert the tool to Mistral's tool format.

        Returns:
            dict: Tool definition in Mistral's expected format

        Note:
            Mistral requires explicit handling of array types and enum values.
        """
        schema = self.arguments.model_json_schema()
        properties = {}
        for prop_name, prop in schema.get("properties", {}).items():
            prop_type = prop["type"]
            if prop_type == "array" and "items" in prop:
                properties[prop_name] = {
                    "type": "array",
                    "items": prop["items"],
                    "description": prop.get("description", ""),
                }
                continue

            properties[prop_name] = {
                "type": prop_type,
                "description": prop.get("description", ""),
            }
            if "enum" in prop:
                properties[prop_name]["enum"] = prop["enum"]

        parameters = {
            "type": "object",
            "properties": properties,
            "required": schema.get("required", []),
            "additionalProperties": False,
        }
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }

    def to_gemini_tool(self) -> dict[str, Any]:
        """
        Convert the tool to Gemini's tool format.

        Returns:
            dict: Tool definition in Gemini's expected format

        Note:
            Gemini uses a simpler format without the nested "function" key.
        """
        schema = self.arguments.model_json_schema()
        # Convert Pydantic schema types to Gemini schema types
        properties = {}
        for name, prop in schema["properties"].items():
            prop_type = prop.get("type", "string")
            if prop_type == "array" and "items" in prop:
                properties[name] = {
                    "type": "array",
                    "items": prop["items"],
                    "description": prop.get("description", ""),
                }
                continue

            properties[name] = {
                "type": prop_type,
                "description": prop.get("description", ""),
            }
            if "enum" in prop:
                properties[name]["enum"] = prop["enum"]

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": schema.get("required", []),
            },
        }
