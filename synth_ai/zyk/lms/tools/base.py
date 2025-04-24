from typing import Type

from pydantic import BaseModel


class BaseTool(BaseModel):
    name: str
    arguments: Type[BaseModel]
    description: str = ""
    strict: bool = True

    def to_openai_tool(self):
        schema = self.arguments.model_json_schema()
        schema["additionalProperties"] = False

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
                "strict": self.strict,
            },
        }

    def to_anthropic_tool(self):
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

    def to_mistral_tool(self):
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

    def to_gemini_tool(self):
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
