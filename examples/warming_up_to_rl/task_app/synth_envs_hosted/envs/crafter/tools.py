"""OpenAI tools schema for Crafter, defined in Python."""

# Pass this list directly to OpenAI/vLLM `tools=`
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "interact_many",
            "description": "Execute a short sequence of Crafter actions in order (1-8).",
            "parameters": {
                "type": "object",
                "properties": {
                    "actions": {
                        "type": "array",
                        "description": "List of Crafter actions to execute sequentially.",
                        "items": {
                            "type": "string",
                            "enum": [
                                "noop",
                                "move_left",
                                "move_right",
                                "move_up",
                                "move_down",
                                "do",
                                "sleep",
                                "place_stone",
                                "place_table",
                                "place_furnace",
                                "place_plant",
                                "make_wood_pickaxe",
                                "make_stone_pickaxe",
                                "make_iron_pickaxe",
                                "make_wood_sword",
                                "make_stone_sword",
                                "make_iron_sword",
                            ],
                        },
                        "minItems": 1,
                        "maxItems": 8,
                    }
                },
                "required": ["actions"],
                "additionalProperties": False,
            },
        },
    }
]
