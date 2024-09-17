from typing import Any, Dict, List, Optional, Tuple, Type, get_type_hints

from pydantic import BaseModel


def generate_type_map() -> Dict[Any, str]:
    base_types = {
        int: "int",
        float: "float",
        bool: "bool",
        str: "str",
        Any: "Any",
    }

    collection_types = {
        List: "List",
        Dict: "Dict",
        Optional: "Optional",
    }

    type_map = {}
    for base_type, name in base_types.items():
        type_map[base_type] = name
        for collection, collection_name in collection_types.items():
            if collection is Optional:
                type_map[Optional[base_type]] = name
            elif collection is Dict:
                # Provide both key and value types for Dict
                type_map[Dict[base_type, base_type]] = f"{collection_name}[{name},{name}]"
            else:
                type_map[collection[base_type]] = f"{collection_name}[{name}]"
    return type_map

def generate_example_dict() -> Dict[str, Any]:
    example_values = {
        "str": "<Your type-str response here>",
        "int": "<Your type-int response here>",
        "float": "<Your type-float response here>",
        "bool": "<Your type-bool response here>",
        "Any": "<Your response here (infer the type from context)>",
    }

    example_dict = {}
    for key, value in example_values.items():
        example_dict[key] = value
        example_dict[f"List[{key}]"] = [value]
        example_dict[f"List[List[{key}]]"] = [[value]]
        if key == "Dict[str,str]":
            example_dict[f"Dict[str,{key.split('[')[1]}"] = {value: value}
        elif key.startswith("Dict"):
            example_dict[key] = {value: value}
    return example_dict

def add_json_instructions_to_messages(
    system_message,
    user_message,
    response_model: Optional[Type[BaseModel]] = None,
    previously_failed_error_messages: List[str] = [],
) -> Tuple[str, str]:
    if response_model:
        dictified = response_model.schema()

        if "$defs" in dictified:
            raise ValueError("Nesting not supported in response model")
        type_hints = get_type_hints(response_model)

        type_map = generate_type_map()
        for k, v in type_hints.items():
            if v in type_map:
                type_hints[k] = type_map[v]
        
        example_dict = generate_example_dict()
        stringified = ""
        for key in type_hints:
            if type_hints[key] not in example_dict.keys():
                raise ValueError(f"Type {type_hints[key]} not supported. key- {key}")
            stringified += f"{key}: {example_dict[type_hints[key]]}\n"
        system_message += f"""\n\n
Please deliver your response in the following json format:
```json
{{
{stringified}
}}
```
"""
    if len(previously_failed_error_messages)!=0:
        system_message += f"""\n\n
Please take special care to follow the format exactly.
Keep in mind the following:
- Always use double quotes for strings

Here are some error traces from previous attempts:
{previously_failed_error_messages}
"""
    return system_message, user_message


def inject_structured_output_instructions(
    messages: List[Dict[str, str]],
    response_model: Optional[Type[BaseModel]] = None,
    previously_failed_error_messages: List[str] = [],
) -> List[Dict[str, str]]:
    prev_system_message_content = messages[0]["content"]
    prev_user_message_content = messages[1]["content"]
    system_message, user_message = add_json_instructions_to_messages(
        prev_system_message_content, prev_user_message_content, response_model, previously_failed_error_messages
    )
    messages[0]["content"] = system_message
    messages[1]["content"] = user_message
    return messages