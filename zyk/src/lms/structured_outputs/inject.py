import json
from typing import Any, Dict, List, Optional, Tuple, Type, get_type_hints, get_args, get_origin, Union
from pydantic import BaseModel
import warnings


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
                # Handle generic Dict type
                type_map[Dict] = "Dict[Any,Any]"
                # Provide both key and value types for Dict
                type_map[Dict[base_type, base_type]] = f"{collection_name}[{name},{name}]"
                # Handle Dict[Any, Any] explicitly
                type_map[Dict[Any, Any]] = "Dict[Any,Any]"
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
    
    # Add example for Dict[Any,Any]
    example_dict["Dict[Any,Any]"] = {"<key>": "<value>"}
    
    return example_dict

base_type_examples = {
    int: ("int", 42),
    float: ("float", 3.14),
    bool: ("bool", True),
    str: ("str", "example"),
    Any: ("Any", "<Any value>"),
}

def get_type_string(type_hint):
    origin = get_origin(type_hint)
    args = get_args(type_hint)

    if origin is None:
        if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
            return type_hint.__name__
        else:
            return base_type_examples.get(type_hint, ("Unknown", "unknown"))[0]
    elif origin in (list, List):
        elem_type = get_type_string(args[0])
        return f"List[{elem_type}]"
    elif origin in (dict, Dict):
        key_type = get_type_string(args[0])
        value_type = get_type_string(args[1])
        return f"Dict[{key_type}, {value_type}]"
    elif origin is Union:
        non_none_types = [t for t in args if t is not type(None)]
        if len(non_none_types) == 1:
            return f"Optional[{get_type_string(non_none_types[0])}]"
        else:
            union_types = ", ".join(get_type_string(t) for t in non_none_types)
            return f"Union[{union_types}]"
    else:
        return "Any"

def get_example_value(type_hint):
    origin = get_origin(type_hint)
    args = get_args(type_hint)
    
    if origin is None:
        if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
            example = {}
            for field_name, field_info in type_hint.model_fields.items():
                # Updated attribute from type_ to annotation
                example[field_name] = get_example_value(field_info.annotation)
            return example
        else:
            return base_type_examples.get(type_hint, ("Unknown", "unknown"))[1]
    elif origin in (list, List):
        return [get_example_value(args[0])]
    elif origin in (dict, Dict):
        if not args or len(args) < 2:
            warnings.warn(
                f"Dictionary type hint {type_hint} missing type arguments. "
                "Defaulting to Dict[str, Any].",
                UserWarning
            )
            return {"example_key": "<Any value>"}  # Default for Dict[str, Any]
        return {get_example_value(args[0]): get_example_value(args[1])}
    elif origin is Union:
        non_none_types = [t for t in args if t is not type(None)]
        return get_example_value(non_none_types[0])
    else:
        return "<Unknown>"

def add_json_instructions_to_messages(
    system_message,
    user_message,
    response_model: Optional[Type[BaseModel]] = None,
    previously_failed_error_messages: List[str] = [],
) -> Tuple[str, str]:
    if response_model:
        type_hints = get_type_hints(response_model)
        stringified_fields = {}
        for key, type_hint in type_hints.items():
            example_value = get_example_value(type_hint)
            field = response_model.model_fields[key]  # Updated for Pydantic v2
            
            # Adjusted for Pydantic v2
            field_description = field.description if hasattr(field, 'description') else None
            
            if field_description:
                stringified_fields[key] = (example_value, field_description)
            else:
                stringified_fields[key] = example_value
        example_json = json.dumps(
            {k: v[0] if isinstance(v, tuple) else v for k, v in stringified_fields.items()},
            indent=4
        )
        description_comments = "\n".join(
            f'// {k}: {v[1]}' for k, v in stringified_fields.items() if isinstance(v, tuple)
        )
        system_message += f"""\n\n
Please deliver your response in the following JSON format:
```json
{example_json}
```
{description_comments}
"""
    if len(previously_failed_error_messages) != 0:
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
