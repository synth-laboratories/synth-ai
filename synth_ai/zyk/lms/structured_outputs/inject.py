import json
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    get_type_hints,
    get_args,
    get_origin,
    Union,
    Literal,
)
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
                type_map[Dict[base_type, base_type]] = (
                    f"{collection_name}[{name},{name}]"
                )
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
            # For Pydantic models, create a dictionary of field types
            field_types = {}
            for field_name, field_info in type_hint.model_fields.items():
                field_type = get_type_string(field_info.annotation)
                # Check for Literal type by looking at the origin
                if get_origin(field_info.annotation) is Literal:
                    literal_args = get_args(field_info.annotation)
                    field_type = f"Literal[{repr(literal_args[0])}]"
                field_types[field_name] = field_type
            return f"{type_hint.__name__}({', '.join(f'{k}: {v}' for k, v in field_types.items())})"
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
            # For unions of Pydantic models (like tool calls), show each variant
            union_types = []
            for t in non_none_types:
                if isinstance(t, type) and issubclass(t, BaseModel):
                    # Include discriminator field if present
                    discriminator = None
                    for field_name, field_info in t.model_fields.items():
                        if get_origin(field_info.annotation) is Literal:
                            literal_args = get_args(field_info.annotation)
                            discriminator = f"{field_name}={repr(literal_args[0])}"
                            break
                    type_str = t.__name__
                    if discriminator:
                        type_str += f"({discriminator})"
                    union_types.append(type_str)
                else:
                    union_types.append(get_type_string(t))
            return f"Union[{', '.join(union_types)}]"
    elif origin is Literal:
        # Handle Literal type directly
        return f"Literal[{repr(args[0])}]"
    else:
        return "Any"


def get_example_value(type_hint):
    origin = get_origin(type_hint)
    args = get_args(type_hint)

    if origin is None:
        if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
            example = {}
            union_docs = []
            for field_name, field_info in type_hint.model_fields.items():
                field_value, field_docs = get_example_value(field_info.annotation)
                if field_docs:
                    union_docs.extend(field_docs)

                # Handle literal fields by checking origin
                if get_origin(field_info.annotation) is Literal:
                    literal_args = get_args(field_info.annotation)
                    field_value = literal_args[0]

                # Include field description if available
                if field_info.description:
                    example[field_name] = {
                        "value": field_value,
                        "description": field_info.description,
                    }
                else:
                    example[field_name] = field_value
            return example, union_docs
        else:
            return base_type_examples.get(type_hint, ("Unknown", "unknown"))[1], []
    elif origin in (list, List):
        value, docs = get_example_value(args[0])
        return [value], docs
    elif origin in (dict, Dict):
        if not args or len(args) < 2:
            warnings.warn(
                f"Dictionary type hint {type_hint} missing type arguments. "
                "Defaulting to Dict[str, Any].",
                UserWarning,
            )
            return {"example_key": "<Any value>"}, []  # Default for Dict[str, Any]
        key_value, key_docs = get_example_value(args[0])
        value_value, value_docs = get_example_value(args[1])
        return {key_value: value_value}, key_docs + value_docs
    elif origin is Union:
        non_none_types = [t for t in args if t is not type(None)]
        # For unions of tool calls, use the first one but preserve the discriminator
        first_type = non_none_types[0]
        union_docs = []

        if all(
            isinstance(t, type) and issubclass(t, BaseModel) for t in non_none_types
        ):
            # Generate examples for all union variants
            for t in non_none_types:
                example = {}
                for field_name, field_info in t.model_fields.items():
                    field_value, _ = get_example_value(field_info.annotation)
                    if get_origin(field_info.annotation) is Literal:
                        literal_args = get_args(field_info.annotation)
                        field_value = literal_args[0]
                    example[field_name] = field_value
                union_docs.append(f"\nExample {t.__name__}:")
                union_docs.append(json.dumps(example, indent=2))

        # Return first type as main example
        if isinstance(first_type, type) and issubclass(first_type, BaseModel):
            example, _ = get_example_value(first_type)
            # Ensure tool_type or other discriminator is preserved
            for field_name, field_info in first_type.model_fields.items():
                if get_origin(field_info.annotation) is Literal:
                    literal_args = get_args(field_info.annotation)
                    if (
                        isinstance(example[field_name], dict)
                        and "value" in example[field_name]
                    ):
                        example[field_name]["value"] = literal_args[0]
                    else:
                        example[field_name] = literal_args[0]
            return example, union_docs
        main_example, docs = get_example_value(first_type)
        return main_example, docs + union_docs
    elif origin is Literal:
        # Handle Literal type directly
        return args[0], []
    else:
        return "<Unknown>", []


def add_json_instructions_to_messages(
    system_message,
    user_message,
    response_model: Optional[Type[BaseModel]] = None,
    previously_failed_error_messages: List[str] = [],
) -> Tuple[str, str]:
    if response_model:
        type_hints = get_type_hints(response_model)
        # print("Type hints", type_hints)
        stringified_fields = {}
        union_docs = []
        for key, type_hint in type_hints.items():
            # print("Key", key, "Type hint", type_hint)
            example_value, docs = get_example_value(type_hint)
            union_docs.extend(docs)
            field = response_model.model_fields[key]  # Updated for Pydantic v2

            # Adjusted for Pydantic v2
            field_description = (
                field.description if hasattr(field, "description") else None
            )

            if field_description:
                stringified_fields[key] = (example_value, field_description)
            else:
                stringified_fields[key] = example_value
        example_json = json.dumps(
            {
                k: v[0] if isinstance(v, tuple) else v
                for k, v in stringified_fields.items()
            },
            indent=4,
        )
        description_comments = "\n".join(
            f"// {k}: {v[1]}"
            for k, v in stringified_fields.items()
            if isinstance(v, tuple)
        )

        # print("Example JSON", example_json)
        # print("Description comments", description_comments)
        # print("Union documentation", "\n".join(union_docs))
        # raise Exception("Stop here")

        system_message += f"""\n\n
Please deliver your response in the following JSON format:
```json
{example_json}
```
{description_comments}
"""
    if len(union_docs) != 0:
        system_message += f"""\n\n
NOTE - the above example included a union type. Here are some additional examples for that union type:
{chr(10).join(union_docs)}
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
        prev_system_message_content,
        prev_user_message_content,
        response_model,
        previously_failed_error_messages,
    )
    messages[0]["content"] = system_message
    messages[1]["content"] = user_message
    return messages
