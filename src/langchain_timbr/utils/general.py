import os
import json

### A global helper functions to use across the project

def parse_list(input_value, separator=',') -> list[str]:
    try:
        if isinstance(input_value, str):
            return [item.strip() for item in input_value.split(separator) if item.strip()]
        elif isinstance(input_value, list):
            return [item.strip() for item in input_value if item.strip()]
        return []
    except Exception as e:
        raise ValueError(f"Failed to parse list value: {e}")


def to_boolean(value) -> bool:
    try:
        if isinstance(value, str):
            return value.lower() in ['true', '1']
        return bool(value)
    except Exception as e:
        raise ValueError(f"Failed to parse boolean value: {e}")


def to_integer(value) -> int:
    try:
        return int(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Failed to parse integer value: {e}")


def is_llm_type(llm_type, enum_value):
    """Check if llm_type equals the enum value or its name, case-insensitive."""
    if llm_type == enum_value:
        return True
    
    if isinstance(llm_type, str):
        llm_type_lower = llm_type.lower()
        enum_name_lower = enum_value.name.lower() if enum_value.name else ""
        enum_value_lower = enum_value.value.lower() if isinstance(enum_value.value, str) else ""

        return (
            llm_type_lower == enum_name_lower or
            llm_type_lower == enum_value_lower or
            llm_type_lower.startswith(enum_name_lower) or # Usecase for snowflake which its type is the provider name + the model name
            llm_type_lower.startswith(enum_value_lower) or
            llm_type_lower in enum_value_lower # Check if the enum value includes the llm type - when providing partial name
        )

    return False
  

def is_support_temperature(llm_type: str, llm_model: str) -> bool:
    """
    Check if the LLM model supports temperature setting.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(current_dir, 'temperature_supported_models.json')

    try:
        with open(json_file_path, 'r') as f:
            temperature_supported_models = json.load(f)
        
        # Check if llm_type exists and llm_model is in its list
        if llm_type in temperature_supported_models:
            return llm_model in temperature_supported_models[llm_type]
        
        return False
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return False