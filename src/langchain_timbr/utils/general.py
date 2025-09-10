import os
from typing import Any, Optional, Union
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


def parse_additional_params(value) -> dict:
    """
    Parse additional parameters from string format 'a=1,b=2' or return dict as-is.
    
    Args:
        value: String in format 'key=value,key2=value2', JSON string, or dict
        
    Returns:
        Dictionary of parsed parameters
    """
    try:
        if isinstance(value, dict):
            return {k.lower(): v for k, v in value.items()}
        elif isinstance(value, str) and value.strip():
            # Try to parse as JSON first
            stripped_value = value.strip()
            if stripped_value.startswith('{') and stripped_value.endswith('}'):
                try:
                    return json.loads(stripped_value)
                except json.JSONDecodeError:
                    pass
            
            # Fall back to key=value parsing
            params = {}
            for pair in (value.split('&') if '&' in value else value.split(',')):
                if '=' in pair:
                    key, val = pair.split('=', 1)
                    params[key.strip().lower()] = val.strip()
                elif ':' in pair:
                    key, val = pair.split(':', 1)
                    params[key.strip().lower()] = val.strip()
            return params
        return {}
    except Exception as e:
        raise ValueError(f"Failed to parse additional parameters: {e}")


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
  

def validate_timbr_connection_params(url: Optional[str] = None, token: Optional[str] = None) -> None:
    """
    Validate that required Timbr connection parameters are provided.
    
    Args:
        url: Timbr server URL
        token: Timbr authentication token
        
    Raises:
        ValueError: If URL or token are not provided with clear instructions
    """
    if not url:
        raise ValueError("URL must be provided either through the 'url' parameter or by setting the 'TIMBR_URL' environment variable")
    if not token:
        raise ValueError("Token must be provided either through the 'token' parameter or by setting the 'TIMBR_TOKEN' environment variable")


def is_support_temperature(llm_type: str, llm_model: str) -> bool:
    """
    Check if the LLM model supports temperature setting.
    """
    supported_models = get_supported_models(llm_type)
    return llm_model in supported_models


def get_supported_models(llm_type: str) -> list[str]:
    """
    Get the list of supported models for a given LLM type.
    
    Args:
        llm_type (str): The LLM type to get supported models for
        
    Returns:
        list[str]: List of supported model names for the given LLM type.
                   Returns empty list if llm_type is not found in the JSON file.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(current_dir, 'temperature_supported_models.json')

    try:
        with open(json_file_path, 'r') as f:
            temperature_supported_models = json.load(f)
        
        # Return the list of models for the given llm_type, or empty list if not found
        return temperature_supported_models.get(llm_type, [])
        
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def pop_param_value(
    params_dict: dict,
    opt_keys: Union[str, list[str]],
    default: Any=None,
):
    """
    Retrieve the value for the first matching key from params_dict.
    
    Args:
        params_dict (dict): Dictionary to search for keys
        opt_keys (str or list[str]): Key or list of keys to look for
        default: Default value to return if no keys are found
        
    Returns:
        The value corresponding to the first found key, or default if none found.
    """
    if isinstance(opt_keys, str):
        opt_keys = [opt_keys]
    
    for key in opt_keys:
        if key in params_dict:
            return params_dict.pop(key)
    return default
