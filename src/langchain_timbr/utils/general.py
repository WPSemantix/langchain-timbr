import os
from typing import Any, Optional, Union
import json
import re

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
    Handles JSON values correctly, including nested structures with commas.
    
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
                    return _try_parse_json_value(stripped_value)
                except json.JSONDecodeError:
                    pass
            
            # Check if complex parsing is needed (presence of nested structures)
            needs_complex_parsing = any(char in value for char in ['{', '}', '[', ']', '(', ')'])
            
            if not needs_complex_parsing:
                # Fast path: simple key=value pairs
                params = {}
                for pair in value.split(','):
                    if '=' in pair:
                        key, val = pair.split('=', 1)
                        params[key.strip().lower()] = _try_parse_json_value(val.strip())
                return params
            
            # Complex parsing that handles JSON values with commas
            params = {}
            i = 0
            while i < len(value):
                # Find the next key=value pair
                equals_pos = value.find('=', i)
                if equals_pos == -1:
                    break
                
                # Extract the key
                key = value[i:equals_pos].strip()
                
                # Find where the value starts
                value_start = equals_pos + 1
                
                # Determine where the value ends (considering nested structures)
                value_end = _find_value_end(value, value_start)
                
                # Extract and parse the value
                val = value[value_start:value_end].strip()
                
                # Try to parse the value as JSON if it looks like JSON
                parsed_val = _try_parse_json_value(val)
                
                params[key.lower()] = parsed_val
                
                # Move to the next parameter (skip comma if present)
                i = value_end
                if i < len(value) and value[i] == ',':
                    i += 1
            
            return params
        return {}
    except Exception as e:
        raise ValueError(f"Failed to parse additional parameters: {e}")


def _find_value_end(text: str, start: int) -> int:
    """
    Find the end position of a parameter value, considering nested structures.
    
    Args:
        text: The full text being parsed
        start: Starting position of the value
        
    Returns:
        End position of the value
    """
    depth = {'brace': 0, 'bracket': 0, 'paren': 0}
    in_quotes = False
    quote_char = None
    i = start
    
    while i < len(text):
        char = text[i]
        
        # Handle quotes
        if char in ('"', "'"):
            if not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char:
                # Check if it's escaped
                if i > 0 and text[i-1] != '\\':
                    in_quotes = False
                    quote_char = None
        
        # Only process structural characters if not in quotes
        elif not in_quotes:
            if char == '{':
                depth['brace'] += 1
            elif char == '}':
                depth['brace'] -= 1
            elif char == '[':
                depth['bracket'] += 1
            elif char == ']':
                depth['bracket'] -= 1
            elif char == '(':
                depth['paren'] += 1
            elif char == ')':
                depth['paren'] -= 1
            elif char == ',' and all(d == 0 for d in depth.values()):
                # Found a comma at the top level - this is the end of the value
                return i
        
        i += 1
    
    return i


def _try_parse_json_value(val: str) -> Any:
    """
    Try to parse a string value as JSON. If it fails, return the original string.
    Supports both single and double quotes for JSON-like structures.
    
    Args:
        val: String value to parse
        
    Returns:
        Parsed JSON value or original string
    """
    val = val.strip()
    
    # Check if it looks like JSON
    if (val.startswith('{') and val.endswith('}')) or \
       (val.startswith('[') and val.endswith(']')) or \
       val in ('true', 'false', 'null') or \
       (val.startswith('"') and val.endswith('"')):
        try:
            return json.loads(val)
        except json.JSONDecodeError:
            # Try converting single quotes to double quotes for Python-style dicts
            if val.startswith('{') or val.startswith('['):
                try:
                    # Replace single quotes with double quotes, handling escaped quotes
                    normalized = val.replace("\\'", "<<<ESCAPED_SINGLE>>>")
                    normalized = normalized.replace("'", '"')
                    normalized = normalized.replace("<<<ESCAPED_SINGLE>>>", "'")
                    return json.loads(normalized)
                except json.JSONDecodeError:
                    pass
    # Try to parse as number - validate format first
    if re.match(r'^-?\d+(\.\d+)?$', val):
        try:
            if '.' in val:
                return float(val)
            return int(val)
        except ValueError:
            pass
        pass
    
    return val


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
