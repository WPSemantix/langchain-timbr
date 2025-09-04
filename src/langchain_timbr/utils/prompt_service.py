import requests
from typing import Dict, Any, Optional, List, Union
from langchain.prompts.chat import ChatPromptTemplate
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
import json
import logging
from ..config import url as default_url, token as default_token, is_jwt, jwt_tenant_id as default_jwt_tenant_id, llm_timeout

logger = logging.getLogger(__name__)

# Global template cache shared across all PromptService instances
_global_template_cache: dict[Any, Any] = {}

class PromptService:
    def __init__(
        self, 
        conn_params: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        Initialize PromptService with connection parameters.
        
        Args:
            conn_params: Dictionary containing connection parameters
            **kwargs: Additional parameters for backward compatibility
        """
        # Extract relevant parameters from conn_params or use defaults
        if conn_params:
            url_value = conn_params.get('url') or default_url
            self.base_url = url_value.rstrip('/') if url_value else ''
            self.token = conn_params.get('token') or default_token
            self.is_jwt = conn_params.get('is_jwt', is_jwt)
            self.jwt_tenant_id = conn_params.get('jwt_tenant_id') or default_jwt_tenant_id
            self.timeout = conn_params.get('timeout') or llm_timeout
        else:
            # Fallback to kwargs for backward compatibility
            url_value = kwargs.get('url') or default_url
            self.base_url = url_value.rstrip('/') if url_value else ''
            self.token = str(kwargs.get('token') or default_token)
            self.is_jwt = kwargs.get('is_jwt', is_jwt)
            self.jwt_tenant_id = kwargs.get('jwt_tenant_id') or default_jwt_tenant_id
            self.timeout = kwargs.get('timeout') or llm_timeout

    
    def _get_headers(self) -> Dict[str, str]:
        """Get headers for API requests"""
        headers = {"Content-Type": "application/json"}

        if self.is_jwt:
          headers["x-jwt-token"] = self.token
          if self.jwt_tenant_id:
              headers["x-jwt-tenant-id"] = self.jwt_tenant_id
        elif self.token:
            headers["x-api-key"] = self.token
            
        return headers
    
    
    def _convert_template_response_to_chat_prompt(self, api_response: Union[List, Dict[str, Any]]) -> ChatPromptTemplate:
        """
        Convert API response containing template strings to ChatPromptTemplate
        
        Expected API response format:
        {
            "status": "success",
            "data": [
                {"type": "SystemMessage", "template": "You are a helpful SQL expert...", "role": "system"},
                {"type": "HumanMessage", "template": "BUSINESS QUESTION: {question}...", "role": "human"}
            ]
        }
        """
        # Handle response with status and data fields
        if isinstance(api_response, dict) and "status" in api_response:
            if api_response["status"] == "error":
                error_msg = api_response.get("data", "Unknown error from prompt service")
                raise Exception(f"Prompt service error: {error_msg}")
            elif api_response["status"] == "success" and "data" in api_response:
                # Process the data array
                data = api_response["data"]
                if isinstance(data, list):
                    return self._parse_template_array(data)
                else:
                    raise ValueError("Expected 'data' to be an array of template objects")
            else:
                raise ValueError(f"Invalid API response: unexpected status '{api_response['status']}' or missing 'data' field")
        
        raise ValueError("Invalid API response format: expected object with 'status' and 'data' fields")
    
    
    def _parse_template_array(self, templates: List[Dict[str, Any]]) -> ChatPromptTemplate:
        """
        Parse an array of template dictionaries into a ChatPromptTemplate
        
        Args:
            templates: List of template dictionaries
            
        Returns:
            ChatPromptTemplate object
        """
        message_templates = []
        for tmpl in templates:
            if not isinstance(tmpl, dict) or "template" not in tmpl:
                continue
                
            # Check type field (SystemMessage/HumanMessage) or role field (system/human)
            msg_type = tmpl.get("type", "").lower()
            msg_role = tmpl.get("role", "").lower()
            template_str = tmpl["template"]
            
            if msg_type == "systemmessage" or msg_role == "system":
                message_templates.append(SystemMessagePromptTemplate.from_template(template_str))
            elif msg_type == "humanmessage" or msg_role == "human":
                message_templates.append(HumanMessagePromptTemplate.from_template(template_str))
            else:
                # Default to HumanMessage for unknown types
                message_templates.append(HumanMessagePromptTemplate.from_template(template_str))
        
        return ChatPromptTemplate.from_messages(message_templates)
    
    
    def _fetch_template(self, endpoint: str) -> ChatPromptTemplate:
        """
        Fetch template from API service without any data parameters
        
        Args:
            endpoint: The API endpoint to call
            
        Returns:
            ChatPromptTemplate object
        """
        # Check global cache first
        if endpoint in _global_template_cache:
            logger.debug(f"Using cached template for endpoint: {endpoint}")
            return _global_template_cache[endpoint]
        
        url = f"{self.base_url}/timbr/api/{endpoint}"
        headers = self._get_headers()
        
        try:
            response = requests.post(
                url,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            api_response = response.json()
            chat_prompt = self._convert_template_response_to_chat_prompt(api_response)
            
            # Cache the template globally
            _global_template_cache[endpoint] = chat_prompt
            logger.debug(f"Cached template for endpoint: {endpoint}")
            
            return chat_prompt
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get template from service {url}: {str(e)}")
            raise Exception(f"Prompt service request failed: {str(e)}")
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON response from prompt service: {str(e)}")
            raise Exception(f"Invalid response from prompt service: {str(e)}")
        except Exception as e:
            logger.error(f"Error processing prompt service response: {str(e)}")
            raise Exception(f"Error processing prompt service response: {str(e)}")
    
    
    def get_identify_concept_template(self) -> ChatPromptTemplate:
        """
        Get identify concept template from API service (cached)
        
        Returns:
            ChatPromptTemplate object
        """
        return self._fetch_template("llm_prompts/identify_concept")
    
    
    def get_generate_sql_template(self) -> ChatPromptTemplate:
        """
        Get generate SQL template from API service (cached)
        
        Returns:
            ChatPromptTemplate object
        """
        return self._fetch_template("llm_prompts/generate_sql")
    
    
    def get_generate_answer_template(self) -> ChatPromptTemplate:
        """
        Get generate answer template from API service (cached)
        
        Returns:
            ChatPromptTemplate object
        """
        return self._fetch_template("llm_prompts/generate_answer")
    
    
    def clear_cache(self):
        """Clear the global template cache"""
        _global_template_cache.clear()
        logger.info("Global prompt template cache cleared")


class PromptTemplateWrapper:
    """
    Wrapper class that mimics the original ChatPromptTemplate behavior
    but uses cached templates from the external API service
    """
    
    def __init__(self, prompt_service: PromptService, template_method: str):
        self.prompt_service = prompt_service
        self.template_method = template_method
        self._cached_template = None
    
    
    def format_messages(self, **kwargs) -> List:
        """
        Format messages using the cached template
        
        Args:
            **kwargs: Parameters for the prompt template
            
        Returns:
            List of LangChain message objects
        """
        # Get the cached template
        if self._cached_template is None:
            method = getattr(self.prompt_service, self.template_method)
            self._cached_template = method()
        
        # Format the template with the provided kwargs
        return self._cached_template.format_messages(**kwargs)


# Individual prompt template getter functions
def get_determine_concept_prompt_template(
    conn_params: Optional[dict] = None
) -> PromptTemplateWrapper:
    """
    Get determine concept prompt template wrapper
    
    Args:
        conn_params: Connection parameters including url, token, is_jwt, and jwt_tenant_id

    Returns:
        PromptTemplateWrapper for determine concept
    """
    prompt_service = PromptService(conn_params=conn_params)
    return PromptTemplateWrapper(prompt_service, "get_identify_concept_template")


def get_generate_sql_prompt_template(
    conn_params: Optional[dict] = None
) -> PromptTemplateWrapper:
    """
    Get generate SQL prompt template wrapper
    
    Args:
        conn_params: Connection parameters including url, token, is_jwt, and jwt_tenant_id
        
    Returns:
        PromptTemplateWrapper for generate SQL
    """
    prompt_service = PromptService(conn_params=conn_params)
    return PromptTemplateWrapper(prompt_service, "get_generate_sql_template")


def get_qa_prompt_template(
    conn_params: Optional[dict] = None
) -> PromptTemplateWrapper:
    """
    Get QA prompt template wrapper
    
    Args:
        conn_params: Connection parameters including url, token, is_jwt, and jwt_tenant_id

    Returns:
        PromptTemplateWrapper for QA
    """
    prompt_service = PromptService(conn_params=conn_params)
    return PromptTemplateWrapper(prompt_service, "get_generate_answer_template")


# Global prompt service instance (updated signature)
def get_prompt_service(
    conn_params: Optional[dict] = None
) -> PromptService:
    """
    Get or create a prompt service instance
    
    Args:
        conn_params: Connection parameters including url, token, is_jwt, and jwt_tenant_id

    Returns:
        PromptService instance
    """
    return PromptService(conn_params=conn_params)


# Global cache management functions
def clear_global_template_cache():
    """Clear the global template cache"""
    _global_template_cache.clear()
    logger.info("Global prompt template cache cleared")


def get_cache_status():
    """Get information about the global template cache"""
    return {
        "cached_endpoints": list(_global_template_cache.keys()),
        "cache_size": len(_global_template_cache)
    }
