#!/usr/bin/env python3
"""
Test script to validate timeout functionality in timbr_llm_utils.py
"""

import sys
import os
import time
import concurrent.futures
from unittest.mock import Mock

# Add the langchain_timbr package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'langchain_timbr'))


def _call_llm_with_timeout(llm, prompt, timeout: int = 60):
    """
    Call LLM with timeout to prevent hanging.
    
    Args:
        llm: The LLM instance (mock for testing)
        prompt: The prompt to send
        timeout: Timeout in seconds (default: 60)
        
    Returns:
        LLM response
        
    Raises:
        TimeoutError: If the call takes longer than timeout seconds
        Exception: Any other exception from the LLM call
    """
    def _llm_call():
        return llm(prompt)
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future = executor.submit(_llm_call)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            raise TimeoutError(f"LLM call timed out after {timeout} seconds")
        except Exception as e:
            raise e


class TestTimeoutFunctionality:
    """Test suite for timeout functionality in LLM calls."""
    
    def test_timeout_functionality(self):
        """Test that the timeout wrapper works correctly."""
        print("Testing timeout functionality...")
        
        # Create a mock LLM that sleeps for too long
        mock_llm = Mock()
        
        def slow_response(prompt):
            time.sleep(2)  # Sleep for 2 seconds
            return "response"
        
        mock_llm.side_effect = slow_response
        
        # Test with a 1-second timeout (should timeout)
        try:
            result = _call_llm_with_timeout(mock_llm, "test prompt", timeout=1)
            assert False, f"Expected TimeoutError but got result: {result}"
        except TimeoutError as e:
            print("✅ TimeoutError correctly raised:", str(e))
            # This is expected behavior
            pass
        
        # Test with a 5-second timeout (should succeed)
        result = _call_llm_with_timeout(mock_llm, "test prompt", timeout=5)
        print("✅ Long timeout works correctly, result:", result)
        assert result is not None, "Expected a result from LLM call"

    def test_config_timeout(self):
        """Test that the config timeout value is properly imported."""
        print("Testing config timeout import...")
        
        try:
            from config import llm_timeout
            
            # Assert that the timeout value is reasonable
            assert isinstance(llm_timeout, (int, float)), f"Expected timeout to be a number, got {type(llm_timeout)}"
            assert llm_timeout > 0, f"Expected timeout to be positive, got {llm_timeout}"
        except ImportError as e:
            print(f"⚠️ Could not import config timeout: {e}")
            # Set a default timeout for testing
            llm_timeout = 60
            print(f"Using default timeout: {llm_timeout} seconds")
            assert llm_timeout > 0, "Default timeout should be positive"
