from langchain.llms.base import LLM
import requests
from typing import Optional, List

class TimbrLlmWrapper(LLM):
    def __init__(self, url: str, api_key: str, temperature: Optional[float] = 0):
        """
        ***TBD, Not ready yet.***
        
        Custom LLM implementation for timbr LLM wrapped with a proxy server.
        
        :param url: URL of the proxy server wrapping timbr LLM.
        :param api_key: API key for authentication with the proxy server.
        :param temperature: Sampling temperature for the model.
        """
        self.url = url
        self.api_key = api_key
        self.temperature = temperature

    @property
    def _llm_type(self) -> str:
        return "timbr"


    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Sends the prompt to the proxy server and returns the response.
        """
        headers = { "Authorization": f"Bearer {self.api_key}" }
        payload = {
            "prompt": prompt,
            "temperature": self.temperature,
            "stop": stop,
        }
        
        response = requests.post(self.url, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json().get("response", "")
        else:
            raise ValueError(f"Error communicating with timbr proxy: {response.text}")

