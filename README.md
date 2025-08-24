![Timbr logo description](https://timbr.ai/wp-content/uploads/2025/01/logotimbrai230125.png)
<!-- [![FOSSA Status](https://app.fossa.com/api/projects/custom%2B50508%2Fgithub.com%2FWPSemantix%2Ftimbr_python_http.svg?type=shield&issueType=license)](https://app.fossa.com/projects/custom%2B50508%2Fgithub.com%2FWPSemantix%2Ftimbr_python_http?ref=badge_shield&issueType=license) -->
[![Python 3.9-3.12](https://img.shields.io/badge/python-3.9--3.12-blue.svg)](https://www.python.org/downloads/)


# Timbr Langchain LLM SDK

Timbr langchain LLM SDK is a Python SDK that extends LangChain and LangGraph with custom agents, chains, and nodes for seamless integration with the Timbr semantic layer. It enables converting natural language prompts into optimized semantic-SQL queries and executing them directly against your data.

## Installation

### Using pip
```bash
python -m pip install langchain-timbr
```

## Documentation

For comprehensive documentation and usage examples, please visit:

- [Timbr LangChain Documentation](https://docs.timbr.ai/doc/docs/integration/langchain-sdk)
- [Timbr LangGraph Documentation](https://docs.timbr.ai/doc/docs/integration/langgraph-sdk)

## Configuration

The SDK requires several environment variables to be configured. See [`src/langchain_timbr/config.py`](src/langchain_timbr/config.py) for all available configuration options.
