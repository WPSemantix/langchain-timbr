![Timbr logo description](https://timbr.ai/wp-content/uploads/2025/01/logotimbrai230125.png)

[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2FWPSemantix%2Flangchain-timbr.svg?type=shield&issueType=security)](https://app.fossa.com/projects/git%2Bgithub.com%2FWPSemantix%2Flangchain-timbr?ref=badge_shield&issueType=security)
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2FWPSemantix%2Flangchain-timbr.svg?type=shield&issueType=license)](https://app.fossa.com/projects/git%2Bgithub.com%2FWPSemantix%2Flangchain-timbr?ref=badge_shield&issueType=license)

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue)](https://www.python.org/downloads/release/python-3921/)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31017/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-31112/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3129/)

# Timbr Langchain LLM SDK

Timbr langchain LLM SDK is a Python SDK that extends LangChain and LangGraph with custom agents, chains, and nodes for seamless integration with the Timbr semantic layer. It enables converting natural language prompts into optimized semantic-SQL queries and executing them directly against your data.

## Dependencies
- Access to a timbr-server
- Python from 3.9.13 or newer

## Installation

### Using pip
```bash
python -m pip install langchain-timbr
```

### Using pip from github
```bash
pip install git+https://github.com/WPSemantix/langchain-timbr
```

## Documentation

For comprehensive documentation and usage examples, please visit:

- [Timbr LangChain Documentation](https://docs.timbr.ai/doc/docs/integration/langchain-sdk)
- [Timbr LangGraph Documentation](https://docs.timbr.ai/doc/docs/integration/langgraph-sdk)

## Configuration

The SDK requires several environment variables to be configured. See [`src/langchain_timbr/config.py`](src/langchain_timbr/config.py) for all available configuration options.
