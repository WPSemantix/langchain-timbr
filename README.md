![Timbr logo description](https://timbr.ai/wp-content/uploads/2025/01/logotimbrai230125.png)

[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2FWPSemantix%2Flangchain-timbr.svg?type=shield&issueType=security)](https://app.fossa.com/projects/git%2Bgithub.com%2FWPSemantix%2Flangchain-timbr?ref=badge_shield&issueType=security)
[![FOSSA Status](https://app.fossa.com/api/projects/git%2Bgithub.com%2FWPSemantix%2Flangchain-timbr.svg?type=shield&issueType=license)](https://app.fossa.com/projects/git%2Bgithub.com%2FWPSemantix%2Flangchain-timbr?ref=badge_shield&issueType=license)


[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31017/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-31112/)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3129/)

# Timbr LangChain LLM SDK

Timbr LangChain LLM SDK is a Python SDK that extends LangChain and LangGraph with custom agents, chains, and nodes for seamless integration with the Timbr semantic layer. It enables converting natural language prompts into optimized semantic-SQL queries and executing them directly against your data.

![Timbr LangGraph pipeline](https://docs.timbr.ai/doc/assets/images/timbr-langgraph-fcf8e2eb7e26dc9dfa8b56b62937281e.png)

## Dependencies

- Access to a timbr-server
- Python 3.10 or newer

## Installation

### Using pip

```bash
python -m pip install langchain-timbr
```

### Install with selected LLM providers

#### One of: openai, anthropic, google, azure_openai, snowflake, databricks, vertex_ai (or 'all')

```bash
python -m pip install 'langchain-timbr[<your selected providers, separated by comma w/o space>]'
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

The SDK uses environment variables for configuration. All configurations are optional - when set, they serve as default values for `langchain-timbr` provided tools. Below are all available configuration options:

### Configuration Options

#### Timbr Connection Settings

- **`TIMBR_URL`** - The URL of your Timbr server
- **`TIMBR_TOKEN`** - Authentication token for accessing the Timbr server
- **`TIMBR_ONTOLOGY`** - The ontology to use (also accepts `ONTOLOGY` as an alias)
- **`IS_JWT`** - Whether the token is a JWT token (true/false)
- **`JWT_TENANT_ID`** - Tenant ID for JWT authentication

#### Cache and Data Processing

- **`CACHE_TIMEOUT`** - Timeout for caching operations in seconds
- **`IGNORE_TAGS`** - Comma-separated list of tags to ignore during processing
- **`IGNORE_TAGS_PREFIX`** - Comma-separated list of tag prefixes to ignore during processing

#### LLM Configuration

- **`LLM_TYPE`** - The type of LLM provider to use
- **`LLM_MODEL`** - The specific model to use with the LLM provider
- **`LLM_API_KEY`** - API key or client secret for the LLM provider
- **`LLM_TEMPERATURE`** - Temperature setting for LLM responses (controls randomness)
- **`LLM_ADDITIONAL_PARAMS`** - Additional parameters to pass to the LLM
- **`LLM_TIMEOUT`** - Timeout for LLM requests in seconds
- **`LLM_TENANT_ID`** - LLM provider tenant/directory ID (Used for Service Principal authentication)
- **`LLM_CLIENT_ID`** - LLM provider client ID (Used for Service Principal authentication)
- **`LLM_CLIENT_SECRET`** - LLM provider client secret (Used for Service Principal authentication)
- **`LLM_ENDPOINT`** - LLM provider OpenAI endpoint URL
- **`LLM_API_VERSION`** - LLM provider API version
- **`LLM_SCOPE`** - LLM provider authentication scope
