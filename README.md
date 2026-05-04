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

#### One of: openai, anthropic, google, azure_openai, snowflake, databricks, vertex_ai, bedrock (or 'all')

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


#### Conversation Memory

- **`TIMBR_ENABLE_MEMORY`** - Enable conversation memory for follow-up question detection (true/false, default: false)
- **`TIMBR_MEMORY_WINDOW_SIZE`** - Number of past conversation turns to consider when detecting follow-ups (default: 3)


#### Technical Context

Technical context enriches SQL generation prompts with per-column statistical annotations

- **`ENABLE_TECHNICAL_CONTEXT`** - Enable or disable technical context enrichment (true/false, default: `true`)
- **`TECHNICAL_CONTEXT_MODE`** - Controls which columns receive annotations:
  - `include_all` — annotate every column that has statistics
  - `filter_matched` — annotate only columns whose values match the user's question
  - `auto` (default) — choose automatically based on token budget
- **`TECHNICAL_CONTEXT_MAX_TOKENS`** - Maximum token budget allocated for technical context annotations (default: `3000`)
- **`TECHNICAL_CONTEXT_PROPERTIES`** - Comma-separated whitelist of property names to fetch statistics for. When set, **only** these properties will have statistics loaded from the ontology. Properties not in this list are skipped, reducing query cost and response size. Empty (default) means all properties are fetched.

These options can also be passed directly to chain/node constructors:

```python
from langchain_timbr import ExecuteTimbrQueryChain

chain = ExecuteTimbrQueryChain(
    llm=llm,
    url="https://your-timbr-server",
    token="your-token",
    ontology="your_ontology",
    concepts_list="organization",
    enable_technical_context=True,
    technical_context_mode="auto",
    technical_context_max_tokens=3000,
    # Only fetch stats for these properties (whitelist):
    technical_context_properties=["region", "status", "country_code"],
    # Exclude these properties from schema display AND stats fetching (blacklist):
    exclude_properties=["entity_id", "entity_type", "entity_label"],
)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `enable_technical_context` | `Optional[bool]` | `True` | Enable/disable technical context enrichment |
| `technical_context_mode` | `Optional[str]` | `"auto"` | Column annotation strategy (`include_all`, `filter_matched`, `auto`) |
| `technical_context_max_tokens` | `Optional[int]` | `3000` | Maximum token budget for annotations |
| `technical_context_properties` | `Optional[list\|str]` | `[]` (all) | Whitelist of property names to fetch statistics for. Empty = no restriction |
| `exclude_properties` | `Optional[list\|str]` | `['entity_id', 'entity_type', 'entity_label']` | Properties excluded from schema display and statistics fetching |

> **Note:** `technical_context_properties` (whitelist) and `exclude_properties` (blacklist) can be used together. The whitelist restricts which properties get statistics fetched; the blacklist further removes properties from the fetched set.



#### Monitoring & History

- **`TIMBR_ENABLE_TRACE`** - Enable detailed trace logging for agent/chain execution (true/false, default: `false`)
- **`TIMBR_ENABLE_HISTORY`** - Enable query history tracking (true/false, default: `false`)
- **`TIMBR_HISTORY_SAVE_RESULTS`** - Whether to save query result rows in history (true/false, default: `false`)

## Monitoring & Tracing

The SDK supports optional execution tracing and query history recording. These can be enabled via environment variables (see above) or set directly on `TimbrSqlAgent`:

```python
from langchain_timbr import TimbrSqlAgent

agent = TimbrSqlAgent(
    llm=llm,
    url="https://your-timbr-server",
    token="your-token",
    ontology="your_ontology",
    enable_trace=True,        # Enable chain-level trace logging
    enable_history=True,      # Enable query history storage
    save_results=True,        # Save result rows in history
    conversation_id="conv-123",  # Group calls into a multi-turn conversation
)
```

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `enable_trace` | `Optional[bool]` | `TIMBR_ENABLE_TRACE` | Enable detailed trace logging per chain step |
| `enable_history` | `Optional[bool]` | `TIMBR_ENABLE_HISTORY` | Store query execution history |
| `save_results` | `Optional[bool]` | `TIMBR_HISTORY_SAVE_RESULTS` | Include result rows in history entries |
| `conversation_id` | `Optional[str]` | `None` | Associate multiple agent calls under one conversation |

## Benchmarking

The SDK includes a benchmarking utility to evaluate LLM query accuracy against a named benchmark defined in your Timbr server.

```python
from langchain_timbr.utils.benchmark import run_benchmark

results = run_benchmark(
    benchmark_name="my_benchmark",
    url="https://your-timbr-server",
    token="your-token",
    ontology="your_ontology",
    execution="full",             # "full" or "generate_sql_only"
    number_of_iterations=1,
    use_deterministic=True,       # Row-comparison scoring
    use_llm_judge=False,          # LLM-as-judge scoring
    llm_params={                  # Optional: override LLM at runtime
        "llm_type": "openai",
        "llm_model": "gpt-4o",
        "api_key": "sk-...",
    },
)
```

The `llm_params` dict accepts: `llm_type`, `llm_model` / `model`, `llm_api_key` / `api_key`. Temperature and timeout are managed automatically.

Results are returned as a dict keyed by question ID, with a `"_summary"` key containing aggregate statistics. Each result includes a `selected_entity` field identifying which ontology entity was used.
