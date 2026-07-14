import os
from .utils.general import to_boolean, to_integer, parse_list

# MUST HAVE VARIABLES
url = os.environ.get('TIMBR_URL')
token = os.environ.get('TIMBR_TOKEN')
ontology = os.environ.get('TIMBR_ONTOLOGY', os.environ.get('ONTOLOGY', 'system_db'))
thrift_host = os.environ.get('THRIFT_HOST', url.split("//")[-1].split(":")[0] if url else 'localhost')
thrift_port = to_integer(os.environ.get('THRIFT_PORT', 11000))

# OPTIONAL VARIABLES
is_jwt = to_boolean(os.environ.get('IS_JWT', 'false'))
jwt_tenant_id = os.environ.get('JWT_TENANT_ID', None)

cache_timeout = to_integer(os.environ.get('CACHE_TIMEOUT', 120))
ignore_tags = parse_list(os.environ.get('IGNORE_TAGS', 'icon'))
ignore_tags_prefix = parse_list(os.environ.get('IGNORE_TAGS_PREFIX', 'mdx.,bli.'))

llm_type = os.environ.get('LLM_TYPE', os.environ.get('TIMBR_LLM_TYPE'))
llm_model = os.environ.get('LLM_MODEL', os.environ.get('TIMBR_LLM_MODEL'))
llm_api_key = os.environ.get('TIMBR_LLM_API_KEY', os.environ.get('TIMBR_LLM_APIKEY', os.environ.get('LLM_API_KEY')))
llm_temperature = os.environ.get('LLM_TEMPERATURE', os.environ.get('TIMBR_LLM_TEMPERATURE', 0.0))
llm_additional_params = os.environ.get('LLM_ADDITIONAL_PARAMS', os.environ.get('TIMBR_LLM_ADDITIONAL_PARAMS', ''))
llm_timeout = to_integer(os.environ.get('LLM_TIMEOUT', os.environ.get('TIMBR_LLM_TIMEOUT', 120)))  # Default 120 seconds timeout

# Optional for Azure OpenAI with Service Principal authentication
llm_tenant_id = os.environ.get('LLM_TENANT_ID', os.environ.get('TIMBR_LLM_TENANT_ID', None))
llm_client_id = os.environ.get('LLM_CLIENT_ID', os.environ.get('TIMBR_LLM_CLIENT_ID', None))
llm_client_secret = os.environ.get('LLM_CLIENT_SECRET', os.environ.get('TIMBR_LLM_CLIENT_SECRET', None))
llm_endpoint = os.environ.get('LLM_ENDPOINT', os.environ.get('TIMBR_LLM_ENDPOINT', None))
llm_api_version = os.environ.get('LLM_API_VERSION', os.environ.get('TIMBR_LLM_API_VERSION', None))
llm_scope = os.environ.get('LLM_SCOPE', os.environ.get('TIMBR_LLM_SCOPE', "https://cognitiveservices.azure.com/.default"))  # e.g. "api://<your-client-id>/.default"

# Whether to enable reasoning during SQL generation
enable_reasoning = to_boolean(os.environ.get('ENABLE_REASONING', 'false'))
reasoning_steps = to_integer(os.environ.get('REASONING_STEPS', 2))

should_validate_sql = to_boolean(os.environ.get('SHOULD_VALIDATE_SQL', os.environ.get('LLM_SHOULD_VALIDATE_SQL', 'true')))
retry_if_no_results = to_boolean(os.environ.get('RETRY_IF_NO_RESULTS', os.environ.get('LLM_RETRY_IF_NO_RESULTS', 'true')))
llm_default_limit = to_integer(os.environ.get('LLM_DEFAULT_LIMIT', os.environ.get('TIMBR_LLM_DEFAULT_LIMIT', 100)))  # Default max result limit for LLM responses

enable_trace = to_boolean(os.environ.get('TIMBR_ENABLE_TRACE', 'true'))
enable_history = to_boolean(os.environ.get('TIMBR_ENABLE_HISTORY', 'true'))
history_save_results = to_boolean(os.environ.get('TIMBR_HISTORY_SAVE_RESULTS', 'false'))

enable_memory = to_boolean(os.environ.get('TIMBR_ENABLE_MEMORY', 'true'))
memory_window_size = to_integer(os.environ.get('TIMBR_MEMORY_WINDOW_SIZE', 3))

# Whether to offer the virtual `ontology_metadata` concept for questions about the
# ontology model itself (concepts / properties / measures, relationships, views,
# and backing source tables).
enable_ontology_questions = to_boolean(os.environ.get('ENABLE_ONTOLOGY_QUESTIONS', 'true'))

# Identify-concept context builder. When ON, the first NL2SQL step (anchor-concept
# selection) renders a hierarchical, signal-rich catalog of the already-filtered
# candidate concepts/views instead of the flat name+description list. Every
# candidate remains visible to the LLM — the builder never drops a candidate, it
# only trims the cheapest signal (descriptions, then relationships/measures) to
# stay under the token budget. Default ON; falls back to the legacy render on any
# error. Set ENABLE_IDENTIFY_CONCEPT_CONTEXT=false to force the legacy path.
enable_identify_concept_context = to_boolean(os.environ.get('ENABLE_IDENTIFY_CONCEPT_CONTEXT', 'true'))
# Token ladder thresholds (cl100k_base) — see identify_concept_context.py.
identify_concept_context_desc_trim_tokens = to_integer(os.environ.get('IDENTIFY_CONCEPT_CONTEXT_DESC_TRIM_TOKENS', 8000))
identify_concept_context_rel_trim_tokens = to_integer(os.environ.get('IDENTIFY_CONCEPT_CONTEXT_REL_TRIM_TOKENS', 12000))
identify_concept_context_hard_limit_tokens = to_integer(os.environ.get('IDENTIFY_CONCEPT_CONTEXT_HARD_LIMIT_TOKENS', 20000))
identify_concept_context_desc_max_chars = to_integer(os.environ.get('IDENTIFY_CONCEPT_CONTEXT_DESC_MAX_CHARS', 200))
identify_concept_context_measure_cap = to_integer(os.environ.get('IDENTIFY_CONCEPT_CONTEXT_MEASURE_CAP', 10))
# Per-parent cap on rendered sub-type hints; overflow is summarized as "+N more".
identify_concept_context_hint_cap = to_integer(os.environ.get('IDENTIFY_CONCEPT_CONTEXT_HINT_CAP', 15))
# Trigram matcher (sub-type hints + relationship-axis trim). Ratio is scaled by 100
# from the env int (e.g. 65 -> 0.65) so it fits the to_integer config convention.
identify_concept_context_trigram_threshold = to_integer(os.environ.get('IDENTIFY_CONCEPT_CONTEXT_TRIGRAM_THRESHOLD', 65)) / 100.0
identify_concept_context_trigram_floor = to_integer(os.environ.get('IDENTIFY_CONCEPT_CONTEXT_TRIGRAM_FLOOR', 3))

enable_technical_context = to_boolean(os.environ.get('ENABLE_TECHNICAL_CONTEXT', 'true'))
technical_context_mode = os.environ.get('TECHNICAL_CONTEXT_MODE', 'auto')
technical_context_max_tokens = to_integer(os.environ.get('TECHNICAL_CONTEXT_MAX_TOKENS', 3000))
technical_context_properties = parse_list(os.environ.get('TECHNICAL_CONTEXT_PROPERTIES', ''))

# Dynamic metadata-context assembly (Plan 2). Default 'static' for backward
# compatibility — the static path is bit-for-bit identical to current behavior.
metadata_context_mode = os.environ.get('METADATA_CONTEXT_MODE', 'dynamic')        # static | dynamic

# SQL-gen metadata budget (final context passed to SQL gen). SOFT cap only —
# triggers the cascade + waypoint filter to compress when exceeded. There is
# NO hard ceiling on this budget: per the "dynamic-over-budget is preferred
# over static-but-much-larger" principle, oversizing past the soft cap is
# logged but emits the rebuilt strings as-is. The old hard-revert-to-static
# branch was removed.
metadata_context_max_tokens = to_integer(os.environ.get('METADATA_CONTEXT_MAX_TOKENS', 12000))    # soft

# DDL prompt budget (filter LLM input — separate from SQL-gen budget). These
# two knobs are config-only (not exposed at the chain/agent surface) since
# they are operator-tuning, not application-level concerns.
metadata_context_filter_max_tokens = to_integer(os.environ.get('METADATA_CONTEXT_FILTER_MAX_TOKENS', 6000))                 # soft
metadata_context_filter_max_tokens_hard_ceiling = to_integer(os.environ.get('METADATA_CONTEXT_FILTER_MAX_TOKENS_HARD_CEILING', 12000))          # hard (log-only — cascade emits stage-4 without failing)

# Planner retry budget. Config-only (not exposed at the chain/agent surface).
# When exhausted, the dynamic pipeline returns empty and the wiring layer's
# outer try/except falls back to static metadata strings. No internal BFS /
# shortest-path / pre-filter rescue exists (see build_filtered.py).
metadata_context_dynamic_retry = to_integer(os.environ.get('METADATA_CONTEXT_DYNAMIC_RETRY', 2))
static_attempt_edge_threshold = to_integer(os.environ.get('STATIC_ATTEMPT_EDGE_THRESHOLD', 100))
include_logic_concepts = to_boolean(os.environ.get('INCLUDE_LOGIC_CONCEPTS', 'false'))

# Concept pre-filter budget — runs when estimated DDL exceeds metadata_context_filter_max_tokens,
# narrowing the candidate concept set via an LLM call before serialization.
max_concept_prefilter_token = to_integer(os.environ.get('MAX_CONCEPT_PREFILTER_TOKEN', 2000))

# Concept pre-filter count trigger — also fires when the detail-band concept
# count would meet/exceed this number, independent of token size. Demotes the
# overflow concepts into the menu band (NOT dropped — they remain visible as
# names the LLM can recover via expand_to).
max_detail_concepts = to_integer(os.environ.get('MAX_DETAIL_CONCEPTS', 20))

# Menu-band outer bound — the max hop-distance from the anchor that the
# subgraph BFS will reach. Concepts at hops 1..detail_depth (the per-chain
# graph_depth) are rendered with full Compact-DDL detail; concepts at
# detail_depth+1..max_graph_depth are rendered as names-only in the `## REACHABLE`
# band, recoverable via expand_to. Beyond max_graph_depth is treated as out of
# scope (the planner can only emit reanchor for those, not expand_to).
# Validation: callers must satisfy graph_depth < max_graph_depth.
max_graph_depth = to_integer(os.environ.get('MAX_GRAPH_DEPTH', 3))

# Knowledge-base search client (kbclient.py). Thin HTTP client over
# POST /timbr/api/kb/search plus opt-in LRU cache invalidated by polling
# MAX(changed_on) on timbr.sys_knowledgebase_examples. Thresholds are scaled
# by 100 from the env int (e.g. 80 -> 0.80) to fit the to_integer convention.
kb_search_timeout = to_integer(os.environ.get('KB_SEARCH_TIMEOUT', 30))            # per-request seconds
kb_top_k = to_integer(os.environ.get('KB_TOP_K', 5))                              # 1 <= top_k <= 20
kb_high_threshold = to_integer(os.environ.get('KB_HIGH_THRESHOLD', 80)) / 100.0    # 0.0 <= x <= 1.0
kb_medium_threshold = to_integer(os.environ.get('KB_MEDIUM_THRESHOLD', 50)) / 100.0  # 0.0 <= x <= high
kb_max_retries = to_integer(os.environ.get('KB_MAX_RETRIES', 2))                  # retries on 502/503/504/conn
# Client-side cache. TTL None (env unset) disables caching entirely.
_kb_cache_ttl_env = os.environ.get('KB_CACHE_TTL_SECONDS', None)
kb_cache_ttl_seconds = to_integer(_kb_cache_ttl_env) if _kb_cache_ttl_env not in (None, '') else None
kb_cache_max_entries = to_integer(os.environ.get('KB_CACHE_MAX_ENTRIES', 128))

# Knowledge-base rules (timbr.sys_knowledgebase_rules) injected into the NL2SQL
# pipeline stages. Curated and small, so cached with a short fixed TTL and
# re-validated by MAX(changed_on). enable flag is a kill-switch; when OFF the
# pipeline behaves exactly as before (rules are never fetched or injected).
kb_rules_cache_ttl_seconds = to_integer(os.environ.get('KB_RULES_CACHE_TTL_SECONDS', 60))

# Knowledge-base example retrieval in the conversation-memory flow. When ON, the
# memory subsystem probes for available knowledge bases (agent-first, else
# ontology) and folds approved reference examples into the SQL-generation prompts.
# Works independently of memory: KB retrieval runs whenever an agent/ontology is
# present, even if TIMBR_ENABLE_MEMORY is false.
enable_knowledge_base = to_boolean(os.environ.get('ENABLE_KNOWLEDGE_BASE', 'true'))
# DEV/test aid: when the live KB search returns no matches, emit a single
# hard-coded example so classifier selection + prompt injection can be validated
# end-to-end without live KB data. Default OFF — never affects production.
kb_fallback_example = to_boolean(os.environ.get('KB_FALLBACK_EXAMPLE', 'false'))