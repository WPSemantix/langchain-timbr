# Chain Execution History Logs & Running Status — Design Plan

## Context

Users need a user-facing status UI to monitor executions in real time, review question history, and optionally drill into per-chain trace details. Three system tables are introduced, following the exact HTTP API pattern of the existing benchmark feature.

- `sys_agents_running` — live status (one row per active question, deleted on completion)
- `sys_agents_history` — permanent record per question, keyed by `query_id`
- `sys_agents_chain_trace_log` — optional per-chain-step detail; links to `sys_agents_history.query_id`

Chain trace logging is **optional**, enabled per agent config (`sys_agents_options.chain_trace_enabled`) and/or per invocation parameter. When disabled, only running + history are written.

---

## Table Schemas

### `sys_agents_running` — Live Status

One row per active execution. Server deletes it when the history POST is received.

| Column | Type | Notes |
|---|---|---|
| `query_id` | VARCHAR | UUID, matches history row when finalized |
| `agent_name` | VARCHAR | Always required |
| `chain_type` | VARCHAR | `"TimbrSqlAgent"`, `"ExecuteTimbrQueryChain"`, etc. |
| `ontology` | VARCHAR | Known at start or empty |
| `schema` | VARCHAR | `dtimbr` / `vtimbr` or empty |
| `prompt` | TEXT | Full user prompt |
| `start_time` | DATETIME | `YYYY-MM-DD HH:MM:SS` |
| `current_step` | VARCHAR | See step vocabulary below |
| `retry_count` | INT | SQL validation retries so far |
| `no_results_retry_count` | INT | No-results retries so far |
| `concept` | VARCHAR | Populated once identified, null before |

**Step vocabulary:** `"identifying_concept"`, `"generating_sql"`, `"validating_sql"`, `"executing_query"`, `"retrying"`, `"generating_answer"`

---

### `sys_agents_history` — Question History

Permanent record per completed or failed execution.

| Column | Type | Notes |
|---|---|---|
| `query_id` | VARCHAR | UUID primary key |
| `agent_name` | VARCHAR | Always required |
| `chain_type` | VARCHAR | Entry-point chain type |
| `ontology` | VARCHAR | Final resolved ontology |
| `schema` | VARCHAR | Final resolved schema |
| `prompt` | TEXT | Full user prompt |
| `start_time` | DATETIME | |
| `end_time` | DATETIME | |
| `duration_ms` | INT | |
| `status` | VARCHAR | `"completed"`, `"completed_no_results"`, `"failed"`, `"timeout"` |
| `failed_at_step` | VARCHAR | Step name at failure, null on success |
| `concept` | VARCHAR | Final identified concept |
| `generated_sql` | TEXT | Final SQL |
| `rows_returned` | INT | `len(rows)`, null on failure |
| `error` | TEXT | Error message on failure |
| `reasoning_status` | VARCHAR | `"correct"`, `"partial"`, `"incorrect"` |
| `total_tokens` | INT | Aggregated across all LLM calls |
| `input_tokens` | INT | |
| `output_tokens` | INT | |
| `retry_count` | INT | Final SQL validation retry count |
| `no_results_retry_count` | INT | Final no-results retry count |
| `answer_generated` | BOOLEAN | True when GenerateAnswerChain ran and returned an answer |
| `chain_trace_enabled` | BOOLEAN | Whether trace rows were written for this query |
| `langchain_timbr_version` | VARCHAR | From `importlib.metadata` |
| `llm_type` | VARCHAR | From `llm._llm_type` |
| `llm_model` | VARCHAR | From `llm.model_name` or equivalent |
| `identify_concept_reason` | TEXT | LLM reasoning for concept selection, null when not applicable |
| `generate_sql_reason` | TEXT | LLM reasoning for SQL generation, null when not applicable |

**Status logic:**
- `"timeout"` — exception contains `"timed out"` (matches existing `TimeoutError`)
- `"failed"` — any other exception
- `"completed_no_results"` — no exception, rows empty or all-None
- `"completed"` — rows returned

---

### `sys_agents_chain_trace_log` — Chain Step Detail

One row per chain invocation within a question. Only written when trace is enabled.
Links to `sys_agents_history` via `query_id`.

| Column | Type | Notes |
|---|---|---|
| `trace_id` | VARCHAR | UUID primary key |
| `query_id` | VARCHAR | FK → `sys_agents_history.query_id` |
| `agent_name` | VARCHAR | |
| `chain_type` | VARCHAR | `"IdentifyTimbrConceptChain"`, `"GenerateTimbrSqlChain"`, `"ValidateTimbrSqlChain"`, `"ExecuteQuery"`, `"GenerateAnswerChain"` |
| `sequence` | INT | Execution order within the question (1, 2, 3…) |
| `retry_attempt` | INT | 0 = first attempt, 1 = first retry, etc. |
| `start_time` | DATETIME | |
| `end_time` | DATETIME | |
| `duration_ms` | INT | |
| `status` | VARCHAR | `"completed"`, `"failed"`, `"timeout"` |
| `concept` | VARCHAR | Identified concept (relevant for identify/generate steps) |
| `schema` | VARCHAR | |
| `generated_sql` | TEXT | SQL produced (relevant for generate/validate steps) |
| `is_sql_valid` | BOOLEAN | Validation result |
| `rows_returned` | INT | For execute steps |
| `error` | TEXT | Error message if failed |
| `reasoning_status` | VARCHAR | |
| `input_tokens` | INT | Tokens for this chain step only |
| `output_tokens` | INT | |
| `total_tokens` | INT | |

---

## API Endpoints

Mirrors the benchmark feature pattern exactly.

| Event | Endpoint | When called |
|---|---|---|
| Start | `POST {url}/timbr-server/log_agent/running` | Before first step executes |
| Step update | `POST {url}/timbr-server/log_agent/running_update_step` | At each step transition / retry |
| Finalize | `POST {url}/timbr-server/log_agent/history` | On completion/failure; triggers server-side delete of running row |
| Trace step | `POST {url}/timbr-server/log_agent/trace` | Per chain completion (only when trace enabled) |

**Auth:** `Authorization: Basic {base64('token:' + token)}` — same as benchmarks.

All calls are fire-and-forget: 5-second timeout, exceptions swallowed (logged at WARNING only). Logging never blocks or crashes main execution.

---

## Step Update Timeline

```
TimbrSqlAgent.invoke()
  │
  ├── POST /log_agent/running               ← start, current_step=""
  │
  ├── ExecuteTimbrQueryChain._call()
  │   ├── POST .../running_update_step      current_step="identifying_concept"
  │   ├── concept resolved                  POST .../running_update_step  (concept populated)
  │   ├── POST .../running_update_step      current_step="generating_sql"
  │   ├── POST .../running_update_step      current_step="validating_sql"  (if enabled)
  │   ├── POST .../running_update_step      current_step="executing_query"
  │   ├── [if retry]
  │   │   ├── POST .../running_update_step  current_step="retrying", retry_count++
  │   │   └── ...repeat generating/executing steps
  │   └── POST /log_agent/trace             (if trace enabled) one row per chain step
  │
  ├── GenerateAnswerChain._call()
  │   ├── POST .../running_update_step      current_step="generating_answer"
  │   └── POST /log_agent/trace             (if trace enabled)
  │
  └── POST /log_agent/history               ← finalize; server deletes running row
```

---

## Trace Enable / Disable Logic

```
chain_trace_enabled = (
    agent_config.chain_trace_enabled    # from sys_agents_options (new column)
    OR invocation param chain_trace=True
)
```

- `False` (default): only `sys_agents_running` and `sys_agents_history` are written
- `True`: all three tables are written; `sys_agents_history.chain_trace_enabled = True`

---

## Nested Invocation Design

When `TimbrSqlAgent` is the entry point, it **owns** the `query_id` record. Sub-chains called internally post trace rows using the same `query_id` — they do not create their own history records.

When any chain is invoked **standalone**, it creates its own history record.

### `AgentLogContext` dataclass

```python
@dataclass
class AgentLogContext:
    query_id: str               # UUID — the history record key
    agent_name: str
    url: str
    token: str
    chain_type: str             # entry-point chain type
    start_time: datetime
    _prompt: str
    chain_trace_enabled: bool
    current_step: Optional[str] = None
    retry_count: int = 0
    no_results_retry_count: int = 0
    concept: Optional[str] = None
    is_delegated: bool = False  # True = sub-chain; writes trace rows, not history
    _trace_sequence: int = 0    # incremented per chain step for ordering
```

- **Owner** (`is_delegated=False`): creates running row + finalizes history
- **Delegated** (`is_delegated=True`): posts step updates + writes trace rows (if enabled)

Context is passed via a new optional `log_ctx=` keyword on `_base_chain.invoke()` / `ainvoke()`, stored as `self._received_log_ctx` so `_call()` implementations can read it.

---

## Implementation Steps

### Step 1 — New file: `src/langchain_timbr/utils/chain_logger.py`

| Component | Purpose |
|---|---|
| `AgentLogContext` | Dataclass with all runtime log state |
| `_build_log_headers(token)` | Auth headers (same pattern as benchmarks) |
| `_safe_post(url, token, path, payload)` | HTTP wrapper; swallows all exceptions |
| `log_agent_start(ctx, ontology, schema)` | POST to `.../log_agent/running` |
| `log_agent_step(ctx)` | POST to `.../log_agent/running_update_step` |
| `log_agent_history(ctx, ...)` | POST to `.../log_agent/history` |
| `log_chain_trace(ctx, chain_type, ...)` | POST to `.../log_agent/trace` (skipped if `ctx.chain_trace_enabled=False`) |
| `new_query_id()` | `str(uuid.uuid4())` |
| `determine_status(rows, error)` | Maps output to status string |
| `get_llm_type(llm)`, `get_llm_model(llm)` | Introspect LLM attributes |
| `_sum_token_field(usage_metadata, field)` | Sum token counts across nested usage dicts |

### Step 2 — Modify `src/langchain_timbr/utils/_base_chain.py`

- Initialize `self._received_log_ctx = None`
- Add `log_ctx=None` to `invoke()` and `ainvoke()`
- Set `self._received_log_ctx = log_ctx` before calling `_call()`

### Step 3 — Modify `src/langchain_timbr/langchain/execute_timbr_query_chain.py`

Most complex: orchestrates all sub-steps and retry loops.

- Add `chain_trace: Optional[bool] = False` to `__init__`
- At start of `_call()`: resolve ownership (standalone vs delegated)
- Insert step updates at each transition (see timeline above)
- After concept resolved: update `ctx.concept`, POST step
- After each logical chain phase completes: call `log_chain_trace()` if trace enabled
- On retry: increment counter, `current_step = "retrying"`, POST step; each retry gets its own trace row (`retry_attempt` incremented)
- On success/failure (if owns log): call `log_agent_history()`

### Step 4 — Modify `src/langchain_timbr/langchain/timbr_sql_agent.py`

- Add `chain_trace: Optional[bool] = False` to `__init__` and `create_timbr_sql_agent()`
- In `_invoke_impl()` / `_ainvoke_impl()`:
  - Read `chain_trace_enabled` from agent config (`sys_agents_options`) OR param
  - Create owner `AgentLogContext` and call `log_agent_start()`
  - Create delegated context (`is_delegated=True`, same `query_id`) and pass as `log_ctx=` to sub-chains
  - On completion: call `log_agent_history()` with `chain_trace_enabled` flag

### Step 5 — Modify remaining four chains

Uniform, simpler pattern (no retry loops):

| File | Trace step name |
|---|---|
| `identify_concept_chain.py` | `"IdentifyTimbrConceptChain"` |
| `generate_timbr_sql_chain.py` | `"GenerateTimbrSqlChain"` |
| `validate_timbr_sql_chain.py` | `"ValidateTimbrSqlChain"` |
| `generate_answer_chain.py` | `"GenerateAnswerChain"` |

For each: add `chain_trace` param, implement standalone-vs-delegated in `_call()`, call `log_chain_trace()` on completion.

### Step 6 — `sys_agents_options` schema addition

Add `chain_trace_enabled` (BOOLEAN, default `False`) column. `TimbrSqlAgent` reads it alongside existing agent options at initialization.

---

## Files Modified / Created

| File | Change |
|---|---|
| `src/langchain_timbr/utils/chain_logger.py` | **New** — all logging logic |
| `src/langchain_timbr/utils/_base_chain.py` | Add `log_ctx` to `invoke` / `ainvoke` |
| `src/langchain_timbr/langchain/execute_timbr_query_chain.py` | Core instrumentation with step tracking and retry counters |
| `src/langchain_timbr/langchain/timbr_sql_agent.py` | Owner context creation, delegation to sub-chains |
| `src/langchain_timbr/langchain/identify_concept_chain.py` | Trace support |
| `src/langchain_timbr/langchain/generate_timbr_sql_chain.py` | Trace support |
| `src/langchain_timbr/langchain/validate_timbr_sql_chain.py` | Trace support |
| `src/langchain_timbr/langchain/generate_answer_chain.py` | Trace support |

**Pattern reference:** `src/langchain_timbr/utils/benchmark.py` (`_build_benchmark_log_headers`, `_log_benchmark_running`, `_log_benchmark_history`)

---

## Verification

1. **Unit test `chain_logger.py`**: mock `requests.post`; verify payloads; verify `_safe_post` swallows exceptions without raising.
2. **Standalone chain test**: invoke `ExecuteTimbrQueryChain` directly — one running row + one history row, zero trace rows.
3. **Agent test (trace off)**: invoke `TimbrSqlAgent` — one running row + one history row, zero trace rows.
4. **Agent test (trace on)**: invoke `TimbrSqlAgent` with `chain_trace=True` — one history row + N trace rows, all with matching `query_id`.
5. **Retry test**: force a SQL validation retry — trace row with `retry_attempt=1`, running row shows `retry_count=1`.
6. **Failure test**: mock `TimeoutError` — history `status="timeout"`, correct `failed_at_step`.
7. **No-results test**: empty query result — history `status="completed_no_results"`.
