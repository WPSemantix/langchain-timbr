"""Technical Context Builder — enriches SQL generation prompts with per-column annotations.

Public API:
    build_technical_context(question, columns, schema, concept, conn_params, config)
"""

from __future__ import annotations

import logging
from typing import Any

from .config import TechnicalContextConfig
from .types import ColumnRef, MatchResult, SemanticType, TechnicalContextResult
from .semantic_type import classify_semantic_type, compute_ontology_distance, compute_priority_band
from .extraction.ngram import extract_prompt_tokens
from .extraction.llm import extract_candidates_with_llm
from .assembly.multi_match import run_all_matchers
from .assembly.per_column import assemble_column_payload, format_annotation
from .assembly.trimming import trim_to_budget
from .modes import estimate_include_all_cost
from .statistics_loader import load_column_statistics
from .statistics_loader.config import StatisticsLoaderConfig

logger = logging.getLogger(__name__)


def build_technical_context(
    question: str,
    columns: list[dict[str, Any]],
    schema: str,
    concept: str,
    conn_params: dict[str, Any],
    config: TechnicalContextConfig | None = None,
    llm=None,
) -> TechnicalContextResult:
    """Build technical context annotations for all columns.

    Orchestrates the full pipeline:
    1. Load column statistics
    2. Classify semantic types
    3. Extract prompt tokens
    4. Run matchers per column
    5. Select columns for annotation (mode)
    6. Assemble per-column annotations
    7. Trim to token budget

    Args:
        question: User's natural language question.
        columns: List of column dicts with 'name' and 'type' keys (from ontology).
        schema: Timbr schema name.
        concept: Concept/table name.
        conn_params: Connection parameters for statistics queries.
        config: Configuration (defaults to TechnicalContextConfig()).

    Returns:
        TechnicalContextResult with column_annotations dict.
    """
    if config is None:
        config = TechnicalContextConfig()

    if not columns or not question:
        return TechnicalContextResult(column_annotations={})

    # 1. Load statistics for all columns
    stats_columns = [{"name": c["name"], "type": c.get("type", "")} for c in columns]
    stats_loader_config = StatisticsLoaderConfig(
        include_properties=config.technical_context_properties,
        exclude_properties=config.exclude_properties,
    )
    try:
        stats_map = load_column_statistics(
            schema=schema,
            table_name=concept,
            columns=stats_columns,
            conn_params=conn_params,
            config=stats_loader_config,
        )
    except Exception as e:
        logger.warning("Failed to load column statistics: %s", e)
        return TechnicalContextResult(column_annotations={}, metadata={"error": str(e)})

    # 2. Classify semantic types and build ColumnRefs
    col_refs: list[ColumnRef] = []
    for c in columns:
        name = c["name"]
        sql_type = c.get("type", "")
        stats = stats_map.get(name)
        sem_type = classify_semantic_type(
            name, sql_type, stats,
            free_text_threshold=config.free_text_distinct_threshold,
            id_unique_ratio=config.id_unique_ratio_threshold,
        )
        distance = compute_ontology_distance(name)
        col_ref = ColumnRef(
            name=name,
            sql_type=sql_type,
            ontology_distance=distance,
            priority_band=compute_priority_band(distance, False),  # updated after matching
            semantic_type=sem_type,
        )
        col_refs.append(col_ref)

    # 3. Determine effective mode (auto-mode fallback logic)
    effective_mode = config.mode
    use_llm = False

    if config.mode == "include_all":
        effective_mode = "include_all"
    elif config.mode == "filter_matched":
        if llm is not None:
            effective_mode = "filter_matched"
            use_llm = True
        else:
            logger.warning("filter_matched mode requires llm; falling back to include_all")
            effective_mode = "include_all"
    elif config.mode == "auto":
        estimated_cost = estimate_include_all_cost(col_refs, stats_map, config)
        if estimated_cost <= config.max_tokens:
            effective_mode = "include_all"
        elif llm is not None:
            effective_mode = "filter_matched"
            use_llm = True
        else:
            effective_mode = "include_all"

    # 4. Extract candidates (LLM or heuristic — same downstream cascade)
    if use_llm:
        try:
            candidates = extract_candidates_with_llm(question, llm=llm)
        except Exception as e:
            logger.warning("LLM candidate extraction failed, falling back to include_all: %s", e)
            candidates = []

        # If LLM returned empty, fall back to include_all with heuristic candidates
        if not candidates:
            logger.warning("LLM returned no candidates, falling back to include_all")
            effective_mode = "include_all"
            use_llm = False
            candidates = extract_prompt_tokens(question)
    else:
        candidates = extract_prompt_tokens(question)

    # 5. Run matchers per column (same cascade in ALL modes)
    matches_by_column: dict[str, list[MatchResult]] = {}
    for col_ref in col_refs:
        stats = stats_map.get(col_ref.name)
        if not stats or not stats.top_k:
            continue
        # Skip ID and FREE_TEXT for matching (unless in include_all mode)
        if effective_mode != "include_all" and col_ref.semantic_type in (
            SemanticType.ID, SemanticType.FREE_TEXT
        ):
            continue

        known_values = [str(e.value) for e in stats.top_k]
        matches = run_all_matchers(
            prompt_text=question,
            prompt_tokens=candidates,
            column_name=col_ref.name,
            known_values=known_values,
            config=config,
            semantic_type=col_ref.semantic_type,
        )
        if matches:
            matches_by_column[col_ref.name] = matches

    # Update priority bands based on match status
    for col_ref in col_refs:
        has_match = col_ref.name in matches_by_column
        col_ref.priority_band = compute_priority_band(col_ref.ontology_distance, has_match)

    # 7. Assemble per-column PAYLOADS (structured, pre-format)
    from .types import ColumnPayload
    payloads: dict[str, ColumnPayload] = {}
    col_ref_map: dict[str, ColumnRef] = {}
    for col_ref in col_refs:
        stats = stats_map.get(col_ref.name)
        if not stats:
            continue
        matches = matches_by_column.get(col_ref.name, [])
        payload = assemble_column_payload(
            col_ref, stats, matches, config, effective_mode=effective_mode,
        )
        if payload is not None:
            payloads[col_ref.name] = payload
            col_ref_map[col_ref.name] = col_ref

    # 8. Trim payloads to token budget
    # Only columns with at least one STRONG match are protected from trimming
    strong_matched_keys: set[str] = set()
    for col_name, payload in payloads.items():
        if payload.matched_values:  # matched_values now only contains strong matches
            strong_matched_keys.add(col_name)
    payloads = trim_to_budget(payloads, col_ref_map, strong_matched_keys, config)

    # 9. Format trimmed payloads into final annotation strings
    annotations: dict[str, str] = {}
    for col_name, payload in payloads.items():
        annotation = format_annotation(payload, config)
        if annotation:
            annotations[col_name] = annotation

    metadata = {
        "total_columns": len(columns),
        "annotated_columns": len(annotations),
        "matched_columns": len(matches_by_column),
        "mode": config.mode,
        "effective_mode": effective_mode,
        "llm_used": use_llm,
    }

    return TechnicalContextResult(column_annotations=annotations, metadata=metadata)
