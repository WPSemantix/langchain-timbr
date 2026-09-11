"""Technical Context Builder — enriches SQL generation prompts with per-column annotations.

Public API:
    build_technical_context(question, columns, schema, concept, conn_params, config)
    technical_context_scope()  — de-duplicates repeat builds within one SQL generation
"""

from __future__ import annotations

import contextvars
import logging
from contextlib import contextmanager
from typing import Any

from .config import TechnicalContextConfig
from .types import ColumnRef, MatchResult, SemanticType, TechnicalContextResult
from .semantic_type import classify_semantic_type, compute_ontology_distance, compute_priority_band
from .extraction.ngram import extract_prompt_tokens
from .extraction.llm import extract_candidates_with_llm, llm_identity
from .assembly.multi_match import run_all_matchers
from .matching.normalize import normalize, normalize_keep_spaces
from .assembly.per_column import assemble_column_payload, format_annotation
from .assembly.trimming import trim_to_budget
from .modes import estimate_include_all_cost
from .statistics_loader import load_column_statistics
from .statistics_loader.config import StatisticsLoaderConfig

logger = logging.getLogger(__name__)


# Per-SQL-generation memo. One ``generate_sql`` call builds the technical
# context two or three times over the same question and column set: the dynamic
# metadata-context top-up re-enters it, and so does the reasoning pass when it
# decides to regenerate. Each rebuild re-reads statistics, re-classifies every
# column and re-runs the matchers, all to produce the identical annotations.
#
# Scoped rather than global on purpose — it lives only for the duration of the
# enclosing ``technical_context_scope()``, so a later invoke still picks up
# fresh statistics. A ContextVar keeps it correct when the pipeline hands work
# to a worker thread (contextvars are copied into the child context).
_scope: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "technical_context_scope", default=None,
)


def match_date_year(
    candidates: list[str],
    date_years: dict[str, tuple[frozenset[str], int]],
    column_name: str,
) -> MatchResult | None:
    """A year named in the question promotes the date columns that cover it.

    Returns a summary — the year, the months present in it, and how many dates —
    never the dates themselves. One year of one column is ~184 values, and across
    every date column listing them would be roughly sixteen times the entire
    technical-context budget. The column already publishes its min/max range, so
    the summary is the increment and the promotion is the point.

    Returns None when no candidate names a year the column holds.
    """
    for candidate in candidates:
        hit = date_years.get(candidate.strip())
        if hit is None:
            continue
        months, count = hit
        return MatchResult(
            column_name=column_name,
            matched_value=(
                f"{candidate.strip()}: months {','.join(sorted(months))} ({count} dates)"
            ),
            score=90,
            match_type="year",
            candidate=candidate,
        )
    return None


@contextmanager
def technical_context_scope():
    """Memoize identical ``build_technical_context`` calls for this block."""
    token = _scope.set({})
    try:
        yield
    finally:
        _scope.reset(token)


def _scope_key(question, columns, schema, concept, config, llm):
    return (
        question, schema, concept, llm_identity(llm), repr(config),
        tuple((c.get("name"), c.get("type", "")) for c in columns),
    )


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

    scope = _scope.get()
    scope_key = None
    if scope is not None:
        scope_key = _scope_key(question, columns, schema, concept, config, llm)
        memoized = scope.get(scope_key)
        if memoized is not None:
            return memoized

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
            categorical_enum_max_distinct=config.categorical_enum_max_distinct,
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
    # We run matchers for EVERY column with stats, including ID and FREE_TEXT.
    # The previous early-skip silently dropped categorical dimension columns
    # whose distinct/non_null ratio happened to be ≈ 1.0 (the unique-ratio
    # signal is grain-dependent; a relationship-joined label column is unique
    # in its own dimension by construction). Running matchers unconditionally
    # means misclassification degrades to name_only with matched values
    # instead of producing no annotation at all.
    matches_by_column: dict[str, list[MatchResult]] = {}
    # The tokens and the question are the same for every column, so normalize
    # them once here rather than once per column inside each matcher.
    norm_candidates = [normalize(t) for t in candidates]
    norm_question = normalize_keep_spaces(question)
    for col_ref in col_refs:
        stats = stats_map.get(col_ref.name)
        if not stats or not stats.top_k:
            continue

        # The normalized forms ride along on the entries, derived once when the
        # row was parsed. One pass builds all three lists, index-aligned by
        # construction — the matcher sees top_k whole and in order, so there is
        # no slicing to keep them in step with.
        known_values: list[str] = []
        norm_values: list[str] | None = []
        norm_space_values: list[str] | None = []
        norm_stripped: list[str] | None = [] if stats.value_kind == "numeric" else None
        for e in stats.top_k:
            known_values.append(str(e.value))
            if norm_values is not None:
                if e.norm is None or e.norm_space is None:
                    # Pre-change row: let the matcher derive them.
                    norm_values = norm_space_values = None
                else:
                    norm_values.append(e.norm)
                    norm_space_values.append(e.norm_space)
            if norm_stripped is not None:
                if e.norm_stripped is None:
                    norm_stripped = None
                else:
                    norm_stripped.append(e.norm_stripped)

        matches = run_all_matchers(
            prompt_text=question,
            prompt_tokens=candidates,
            column_name=col_ref.name,
            known_values=known_values,
            config=config,
            semantic_type=col_ref.semantic_type,
            normalized=norm_values,
            normalized_space=norm_space_values,
            normalized_tokens=norm_candidates,
            normalized_prompt=norm_question,
            normalized_stripped=norm_stripped,
            value_kind=stats.value_kind,
            # Identity for the structure cache. `column_name` alone is not
            # unique — the same property exists under different concepts and
            # schemas, and on a multi-tenant server under different ontologies.
            ontology=conn_params.get("ontology", ""),
            schema=schema,
            concept=concept,
        )

        if stats.value_kind == "date" and stats.date_years:
            year_hit = match_date_year(candidates, stats.date_years, col_ref.name)
            if year_hit is not None:
                matches.append(year_hit)

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

    result = TechnicalContextResult(column_annotations=annotations, metadata=metadata)
    if scope is not None:
        scope[scope_key] = result
    return result
