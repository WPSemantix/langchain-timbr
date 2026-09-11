"""Fuzzy matching using rapidfuzz for approximate string comparison."""

from __future__ import annotations

from ...config import fuzzy_use_cdist, match_max_per_token, match_max_per_column
from ..types import MatchResult
from .normalize import normalize


def build_fuzzy_choices(
    known_values: list[str],
    normalized: list[str] | None = None,
) -> tuple[list[str], list[str]]:
    """Split a column's values into (normalized choices, originals) for fuzzy.

    Question-independent — a pure function of the column's values — which is
    why :mod:`..assembly.structure_cache` can hold the result across requests.
    """
    norm_choices: list[str] = []
    originals: list[str] = []
    if normalized is None:
        normalized = (normalize(str(v)) for v in known_values)
    for v, nv in zip(known_values, normalized):
        if nv and len(nv) >= 3:  # Skip very short values for fuzzy
            norm_choices.append(nv)
            originals.append(str(v))
    return norm_choices, originals


def _use_cdist() -> bool:
    return fuzzy_use_cdist


def config_max_per_token() -> int:
    return match_max_per_token


def config_max_per_column() -> int:
    return match_max_per_column


def fuzzy_match(
    prompt_tokens: list[str],
    column_name: str,
    known_values: list[str],
    *,
    threshold: int = 88,
    strong_threshold: int | None = None,
    normalized: list[str] | None = None,
    normalized_tokens: list[str] | None = None,
    choices: tuple[list[str], list[str]] | None = None,
    exclude: set[str] | None = None,
) -> list[MatchResult]:
    """Match prompt tokens against known values using fuzzy string similarity.

    Scoring runs through ``process.cdist``, which computes the whole
    tokens-x-values score matrix for a column in one C++ call. Two reasons, both
    measured on the captured fixture: it is 4.4x faster than a ``extractOne``
    call per token, and it **releases the GIL**, which ``extractOne`` does not —
    under 8 concurrent questions that moved CPU/wall from 0.96 (fully
    serialised) to 3.24. In a ``--threads 100`` server that is the difference
    that matters. Only imports rapidfuzz when called (lazy import for optional
    dependency).

    Args:
        prompt_tokens: Tokens extracted from the user prompt.
        column_name: Column name for result attribution.
        known_values: Known values from column statistics top_k.
        threshold: Minimum similarity score (0-100) to consider a match.
        normalized: Optional pre-normalized forms of ``known_values``, index-aligned.
            Supplied by :func:`run_all_matchers` so the three matchers normalize
            each value once between them. Computed here when omitted.
        choices: Optional prebuilt ``(norm_choices, originals)`` from
            :func:`build_fuzzy_choices`, supplied by the structure cache so the
            split is not redone per request. Derived here when omitted.

    Returns:
        List of MatchResult with match_type="fuzzy" and the similarity score.
        A token may contribute more than one value — see ``match_max_per_token``.
    """
    try:
        from rapidfuzz import fuzz, process
    except ImportError:
        # rapidfuzz not available — skip fuzzy matching
        return []

    # Pre-normalize known values
    if choices is None:
        choices = build_fuzzy_choices(known_values, normalized)
    norm_choices, originals = choices

    if not norm_choices:
        return []

    norm_tokens = (
        normalized_tokens if normalized_tokens is not None
        else [normalize(t) for t in prompt_tokens]
    )
    # Only tokens long enough to be worth comparing, paired with the original
    # spelling so the MatchResult can still report what the user wrote.
    scored = [(t, nt) for t, nt in zip(prompt_tokens, norm_tokens)
              if nt and len(nt) >= 3]
    if not scored:
        return []

    # The uncached code handed in only the values earlier stages had not claimed.
    # The choices list is now cached whole, so the exclusion moves here — as a
    # mask over the scores rather than a rebuild of the list.
    live = None
    if exclude:
        live = [i for i, v in enumerate(originals) if v not in exclude]
        if not live:
            return []

    if not _use_cdist():
        return _fuzzy_extract_one(
            process, fuzz, scored, norm_choices, originals, column_name,
            threshold, live,
        )

    import numpy as np

    # cdist over extractOne, for two reasons measured on the captured fixture:
    #   - one batched C++ call per column instead of one per token: 4.4x
    #   - it RELEASES THE GIL, which extractOne does not. Under 8 concurrent
    #     questions this took CPU/wall from 0.96 (fully serialised) to 3.24.
    # dtype=float64 is deliberate, NOT the float32 default: the score reaches
    # the prompt through int(), and float32 rounding disagreed with extractOne
    # on 86 of 20,000 random comparisons. float64 disagreed on none.
    # workers=1 is rapidfuzz's default and means NO thread pool: one section,
    # computed inline. Stated explicitly because workers=-1 looks like the
    # obvious speed knob and measured 13x SLOWER — it spawns a pool per call,
    # and each call here is tiny (a handful of tokens x ~1k values). The GIL
    # release is a property of cdist itself, not of the worker count, so the
    # concurrency win survives running single-threaded inside rapidfuzz.
    matrix = process.cdist(
        [nt for _, nt in scored], norm_choices,
        scorer=fuzz.ratio, dtype=np.float64, score_cutoff=threshold, workers=1,
    )

    max_per_token = config_max_per_token()
    live_arr = np.asarray(live) if live is not None else None
    results: list[MatchResult] = []
    emitted: set[str] = set()
    max_per_column = config_max_per_column()

    for row, (token, _) in zip(matrix, scored):
        if live_arr is not None:
            # Restrict to values earlier stages left alone. argmax below then
            # indexes into this view, so map back through live_arr.
            view = row[live_arr]
        else:
            view = row
        # score_cutoff zeroes everything below the threshold, so a row that
        # never cleared it is all zeros — cheaper to test than to scan.
        if view.size == 0 or view.max() < threshold:
            continue

        # Multiplicity is for STRONG matches only. The weak band (sort_threshold
        # to surface_threshold) is where fuzzy invents values — 'Damage Stock'
        # and 'MRP ASI Versteck' both clear it for the token 'master stock' —
        # so returning more of them would multiply noise, not signal. Below the
        # strong bar the historical single-best behaviour stands.
        strong_bar = threshold if strong_threshold is None else strong_threshold
        if max_per_token <= 1 or view.max() < strong_bar:
            # argmax returns the FIRST maximum, which is extractOne's tie-break.
            idx = [int(view.argmax())]
        else:
            hits = np.flatnonzero(view >= strong_bar)
            if len(hits) > max_per_token:
                # A token matching this many values is not a filter literal, it
                # is noise. Showing an arbitrary N of them would imply those N
                # are special. Fall back to the single best — which is exactly
                # the pre-Part-4 behaviour, in exactly the dangerous case.
                idx = [int(view.argmax())]
            else:
                # Highest first, ties broken by position, as extractOne would.
                idx = sorted(hits.tolist(), key=lambda j: (-view[j], j))

        for j in idx:
            real = int(live_arr[j]) if live_arr is not None else j
            value = originals[real]
            if value in emitted:
                continue
            if len(emitted) >= max_per_column:
                # Backstop: trimming is forbidden from shrinking a matched
                # column, so this is the only bound on how many values one
                # column can push into the prompt.
                break
            emitted.add(value)
            results.append(MatchResult(
                column_name=column_name,
                matched_value=value,
                score=int(view[j]),
                match_type="fuzzy",
                candidate=token,
            ))

    return results


def _fuzzy_extract_one(process, fuzz, scored, norm_choices, originals,
                       column_name, threshold, live=None):
    """The pre-cdist path, kept behind ``TIMBR_FUZZY_USE_CDIST=false``."""
    if live is not None:
        norm_choices = [norm_choices[i] for i in live]
        originals = [originals[i] for i in live]
    results: list[MatchResult] = []
    for token, norm_token in scored:
        best = process.extractOne(norm_token, norm_choices, scorer=fuzz.ratio)
        if best is None:
            continue
        best_score, best_index = best[1], best[2]
        if best_score >= threshold:
            results.append(MatchResult(
                column_name=column_name,
                matched_value=originals[best_index],
                score=int(best_score),
                match_type="fuzzy",
                candidate=token,
            ))
    return results
