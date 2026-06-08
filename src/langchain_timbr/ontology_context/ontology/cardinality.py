"""Pure cardinality derivation from RelationshipMeta + source/target PKs."""

from __future__ import annotations

from .models import RelationshipMeta


def derive_cardinality(
    rel: RelationshipMeta,
    *,
    source_pks: set[str],
    target_pks: set[str],
) -> str:
    """Return one of 'N:M' | 'N:1' | '1:N' | '1:1'.

    Rules (first match wins):
      1. is_mtm=True                                -> 'N:M'
      2. join keys fully match both sides' PKs      -> '1:1'
      3. target join keys == target PKs (and source side does not match) -> 'N:1'
      4. source join keys == source PKs (and target side does not match) -> '1:N'
      5. default                                    -> '1:N'

    Empty join-key tuples never match the equality check (rule degrades to default).
    """
    if rel.is_mtm:
        return "N:M"
    src_keys = set(rel.source_join_keys)
    tgt_keys = set(rel.target_join_keys)
    source_match = bool(src_keys) and src_keys == source_pks
    target_match = bool(tgt_keys) and tgt_keys == target_pks
    if source_match and target_match:
        return "1:1"
    if target_match:
        return "N:1"
    if source_match:
        return "1:N"
    return "1:N"
