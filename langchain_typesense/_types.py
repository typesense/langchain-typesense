"""Shared public and internal types for the Typesense vector-store adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal, TypeAlias, TypedDict

FilterScalar: TypeAlias = str | int | float | bool
FilterValue: TypeAlias = FilterScalar | Sequence[FilterScalar]
Filter: TypeAlias = str | Mapping[str, FilterValue]
VectorDistance: TypeAlias = Literal["cosine", "ip"]
ClientMode: TypeAlias = Literal["sync", "async", "both"]


class TypesenseSearchParameters(TypedDict, total=False):
    """Safe Typesense search options that can be forwarded by this adapter.

    The integration owns the query, vector query, pagination, filtering, and
    returned-field selection. Options that can change the response shape or
    replace vector-distance ordering are intentionally not part of this type.
    """

    max_filter_by_candidates: int
    enable_lazy_filter: bool
    facet_by: str | list[str]
    max_facet_values: int
    facet_query: str
    facet_query_num_typos: int
    facet_return_parent: str
    facet_sample_percent: int
    facet_sample_threshold: int
    facet_strategy: Literal["exhaustive", "top_values", "automatic"]
    highlight_fields: Literal["none"] | str | list[str]
    highlight_full_fields: Literal["none"] | str | list[str]
    highlight_affix_num_tokens: int
    highlight_start_tag: str
    highlight_end_tag: str
    enable_highlight_v1: bool
    snippet_threshold: int
    limit_hits: int
    search_cutoff_ms: int
    exhaustive_search: bool
    use_cache: bool
    cache_ttl: int
    curation_tags: str
    diversity_lambda: float


class TypesenseHybridSearchParameters(TypesenseSearchParameters, total=False):
    """Safe Typesense options for the keyword side of hybrid search."""

    prefix: str | bool | list[bool]
    infix: Literal["off", "always", "fallback"] | list[Literal["off", "always", "fallback"]]
    pre_segmented_query: bool
    stopwords: str | list[str]
    validate_field_names: bool
    query_by_weights: str | list[int]
    text_match_type: Literal["max_score", "max_weight"]
    prioritize_exact_match: bool
    prioritize_token_position: bool
    prioritize_num_matching_fields: bool
    max_candidates: int
    enable_synonyms: bool
    filter_curated_hits: bool
    synonym_prefix: bool
    num_typos: int
    min_len_1typo: int
    min_len_2typo: int
    split_join_tokens: Literal["off", "fallback", "always"]
    typo_tokens_threshold: int
    drop_tokens_threshold: int
    drop_tokens_mode: Literal["right_to_left", "left_to_right", "both_sides:3"]
    enable_typos_for_numerical_tokens: bool
    enable_typos_for_alpha_numerical_tokens: bool
    synonym_num_typos: int
    rerank_hybrid_matches: bool


MANAGED_SEARCH_PARAMETERS = frozenset(
    {
        "q",
        "vector_query",
        "per_page",
        "page",
        "filter_by",
        "include_fields",
        "exclude_fields",
        "query_by",
    }
)
SAFE_SEARCH_PARAMETERS = frozenset(TypesenseSearchParameters.__annotations__)
SAFE_HYBRID_SEARCH_PARAMETERS = frozenset(
    {
        *TypesenseSearchParameters.__annotations__,
        *TypesenseHybridSearchParameters.__annotations__,
    }
)


__all__ = [
    "ClientMode",
    "Filter",
    "FilterScalar",
    "FilterValue",
    "TypesenseHybridSearchParameters",
    "TypesenseSearchParameters",
    "VectorDistance",
]
