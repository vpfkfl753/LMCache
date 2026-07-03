# SPDX-License-Identifier: Apache-2.0
"""Tests for span-level lookup results."""

# First Party
from lmcache.v1.span_lookup import (
    SpanLookupHit,
    SpanLookupResult,
    intersect_span_lookup_results,
    span_lookup_result_from_cb_unified_lookup,
)


def test_span_lookup_roundtrip_and_prefix_compatibility() -> None:
    result = SpanLookupResult(
        prefix_hit_tokens=4,
        spans=[
            SpanLookupHit(0, 4, True, backend="LocalCPUBackend", key_index=0),
            SpanLookupHit(4, 8, False, key_index=1),
            SpanLookupHit(8, 12, True, backend="LocalCPUBackend", key_index=2),
        ],
    )

    restored = SpanLookupResult.from_bytes(result.to_bytes())

    assert restored.to_prefix_tokens() == 4
    assert restored.hit_tokens == 8
    assert restored.nonprefix_hit_tokens == 4


def test_prefix_only_wrapper_hides_nonprefix_hits() -> None:
    result = SpanLookupResult.from_prefix_tokens(4)

    assert result.to_prefix_tokens() == 4
    assert result.hit_tokens == 4
    assert result.nonprefix_hit_tokens == 0


def test_intersection_requires_all_ranks_to_hit_span() -> None:
    rank0 = SpanLookupResult(
        prefix_hit_tokens=4,
        spans=[
            SpanLookupHit(0, 4, True, backend="rank0", key_index=0),
            SpanLookupHit(4, 8, False, key_index=1),
            SpanLookupHit(8, 12, True, backend="rank0", key_index=2),
        ],
    )
    rank1 = SpanLookupResult(
        prefix_hit_tokens=4,
        spans=[
            SpanLookupHit(0, 4, True, backend="rank1", key_index=0),
            SpanLookupHit(4, 8, False, key_index=1),
            SpanLookupHit(8, 12, False, backend=None, key_index=2),
        ],
    )

    result = intersect_span_lookup_results([rank0, rank1])

    assert result.to_prefix_tokens() == 4
    assert result.nonprefix_hit_tokens == 0
    assert result.spans[2].hit is False


def test_cb_unified_lookup_conversion_preserves_hash_metadata() -> None:
    class Match:
        def __init__(self, old_st: int, old_ed: int, cur_st: int, cur_ed: int, chunk_hash: bytes):
            self.old_st = old_st
            self.old_ed = old_ed
            self.cur_st = cur_st
            self.cur_ed = cur_ed
            self.hash = chunk_hash

    class Result:
        prefix_coverage_tokens = 4
        non_prefix_segments = [Match(64, 68, 12, 16, bytes.fromhex("00ff"))]
        segmented_prefix_segments = [Match(8, 12, 8, 12, bytes.fromhex("aa55"))]

    result = span_lookup_result_from_cb_unified_lookup(Result())

    assert result.to_prefix_tokens() == 4
    assert result.nonprefix_hit_tokens == 8
    by_kind = {span.metadata.get("coherentkv_span_kind"): span for span in result.spans}
    assert by_kind["non_prefix"].metadata["coherentkv_cb_hash_hex"] == "00ff"
    assert by_kind["non_prefix"].metadata["coherentkv_source_start_token"] == "64"
    assert by_kind["segmented_prefix"].metadata["coherentkv_cb_hash_hex"] == "aa55"


def test_intersection_preserves_matching_cb_hash_and_rejects_conflict() -> None:
    left = SpanLookupResult(
        prefix_hit_tokens=0,
        spans=[SpanLookupHit(8, 12, True, metadata={"coherentkv_cb_hash_hex": "00ff"})],
    )
    same = SpanLookupResult(
        prefix_hit_tokens=0,
        spans=[SpanLookupHit(8, 12, True, metadata={"coherentkv_cb_hash_hex": "00ff"})],
    )
    different = SpanLookupResult(
        prefix_hit_tokens=0,
        spans=[SpanLookupHit(8, 12, True, metadata={"coherentkv_cb_hash_hex": "aa55"})],
    )

    kept = intersect_span_lookup_results([left, same])
    rejected = intersect_span_lookup_results([left, different])

    assert kept.spans[0].hit is True
    assert kept.spans[0].metadata["coherentkv_cb_hash_hex"] == "00ff"
    assert rejected.spans[0].hit is False
    assert rejected.spans[0].metadata["metadata_compatible"] is False
