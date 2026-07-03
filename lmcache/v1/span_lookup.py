# SPDX-License-Identifier: Apache-2.0
"""Span-level lookup results for non-prefix KV admission.

This module is intentionally independent of vLLM.  It lets LMCache keep the
existing integer prefix-hit API while exposing enough segment metadata for
CacheBlend-style non-prefix reuse and external admission gates.
"""

# Standard
from dataclasses import asdict, dataclass, field
import json
from typing import Any


@dataclass(frozen=True)
class SpanLookupHit:
    """One token span considered during lookup."""

    start: int
    end: int
    hit: bool
    backend: str | None = None
    key_index: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        """Return the number of tokens covered by this span."""

        return max(0, self.end - self.start)


@dataclass(frozen=True)
class SpanLookupResult:
    """Lookup result with both prefix-compatible and span-aware views."""

    prefix_hit_tokens: int
    spans: list[SpanLookupHit]

    @property
    def hit_tokens(self) -> int:
        """Return total hit tokens across all spans."""

        return sum(span.token_count for span in self.spans if span.hit)

    @property
    def nonprefix_hit_tokens(self) -> int:
        """Return hit tokens after the first miss in the request."""

        total = 0
        in_prefix = True
        for span in self.spans:
            if in_prefix and span.hit and span.end <= self.prefix_hit_tokens:
                continue
            in_prefix = False
            if span.hit:
                total += span.token_count
        return total

    def to_prefix_tokens(self) -> int:
        """Return the old LMCache lookup value."""

        return self.prefix_hit_tokens

    def to_bytes(self) -> bytes:
        """Serialize the result for the existing lookup RPC transport."""

        payload = {
            "prefix_hit_tokens": self.prefix_hit_tokens,
            "spans": [asdict(span) for span in self.spans],
        }
        return json.dumps(payload, sort_keys=True).encode("utf-8")

    @classmethod
    def from_bytes(cls, payload: bytes) -> "SpanLookupResult":
        """Deserialize a span lookup result."""

        raw = json.loads(payload.decode("utf-8"))
        return cls(
            prefix_hit_tokens=int(raw["prefix_hit_tokens"]),
            spans=[SpanLookupHit(**span) for span in raw["spans"]],
        )

    @classmethod
    def from_prefix_tokens(cls, prefix_hit_tokens: int) -> "SpanLookupResult":
        """Create a prefix-only result for backward-compatible clients."""

        spans: list[SpanLookupHit] = []
        if prefix_hit_tokens > 0:
            spans.append(SpanLookupHit(0, prefix_hit_tokens, True, key_index=0))
        return cls(prefix_hit_tokens=prefix_hit_tokens, spans=spans)


def span_lookup_result_from_cb_unified_lookup(result: Any) -> SpanLookupResult:
    """Convert a Blend V3 CB_UNIFIED_LOOKUP payload into span lookup metadata.

    The conversion keeps the old prefix-token view while attaching the
    CBMatchResult hash needed later by CB_RETRIEVE_PRE_COMPUTED_V3.  It is a
    pure metadata bridge: admission, parity, and physical retrieval remain
    separate fail-closed steps.
    """

    prefix_hit_tokens = int(getattr(result, "prefix_coverage_tokens", 0) or 0)
    spans: list[SpanLookupHit] = []
    next_index = 0
    if prefix_hit_tokens > 0:
        spans.append(
            SpanLookupHit(
                0,
                prefix_hit_tokens,
                True,
                backend="CacheBlendV3",
                key_index=next_index,
                metadata={"coherentkv_span_kind": "prefix"},
            )
        )
        next_index += 1

    for kind, segments in (
        ("non_prefix", getattr(result, "non_prefix_segments", ()) or ()),
        ("segmented_prefix", getattr(result, "segmented_prefix_segments", ()) or ()),
    ):
        for segment in segments:
            chunk_hash = getattr(segment, "hash", b"") or b""
            spans.append(
                SpanLookupHit(
                    start=int(getattr(segment, "cur_st")),
                    end=int(getattr(segment, "cur_ed")),
                    hit=True,
                    backend="CacheBlendV3",
                    key_index=next_index,
                    metadata={
                        "coherentkv_span_kind": kind,
                        "coherentkv_cb_hash_hex": bytes(chunk_hash).hex(),
                        "coherentkv_source_start_token": str(
                            int(getattr(segment, "old_st"))
                        ),
                        "coherentkv_source_end_token": str(
                            int(getattr(segment, "old_ed"))
                        ),
                    },
                )
            )
            next_index += 1

    spans.sort(key=lambda span: (span.start, span.end, span.key_index or -1))
    return SpanLookupResult(prefix_hit_tokens=prefix_hit_tokens, spans=spans)


def intersect_span_lookup_results(
    results: list[SpanLookupResult],
) -> SpanLookupResult:
    """Return a conservative cross-rank span lookup result.

    A span is a hit only when every rank reports the same span boundary as a
    hit. This mirrors LMCache's existing prefix rule, which uses the minimum
    hit count across ranks.
    """

    if not results:
        return SpanLookupResult(prefix_hit_tokens=0, spans=[])
    prefix_hit_tokens = min(result.prefix_hit_tokens for result in results)
    min_spans = min(len(result.spans) for result in results)
    spans: list[SpanLookupHit] = []
    for idx in range(min_spans):
        grouped = [result.spans[idx] for result in results]
        first = grouped[0]
        same_boundary = all(
            span.start == first.start and span.end == first.end for span in grouped
        )
        hash_values = [
            (span.metadata or {}).get("coherentkv_cb_hash_hex") for span in grouped
        ]
        hash_compatible = all(value == hash_values[0] for value in hash_values)
        hit = same_boundary and hash_compatible and all(span.hit for span in grouped)
        metadata: dict[str, Any] = {
            "rank_count": len(results),
            "same_boundary": same_boundary,
            "metadata_compatible": hash_compatible,
        }
        if hit:
            common_keys = set(grouped[0].metadata or {})
            for span in grouped[1:]:
                common_keys &= set(span.metadata or {})
            for key in sorted(common_keys):
                values = {str((span.metadata or {})[key]) for span in grouped}
                if len(values) == 1:
                    metadata[key] = values.pop()
        spans.append(
            SpanLookupHit(
                start=first.start,
                end=first.end,
                hit=hit,
                backend=first.backend if hit else None,
                key_index=idx,
                metadata=metadata,
            )
        )
    return SpanLookupResult(prefix_hit_tokens=prefix_hit_tokens, spans=spans)
