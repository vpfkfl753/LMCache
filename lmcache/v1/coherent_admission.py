# SPDX-License-Identifier: Apache-2.0
"""Fail-closed span admission for CoherentKV materializations.

This module deliberately has no vLLM dependency.  A connector can use it after
span lookup and before any physical KV load/publication.  Unknown freshness or
compatibility always rejects the candidate and leaves the corresponding token
range to be recomputed.
"""

# Standard
from dataclasses import dataclass, field
from typing import Mapping


@dataclass(frozen=True)
class SpanAdmissionCandidate:
    """One candidate KV span discovered by a cache lookup."""

    name: str
    start_token: int
    end_token: int
    committed: bool
    compatible: bool
    cached_deps: Mapping[str, str]
    metadata: Mapping[str, str] = field(default_factory=dict)

    @property
    def token_count(self) -> int:
        return max(0, self.end_token - self.start_token)


@dataclass(frozen=True)
class SpanDecision:
    """Admission decision for one span."""

    candidate: SpanAdmissionCandidate
    reason: str

    @property
    def admitted(self) -> bool:
        return self.reason == "admit"


@dataclass(frozen=True)
class TokenInterval:
    """Half-open token interval [start_token, end_token)."""

    start_token: int
    end_token: int

    @property
    def token_count(self) -> int:
        return max(0, self.end_token - self.start_token)

    def contains(self, other: "TokenInterval") -> bool:
        return self.start_token <= other.start_token and other.end_token <= self.end_token


@dataclass(frozen=True)
class SpanAdmissionPlan:
    """Fail-closed load/recompute plan for a request."""

    total_context_tokens: int
    request_snapshot: Mapping[str, str]
    decisions: list[SpanDecision]
    recompute_intervals: list[TokenInterval]

    @property
    def admitted_tokens(self) -> int:
        return sum(d.candidate.token_count for d in self.decisions if d.admitted)

    @property
    def rejected_tokens(self) -> int:
        return sum(d.candidate.token_count for d in self.decisions if not d.admitted)

    @property
    def recompute_tokens(self) -> int:
        return sum(interval.token_count for interval in self.recompute_intervals)


def admission_reason(
    candidate: SpanAdmissionCandidate,
    request_snapshot: Mapping[str, str],
) -> str:
    """Return the fail-closed admission reason for one candidate."""

    if not candidate.committed:
        return "not_committed"
    if not candidate.compatible:
        return "not_compatible"
    for artifact_id, cached_version in candidate.cached_deps.items():
        request_version = request_snapshot.get(artifact_id)
        if request_version is None:
            return f"missing_snapshot_dep:{artifact_id}:{cached_version}"
        if request_version != cached_version:
            return f"stale:{artifact_id}:{cached_version}->{request_version}"
    return "admit"


def merge_intervals(intervals: list[TokenInterval]) -> list[TokenInterval]:
    ordered = sorted(
        [item for item in intervals if item.end_token > item.start_token],
        key=lambda item: (item.start_token, item.end_token),
    )
    merged: list[TokenInterval] = []
    for interval in ordered:
        if not merged or interval.start_token > merged[-1].end_token:
            merged.append(interval)
            continue
        previous = merged[-1]
        merged[-1] = TokenInterval(
            previous.start_token,
            max(previous.end_token, interval.end_token),
        )
    return merged


def complement_intervals(
    total_context_tokens: int, admitted_intervals: list[TokenInterval]
) -> list[TokenInterval]:
    out: list[TokenInterval] = []
    cursor = 0
    for interval in merge_intervals(admitted_intervals):
        if interval.start_token > cursor:
            out.append(TokenInterval(cursor, interval.start_token))
        cursor = max(cursor, interval.end_token)
    if cursor < total_context_tokens:
        out.append(TokenInterval(cursor, total_context_tokens))
    return out


def build_span_admission_plan(
    *,
    candidates: list[SpanAdmissionCandidate],
    request_snapshot: Mapping[str, str],
    total_context_tokens: int,
) -> SpanAdmissionPlan:
    """Build a fail-closed span plan from lookup candidates."""

    decisions = [
        SpanDecision(candidate, admission_reason(candidate, request_snapshot))
        for candidate in candidates
    ]
    admitted = [
        TokenInterval(decision.candidate.start_token, decision.candidate.end_token)
        for decision in decisions
        if decision.admitted
    ]
    recompute = complement_intervals(total_context_tokens, admitted)
    return SpanAdmissionPlan(
        total_context_tokens=total_context_tokens,
        request_snapshot=request_snapshot,
        decisions=decisions,
        recompute_intervals=recompute,
    )


def validate_span_admission_plan(plan: SpanAdmissionPlan) -> list[str]:
    """Validate coverage and fail-closed admission invariants."""

    errors: list[str] = []
    admitted = [
        TokenInterval(decision.candidate.start_token, decision.candidate.end_token)
        for decision in plan.decisions
        if decision.admitted
    ]
    for decision in plan.decisions:
        if decision.admitted:
            reason = admission_reason(decision.candidate, plan.request_snapshot)
            if reason != "admit":
                errors.append(
                    "admitted_invalid_span:"
                    f"{decision.candidate.name}:{decision.candidate.start_token}:"
                    f"{reason}"
                )

    partition = sorted(
        [*admitted, *plan.recompute_intervals],
        key=lambda item: (item.start_token, item.end_token),
    )
    cursor = 0
    for interval in partition:
        if interval.start_token != cursor:
            errors.append(
                f"coverage_gap_or_overlap:{interval.start_token}:{interval.end_token}:"
                f"expected_start:{cursor}"
            )
            cursor = max(cursor, interval.end_token)
            continue
        cursor = interval.end_token
    if cursor != plan.total_context_tokens:
        errors.append(f"coverage_gap_or_overlap:end:{cursor}:{plan.total_context_tokens}")

    for decision in plan.decisions:
        if decision.admitted:
            continue
        rejected = TokenInterval(
            decision.candidate.start_token, decision.candidate.end_token
        )
        if not any(interval.contains(rejected) for interval in plan.recompute_intervals):
            errors.append(
                "rejected_not_recomputed:"
                f"{decision.candidate.name}:{rejected.start_token}:{rejected.end_token}"
            )
    return errors
