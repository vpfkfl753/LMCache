# SPDX-License-Identifier: Apache-2.0
"""Tests for CoherentKV span admission."""

# First Party
from lmcache.v1.coherent_admission import (
    SpanAdmissionCandidate,
    SpanAdmissionPlan,
    SpanDecision,
    TokenInterval,
    build_span_admission_plan,
    validate_span_admission_plan,
)


def test_fail_closed_rejects_stale_and_recomputes_gap() -> None:
    plan = build_span_admission_plan(
        total_context_tokens=12,
        request_snapshot={"repo": "r1", "file": "v2"},
        candidates=[
            SpanAdmissionCandidate(
                "prefix",
                0,
                4,
                committed=True,
                compatible=True,
                cached_deps={"repo": "r1"},
            ),
            SpanAdmissionCandidate(
                "downstream",
                8,
                12,
                committed=True,
                compatible=True,
                cached_deps={"file": "v1"},
            ),
        ],
    )

    assert [decision.reason for decision in plan.decisions] == [
        "admit",
        "stale:file:v1->v2",
    ]
    assert plan.recompute_intervals == [TokenInterval(4, 12)]
    assert validate_span_admission_plan(plan) == []


def test_missing_snapshot_dependency_rejects() -> None:
    plan = build_span_admission_plan(
        total_context_tokens=4,
        request_snapshot={"repo": "r1"},
        candidates=[
            SpanAdmissionCandidate(
                "candidate",
                0,
                4,
                committed=True,
                compatible=True,
                cached_deps={"repo": "r1", "tool_obs": "o1"},
            )
        ],
    )

    assert plan.decisions[0].reason == "missing_snapshot_dep:tool_obs:o1"
    assert plan.recompute_intervals == [TokenInterval(0, 4)]
    assert validate_span_admission_plan(plan) == []


def test_validation_catches_corrupted_stale_admission() -> None:
    candidate = SpanAdmissionCandidate(
        "bad",
        0,
        4,
        committed=True,
        compatible=True,
        cached_deps={"file": "v1"},
    )
    plan = SpanAdmissionPlan(
        total_context_tokens=4,
        request_snapshot={"file": "v2"},
        decisions=[SpanDecision(candidate, "admit")],
        recompute_intervals=[],
    )

    errors = validate_span_admission_plan(plan)

    assert any(error.startswith("admitted_invalid_span:") for error in errors)


def test_validation_catches_missing_dependency_recompute_gap() -> None:
    candidate = SpanAdmissionCandidate(
        "missing",
        2,
        6,
        committed=True,
        compatible=True,
        cached_deps={"tool_obs": "o1"},
    )
    plan = SpanAdmissionPlan(
        total_context_tokens=8,
        request_snapshot={},
        decisions=[SpanDecision(candidate, "missing_snapshot_dep:tool_obs:o1")],
        recompute_intervals=[TokenInterval(0, 2), TokenInterval(6, 8)],
    )

    errors = validate_span_admission_plan(plan)

    assert any(error.startswith("coverage_gap_or_overlap:") for error in errors)
    assert any(error.startswith("rejected_not_recomputed:") for error in errors)


def test_validation_catches_overlap_corruption() -> None:
    admitted = SpanAdmissionCandidate(
        "prefix",
        0,
        4,
        committed=True,
        compatible=True,
        cached_deps={"repo": "r1"},
    )
    plan = SpanAdmissionPlan(
        total_context_tokens=8,
        request_snapshot={"repo": "r1"},
        decisions=[SpanDecision(admitted, "admit")],
        recompute_intervals=[TokenInterval(2, 8)],
    )

    errors = validate_span_admission_plan(plan)

    assert any(error.startswith("coverage_gap_or_overlap:") for error in errors)
