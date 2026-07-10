# SPDX-License-Identifier: Apache-2.0
"""Tests for CoherentKV span admission."""

# First Party
from lmcache.v1.coherent_admission import (
    EXECUTION_C0,
    EXECUTION_C1,
    EXECUTION_F,
    SpanAdmissionCandidate,
    SpanAdmissionPlan,
    SpanDecision,
    TokenInterval,
    build_span_admission_plan,
    validate_span_admission_plan,
)


BASE_PATH_KEY = {
    "model_fp": "model@commit",
    "tokenizer_fp": "tokenizer@commit",
    "adapter_fp": "none",
    "dtype": "bfloat16",
    "rotary_config": {"rope_theta": 1_000_000.0},
    "engine_commit": "vllm@commit+overlay",
    "attention_backend": "FLEX_ATTENTION",
    "kernel_config": {"enforce_eager": True},
    "physical_class": "exact_prefix_tensor",
    "cache_state_class": "exact_prefix_hit",
}


def prefix_candidate(
    name: str,
    start: int,
    end: int,
    *,
    committed: bool = True,
    compatible: bool = True,
    cached_deps: dict[str, str] | None = None,
) -> SpanAdmissionCandidate:
    return SpanAdmissionCandidate(
        name,
        start,
        end,
        committed=committed,
        compatible=compatible,
        cached_deps=cached_deps or {"repo": "r1"},
        physical_class="exact_prefix_tensor",
        cache_state_class="exact_prefix_hit",
        runtime_path_key=BASE_PATH_KEY,
        compatibility_certificate={
            "certificate_id": "exact-prefix-cert",
            "status": "pass",
            "path_key": BASE_PATH_KEY,
            "allowed_execution_classes": [EXECUTION_C1],
            "source_commit": "source-commit",
            "source_dirty": False,
            "output_parity_validated": True,
            "logit_diff_validated": True,
        },
    )


def test_fail_closed_rejects_stale_and_recomputes_gap() -> None:
    plan = build_span_admission_plan(
        total_context_tokens=12,
        request_snapshot={"repo": "r1", "file": "v2"},
        candidates=[
            prefix_candidate("prefix", 0, 4),
            SpanAdmissionCandidate(
                "downstream",
                8,
                12,
                committed=True,
                compatible=True,
                cached_deps={"file": "v1"},
                physical_class="same_position_nonprefix_tensor",
                cache_state_class="selective_nonprefix_hit",
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
                physical_class="exact_prefix_tensor",
                cache_state_class="exact_prefix_hit",
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
        physical_class="exact_prefix_tensor",
        cache_state_class="exact_prefix_hit",
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
        physical_class="same_position_nonprefix_tensor",
        cache_state_class="selective_nonprefix_hit",
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
    admitted = prefix_candidate("prefix", 0, 4)
    plan = SpanAdmissionPlan(
        total_context_tokens=8,
        request_snapshot={"repo": "r1"},
        decisions=[SpanDecision(admitted, "admit")],
        recompute_intervals=[TokenInterval(2, 8)],
    )

    errors = validate_span_admission_plan(plan)

    assert any(error.startswith("coverage_gap_or_overlap:") for error in errors)


def test_strict_nonprefix_falls_back_to_c0() -> None:
    path_key = {
        **BASE_PATH_KEY,
        "physical_class": "shifted_nonprefix_tensor",
        "cache_state_class": "selective_nonprefix_hit",
    }
    candidate = SpanAdmissionCandidate(
        "shifted",
        4,
        8,
        committed=True,
        compatible=True,
        cached_deps={"repo": "r1"},
        physical_class="shifted_nonprefix_tensor",
        cache_state_class="selective_nonprefix_hit",
        runtime_path_key=path_key,
        compatibility_certificate={
            "certificate_id": "f-diagnostic",
            "status": "diagnostic",
            "path_key": path_key,
            "allowed_execution_classes": [EXECUTION_F],
            "source_commit": "source-commit",
            "source_dirty": False,
            "approximate": True,
        },
    )

    plan = build_span_admission_plan(
        candidates=[candidate],
        request_snapshot={"repo": "r1"},
        total_context_tokens=12,
    )

    assert plan.decisions[0].reason == "strict_nonprefix_requires_dense_recompute"
    assert plan.decisions[0].execution_class == EXECUTION_C0
    assert plan.recompute_intervals == [TokenInterval(0, 12)]


def test_explicit_diagnostic_f_path_is_labeled_and_admitted() -> None:
    path_key = {
        **BASE_PATH_KEY,
        "physical_class": "shifted_nonprefix_tensor",
        "cache_state_class": "selective_nonprefix_hit",
    }
    candidate = SpanAdmissionCandidate(
        "shifted",
        4,
        8,
        committed=True,
        compatible=True,
        cached_deps={"repo": "r1"},
        physical_class="shifted_nonprefix_tensor",
        cache_state_class="selective_nonprefix_hit",
        strict_mode=False,
        allow_approximate_nonprefix=True,
        diagnostic_mode=True,
        runtime_path_key=path_key,
        compatibility_certificate={
            "certificate_id": "f-diagnostic",
            "status": "diagnostic",
            "path_key": path_key,
            "allowed_execution_classes": [EXECUTION_F],
            "source_commit": "source-commit",
            "source_dirty": False,
            "approximate": True,
        },
    )

    plan = build_span_admission_plan(
        candidates=[candidate],
        request_snapshot={"repo": "r1"},
        total_context_tokens=12,
    )

    assert plan.decisions[0].reason == "admit"
    assert plan.decisions[0].execution_class == EXECUTION_F
    assert plan.recompute_intervals == [TokenInterval(0, 4), TokenInterval(8, 12)]


def test_prefix_certificate_path_mismatch_fails_closed() -> None:
    candidate = prefix_candidate("prefix", 0, 4)
    candidate = SpanAdmissionCandidate(
        **{
            **candidate.__dict__,
            "runtime_path_key": {**BASE_PATH_KEY, "attention_backend": "TRITON_ATTN"},
        }
    )

    plan = build_span_admission_plan(
        candidates=[candidate],
        request_snapshot={"repo": "r1"},
        total_context_tokens=8,
    )

    assert plan.decisions[0].reason == "certificate_path_mismatch"
    assert plan.recompute_intervals == [TokenInterval(0, 8)]
