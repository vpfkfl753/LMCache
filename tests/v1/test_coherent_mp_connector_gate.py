# SPDX-License-Identifier: Apache-2.0
"""Tests for scheduler-side C0/C1/F binding in the MP connector."""

# Standard
from types import SimpleNamespace

# Third Party
import pytest

pytest.importorskip("vllm")

# First Party
from lmcache.integration.vllm.lmcache_mp_connector import (
    COHERENTKV_ALLOW_APPROXIMATE_KEY,
    COHERENTKV_DIAGNOSTIC_MODE_KEY,
    COHERENTKV_ENABLED_KEY,
    COHERENTKV_REQUEST_SNAPSHOT_KEY,
    COHERENTKV_SPAN_CERTIFICATES_KEY,
    COHERENTKV_SPAN_COMMITTED_KEY,
    COHERENTKV_SPAN_COMPATIBLE_KEY,
    COHERENTKV_SPAN_DEPS_KEY,
    COHERENTKV_STRICT_MODE_KEY,
    _build_coherentkv_mp_admission_plan,
    _coherentkv_mp_admitted_nonprefix_span_matches,
)
from lmcache.v1.coherent_admission import EXECUTION_C0, EXECUTION_F


RUNTIME_PATH_BASE = {
    "model_fp": "model@commit",
    "tokenizer_fp": "tokenizer@commit",
    "adapter_fp": "none",
    "dtype": "bfloat16",
    "rotary_config": {"rope_theta": 1_000_000.0},
    "engine_commit": "vllm@commit+overlay",
    "attention_backend": "FLEX_ATTENTION",
    "kernel_config": {"enforce_eager": True},
}


def span_result() -> SimpleNamespace:
    return SimpleNamespace(
        prefix_hit_tokens=0,
        spans=[
            SimpleNamespace(
                start=16,
                end=32,
                key_index=0,
                hit=True,
                metadata={
                    "coherentkv_cb_hash_hex": "00ff",
                    "coherentkv_source_start_token": "0",
                },
            )
        ],
    )


def request_config(*, strict: bool, source_dirty: bool = False) -> dict:
    span_key = "16:32"
    path_key = {
        **RUNTIME_PATH_BASE,
        "physical_class": "shifted_nonprefix_tensor",
        "cache_state_class": "selective_nonprefix_hit",
    }
    return {
        COHERENTKV_ENABLED_KEY: True,
        COHERENTKV_STRICT_MODE_KEY: strict,
        COHERENTKV_ALLOW_APPROXIMATE_KEY: not strict,
        COHERENTKV_DIAGNOSTIC_MODE_KEY: not strict,
        COHERENTKV_REQUEST_SNAPSHOT_KEY: {"repo": "v1"},
        COHERENTKV_SPAN_DEPS_KEY: {span_key: {"repo": "v1"}},
        COHERENTKV_SPAN_COMMITTED_KEY: {span_key: True},
        COHERENTKV_SPAN_COMPATIBLE_KEY: {span_key: True},
        COHERENTKV_SPAN_CERTIFICATES_KEY: {
            span_key: {
                "certificate_id": "f-diagnostic",
                "status": "diagnostic",
                "path_key": path_key,
                "allowed_execution_classes": [EXECUTION_F],
                "source_commit": "source-commit",
                "source_dirty": source_dirty,
                "approximate": True,
            }
        },
    }


def test_strict_nonprefix_never_reaches_span_match() -> None:
    plan = _build_coherentkv_mp_admission_plan(
        span_result(),
        request_config(strict=True),
        RUNTIME_PATH_BASE,
    )

    assert plan.decisions[0].execution_class == EXECUTION_C0
    assert plan.decisions[0].reason == "strict_nonprefix_requires_dense_recompute"
    assert _coherentkv_mp_admitted_nonprefix_span_matches(plan, 0) == ()


def test_diagnostic_f_reaches_worker_with_non_strict_labels() -> None:
    plan = _build_coherentkv_mp_admission_plan(
        span_result(),
        request_config(strict=False),
        RUNTIME_PATH_BASE,
    )

    assert plan.decisions[0].execution_class == EXECUTION_F
    assert plan.decisions[0].reason == "admit"
    matches = _coherentkv_mp_admitted_nonprefix_span_matches(plan, 0)
    assert len(matches) == 1
    metadata = matches[0].metadata
    assert metadata["coherentkv_execution_class"] == EXECUTION_F
    assert metadata["coherentkv_approximate"] == "true"
    assert metadata["coherentkv_strict_mode"] == "false"
    assert metadata["coherentkv_certificate_id"] == "f-diagnostic"
    assert len(metadata["coherentkv_path_key_sha256"]) == 64


def test_dirty_diagnostic_certificate_fails_closed() -> None:
    plan = _build_coherentkv_mp_admission_plan(
        span_result(),
        request_config(strict=False, source_dirty=True),
        RUNTIME_PATH_BASE,
    )

    assert plan.decisions[0].execution_class == EXECUTION_C0
    assert plan.decisions[0].reason == "dirty_certificate_source"
    assert _coherentkv_mp_admitted_nonprefix_span_matches(plan, 0) == ()
