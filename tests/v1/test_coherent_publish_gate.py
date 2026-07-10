# SPDX-License-Identifier: Apache-2.0
"""Tests for the worker-side CoherentKV execution proof gate."""

# Standard
from types import SimpleNamespace

# First Party
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    _coherentkv_validate_cb_v3_publish_span_metadata,
)
from lmcache.v1.coherent_admission import EXECUTION_C1, EXECUTION_F


def span_load(metadata: dict[str, str]) -> SimpleNamespace:
    return SimpleNamespace(
        source_start_token=16,
        source_end_token=32,
        destination_start_token=48,
        destination_end_token=64,
        destination_block_ids=(3,),
        destination_block_ids_by_group=(),
        metadata={
            "coherentkv_cb_hash_hex": "00ff",
            "coherentkv_admitted": "true",
            "coherentkv_gap_validated": "true",
            **metadata,
        },
    )


def test_c1_requires_parity_artifact() -> None:
    valid, errors = _coherentkv_validate_cb_v3_publish_span_metadata(
        [
            span_load(
                {
                    "coherentkv_execution_class": EXECUTION_C1,
                    "coherentkv_parity_validated": "true",
                    "coherentkv_parity_artifact_id": "parity-artifact",
                }
            )
        ]
    )
    assert valid
    assert errors == []

    valid, errors = _coherentkv_validate_cb_v3_publish_span_metadata(
        [span_load({"coherentkv_execution_class": EXECUTION_C1})]
    )
    assert not valid
    assert "span_0:missing_parity_validation_proof" in errors
    assert "span_0:missing_parity_artifact_id" in errors


def test_f_requires_approximate_path_certificate_and_non_strict_label() -> None:
    valid, errors = _coherentkv_validate_cb_v3_publish_span_metadata(
        [
            span_load(
                {
                    "coherentkv_execution_class": EXECUTION_F,
                    "coherentkv_approximate": "true",
                    "coherentkv_approximate_opt_in": "true",
                    "coherentkv_strict_mode": "false",
                    "coherentkv_certificate_id": "f-diagnostic",
                    "coherentkv_path_key_sha256": "a" * 64,
                }
            )
        ]
    )
    assert valid
    assert errors == []

    valid, errors = _coherentkv_validate_cb_v3_publish_span_metadata(
        [
            span_load(
                {
                    "coherentkv_execution_class": EXECUTION_F,
                    "coherentkv_strict_mode": "true",
                }
            )
        ]
    )
    assert not valid
    assert "span_0:missing_approximate_opt_in" in errors
    assert "span_0:approximate_path_marked_strict" in errors
    assert "span_0:missing_compatibility_certificate_id" in errors
    assert "span_0:missing_path_key_digest" in errors
