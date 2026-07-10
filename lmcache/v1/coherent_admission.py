# SPDX-License-Identifier: Apache-2.0
"""Fail-closed span admission for CoherentKV materializations.

This module deliberately has no vLLM dependency.  A connector can use it after
span lookup and before any physical KV load/publication.  Unknown freshness or
compatibility always rejects the candidate and leaves the corresponding token
range to be recomputed.
"""

# Standard
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


EXECUTION_C0 = "C0_dense_recompute"
EXECUTION_C1 = "C1_exact_prefix_dense_suffix"
EXECUTION_F = "F_approximate_selective_nonprefix"

PATH_KEY_FIELDS = (
    "model_fp",
    "tokenizer_fp",
    "adapter_fp",
    "dtype",
    "rotary_config",
    "engine_commit",
    "attention_backend",
    "kernel_config",
    "physical_class",
    "cache_state_class",
)


def _coerce_bool(value: Any, *, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on", "pass", "validated"}:
        return True
    if text in {"0", "false", "no", "n", "off", "", "fail"}:
        return False
    return default


def _canonical_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list, tuple, bool, int, float)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value).strip()


def normalize_path_key(raw: Mapping[str, Any] | None) -> dict[str, str]:
    source = raw or {}
    return {field: _canonical_value(source.get(field)) for field in PATH_KEY_FIELDS}


def missing_path_key_fields(raw: Mapping[str, Any] | None) -> tuple[str, ...]:
    normalized = normalize_path_key(raw)
    missing = []
    for field, value in normalized.items():
        lowered = value.lower()
        unresolved = (
            lowered in {"unknown", "unresolved"}
            or lowered.endswith("@unresolved")
            or lowered.startswith("unknown-")
            or (field == "attention_backend" and lowered == "auto")
        )
        if not value or unresolved:
            missing.append(field)
    return tuple(missing)


def path_key_digest(raw: Mapping[str, Any] | None) -> str:
    payload = json.dumps(
        normalize_path_key(raw),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class CompatibilityCertificate:
    certificate_id: str
    status: str
    path_key: Mapping[str, Any]
    allowed_execution_classes: Sequence[str]
    source_commit: str
    source_dirty: bool = True
    output_parity_validated: bool = False
    logit_diff_validated: bool = False
    approximate: bool = False

    @classmethod
    def from_mapping(
        cls, raw: Mapping[str, Any] | None
    ) -> "CompatibilityCertificate | None":
        if not raw:
            return None
        allowed = raw.get("allowed_execution_classes") or ()
        if isinstance(allowed, str):
            allowed = (allowed,)
        return cls(
            certificate_id=str(raw.get("certificate_id") or ""),
            status=str(raw.get("status") or ""),
            path_key=(
                raw.get("path_key")
                if isinstance(raw.get("path_key"), Mapping)
                else {}
            ),
            allowed_execution_classes=tuple(str(value) for value in allowed),
            source_commit=str(raw.get("source_commit") or ""),
            source_dirty=_coerce_bool(raw.get("source_dirty"), default=True),
            output_parity_validated=_coerce_bool(
                raw.get("output_parity_validated")
            ),
            logit_diff_validated=_coerce_bool(raw.get("logit_diff_validated")),
            approximate=_coerce_bool(raw.get("approximate")),
        )


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
    physical_class: str = "unknown"
    cache_state_class: str = "unknown"
    strict_mode: bool = True
    allow_approximate_nonprefix: bool = False
    diagnostic_mode: bool = False
    runtime_path_key: Mapping[str, Any] = field(default_factory=dict)
    compatibility_certificate: Mapping[str, Any] = field(default_factory=dict)

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

    @property
    def execution_class(self) -> str:
        if not self.admitted:
            return EXECUTION_C0
        return execution_class_for(self.candidate)


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


def execution_class_for(candidate: SpanAdmissionCandidate) -> str:
    exact_prefix = (
        candidate.start_token == 0
        and candidate.physical_class == "exact_prefix_tensor"
        and candidate.cache_state_class == "exact_prefix_hit"
    )
    if exact_prefix:
        return EXECUTION_C1
    if not candidate.strict_mode and candidate.allow_approximate_nonprefix:
        return EXECUTION_F
    return EXECUTION_C0


def compatibility_reason(candidate: SpanAdmissionCandidate) -> str | None:
    execution_class = execution_class_for(candidate)
    if execution_class == EXECUTION_C0:
        if candidate.start_token == 0:
            return "uncertified_prefix_class"
        return "strict_nonprefix_requires_dense_recompute"
    missing = missing_path_key_fields(candidate.runtime_path_key)
    if missing:
        return "incomplete_path_key:" + ",".join(missing)
    certificate = CompatibilityCertificate.from_mapping(
        candidate.compatibility_certificate
    )
    if certificate is None:
        return "missing_compatibility_certificate"
    if not certificate.certificate_id:
        return "missing_certificate_id"
    if not certificate.source_commit:
        return "missing_certificate_source_commit"
    if certificate.source_dirty:
        return "dirty_certificate_source"
    certificate_missing = missing_path_key_fields(certificate.path_key)
    if certificate_missing:
        return "incomplete_certificate_key:" + ",".join(certificate_missing)
    if normalize_path_key(certificate.path_key) != normalize_path_key(
        candidate.runtime_path_key
    ):
        return "certificate_path_mismatch"
    if execution_class not in certificate.allowed_execution_classes:
        return "certificate_execution_class_mismatch"
    if execution_class == EXECUTION_C1:
        if certificate.status != "pass":
            return "certificate_not_passed"
        if not certificate.output_parity_validated:
            return "missing_output_parity"
        if not certificate.logit_diff_validated:
            return "missing_logit_bound"
    else:
        allowed_statuses = {"pass"}
        if candidate.diagnostic_mode:
            allowed_statuses.add("diagnostic")
        if certificate.status not in allowed_statuses:
            return "approximate_certificate_not_enabled"
        if not certificate.approximate:
            return "approximate_label_missing"
    return None


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
    physical_reason = compatibility_reason(candidate)
    if physical_reason:
        return physical_reason
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
