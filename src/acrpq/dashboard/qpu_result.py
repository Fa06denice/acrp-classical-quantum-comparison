"""Robust IBM Sampler result adapter + scientific decode (Phase 5.3) — no IBM.

A single public normaliser turns the many shapes a Runtime SamplerV2 result can take
(counts, quasi-distributions, hex or spaced bitstrings, single-register/single-PUB) into
one strict, immutable internal model in QUBO bit order, failing CLOSED on empty / unknown
kind / wrong bit length / non-finite / zero-sum / ambiguous (multiple PUB or register)
inputs. It never silently corrects: negative quasi-probabilities are PRESERVED (with a
warning), not clamped. Then a scientific decode publishes the distinct solution candidates
without ever conflating one-hot validity, geometric feasibility, energy minimum, discrete
optimum, and continuous optimum.

This module contacts nothing and reads no credential — it only transforms recorded /
synthetic result envelopes (clearly labelled) and the local QUBO. The count→prob core and
the candidate ranking reuse the tested ``hardware_validation`` helpers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from . import hardware_validation as H


class ResultAdapterError(ValueError):
    """The result envelope is invalid/ambiguous; fail closed."""


_KINDS = frozenset({"counts", "quasi"})
_ORDERS = frozenset({"qiskit", "qubo"})


@dataclass(frozen=True)
class NormalizedResult:
    prob_qubo_order: dict[str, float]      # QUBO-order bitstring -> probability (or quasi value)
    kind: str
    bit_order_in: str
    expected_n_bits: int
    n_unique: int
    n_shots: int | None
    endianness_applied: str                # "reversed" (qiskit->qubo) | "identity"
    negative_quasi_policy: str             # "preserved" | "not_applicable"
    warnings: tuple[str, ...]
    format_provenance: str
    provider_metrics: dict[str, Any] = field(default_factory=dict)


def _clean_bitstring(bs: Any, n: int) -> str:
    if not isinstance(bs, str):
        raise ResultAdapterError(f"bitstring key must be a string; got {type(bs).__name__}")
    s = bs.replace(" ", "")
    if s.lower().startswith("0x"):
        try:
            s = format(int(s, 16), "b").zfill(n)
        except ValueError as exc:
            raise ResultAdapterError(f"bad hex bitstring {bs!r}: {exc}") from exc
    if len(s) != n or any(c not in "01" for c in s):
        raise ResultAdapterError(f"bitstring {bs!r} is not {n} binary characters")
    return s


def _extract_single_counts(envelope: dict, *, expect_single: bool) -> dict:
    """Reduce a possibly-nested SamplerV2-shaped envelope to one raw {bitstring: value}
    map, failing closed on multiple PUBs/registers when a single one is expected."""
    if "pubs" in envelope:
        pubs = envelope["pubs"]
        if not isinstance(pubs, list) or not pubs:
            raise ResultAdapterError("envelope 'pubs' must be a non-empty list")
        if expect_single and len(pubs) != 1:
            raise ResultAdapterError(f"expected a single PUB, got {len(pubs)}")
        pub = pubs[0]
        regs = pub.get("registers") if isinstance(pub, dict) else None
        if isinstance(regs, dict):
            if expect_single and len(regs) != 1:
                raise ResultAdapterError(f"expected a single register, got {len(regs)}")
            return next(iter(regs.values()))
        if isinstance(pub, dict) and isinstance(pub.get("counts"), dict):
            return pub["counts"]
        raise ResultAdapterError("PUB has no usable counts/registers")
    if "registers" in envelope:
        regs = envelope["registers"]
        if not isinstance(regs, dict) or not regs:
            raise ResultAdapterError("'registers' must be a non-empty object")
        if expect_single and len(regs) != 1:
            raise ResultAdapterError(f"expected a single register, got {len(regs)}")
        return next(iter(regs.values()))
    counts = envelope.get("counts")
    if not isinstance(counts, dict):
        raise ResultAdapterError("envelope has no 'counts' object")
    return counts


def normalize_ibm_sampler_result(
    envelope: Any, *, expected_n_bits: int, expect_single_result: bool = True,
) -> NormalizedResult:
    """Normalise a recorded/synthetic Sampler result envelope into a strict model."""
    if not isinstance(envelope, dict):
        raise ResultAdapterError("result envelope must be a dict")
    kind = envelope.get("kind")
    bit_order = envelope.get("bit_order")
    if kind not in _KINDS:
        raise ResultAdapterError(f"kind must be one of {sorted(_KINDS)}; got {kind!r}")
    if bit_order not in _ORDERS:
        raise ResultAdapterError(f"bit_order must be one of {sorted(_ORDERS)}; got {bit_order!r}")
    raw = _extract_single_counts(envelope, expect_single=expect_single_result)
    if not isinstance(raw, dict) or not raw:
        raise ResultAdapterError("result distribution is empty")

    warnings: list[str] = []
    folded: dict[str, float] = {}
    endianness = "reversed" if bit_order == "qiskit" else "identity"

    if kind == "counts":
        total = 0
        cleaned: list[tuple[str, int]] = []
        for bs, c in raw.items():
            if isinstance(c, bool) or not isinstance(c, int) or c < 0:
                raise ResultAdapterError(f"count for {bs!r} must be a non-negative int")
            key = _clean_bitstring(bs, expected_n_bits)
            key = key[::-1] if bit_order == "qiskit" else key
            cleaned.append((key, c))
            total += c
        if total <= 0:
            raise ResultAdapterError("counts sum to zero")
        for key, c in cleaned:
            folded[key] = folded.get(key, 0.0) + c / total
        n_shots: int | None = total
        neg_policy = "not_applicable"
    else:  # quasi — PRESERVE negatives, never clamp
        s = 0.0
        any_negative = False
        for bs, pv in raw.items():
            if isinstance(pv, bool) or not isinstance(pv, (int, float)) or not math.isfinite(pv):
                raise ResultAdapterError(f"quasi value for {bs!r} must be a finite number")
            key = _clean_bitstring(bs, expected_n_bits)
            key = key[::-1] if bit_order == "qiskit" else key
            folded[key] = folded.get(key, 0.0) + float(pv)
            s += float(pv)
            any_negative = any_negative or pv < 0
        if not math.isfinite(s) or abs(s) < 1e-12:
            raise ResultAdapterError("quasi-distribution sums to (near) zero")
        if abs(s - 1.0) > 0.05:
            warnings.append(f"quasi_sum_off_by={s - 1.0:.4g}")
        if any_negative:
            warnings.append("negative_quasi_preserved")   # explicit: no silent correction
        n_shots = envelope.get("shots") if isinstance(envelope.get("shots"), int) else None
        neg_policy = "preserved"

    metrics = envelope.get("provider_metrics")
    metrics = metrics if isinstance(metrics, dict) else {}
    return NormalizedResult(
        prob_qubo_order=folded, kind=kind, bit_order_in=bit_order,
        expected_n_bits=expected_n_bits, n_unique=len(folded), n_shots=n_shots,
        endianness_applied=endianness, negative_quasi_policy=neg_policy,
        warnings=tuple(warnings),
        format_provenance=f"normalized:{kind}:{bit_order}", provider_metrics=metrics)


@dataclass(frozen=True)
class ScientificDecode:
    """Distinct candidates, never conflated: best_raw (energy minimum), modal (most
    probable), best_onehot_valid (raw one-hot), best_feasible (geometrically conflict
    free). Plus masses/rates and the energy decomposition of the best feasible."""

    best_raw_bitstring: str | None
    best_raw_energy: float | None
    modal_bitstring: str | None
    best_onehot_valid_bitstring: str | None
    best_feasible_bitstring: str | None
    best_feasible_primary_cost: float | None
    onehot_valid_mass: float
    feasibility_rate: float
    best_feasible_qubo_energy: float | None
    onehot_penalty: float | None
    conflict_penalty: float | None
    residual_conflicts: int | None
    decoded_q_theta: tuple[tuple[float, float], ...] | None
    delta_vs_reference: float | None
    comparability: str
    comparability_reason: str
    warnings: tuple[str, ...]


def decode_scientific(
    qubo: Any, envelope: Any, *, objective_id: str, reference_objective: float | None = None,
    max_candidates: int = 256,
) -> ScientificDecode:
    """Decode a COUNTS envelope into the distinct scientific candidates. A quasi envelope
    with negatives is not a valid distribution to decode — it is returned as not comparable
    rather than silently coerced."""
    from ..quantum import explain
    kind = envelope.get("kind")
    if kind == "quasi":
        norm = normalize_ibm_sampler_result(envelope, expected_n_bits=qubo.n_qubits)
        if "negative_quasi_preserved" in norm.warnings:
            return ScientificDecode(
                None, None, None, None, None, None, 0.0, 0.0, None, None, None, None, None,
                None, "not_comparable", "negative_quasi_probabilities_not_decodable",
                norm.warnings)
    raw = _extract_single_counts(envelope, expect_single=True)
    summary = H.decode_samples(
        qubo, raw, bit_order=envelope["bit_order"], kind=kind,
        reference_objective=reference_objective, max_candidates=max_candidates)
    warns = list(summary.warnings)

    best_raw = summary.best_raw
    modal = summary.modal
    feasible = summary.best_feasible
    onehot = min((c for c in summary.candidates if c.onehot_valid),
                 key=lambda c: (c.energy, c.bitstring), default=None)
    onehot_mass = sum(c.probability for c in summary.candidates if c.onehot_valid)
    if summary.n_candidates_returned < summary.n_unique_total:
        warns.append("onehot_mass_over_returned_candidates_only")

    qtheta = None
    conflict_pen = onehot_pen = residual = fe_energy = fe_cost = None
    delta = None
    comparability, reason = "not_comparable", "no_feasible_candidate"
    if feasible is not None:
        eb = explain.energy_breakdown(qubo, feasible.bitstring)
        conflict_pen, onehot_pen = eb["conflict_penalty"], eb["onehot_penalty"]
        residual, fe_energy, fe_cost = eb["conflict_activations"], feasible.energy, feasible.objective
        exp = explain.bitstring_explanation(qubo, feasible.bitstring)
        qtheta = tuple((float(r["q"]), float(r["theta"])) for r in exp["aircraft"])
        if reference_objective is not None:
            delta = feasible.objective - reference_objective
            comparability, reason = "comparable", "feasible_candidate_vs_same_objective_reference"
        else:
            comparability, reason = "no_reference", "no_discrete_reference_provided"
    return ScientificDecode(
        best_raw_bitstring=(best_raw.bitstring if best_raw else None),
        best_raw_energy=(best_raw.energy if best_raw else None),
        modal_bitstring=(modal.bitstring if modal else None),
        best_onehot_valid_bitstring=(onehot.bitstring if onehot else None),
        best_feasible_bitstring=(feasible.bitstring if feasible else None),
        best_feasible_primary_cost=fe_cost,
        onehot_valid_mass=onehot_mass, feasibility_rate=summary.feasibility_rate,
        best_feasible_qubo_energy=fe_energy, onehot_penalty=onehot_pen,
        conflict_penalty=conflict_pen, residual_conflicts=residual,
        decoded_q_theta=qtheta, delta_vs_reference=delta,
        comparability=comparability, comparability_reason=reason, warnings=tuple(warns))
