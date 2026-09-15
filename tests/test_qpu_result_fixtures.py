"""Validate qpu_result against anonymized synthetic Runtime-contract fixtures.

The fixtures under tests/fixtures/runtime_results/ are clearly synthetic, anonymized
representations of qiskit-ibm-runtime SamplerV2 result SHAPES — no account, job id, PII, or
token. This locks the offline adapter's behaviour to the documented result shapes before
any pilot (where a REAL recorded result would replace these, never a live call).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from acrpq.dashboard import qpu_result as R

_DIR = Path(__file__).resolve().parent / "fixtures" / "runtime_results"
_INDEX = json.loads((_DIR / "INDEX.json").read_text())


def _load(name):
    return json.loads((_DIR / name).read_text())


@pytest.mark.parametrize("name,n_bits", list(_INDEX["valid"].items()))
def test_valid_recorded_shapes_normalise(name, n_bits):
    norm = R.normalize_ibm_sampler_result(_load(name), expected_n_bits=n_bits)
    assert norm.n_unique >= 1 and norm.kind in ("counts", "quasi")
    assert norm.expected_n_bits == n_bits
    if "neg_quasi" in name:
        assert "negative_quasi_preserved" in norm.warnings          # never clamped
        assert norm.negative_quasi_policy == "preserved"


@pytest.mark.parametrize("name,n_bits", list(_INDEX["invalid"].items()))
def test_invalid_recorded_shapes_fail_closed(name, n_bits):
    with pytest.raises(R.ResultAdapterError):
        R.normalize_ibm_sampler_result(_load(name), expected_n_bits=n_bits)


def test_no_fixture_carries_pii_secret_or_absolute_path():
    path_re = re.compile(r'(?:^|["\s,:=\[(])(?:/[A-Za-z0-9._~-]+/[^"\s,\])]*|[A-Za-z]:\\\\)')
    secret_re = re.compile(r'QISKIT_IBM_TOKEN|api[_-]?token|Bearer\s|ghp_[A-Za-z0-9]{16,}|@')
    for f in _DIR.glob("*.json"):
        blob = f.read_text()
        assert not path_re.search(blob), f"{f.name} has an absolute path"
        assert not secret_re.search(blob), f"{f.name} has a secret/PII marker"
