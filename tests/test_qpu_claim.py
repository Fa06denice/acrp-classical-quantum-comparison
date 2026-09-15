"""Phase 5.2.C — inter-process single-winner claim.

Covers the claim primitive directly, its fail-closed handling of corrupt/foreign
claim and runid files, crash-recovery, and REAL multi-process concurrency (2/5/10
separate interpreters via subprocess — not threads) proving exactly one designated
``local_run_id`` and exactly one preparation.
"""
from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from acrpq.dashboard import qpu_claim

# a well-formed config hash for the claim (sha256:<64 hex>)
CH = "sha256:" + "a" * 64
CH2 = "sha256:" + "b" * 64


def _prep(ids):
    """A prepare_fn that records each invocation and returns a fresh id."""
    def _fn():
        rid = f"run-{len(ids)}-{id(ids)}"
        ids.append(rid)
        return rid
    return _fn


# --- basic winner / adopt / distinct-config -------------------------------
def test_first_caller_wins_second_adopts(tmp_path):
    prepared_a, prepared_b = [], []
    a = qpu_claim.claim_or_adopt(CH, claims_dir=str(tmp_path), prepare_fn=_prep(prepared_a))
    b = qpu_claim.claim_or_adopt(CH, claims_dir=str(tmp_path), prepare_fn=_prep(prepared_b))
    assert a.role == "winner" and b.role == "adopter"
    assert a.local_run_id == b.local_run_id           # same designated run
    assert len(prepared_a) == 1 and len(prepared_b) == 0   # only the winner prepared


def test_distinct_configs_get_distinct_claims(tmp_path):
    a = qpu_claim.claim_or_adopt(CH, claims_dir=str(tmp_path), prepare_fn=lambda: "rid-A")
    b = qpu_claim.claim_or_adopt(CH2, claims_dir=str(tmp_path), prepare_fn=lambda: "rid-B")
    assert a.local_run_id == "rid-A" and b.local_run_id == "rid-B"
    ca, ra = qpu_claim.claim_paths(tmp_path, CH)
    cb, rb = qpu_claim.claim_paths(tmp_path, CH2)
    assert {ca, ra}.isdisjoint({cb, rb})


def test_config_hash_must_be_wellformed(tmp_path):
    for bad in ["", "nope", "sha256:xyz", "sha256:" + "a" * 63, "sha256:" + "A" * 64]:
        with pytest.raises(qpu_claim.ClaimError):
            qpu_claim.claim_or_adopt(bad, claims_dir=str(tmp_path), prepare_fn=lambda: "r")


# --- fail-closed on corruption --------------------------------------------
def test_corrupt_runid_fails_closed(tmp_path):
    qpu_claim.claim_or_adopt(CH, claims_dir=str(tmp_path), prepare_fn=lambda: "rid-1")
    _claim, runid = qpu_claim.claim_paths(tmp_path, CH)
    runid.write_text("{ not json")
    with pytest.raises(qpu_claim.ClaimError, match="corrupt runid"):
        qpu_claim.claim_or_adopt(CH, claims_dir=str(tmp_path), prepare_fn=lambda: "rid-2")


def test_tampered_runid_fails_closed(tmp_path):
    qpu_claim.claim_or_adopt(CH, claims_dir=str(tmp_path), prepare_fn=lambda: "rid-1")
    _claim, runid = qpu_claim.claim_paths(tmp_path, CH)
    obj = json.loads(runid.read_text())
    obj["local_run_id"] = "rid-EVIL"                  # break the content hash
    runid.write_text(json.dumps(obj))
    with pytest.raises(qpu_claim.ClaimError, match="content hash"):
        qpu_claim.claim_or_adopt(CH, claims_dir=str(tmp_path), prepare_fn=lambda: "rid-2")


def test_corrupt_claim_fails_closed_never_prepares(tmp_path):
    """A corrupt claim (present, no valid content) must fail closed for a loser — it
    must NEVER be mistaken as absent and drive a fresh preparation."""
    claim, _runid = qpu_claim.claim_paths(tmp_path, CH)
    tmp_path.mkdir(parents=True, exist_ok=True)
    claim.write_text("garbage-not-json")             # a corrupt lock, no runid yet
    prepared = []
    with pytest.raises(qpu_claim.ClaimError, match="corrupt claim"):
        qpu_claim.claim_or_adopt(CH, claims_dir=str(tmp_path), prepare_fn=_prep(prepared),
                                 wait_timeout=0.0)
    assert prepared == []                             # no preparation triggered


def test_foreign_claim_for_other_config_fails_closed(tmp_path):
    claim, _runid = qpu_claim.claim_paths(tmp_path, CH)
    tmp_path.mkdir(parents=True, exist_ok=True)
    # a well-formed claim but for a DIFFERENT config, placed at THIS config's path
    from acrpq.dashboard.persist import _sha256
    body = {"schema": qpu_claim.CLAIM_SCHEMA, "config_hash": CH2, "nonce": "x"}
    body["content_sha256"] = _sha256(qpu_claim._canonical(body))
    claim.write_text(qpu_claim._canonical(body))
    with pytest.raises(qpu_claim.ClaimError, match="different config"):
        qpu_claim.claim_or_adopt(CH, claims_dir=str(tmp_path), prepare_fn=lambda: "r",
                                 wait_timeout=0.0)


# --- crash recovery -------------------------------------------------------
def test_committed_runid_is_adopted_without_preparing(tmp_path):
    """Crash-after-runid: a committed runid exists -> adopt, never prepare."""
    qpu_claim.claim_or_adopt(CH, claims_dir=str(tmp_path), prepare_fn=lambda: "rid-winner")
    prepared = []
    r = qpu_claim.claim_or_adopt(CH, claims_dir=str(tmp_path), prepare_fn=_prep(prepared))
    assert r.local_run_id == "rid-winner" and r.role == "adopter" and prepared == []


def test_stale_claim_without_runid_recovers(tmp_path):
    """Crash-after-claim-before-runid: a valid claim exists but no runid. After the
    wait window a recoverer prepares and commits — exactly one runid still wins."""
    claim, _runid = qpu_claim.claim_paths(tmp_path, CH)
    tmp_path.mkdir(parents=True, exist_ok=True)
    from acrpq.dashboard.persist import _sha256
    body = {"schema": qpu_claim.CLAIM_SCHEMA, "config_hash": CH, "nonce": "winner-nonce"}
    body["content_sha256"] = _sha256(qpu_claim._canonical(body))
    claim.write_text(qpu_claim._canonical(body))      # a stale claim, no runid
    prepared = []
    # sleeper is a no-op and monotonic advances past the deadline on the 2nd read
    ticks = iter([0.0, 0.0, 100.0, 100.0, 100.0])
    r = qpu_claim.claim_or_adopt(CH, claims_dir=str(tmp_path), prepare_fn=_prep(prepared),
                                 wait_timeout=1.0, sleeper=lambda _s: None,
                                 monotonic=lambda: next(ticks))
    assert r.role == "recovered" and len(prepared) == 1
    assert r.local_run_id == prepared[0]


def test_recovery_race_only_one_runid_wins(tmp_path):
    """Two recoverers both prepare; only one runid is committed; the other adopts it."""
    claim, runid = qpu_claim.claim_paths(tmp_path, CH)
    # pre-commit a runid as if recoverer-1 won the publish race
    qpu_claim._commit_runid(runid, CH, "rid-recoverer-1")
    prepared = []
    r = qpu_claim.claim_or_adopt(CH, claims_dir=str(tmp_path), prepare_fn=_prep(prepared))
    assert r.local_run_id == "rid-recoverer-1" and r.role == "adopter" and prepared == []


# --- REAL multi-process concurrency (spawned interpreters, not threads) ---
_WORKER = textwrap.dedent(
    """
    import json, sys, time
    from acrpq.dashboard import qpu_claim
    claims_dir, config_hash, out_path, gate = sys.argv[1:5]
    # spin on a shared gate file so the workers contend as simultaneously as possible
    import os
    while not os.path.exists(gate):
        time.sleep(0.001)
    prepared = []
    def prep():
        rid = "run-%d-%d" % (os.getpid(), time.time_ns())
        prepared.append(rid)
        return rid
    res = qpu_claim.claim_or_adopt(config_hash, claims_dir=claims_dir, prepare_fn=prep,
                                   wait_timeout=30.0, poll_interval=0.005)
    json.dump({"role": res.role, "rid": res.local_run_id, "prepared": prepared},
              open(out_path, "w"))
    """
)


def _run_concurrent(tmp_path, n):
    worker = tmp_path / "worker.py"
    worker.write_text(_WORKER)
    claims_dir = str(tmp_path / "claims")
    gate = tmp_path / "GO"
    outs = [str(tmp_path / f"out-{i}.json") for i in range(n)]
    procs = [subprocess.Popen([sys.executable, str(worker), claims_dir, CH, outs[i], str(gate)])
             for i in range(n)]
    gate.write_text("go")                             # release all workers at once
    for p in procs:
        assert p.wait(timeout=60) == 0
    results = [json.loads(Path(o).read_text()) for o in outs]
    return results


@pytest.mark.parametrize("n", [2, 5, 10])
def test_concurrent_processes_single_winner(tmp_path, n):
    """NORMAL concurrent path (no crash): exactly one candidate is prepared and exactly
    one run is designated. (After a crash several candidates may exist but still exactly
    one is designated — see test_crash_before_runid_... in the offline suite.)"""
    results = _run_concurrent(tmp_path, n)
    assert len(results) == n
    rids = {r["rid"] for r in results}
    assert len(rids) == 1                              # ONE designated run for all
    total_prepared = sum(len(r["prepared"]) for r in results)
    assert total_prepared == 1                         # normal path: one preparation
    winners = [r for r in results if r["role"] == "winner"]
    assert len(winners) == 1                            # exactly one winner
    assert all(r["role"] in ("winner", "adopter") for r in results)   # none had to recover
