"""Persistent, event-sourced QPU run store — no IBM contact, pure filesystem."""

from __future__ import annotations

import json

import pytest

from acrpq.dashboard import qpu_runs as Q
from acrpq.dashboard.qpu_runs import CorruptRunError, IllegalTransitionError, QpuState

_VALID_SHA = "sha256:" + "a" * 64


def _params(mode="hardware_validation", **over):
    p = {
        "mode": mode, "instance": "CP_4", "subset": None, "n_theta": 3, "n_q": 1,
        "qubo_sha256": _VALID_SHA, "backend_requested": "least_busy",
        "shots": 4096, "reps": 1, "optimizer": "COBYLA", "maxiter": 40,
        "max_quantum_seconds": 10.0, "max_jobs": 1, "transpiler_seed": 7,
    }
    p.update(over)
    return p


def _submit(rid, root):
    """Drive a run to SUBMITTED via valid transitions (awaiting_confirmation first)."""
    Q.transition(rid, QpuState.AWAITING_CONFIRMATION, root=root)
    Q.transition(rid, QpuState.SUBMITTING, root=root)
    Q.transition(rid, QpuState.SUBMITTED, root=root)


# --- basics ----------------------------------------------------------------
def test_prepare_then_load_is_prepared(tmp_path):
    rid = Q.prepare(_params(), root=str(tmp_path))
    run = Q.load(rid, root=str(tmp_path))
    assert run["state"] == "prepared" and run["terminal"] is False


def test_valid_state_machine_survives_reload(tmp_path):
    root = str(tmp_path)
    rid = Q.prepare(_params(), root=root)
    _submit(rid, root)
    Q.transition(rid, QpuState.QUEUED, root=root)
    Q.transition(rid, QpuState.RUNNING, root=root)
    Q.transition(rid, QpuState.COMPLETED, root=root)
    run = Q.load(rid, root=root)  # fresh read == restart replay
    assert run["state"] == "completed" and run["terminal"] is True


def test_job_ids_accumulated_ordered_and_deduped(tmp_path):
    root = str(tmp_path)
    rid = Q.prepare(_params(mode="full_qaoa_hardware"), root=root)
    for jid in ["cxyz-1", "cxyz-2", "cxyz-2", "cxyz-3"]:
        Q.record_job_id(rid, jid, root=root)
    assert Q.load(rid, root=root)["ibm_job_ids"] == ["cxyz-1", "cxyz-2", "cxyz-3"]


def test_job_id_is_stripped_and_length_bounded(tmp_path):
    root = str(tmp_path)
    rid = Q.prepare(_params(mode="full_qaoa_hardware"), root=root)
    Q.record_job_id(rid, "  cjob-1  ", root=root)
    assert Q.load(rid, root=root)["ibm_job_ids"] == ["cjob-1"]
    with pytest.raises(ValueError, match="non-empty after stripping"):
        Q.record_job_id(rid, "   ", root=root)
    with pytest.raises(ValueError, match="exceeds"):
        Q.record_job_id(rid, "x" * (Q.MAX_JOB_ID_LEN + 1), root=root)


# --- mode-dependent param schema (item 2) ----------------------------------
def test_hardware_mode_requires_budget_hash_backend(tmp_path):
    root = str(tmp_path)
    for missing in ("qubo_sha256", "backend_requested", "max_quantum_seconds",
                    "max_jobs", "transpiler_seed"):
        p = _params()
        p.pop(missing)
        with pytest.raises(ValueError):
            Q.prepare(p, root=root)
    # aer_simulation has no such hardware requirements
    assert Q.prepare(
        {"mode": "aer_simulation", "instance": "CP_4", "n_theta": 3, "shots": 256}, root=root
    )


# --- hardware transition rules (item 3) ------------------------------------
def test_hardware_forbids_prepared_to_submitting(tmp_path):
    root = str(tmp_path)
    rid = Q.prepare(_params(), root=root)  # hardware_validation
    with pytest.raises(IllegalTransitionError, match="not allowed"):
        Q.transition(rid, QpuState.SUBMITTING, root=root)  # awaiting_confirmation is mandatory
    Q.transition(rid, QpuState.AWAITING_CONFIRMATION, root=root)
    Q.transition(rid, QpuState.SUBMITTING, root=root)  # now allowed
    assert Q.load(rid, root=root)["state"] == "submitting"


# --- submission_unknown may not fail without reconciliation (item 4) -------
def test_submission_unknown_cannot_fail_directly(tmp_path):
    root = str(tmp_path)
    rid = Q.prepare(_params(), root=root)
    Q.transition(rid, QpuState.AWAITING_CONFIRMATION, root=root)
    Q.transition(rid, QpuState.SUBMITTING, root=root)
    Q.transition(rid, QpuState.SUBMISSION_UNKNOWN, root=root)
    with pytest.raises(IllegalTransitionError, match="not allowed"):
        Q.transition(rid, QpuState.FAILED, root=root)  # must reconcile first
    Q.transition(rid, QpuState.RECONCILIATION_REQUIRED, root=root)
    Q.transition(rid, QpuState.FAILED, root=root)  # determined failure is fine


# --- reserved fields + nested secret ---------------------------------------
def test_reserved_fields_and_nested_secrets_rejected(tmp_path):
    root = str(tmp_path)
    rid = Q.prepare(_params(), root=root)
    for field in ("seq", "stamp_utc", "prev_sha256", "content_sha256"):
        with pytest.raises(ValueError, match="reserved event field"):
            Q.append_event(rid, {"event_kind": "provider_observation", field: "x"}, root=root)
    with pytest.raises(ValueError, match="secret-like"):
        Q.prepare(_params(auth={"nested": {"bearer_token": "x"}}), root=root)


# --- tuples refused (item 8) -----------------------------------------------
def test_tuple_is_refused_not_coerced(tmp_path):
    root = str(tmp_path)
    with pytest.raises(ValueError, match="tuple not accepted"):
        Q.prepare(_params(coords=(1, 2)), root=root)
    rid = Q.prepare(_params(), root=root)
    with pytest.raises(ValueError, match="tuple not accepted"):
        Q.append_event(rid, {"event_kind": "provider_observation", "observation": (1, 2)}, root=root)


# --- deep snapshot of params (item 10) -------------------------------------
def test_params_are_deep_snapshotted_before_persist(tmp_path):
    root = str(tmp_path)
    p = _params(subset=[1, 2])
    rid = Q.prepare(p, root=root)
    p["shots"] = 999_999          # mutate the caller's dict AFTER prepare
    p["subset"].append(3)         # and a nested container
    run = Q.load(rid, root=root)
    assert run["params"]["shots"] == 4096
    assert run["params"]["subset"] == [1, 2]


# --- terminal immutability + probative late job id (item 6) ----------------
def test_terminal_is_immutable_but_accepts_probative_evidence(tmp_path):
    root = str(tmp_path)
    rid = Q.prepare(_params(), root=root)
    _submit(rid, root)
    Q.transition(rid, QpuState.COMPLETED, root=root)

    # a real transition after terminal is always refused
    with pytest.raises(IllegalTransitionError, match="terminal"):
        Q.transition(rid, QpuState.RUNNING, root=root)
    # a plain job_id after terminal is refused (must be a late_job_id)
    with pytest.raises(IllegalTransitionError, match="terminal"):
        Q.record_job_id(rid, "j-late", root=root)

    # probative evidence is accepted and never changes state/result
    Q.record_late_job_id(rid, "  discovered-1  ", root=root)
    Q.record_provider_metrics(rid, {"quantum_seconds": 3.2}, root=root)
    run = Q.load(rid, root=root)
    assert run["state"] == "completed"          # unchanged
    assert run["ibm_job_ids"] == []             # main list untouched
    assert run["late_ibm_job_ids"] == ["discovered-1"]
    assert any(o["event_kind"] == "provider_metrics" for o in run["provider_observations"])
    # a probative event may not smuggle a state (rejected as an unexpected field)
    with pytest.raises(ValueError, match="unexpected field"):
        Q.append_event(
            rid, {"event_kind": "provider_observation", "observation": 1, "state": "running"},
            root=root,
        )


# --- Phase 2.3: strict per-kind field whitelist (adversarial) --------------
def test_strict_event_field_whitelist_adversarial(tmp_path):
    root = str(tmp_path)
    rid = Q.prepare(_params(), root=root)

    # error may not carry state ; job_id may not carry state
    with pytest.raises(ValueError, match="unexpected field"):
        Q.append_event(rid, {"event_kind": "error", "error": "x", "state": "failed"}, root=root)
    with pytest.raises(ValueError, match="unexpected field"):
        Q.append_event(rid, {"event_kind": "job_id", "job_id": "j", "state": "running"}, root=root)
    # a transition with an injected extra field is refused
    with pytest.raises(ValueError, match="unexpected field"):
        Q.append_event(
            rid, {"event_kind": "transition", "state": "awaiting_confirmation", "injected": 1},
            root=root,
        )
    # an arbitrary extra field on any kind is refused
    with pytest.raises(ValueError, match="unexpected field"):
        Q.append_event(
            rid, {"event_kind": "provider_observation", "observation": 1, "foo": 2}, root=root
        )
    # provider_metrics / reconciliation_evidence require a NON-EMPTY dict
    with pytest.raises(ValueError, match="non-empty dict"):
        Q.record_provider_metrics(rid, {}, root=root)
    with pytest.raises(ValueError, match="non-empty dict"):
        Q.record_reconciliation_evidence(rid, {}, root=root)
    # result on a transition is allowed ONLY for completed
    with pytest.raises(ValueError, match="only when state == 'completed'"):
        Q.transition(rid, QpuState.RUNNING, root=root, result={"x": 1})


def test_load_never_changes_state_via_non_transition_event(tmp_path):
    root = str(tmp_path)
    rid = Q.prepare(_params(), root=root)
    Q.transition(rid, QpuState.AWAITING_CONFIRMATION, root=root)
    # non-transition events must not move the state, whatever they contain
    Q.record_job_id(rid, "j-1", root=root)
    Q.record_result(rid, {"objective": 1.0}, root=root)
    Q.record_provider_observation(rid, {"queue": 5}, root=root)
    run = Q.load(rid, root=root)
    assert run["state"] == "awaiting_confirmation"      # unchanged by non-transition events
    assert run["result"] == {"objective": 1.0}          # folded only from the RESULT event
    assert run["ibm_job_ids"] == ["j-1"]

    # the documented atomic completed+result folds the result from the transition
    _sub = rid  # continue the same run to a completed terminal
    Q.transition(_sub, QpuState.SUBMITTING, root=root)
    Q.transition(_sub, QpuState.SUBMITTED, root=root)
    Q.transition(_sub, QpuState.COMPLETED, root=root, result={"final": True})
    done = Q.load(_sub, root=root)
    assert done["state"] == "completed" and done["result"] == {"final": True}


# --- corrupted: refused by default load, split reconciler views (items 5,9) -
def test_corrupted_run_default_load_refused_and_reconciler_split(tmp_path):
    root = str(tmp_path)
    healthy = Q.prepare(_params(), root=root)
    Q.transition(healthy, QpuState.AWAITING_CONFIRMATION, root=root)
    Q.transition(healthy, QpuState.SUBMITTING, root=root)  # a reconcilable state
    bad = Q.prepare(_params(instance="CP_5"), root=root)
    Q.transition(bad, QpuState.AWAITING_CONFIRMATION, root=root)
    Q.transition(bad, QpuState.SUBMITTING, root=root)

    ev = tmp_path / "qpu" / bad / f"{1:0{Q._EVENT_WIDTH}d}.event.json"
    doc = json.loads(ev.read_text())
    doc["state"] = "running"  # tamper
    ev.write_text(json.dumps(doc, indent=2, sort_keys=True))

    # load verifies by DEFAULT -> corrupt run refused
    with pytest.raises(CorruptRunError):
        Q.load(bad, root=root)

    cand = {r["local_run_id"] for r in Q.reconciliation_candidates(root=root)}
    assert healthy in cand and bad not in cand  # reconciler never sees the corrupt run
    attn = {r["local_run_id"]: r for r in Q.attention_runs(root=root)}
    assert bad in attn and attn[bad]["state"] == "corrupted"


def test_incomplete_manifest_invisible_but_flagged_for_attention(tmp_path):
    root = str(tmp_path)
    d = tmp_path / "qpu" / "20260101T000000000000Z-deadbeef-cafe1234"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(json.dumps({"schema": "acrpq-qpu-run/1"}))
    assert Q.list_runs(root=root) == []               # invisible
    assert d.name not in {r["local_run_id"] for r in Q.reconciliation_candidates(root=root)}
    attn = {r["local_run_id"]: r for r in Q.attention_runs(root=root)}
    assert attn[d.name]["state"] == "incomplete"      # recoverable / flagged


# --- >9999 events + explicit cap (item 9 carried over) ---------------------
def test_high_sequence_ordering_and_event_cap(tmp_path, monkeypatch):
    d = tmp_path / "qpu" / "run"
    d.mkdir(parents=True)
    for seq in (9998, 9999, 10000, 10001):
        (d / f"{seq:0{Q._EVENT_WIDTH}d}.event.json").write_text("{}")
    assert [int(p.name.split(".", 1)[0]) for p in Q._event_files(d)] == [9998, 9999, 10000, 10001]

    root = str(tmp_path / "capped")
    monkeypatch.setattr(Q, "MAX_EVENTS", 3)
    rid = Q.prepare(_params(mode="full_qaoa_hardware"), root=root)  # event 0
    Q.record_job_id(rid, "j1", root=root)
    Q.record_job_id(rid, "j2", root=root)
    with pytest.raises(RuntimeError, match="event cap"):
        Q.record_job_id(rid, "j3", root=root)


# --- concurrent incompatible transitions: exactly one winner ---------------
def _mp_transition_worker(args):
    root, rid, target = args
    from acrpq.dashboard import qpu_runs as _Q

    try:
        _Q.transition(rid, _Q.QpuState(target), root=root)
        return "ok"
    except _Q.IllegalTransitionError:
        return "rejected"


def test_concurrent_incompatible_transitions_have_one_winner(tmp_path):
    import concurrent.futures as cf
    import multiprocessing as mp

    root = str(tmp_path)
    rid = Q.prepare(_params(), root=root)
    _submit(rid, root)  # -> submitted; completed and failed are both legal from here
    ctx = mp.get_context("spawn")
    args = [(root, rid, "completed"), (root, rid, "failed")]
    with cf.ProcessPoolExecutor(max_workers=2, mp_context=ctx) as ex:
        results = sorted(ex.map(_mp_transition_worker, args))

    assert results == ["ok", "rejected"]
    run = Q.load(rid, root=root, verify_integrity=True)
    assert run["terminal"] is True and run["state"] in {"completed", "failed"}
    assert Q.verify(rid, root=root)["ok"] is True
