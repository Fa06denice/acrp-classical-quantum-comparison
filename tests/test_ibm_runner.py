"""IBM submission/reconciliation logic — fully faked, never contacts IBM."""

from __future__ import annotations

import json

import pytest

from acrpq.dashboard import ibm_runner as R
from acrpq.dashboard import qpu_runs as Q

_VALID_SHA = "sha256:" + "a" * 64
_NOSLEEP = {"sleeper": (lambda d: None), "jitter": (lambda: 0.0)}


def _hw_params(**over):
    p = {
        "mode": "hardware_validation", "instance": "CP_4", "n_theta": 3, "shots": 4096,
        "qubo_sha256": _VALID_SHA, "backend_requested": "fake_test_backend",
        "transpiler_seed": 7, "max_quantum_seconds": 10.0, "max_jobs": 1,
    }
    p.update(over)
    return p


def _awaiting(root, **over):
    rid = Q.prepare(_hw_params(**over), root=root)
    Q.transition(rid, Q.QpuState.AWAITING_CONFIRMATION, root=root)
    return rid


def _chash(rid, root):
    return Q.load(rid, root=root)["config_hash"]


def _submit(rid, gw, root, **kw):
    kw.setdefault("expected_config_hash", _chash(rid, root))
    return R.submit_run(rid, gw, isa_circuit=object(), root=root, **{**_NOSLEEP, **kw})


class FakeJob:
    def __init__(self, jid, status="QUEUED"):
        self._id, self._status = jid, status

    def job_id(self):
        return self._id

    def status(self):
        return self._status

    def cancel(self):
        self._status = "CANCELLED"


class FakeGateway:
    def __init__(self, *, submit_result=None, submit_exc=None, existing=None,
                 find_sequence=None, get_status="DONE", cancel_exc=None):
        self.submit_result = submit_result
        self.submit_exc = submit_exc
        self.existing = list(existing or [])
        # find_sequence: list of per-call results; an item may be an Exception to raise
        self.find_sequence = find_sequence
        self._find_i = 0
        self.get_status = get_status
        self.cancel_exc = cancel_exc
        self.calls: list[tuple] = []

    def run_sampler(self, *, isa_circuit, shots, tags, max_execution_time):
        self.calls.append(("run", tags, shots))
        if self.submit_exc is not None:
            raise self.submit_exc
        return self.submit_result

    def find_jobs(self, *, tags, created_after_iso):
        self.calls.append(("find", tags, created_after_iso))
        if self.find_sequence is not None:
            item = self.find_sequence[min(self._find_i, len(self.find_sequence) - 1)]
            self._find_i += 1
            if isinstance(item, BaseException):
                raise item
            return list(item)
        return list(self.existing)

    def get_job(self, job_id):
        job = FakeJob(job_id, status=self.get_status)
        if self.cancel_exc is not None:
            def _boom():
                raise self.cancel_exc
            job.cancel = _boom  # type: ignore[method-assign]
        return job


def _run_calls(gw):
    return [c for c in gw.calls if c[0] == "run"]


# ===========================================================================
# Area 3 — mandatory confirmation (must run BEFORE any gateway call)
# ===========================================================================
@pytest.mark.parametrize("bad", [None, "", "not-a-hash", "sha256:" + "b" * 64])
def test_submit_refused_without_valid_confirmation_and_zero_gateway_calls(tmp_path, bad):
    root = str(tmp_path)
    rid = _awaiting(root)
    gw = FakeGateway(submit_result=FakeJob("x"))
    with pytest.raises(R.SubmissionRefused):
        R.submit_run(rid, gw, isa_circuit=object(), root=root, expected_config_hash=bad, **_NOSLEEP)
    assert gw.calls == []                       # neither find_jobs nor run_sampler called
    assert Q.load(rid, root=root)["state"] == "awaiting_confirmation"


def test_submit_succeeds_with_correct_confirmation(tmp_path):
    root = str(tmp_path)
    rid = _awaiting(root)
    v = _submit(rid, FakeGateway(submit_result=FakeJob("cjob-1")), root)
    assert v["verdict"] == "submitted" and v["job_id"] == "cjob-1"
    assert Q.load(rid, root=root)["ibm_job_ids"] == ["cjob-1"]


# ===========================================================================
# Area 1 — error sanitisation (no secret ever persisted)
# ===========================================================================
def test_redact_secrets_covers_all_forms():
    assert "SECRET" not in R.redact_secrets("Authorization: Bearer abc-SECRET-123")
    assert "SECRET" not in R.redact_secrets("Authorization: abc-SECRET")
    assert "SECRET" not in R.redact_secrets("https://user:paSECRETss@host/x")
    assert "SECRET" not in R.redact_secrets("https://h/x?token=SECRET&a=1")
    assert "SECRET" not in R.redact_secrets('{"outer": {"api_key": "SECRET-999"}}')
    kept = R.redact_secrets("backend temporarily unavailable")
    assert kept == "backend temporarily unavailable"    # useful, non-secret text preserved


def test_secret_in_exception_is_never_persisted(tmp_path):
    root = str(tmp_path)
    rid = _awaiting(root)
    gw = FakeGateway(submit_exc=ConnectionError(
        "POST /jobs failed; headers={'Authorization': 'Bearer abc-SECRET-123'}"))
    v = _submit(rid, gw, root)
    assert v["verdict"] == "submission_unknown"
    run = Q.load(rid, root=root)
    assert "SECRET-123" not in json.dumps(run)                  # not in folded state
    blob = "".join(p.read_text() for p in (tmp_path / "qpu" / rid).glob("*.json"))
    assert "SECRET-123" not in blob                              # not in any event file
    assert run["error"]["message"]                              # a (redacted) message is kept


# ===========================================================================
# Area 2 — unknown IBM status is non-terminal
# ===========================================================================
def _submission_unknown(root):
    rid = _awaiting(root)
    _submit(rid, FakeGateway(submit_exc=ConnectionError("network reset")), root)
    return rid


def test_unknown_status_stays_reconciliation_required(tmp_path):
    root = str(tmp_path)
    rid = _submission_unknown(root)
    gw = FakeGateway(find_sequence=[[FakeJob("j", status="VALIDATING")]])
    v = R.reconcile_run(rid, gw, root=root, **_NOSLEEP)
    assert v["verdict"] == "unknown_provider_status"
    assert Q.load(rid, root=root)["state"] == "reconciliation_required"  # never failed


def test_subsequent_reconcile_after_unknown_can_advance(tmp_path):
    root = str(tmp_path)
    rid = _submission_unknown(root)
    R.reconcile_run(rid, FakeGateway(find_sequence=[[FakeJob("j", "VALIDATING")]]), root=root, **_NOSLEEP)
    # the same job now RUNNING -> advances (known job id, polled)
    v = R.reconcile_run(rid, FakeGateway(get_status="RUNNING"), root=root, **_NOSLEEP)
    assert v["verdict"] == "reconciled" and Q.load(rid, root=root)["state"] == "running"


def test_explicit_error_status_fails(tmp_path):
    root = str(tmp_path)
    rid = _submission_unknown(root)
    v = R.reconcile_run(rid, FakeGateway(find_sequence=[[FakeJob("j", "ERROR")]]), root=root, **_NOSLEEP)
    assert v["verdict"] == "reconciled" and Q.load(rid, root=root)["state"] == "failed"


def test_normalise_status_aliases_and_unknown():
    assert R.normalise_status("done") == "DONE"
    assert R.normalise_status("Completed") == "DONE"
    assert R.normalise_status("CANCELED") == "CANCELLED"
    assert R.normalise_status("VALIDATING") == "UNKNOWN"
    assert R.normalise_status("weird") == "UNKNOWN"


# ===========================================================================
# Area 4 — clock skew + spaced pre-submit search
# ===========================================================================
def test_pre_submit_search_uses_skewed_cutoff(tmp_path):
    root = str(tmp_path)
    rid = _awaiting(root)
    created = Q.load(rid, root=root)["created_utc"]
    gw = FakeGateway(submit_result=FakeJob("cjob-1"))
    _submit(rid, gw, root, clock_skew_s=120.0)
    cutoff = next(c[2] for c in gw.calls if c[0] == "find")
    assert cutoff < created                       # cutoff is earlier than created_utc (skew applied)


def test_job_visible_on_later_read_is_adopted_without_submitting(tmp_path):
    root = str(tmp_path)
    rid = _awaiting(root)
    gw = FakeGateway(find_sequence=[[], [], [FakeJob("late-1")]])
    v = _submit(rid, gw, root)
    assert v["verdict"] == "existing_job_found" and v["job_ids"] == ["late-1"]
    assert _run_calls(gw) == []                   # never submitted


def test_multiple_jobs_pre_submit_is_anomaly_without_submitting(tmp_path):
    root = str(tmp_path)
    rid = _awaiting(root)
    gw = FakeGateway(find_sequence=[[FakeJob("d1"), FakeJob("d2")]])
    v = _submit(rid, gw, root)
    assert v["verdict"] == "duplicate_remote_jobs" and set(v["job_ids"]) == {"d1", "d2"}
    assert _run_calls(gw) == []


@pytest.mark.parametrize("kw", [{"reads": 0}, {"clock_skew_s": -1.0}, {"backoff_base_s": -1.0}])
def test_invalid_search_params_refused_before_remote(tmp_path, kw):
    root = str(tmp_path)
    rid = _awaiting(root)
    gw = FakeGateway(submit_result=FakeJob("x"))
    with pytest.raises(ValueError):
        _submit(rid, gw, root, **kw)
    assert gw.calls == []


# ===========================================================================
# Area 5 — bounded read retries (network/timeout only)
# ===========================================================================
def test_reconcile_retries_network_then_succeeds(tmp_path):
    root = str(tmp_path)
    rid = _submission_unknown(root)
    sleeps: list = []
    gw = FakeGateway(find_sequence=[ConnectionError("network reset"), [FakeJob("ok-1", "DONE")]])
    v = R.reconcile_run(rid, gw, root=root, sleeper=lambda d: sleeps.append(d), jitter=lambda: 0.0)
    assert v["verdict"] == "reconciled" and Q.load(rid, root=root)["state"] == "completed"
    assert sleeps and all(s >= 0 for s in sleeps)   # bounded, non-negative backoff


def test_auth_read_error_is_not_retried(tmp_path):
    root = str(tmp_path)
    rid = _submission_unknown(root)
    gw = FakeGateway(find_sequence=[PermissionError("Unauthorized; Authorization: Bearer abc-SECRET")])
    v = R.reconcile_run(rid, gw, root=root, **_NOSLEEP)
    assert v["verdict"] == "read_retry_exhausted" and v["error_category"] == "auth"
    finds = [c for c in gw.calls if c[0] == "find"]
    assert len(finds) == 1                          # auth error not retried
    assert "SECRET" not in json.dumps(Q.load(rid, root=root))  # sanitised


def test_timeout_read_exhausts_bounded(tmp_path):
    root = str(tmp_path)
    rid = _submission_unknown(root)
    gw = FakeGateway(find_sequence=[TimeoutError("timed out")])
    v = R.reconcile_run(rid, gw, root=root, read_attempts=3, reads=1, **_NOSLEEP)
    assert v["verdict"] == "read_retry_exhausted" and v["error_category"] == "timeout"
    assert Q.load(rid, root=root)["state"] == "reconciliation_required"  # honest non-terminal


def test_submit_calls_run_sampler_at_most_once(tmp_path):
    root = str(tmp_path)
    rid = _awaiting(root)
    gw = FakeGateway(submit_exc=ConnectionError("network reset"))
    _submit(rid, gw, root)
    assert len(_run_calls(gw)) == 1                 # never retried around run_sampler


# ===========================================================================
# Area 6 — safe cancellation with multiple job ids
# ===========================================================================
def _submitted(root, jobs=("cjob-1",)):
    rid = _awaiting(root)
    _submit(rid, FakeGateway(submit_result=FakeJob(jobs[0])), root)
    for extra in jobs[1:]:
        Q.record_job_id(rid, extra, root=root)  # simulate a second discovered job
    return rid


def test_cancel_with_multiple_jobs_blocks_and_calls_no_cancel(tmp_path):
    root = str(tmp_path)
    rid = _submitted(root, jobs=("cjob-1", "cjob-2"))
    gw = FakeGateway(get_status="RUNNING")
    v = R.request_cancel(rid, gw, root=root, **_NOSLEEP)
    assert v["verdict"] == "duplicate_remote_jobs_requires_resolution"
    assert set(v["job_ids"]) == {"cjob-1", "cjob-2"}
    assert Q.load(rid, root=root)["state"] == "submitted"   # unchanged; no cancel executed


def test_cancel_single_confirmed(tmp_path):
    root = str(tmp_path)
    rid = _submitted(root)
    v = R.request_cancel(rid, FakeGateway(get_status="CANCELLED"), root=root, **_NOSLEEP)
    assert v["verdict"] == "cancelled" and Q.load(rid, root=root)["state"] == "cancelled"


def test_cancel_of_done_completes(tmp_path):
    root = str(tmp_path)
    rid = _submitted(root)
    v = R.request_cancel(rid, FakeGateway(get_status="DONE"), root=root, **_NOSLEEP)
    assert v["verdict"] == "completed" and Q.load(rid, root=root)["state"] == "completed"


def test_cancel_status_network_error_stays_cancel_requested(tmp_path):
    root = str(tmp_path)
    rid = _submitted(root)
    gw = FakeGateway(cancel_exc=ConnectionError("network reset"), get_status="RUNNING")
    v = R.request_cancel(rid, gw, root=root, **_NOSLEEP)
    assert v["verdict"] == "cancel_requested"
    assert Q.load(rid, root=root)["state"] == "cancel_requested"  # never a false 'cancelled'


# ===========================================================================
# Area 7 — do not capture BaseException
# ===========================================================================
@pytest.mark.parametrize("exc", [KeyboardInterrupt(), SystemExit()])
def test_base_exceptions_propagate_and_leave_run_reconcilable(tmp_path, exc):
    root = str(tmp_path)
    rid = _awaiting(root)
    gw = FakeGateway(submit_exc=exc)
    with pytest.raises(type(exc)):
        _submit(rid, gw, root)
    # left in submitting -> reconcilable (not silently swallowed to submission_unknown)
    assert Q.load(rid, root=root)["state"] == "submitting"
    assert rid in {r["local_run_id"] for r in Q.reconciliation_candidates(root=root)}


# ===========================================================================
# Area 8 — real max_jobs budget
# ===========================================================================
def test_budget_consumed_locally_blocks_submission(tmp_path):
    root = str(tmp_path)
    rid = _awaiting(root)
    Q.record_job_id(rid, "prior-local", root=root)      # a job already known locally
    gw = FakeGateway(submit_result=FakeJob("new"))
    v = _submit(rid, gw, root)
    assert v["verdict"] == "existing_job_found" and _run_calls(gw) == []


def test_budget_consumed_remotely_blocks_submission(tmp_path):
    root = str(tmp_path)
    rid = _awaiting(root)
    gw = FakeGateway(existing=[FakeJob("remote-prior")])
    v = _submit(rid, gw, root)
    assert v["verdict"] == "existing_job_found" and _run_calls(gw) == []


# ===========================================================================
# Area 9 — reconciliation candidate states
# ===========================================================================
def test_reconciliation_candidates_exclude_prepared_awaiting_and_corrupted(tmp_path):
    root = str(tmp_path)
    prepared = Q.prepare(_hw_params(), root=root)
    awaiting = _awaiting(root)
    active = _submitted(root)
    states = {r["local_run_id"]: r["state"] for r in Q.reconciliation_candidates(root=root)}
    assert prepared not in states and awaiting not in states
    assert active in states and states[active] == "submitted"


def test_reconcile_prepared_without_job_is_not_submitted(tmp_path):
    root = str(tmp_path)
    rid = _awaiting(root)  # awaiting_confirmation, no job
    v = R.reconcile_run(rid, FakeGateway(get_status="DONE"), root=root, **_NOSLEEP)
    assert v["verdict"] == "not_submitted"
    assert Q.load(rid, root=root)["state"] == "awaiting_confirmation"


@pytest.mark.parametrize("state", ["submitting", "submitted", "queued", "running",
                                   "submission_unknown", "reconciliation_required"])
def test_reconcile_from_each_candidate_state_advances_legally(tmp_path, state):
    root = str(tmp_path)
    rid = _awaiting(root)
    # build each candidate state via a VALID path, then reconcile
    if state == "submitting":                       # no job yet -> search path
        Q.transition(rid, Q.QpuState.SUBMITTING, root=root)
        gw = FakeGateway(find_sequence=[[FakeJob("cjob-1", "DONE")]])
    elif state == "submission_unknown":             # ambiguous failure -> no job -> search
        _submit(rid, FakeGateway(submit_exc=ConnectionError("network reset")), root)
        gw = FakeGateway(find_sequence=[[FakeJob("cjob-1", "DONE")]])
    else:                                           # has a job -> poll path
        _submit(rid, FakeGateway(submit_result=FakeJob("cjob-1")), root)  # -> submitted
        if state == "queued":
            Q.transition(rid, Q.QpuState.QUEUED, root=root)
        elif state == "running":
            Q.transition(rid, Q.QpuState.QUEUED, root=root)
            Q.transition(rid, Q.QpuState.RUNNING, root=root)
        elif state == "reconciliation_required":
            Q.transition(rid, Q.QpuState.RECONCILIATION_REQUIRED, root=root)
        gw = FakeGateway(get_status="DONE")
    assert Q.load(rid, root=root)["state"] == state
    v = R.reconcile_run(rid, gw, root=root, **_NOSLEEP)
    assert v["verdict"] == "reconciled"
    assert Q.load(rid, root=root)["state"] == "completed"  # DONE -> completed, no illegal transition


@pytest.mark.parametrize("state,status", [("queued", "QUEUED"), ("running", "RUNNING")])
def test_reconcile_unchanged_provider_state_writes_no_events(tmp_path, state, status):
    root = str(tmp_path)
    rid = _submitted(root)
    if state == "queued":
        Q.transition(rid, Q.QpuState.QUEUED, root=root)
    else:
        Q.transition(rid, Q.QpuState.QUEUED, root=root)
        Q.transition(rid, Q.QpuState.RUNNING, root=root)
    before = Q.load(rid, root=root)["n_events"]
    verdict = R.reconcile_run(rid, FakeGateway(get_status=status), root=root, **_NOSLEEP)
    after = Q.load(rid, root=root)
    assert verdict["verdict"] == "state_unchanged"
    assert after["state"] == state and after["n_events"] == before


# ===========================================================================
# Area 10 — robustness
# ===========================================================================
def test_exec_time_ceils_positive_subsecond_to_one():
    assert R._exec_time_seconds(None) is None
    assert R._exec_time_seconds(0.4) == 1        # never truncates a positive budget to 0
    assert R._exec_time_seconds(9.2) == 10


def test_gateway_empty_job_id_becomes_submission_unknown(tmp_path):
    root = str(tmp_path)
    rid = _awaiting(root)
    v = _submit(rid, FakeGateway(submit_result=FakeJob("   ")), root)  # blank id
    assert v["verdict"] == "submission_unknown"
    assert Q.load(rid, root=root)["ibm_job_ids"] == []


def test_error_taxonomy_classification():
    cases = {
        "auth": PermissionError("Unauthorized: invalid token"),
        "timeout": TimeoutError("request timed out"),
        "network": ConnectionError("connection reset"),
        "quota": RuntimeError("monthly quota exceeded limit"),
        "validation": ValueError("circuit not supported by backend"),
        "unknown": RuntimeError("something odd"),
    }
    for expected, exc in cases.items():
        assert R.classify_remote_error(exc) == expected


# ===========================================================================
# Phase 3.2-A — crash window after job adoption
# ===========================================================================
def _crashed_awaiting_with_jobs(root, jobs=("cjob-crash",)):
    """The exact post-crash state: awaiting_confirmation with adopted job id(s)."""
    rid = _awaiting(root)
    for jid in jobs:
        Q.record_job_id(rid, jid, root=root)  # persisted just before the crash
    assert Q.load(root=root, local_run_id=rid)["state"] == "awaiting_confirmation"
    return rid


def test_crash_adopted_job_becomes_reconcilable(tmp_path):
    root = str(tmp_path)
    rid = _crashed_awaiting_with_jobs(root)
    assert rid in {r["local_run_id"] for r in Q.reconciliation_candidates(root=root)}


@pytest.mark.parametrize("status,expected", [
    ("RUNNING", "running"), ("DONE", "completed"), ("ERROR", "failed"),
    ("VALIDATING", "reconciliation_required"),
])
def test_crash_adopted_job_reconciles_without_illegal_transition(tmp_path, status, expected):
    root = str(tmp_path)
    rid = _crashed_awaiting_with_jobs(root)
    R.reconcile_run(rid, FakeGateway(get_status=status), root=root, **_NOSLEEP)
    run = Q.load(rid, root=root)
    assert run["state"] == expected
    seq = [t["state"] for t in run["transitions"]]
    # routed via submission_unknown -> reconciliation_required, never awaiting->IBM
    assert "submission_unknown" in seq and "reconciliation_required" in seq
    i = seq.index("awaiting_confirmation")
    assert seq[i + 1] == "submission_unknown"  # the very next state is the ambiguity hub


def test_crash_adopted_multiple_jobs_is_anomaly(tmp_path):
    root = str(tmp_path)
    rid = _crashed_awaiting_with_jobs(root, jobs=("dup-1", "dup-2"))
    v = R.reconcile_run(rid, FakeGateway(get_status="DONE"), root=root, **_NOSLEEP)
    assert v["verdict"] == "duplicate_remote_jobs" and set(v["job_ids"]) == {"dup-1", "dup-2"}
    assert Q.load(rid, root=root)["state"] == "reconciliation_required"  # no silent winner


# ===========================================================================
# Phase 3.2-B — honest raw provider status
# ===========================================================================
class _EnumStatus:
    def __init__(self, name):
        self.name = name


def test_unknown_status_preserves_raw_value(tmp_path):
    root = str(tmp_path)
    rid = _submission_unknown(root)
    v = R.reconcile_run(rid, FakeGateway(find_sequence=[[FakeJob("j", "VALIDATING")]]),
                        root=root, **_NOSLEEP)
    assert v["verdict"] == "unknown_provider_status" and v["raw_status"] == "VALIDATING"
    obs = Q.load(rid, root=root)["provider_observations"][-1]["payload"]
    assert obs["canonical"] == "UNKNOWN" and obs["raw_status"] == "VALIDATING"


def test_unknown_enum_status_preserved(tmp_path):
    root = str(tmp_path)
    rid = _submission_unknown(root)
    v = R.reconcile_run(rid, FakeGateway(find_sequence=[[FakeJob("j", _EnumStatus("VALIDATING"))]]),
                        root=root, **_NOSLEEP)
    assert v["raw_status"] == "VALIDATING"


def test_unknown_status_with_secret_is_sanitised(tmp_path):
    root = str(tmp_path)
    rid = _submission_unknown(root)
    R.reconcile_run(rid, FakeGateway(find_sequence=[[FakeJob("j", "VALIDATING token=abc-SECRET")]]),
                    root=root, **_NOSLEEP)
    blob = "".join(p.read_text() for p in (tmp_path / "qpu" / rid).glob("*.json"))
    assert "abc-SECRET" not in blob and "SECRET" not in json.dumps(Q.load(rid, root=root))


def test_validating_then_running_then_done_sequence(tmp_path):
    root = str(tmp_path)
    rid = _submission_unknown(root)
    R.reconcile_run(rid, FakeGateway(find_sequence=[[FakeJob("j", "VALIDATING")]]), root=root, **_NOSLEEP)
    R.reconcile_run(rid, FakeGateway(get_status="RUNNING"), root=root, **_NOSLEEP)
    assert Q.load(rid, root=root)["state"] == "running"
    R.reconcile_run(rid, FakeGateway(get_status="DONE"), root=root, **_NOSLEEP)
    assert Q.load(rid, root=root)["state"] == "completed"


# ===========================================================================
# Phase 3.2-C — retry parameter validation (deterministic, zero gateway/sleep)
# ===========================================================================
_BAD_PARAMS = [
    {"reads": 0}, {"reads": -1}, {"reads": True}, {"reads": 1.5},
    {"read_attempts": 0}, {"read_attempts": True},
    {"clock_skew_s": -1.0}, {"clock_skew_s": float("nan")}, {"clock_skew_s": float("inf")},
    {"backoff_base_s": -1.0}, {"backoff_base_s": float("inf")},
]


@pytest.mark.parametrize("bad", _BAD_PARAMS)
def test_submit_rejects_invalid_retry_params_before_gateway(tmp_path, bad):
    root = str(tmp_path)
    rid = _awaiting(root)
    gw = FakeGateway(submit_result=FakeJob("x"))
    sleeps: list = []
    kw = {"sleeper": lambda d: sleeps.append(d), "jitter": lambda: 0.0,
          "expected_config_hash": _chash(rid, root), **bad}
    with pytest.raises(ValueError):
        R.submit_run(rid, gw, isa_circuit=object(), root=root, **kw)
    assert gw.calls == [] and sleeps == []
    assert Q.load(rid, root=root)["state"] == "awaiting_confirmation"


@pytest.mark.parametrize("bad", _BAD_PARAMS)
def test_reconcile_rejects_invalid_retry_params_before_gateway(tmp_path, bad):
    root = str(tmp_path)
    rid = _submission_unknown(root)
    gw = FakeGateway(find_sequence=[[FakeJob("j", "DONE")]])
    sleeps: list = []
    before = Q.load(rid, root=root)["n_events"]
    with pytest.raises(ValueError):
        R.reconcile_run(rid, gw, root=root, sleeper=lambda d: sleeps.append(d),
                        jitter=lambda: 0.0, **bad)
    assert gw.calls == [] and sleeps == []
    assert Q.load(rid, root=root)["n_events"] == before  # no event written


def test_bad_jitter_value_rejected(tmp_path):
    root = str(tmp_path)
    rid = _awaiting(root)
    gw = FakeGateway(submit_result=FakeJob("x"))
    with pytest.raises(ValueError, match="jitter"):
        R.submit_run(rid, gw, isa_circuit=object(), root=root,
                     expected_config_hash=_chash(rid, root),
                     sleeper=lambda d: None, jitter=lambda: -1.0)
    assert gw.calls == []


# ===========================================================================
# Phase 3.2-D — cancel retry discipline (cancel() never retried)
# ===========================================================================
class _CountingCancelGateway(FakeGateway):
    def __init__(self, *, get_status="RUNNING", cancel_raises=None, get_job_raises_times=0):
        super().__init__(get_status=get_status)
        self.cancel_raises = cancel_raises
        self.get_job_raises_times = get_job_raises_times
        self.cancel_count = 0
        self.get_job_count = 0

    def get_job(self, job_id):
        self.get_job_count += 1
        if self.get_job_count <= self.get_job_raises_times:
            raise ConnectionError("network reset")
        parent = super().get_job(job_id)
        outer = self

        class _J:
            def job_id(self_inner):
                return job_id
            def status(self_inner):
                return parent.status()
            def cancel(self_inner):
                outer.cancel_count += 1
                if outer.cancel_raises is not None:
                    raise outer.cancel_raises
        return _J()


def test_cancel_get_job_read_is_retried(tmp_path):
    root = str(tmp_path)
    rid = _submitted(root)
    gw = _CountingCancelGateway(get_status="CANCELLED", get_job_raises_times=1)
    v = R.request_cancel(rid, gw, root=root, sleeper=lambda d: None, jitter=lambda: 0.0)
    assert v["verdict"] == "cancelled"       # recovered after one network retry on get_job
    assert gw.cancel_count == 1              # cancel executed once


def test_cancel_mutation_is_never_retried_on_network_error(tmp_path):
    root = str(tmp_path)
    rid = _submitted(root)
    gw = _CountingCancelGateway(get_status="RUNNING", cancel_raises=ConnectionError("network"))
    v = R.request_cancel(rid, gw, root=root, sleeper=lambda d: None, jitter=lambda: 0.0)
    assert gw.cancel_count == 1                                  # exactly one cancel attempt
    assert v["verdict"] == "cancel_requested"                   # honest, not a false cancelled
    assert Q.load(rid, root=root)["state"] == "cancel_requested"


# ===========================================================================
# Phase 3.2-E — pre-submit search exhaustion (no paid call, stays awaiting)
# ===========================================================================
@pytest.mark.parametrize("exc,cat", [
    (ConnectionError("network reset"), "network"),
    (TimeoutError("timed out"), "timeout"),
    (PermissionError("Unauthorized; Authorization: Bearer abc-SECRET"), "auth"),
])
def test_presubmit_search_failure_stays_awaiting_no_paid_call(tmp_path, exc, cat):
    root = str(tmp_path)
    rid = _awaiting(root)
    gw = FakeGateway(find_sequence=[exc])
    v = _submit(rid, gw, root)
    assert v["verdict"] == "pre_submit_search_failed" and v["error_category"] == cat
    assert _run_calls(gw) == []                                  # zero run_sampler
    assert Q.load(rid, root=root)["state"] == "awaiting_confirmation"  # unchanged
    assert "SECRET" not in "".join(p.read_text() for p in (tmp_path / "qpu" / rid).glob("*.json"))


# --- real gateway, LOCAL execution via the official fake backend (no account) -
def test_runtime_gateway_runs_locally_on_fake_backend():
    pytest.importorskip("qiskit_ibm_runtime")
    from qiskit import QuantumCircuit
    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2

    fake = FakeManilaV2()
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.measure_all()
    isa = generate_preset_pass_manager(backend=fake, optimization_level=1).run(qc)

    gw = R.RuntimeGateway(backend=fake)  # local: no service, no token, no network
    job = gw.run_sampler(isa_circuit=isa, shots=128, tags=["local-seam"], max_execution_time=0.4)
    assert isinstance(job.job_id(), str) and job.job_id()
    assert R.normalise_status(job.status()) in {"DONE", "QUEUED", "RUNNING", "INITIALIZING", "UNKNOWN"}
    assert job.result() is not None
    with pytest.raises(RuntimeError, match="requires a QiskitRuntimeService"):
        gw.find_jobs(tags=["x"], created_after_iso="2026-01-01T00:00:00")
