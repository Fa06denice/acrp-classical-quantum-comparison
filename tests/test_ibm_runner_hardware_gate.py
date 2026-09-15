"""Codex items 1-2 — the real submit path enforces the hardware gate at the point of use.

Uses a RuntimeGateway SUBCLASS with a FAKE run_sampler (so isinstance(...,RuntimeGateway) is
True) — never a real service, token, or network. Proves: a real gateway REQUIRES a
hardware_authorizer; with all invariants satisfied it submits under EXACTLY the fresh
point-of-use dedup tag; and it is refused (run_sampler never called) when the authorizer's
own fresh dedup finds a remote job or any gate invariant is absent.
"""
from __future__ import annotations

import warnings

import pytest

from acrpq.dashboard import ibm_runner, qpu_dedup as D, qpu_runs
from acrpq.dashboard import qpu_budget as B
from acrpq.dashboard import qpu_submit_gate as G

warnings.filterwarnings("ignore")

CFG = "sha256:" + "a" * 64
JOB = "sha256:" + "b" * 64


class _FakeJob:
    def job_id(self):
        return "job-xyz"


class FakeRealGateway(ibm_runner.RuntimeGateway):
    """isinstance(...,RuntimeGateway) is True, but nothing real happens."""

    def __init__(self):
        self.tags = None

    def find_jobs(self, **_k):
        return []                                    # empty pre-submit search

    def run_sampler(self, *, isa_circuit, shots, tags, max_execution_time):
        self.tags = list(tags)
        return _FakeJob()


class ComposedRealGateway:
    """Runtime-like composition wrapper: not isinstance(RuntimeGateway)."""

    def __init__(self):
        self.tags = None

    def find_jobs(self, **_k):
        return []

    def run_sampler(self, *, isa_circuit, shots, tags, max_execution_time):
        self.tags = list(tags)
        return _FakeJob()


def _prepare_run(tmp_path):
    runs = str(tmp_path / "runs")
    params = {"mode": "hardware_validation", "instance": "CP_3", "n_theta": 3, "n_q": 1,
              "shots": 128, "transpiler_seed": 1, "max_jobs": 1, "max_quantum_seconds": 20.0,
              "qubo_sha256": "sha256:" + "c" * 64, "backend_requested": "ibm_x"}
    rid = qpu_runs.prepare(params, root=runs)
    qpu_runs.record_batch_identity(rid, CFG, root=runs)
    qpu_runs.transition(rid, qpu_runs.QpuState.AWAITING_CONFIRMATION, root=runs)
    run_cfg = qpu_runs.load(rid, root=runs)["config_hash"]
    return runs, rid, run_cfg


def _authorizer(tmp_path, runs, rid, *, search_jobs, commit=True, consume=True):
    store = B.ConfirmationStore(str(tmp_path / "conf"))
    conf = B.Confirmation(batch_config_hash=CFG, job_config_hash=JOB, instance="CP_3",
                          objective_id="maneuver_count_v1", backend="ibm_x", shots=128,
                          n_qubits=9, reserved_shots=128, reserved_qpu_seconds=10.0,
                          qubo_sha256="sha256:" + "c" * 64, isa_sha256="sha256:" + "d" * 64,
                          nonce="n1", expires_epoch=1e12)
    store.issue(conf)
    if consume:
        store.consume(conf, now_epoch=0.0)
    pol = B.BudgetPolicy(max_jobs=1, shots_per_job=128, total_shot_budget=1000,
                         qpu_seconds_per_job=10.0, max_total_qpu_seconds=600.0,
                         max_wait_seconds=600.0, max_jobs_per_backend=1, max_jobs_per_instance=1,
                         stop_after_consecutive_errors=2)
    ledger = B.BudgetLedger(str(tmp_path / "led"), pol)
    res = ledger.reserve(reservation_id="r1", batch_config_hash=CFG, job_config_hash=JOB,
                         backend="ibm_x", instance="CP_3")
    if commit:
        ledger.commit(res)
    return G.HardwareAuthorizer(root=runs, local_run_id=rid, batch_config_hash=CFG,
                                job_config_hash=JOB, confirmation=conf, confirmation_store=store,
                                ledger=ledger, reservation=res,
                                search=D.FakeRemoteJobService(jobs=search_jobs))


def _submit(rid, gw, run_cfg, runs, authorizer):
    return ibm_runner.submit_run(rid, gw, isa_circuit=object(), expected_config_hash=run_cfg,
                                 root=runs, sleeper=lambda _s: None, jitter=lambda: 0.0,
                                 hardware_authorizer=authorizer)


def test_real_gateway_submits_under_the_fresh_dedup_tag(tmp_path):
    runs, rid, run_cfg = _prepare_run(tmp_path)
    authz = _authorizer(tmp_path, runs, rid, search_jobs=[])   # zero remote jobs -> submit_allowed
    gw = FakeRealGateway()
    _submit(rid, gw, run_cfg, runs, authz)
    # the fresh dedup tag FIRST, plus the local_run_id so reconcile_run's
    # [local_run_id] tag search can rediscover the job after a crash (MEDIUM-2 fix)
    assert gw.tags == [D.job_tag(CFG, JOB), rid]
    assert qpu_runs.load(rid, root=runs)["ibm_job_ids"] == ["job-xyz"]


def test_real_gateway_without_authorizer_is_refused(tmp_path):
    runs, rid, run_cfg = _prepare_run(tmp_path)
    gw = FakeRealGateway()
    with pytest.raises(ibm_runner.SubmissionRefused, match="exact HardwareAuthorizer"):
        _submit(rid, gw, run_cfg, runs, None)
    assert gw.tags is None                                     # run_sampler NEVER called


def test_composed_real_gateway_without_authorizer_is_refused(tmp_path):
    """A wrapper around a payable gateway cannot evade the gate via composition."""
    runs, rid, run_cfg = _prepare_run(tmp_path)
    gw = ComposedRealGateway()
    with pytest.raises(ibm_runner.SubmissionRefused, match="exact HardwareAuthorizer"):
        _submit(rid, gw, run_cfg, runs, None)
    assert gw.tags is None                                     # run_sampler NEVER called


def test_composed_real_gateway_with_authorizer_uses_fresh_tag(tmp_path):
    runs, rid, run_cfg = _prepare_run(tmp_path)
    authz = _authorizer(tmp_path, runs, rid, search_jobs=[])
    gw = ComposedRealGateway()
    _submit(rid, gw, run_cfg, runs, authz)
    assert gw.tags == [D.job_tag(CFG, JOB), rid]   # dedup tag + run id (MEDIUM-2)


def test_authorizer_refuses_when_a_remote_job_exists(tmp_path):
    runs, rid, run_cfg = _prepare_run(tmp_path)
    tag = D.job_tag(CFG, JOB)
    authz = _authorizer(tmp_path, runs, rid,
                        search_jobs=[D.RemoteJobRef("existing", (tag,), JOB)])  # fresh dedup finds a job
    gw = FakeRealGateway()
    with pytest.raises(G.SubmissionForbidden):
        _submit(rid, gw, run_cfg, runs, authz)
    assert gw.tags is None


@pytest.mark.parametrize("break_", ["no_commit", "no_consume"])
def test_authorizer_refuses_when_a_gate_invariant_absent(tmp_path, break_):
    runs, rid, run_cfg = _prepare_run(tmp_path)
    authz = _authorizer(tmp_path, runs, rid, search_jobs=[],
                        commit=(break_ != "no_commit"), consume=(break_ != "no_consume"))
    gw = FakeRealGateway()
    with pytest.raises(G.SubmissionForbidden):
        _submit(rid, gw, run_cfg, runs, authz)
    assert gw.tags is None


class WrappedRealGateway:
    """MoE Expert E, IMPORTANT-1: a composition wrapper delegating to a REAL
    RuntimeGateway. It is NOT an isinstance of RuntimeGateway, so the old
    `startswith("fake")` exemption let it reach run_sampler ungated."""

    def __init__(self):
        self._inner = FakeRealGateway()          # stands in for a real gateway
        self.tags = None

    def find_jobs(self, **_k):
        return []

    def run_sampler(self, **kw):
        self.tags = list(kw["tags"])
        return self._inner.run_sampler(**kw)


def _prepare_run_with_backend(tmp_path, backend):
    runs = str(tmp_path / "runs")
    params = {"mode": "hardware_validation", "instance": "CP_3", "n_theta": 3, "n_q": 1,
              "shots": 128, "transpiler_seed": 1, "max_jobs": 1, "max_quantum_seconds": 20.0,
              "qubo_sha256": "sha256:" + "c" * 64, "backend_requested": backend}
    rid = qpu_runs.prepare(params, root=runs)
    qpu_runs.record_batch_identity(rid, CFG, root=runs)
    qpu_runs.transition(rid, qpu_runs.QpuState.AWAITING_CONFIRMATION, root=runs)
    return runs, rid, qpu_runs.load(rid, root=runs)["config_hash"]


def test_wrapper_around_real_gateway_cannot_claim_a_fake_destination(tmp_path):
    """A persisted backend_requested='fake_manila' must NOT exempt a gateway that
    wraps a real RuntimeGateway: the authorizer stays mandatory."""
    runs, rid, run_cfg = _prepare_run_with_backend(tmp_path, "fake_manila")
    gw = WrappedRealGateway()
    with pytest.raises(ibm_runner.SubmissionRefused, match="exact HardwareAuthorizer"):
        _submit(rid, gw, run_cfg, runs, None)
    assert gw.tags is None and gw._inner.tags is None     # run_sampler NEVER reached


def test_authorizer_minted_for_another_run_is_refused(tmp_path):
    """MoE Expert E, IMPORTANT-2: the type check alone allowed an authorizer built for
    run B to authorize run A. The authorizer must be bound to THIS local_run_id."""
    runs, rid_a, cfg_a = _prepare_run_with_backend(tmp_path, "ibm_x")
    _runs_b, rid_b, _cfg_b = _prepare_run_with_backend(tmp_path, "ibm_x")
    authz_for_b = _authorizer(tmp_path, runs, rid_b, search_jobs=[])   # bound to run B
    gw = FakeRealGateway()
    with pytest.raises(ibm_runner.SubmissionRefused, match="not bound to this local_run_id"):
        _submit(rid_a, gw, cfg_a, runs, authz_for_b)
    assert gw.tags is None                                 # run_sampler NEVER reached
