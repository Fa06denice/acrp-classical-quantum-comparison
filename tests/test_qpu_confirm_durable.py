"""Phase 6.2 tests: durable, atomic, restart-surviving confirmation.

Threading + multiprocessing(spawn) single-winner, idempotent winner retry,
divergent replay refusal, expiry, restart (a fresh store instance), and tamper.
"""

from __future__ import annotations

import multiprocessing as mp

import pytest

from acrpq.dashboard import qpu_confirm, qpu_runs, qpu_store

CONSENT = qpu_confirm.CONSENT_PHRASE


def _params():
    return {
        "mode": "hardware_validation", "instance": "CP_3", "n_theta": 2, "n_q": 1, "shots": 512,
        "transpiler_seed": 1, "max_jobs": 1, "max_quantum_seconds": 20.0,
        "qubo_sha256": "sha256:" + "a" * 64, "backend_requested": "fake_lagos",
    }


def _prepared(root):
    return qpu_runs.prepare(_params(), root=root)


_CHAIN = dict(config_hash="sha256:" + "c" * 64, artefact_sha256="sha256:" + "b" * 64,
              decision_sha256="sha256:" + "d" * 64, plan_sha256="sha256:" + "p" * 64,
              isa_fingerprint_sha256="sha256:" + "f" * 64, backend="fake_lagos",
              total_shots=512, max_jobs=1, max_quantum_seconds=20.0)


def _issue(store, run_id):
    return store.issue(run_id=run_id, **_CHAIN)


def _verify_kwargs(run_id):
    return dict(run_id=run_id, expected_config_hash=_CHAIN["config_hash"],
                expected_decision_sha256=_CHAIN["decision_sha256"],
                expected_artefact_sha256=_CHAIN["artefact_sha256"],
                expected_plan_sha256=_CHAIN["plan_sha256"],
                expected_isa_fingerprint=_CHAIN["isa_fingerprint_sha256"],
                expected_backend=_CHAIN["backend"], expected_shots=_CHAIN["total_shots"],
                expected_max_jobs=_CHAIN["max_jobs"],
                expected_max_quantum_seconds=_CHAIN["max_quantum_seconds"])


def _consume_kwargs(nonce):
    return dict(run_id=None, nonce=nonce, config_hash="sha256:" + "c" * 64,
                artefact_sha256="sha256:" + "b" * 64, decision_sha256="sha256:" + "d" * 64,
                backend="fake_lagos", total_shots=512, max_jobs=1, max_quantum_seconds=20.0,
                consent=CONSENT)


def test_durable_consume_and_idempotent_retry(tmp_path):
    root = str(tmp_path / "runs")
    rid = _prepared(root)
    store = qpu_confirm.DurableConfirmationStore(root=root)
    nonce, _pub = _issue(store, rid)
    kw = {**_consume_kwargs(nonce), "run_id": rid}
    v1 = store.consume(**kw)
    assert v1["confirmed"] is True and v1["idempotent"] is False
    v2 = store.consume(**kw)                       # winner retry
    assert v2["confirmed"] is True and v2["idempotent"] is True


def test_survives_restart(tmp_path):
    root = str(tmp_path / "runs")
    rid = _prepared(root)
    nonce, _pub = _issue(qpu_confirm.DurableConfirmationStore(root=root), rid)
    # a brand-new store instance (== a process restart) still knows the challenge
    fresh = qpu_confirm.DurableConfirmationStore(root=root)
    v = fresh.consume(**{**_consume_kwargs(nonce), "run_id": rid})
    assert v["confirmed"] is True


def test_divergent_replay_refused(tmp_path):
    root = str(tmp_path / "runs")
    rid = _prepared(root)
    store = qpu_confirm.DurableConfirmationStore(root=root)
    nonce, _pub = _issue(store, rid)
    store.consume(**{**_consume_kwargs(nonce), "run_id": rid})
    with pytest.raises(qpu_confirm.ConfirmationError):
        store.consume(**{**_consume_kwargs(nonce), "run_id": rid, "total_shots": 999})


def test_expiry(tmp_path):
    root = str(tmp_path / "runs")
    rid = _prepared(root)
    clock = [1000.0]
    store = qpu_confirm.DurableConfirmationStore(root=root, now=lambda: clock[0], ttl_s=10.0)
    nonce, _pub = _issue(store, rid)
    clock[0] = 2000.0
    with pytest.raises(qpu_confirm.ConfirmationError):
        store.consume(**{**_consume_kwargs(nonce), "run_id": rid})


def test_config_drift_refused(tmp_path):
    root = str(tmp_path / "runs")
    rid = _prepared(root)
    store = qpu_confirm.DurableConfirmationStore(root=root)
    nonce, _pub = _issue(store, rid)
    with pytest.raises(qpu_confirm.ConfirmationError):
        store.consume(**{**_consume_kwargs(nonce), "run_id": rid},
                      current_config_hash="sha256:" + "9" * 64)


def test_tampered_challenge_refused(tmp_path):
    root = str(tmp_path / "runs")
    rid = _prepared(root)
    store = qpu_confirm.DurableConfirmationStore(root=root)
    nonce, _pub = _issue(store, rid)
    # tamper the persisted challenge -> store hash no longer verifies -> refused
    import json
    p = qpu_store._doc_path(root, rid, "confirmation")
    doc = json.loads(p.read_text())
    doc["payload"]["total_shots"] = 999999
    p.write_text(json.dumps(doc))
    with pytest.raises(qpu_confirm.ConfirmationError):
        store.consume(**{**_consume_kwargs(nonce), "run_id": rid})


def test_threaded_single_winner(tmp_path):
    import threading
    root = str(tmp_path / "runs")
    rid = _prepared(root)
    store = qpu_confirm.DurableConfirmationStore(root=root)
    nonce, _pub = _issue(store, rid)
    results = []
    barrier = threading.Barrier(16)

    def worker():
        barrier.wait()
        try:
            results.append(store.consume(**{**_consume_kwargs(nonce), "run_id": rid}))
        except qpu_confirm.ConfirmationError:
            results.append(None)

    threads = [threading.Thread(target=worker) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    winners = [r for r in results if r and not r["idempotent"]]
    assert len(winners) == 1                       # exactly one non-idempotent winner
    assert all(r is not None for r in results)      # the rest are idempotent (same payload)


def test_verify_consumed_happy_path(tmp_path):
    root = str(tmp_path / "runs")
    rid = _prepared(root)
    store = qpu_confirm.DurableConfirmationStore(root=root)
    nonce, _pub = _issue(store, rid)
    store.consume(**{**_consume_kwargs(nonce), "run_id": rid})
    cc = store.verify_consumed_confirmation(**_verify_kwargs(rid))
    assert cc["run_id"] == rid and cc["backend"] == "fake_lagos"


def test_verify_rejects_empty_marker(tmp_path):
    # the exact Codex bypass: a {} marker must NOT verify
    root = str(tmp_path / "runs")
    rid = _prepared(root)
    store = qpu_confirm.DurableConfirmationStore(root=root)
    _issue(store, rid)
    marker = qpu_store._artefacts_dir(root, rid) / "confirmation_consumed.json"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("{}")
    with pytest.raises(qpu_confirm.ConfirmationError):
        store.verify_consumed_confirmation(**_verify_kwargs(rid))


def test_verify_rejects_truncated_and_rehashed_wrong_binding(tmp_path):
    import json
    root = str(tmp_path / "runs")
    rid = _prepared(root)
    store = qpu_confirm.DurableConfirmationStore(root=root)
    nonce, _pub = _issue(store, rid)
    store.consume(**{**_consume_kwargs(nonce), "run_id": rid})
    marker = qpu_store._artefacts_dir(root, rid) / "confirmation_consumed.json"
    doc = json.loads(marker.read_text())
    # forge a marker with a WRONG binding but a correctly recomputed self-hash
    doc["backend"] = "ibm_real_device"
    doc["marker_sha256"] = qpu_confirm.DurableConfirmationStore._marker_self_hash(doc)
    marker.write_text(json.dumps(doc))
    with pytest.raises(qpu_confirm.ConfirmationError):   # backend != expected chain
        store.verify_consumed_confirmation(**_verify_kwargs(rid))


def test_verify_rejects_challenge_tampered_after_consume(tmp_path):
    import json
    root = str(tmp_path / "runs")
    rid = _prepared(root)
    store = qpu_confirm.DurableConfirmationStore(root=root)
    nonce, _pub = _issue(store, rid)
    store.consume(**{**_consume_kwargs(nonce), "run_id": rid})
    # tamper the persisted challenge AFTER consumption -> challenge_doc_sha256 mismatch
    cp = qpu_store._doc_path(root, rid, "confirmation")
    cd = json.loads(cp.read_text())
    cd["payload"]["backend"] = "evil"
    cp.write_text(json.dumps(cd))
    with pytest.raises(qpu_confirm.ConfirmationError):
        store.verify_consumed_confirmation(**_verify_kwargs(rid))


def _mp_consume(root, rid, nonce, q):
    from acrpq.dashboard import qpu_confirm as C
    store = C.DurableConfirmationStore(root=root)
    kw = dict(run_id=rid, nonce=nonce, config_hash="sha256:" + "c" * 64,
              artefact_sha256="sha256:" + "b" * 64, decision_sha256="sha256:" + "d" * 64,
              backend="fake_lagos", total_shots=512, max_jobs=1, max_quantum_seconds=20.0,
              consent=C.CONSENT_PHRASE)
    try:
        q.put(store.consume(**kw)["idempotent"])
    except C.ConfirmationError:
        q.put("error")


def test_multiprocess_spawn_single_winner(tmp_path):
    root = str(tmp_path / "runs")
    rid = _prepared(root)
    nonce, _pub = _issue(qpu_confirm.DurableConfirmationStore(root=root), rid)
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    procs = [ctx.Process(target=_mp_consume, args=(root, rid, nonce, q)) for _ in range(6)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    outs = [q.get() for _ in procs]
    assert outs.count(False) == 1                  # exactly one winner across processes
    assert "error" not in outs                     # the rest are idempotent, never ambiguous
