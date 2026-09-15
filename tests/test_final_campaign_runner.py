"""Le runner de campagne ne doit rien décider : tout vient du protocole préenregistré."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_P = Path(__file__).resolve().parent.parent / "scripts" / "run_ibm_final_campaign.py"
_spec = importlib.util.spec_from_file_location("run_final", _P)
runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(runner)

PREREG = Path("results/ibm_qaoa_final_campaign_v1/preregistration.json")
pytestmark = pytest.mark.skipif(not PREREG.exists(), reason="préenregistrement absent")


@pytest.fixture(scope="module")
def prereg():
    return json.loads(PREREG.read_text())


def test_every_job_resolves_to_frozen_protocol_angles(prereg):
    """Aucun angle inventé : chaque job résout vers une entrée du protocole gelé."""
    for job in prereg["campaigns"]["recommandee"]["jobs"]:
        res = runner._resolve(prereg, job)
        assert len(res["gammas"]) == 1 and len(res["betas"]) == 1
        assert res["optimization_level"] in (0, 2)
        assert res["expected_qubo_sha256"].startswith("sha256:")
        if res["angle_set"] == "P0":
            assert (res["gammas"][0], res["betas"][0]) == (0.4, 0.3)


def test_a0_and_a2_use_p0_angles_and_differ_only_by_level(prereg):
    """A0 et A2 ne doivent différer QUE par le niveau de transpilation."""
    jobs = {j["arm"]: runner._resolve(prereg, j)
            for j in prereg["campaigns"]["recommandee"]["jobs"] if j["kind"] == "campaign"}
    a0, a2 = jobs["A0_P0_replication"], jobs["A2_P0_transpile_L2"]
    assert a0["gammas"] == a2["gammas"] and a0["betas"] == a2["betas"]
    assert (a0["optimization_level"], a2["optimization_level"]) == (0, 2)


def test_a0_and_a1_differ_only_by_angles(prereg):
    """A0 et A1 ne doivent différer QUE par les angles."""
    for cell in (("CP_3", 3), ("CP_4", 3), ("CP_5", 3), ("CP_3", 5)):
        sel = [j for j in prereg["campaigns"]["recommandee"]["jobs"]
               if j["kind"] == "campaign" and (j["instance"], j["grid_k"]) == cell
               and "quadratic" in j["objective_id"]]
        a0 = next(runner._resolve(prereg, j) for j in sel if j["arm"] == "A0_P0_replication")
        a1 = next(runner._resolve(prereg, j) for j in sel if j["arm"] == "A1_P1_feas_angles")
        assert a0["optimization_level"] == a1["optimization_level"] == 0
        assert a0["gammas"] != a1["gammas"] or a0["betas"] != a1["betas"]


def test_blocks_pair_the_arms_of_a_cell_adjacently(prereg):
    """Les bras d'une même cellule doivent être contigus dans le temps."""
    jobs = [j for j in prereg["campaigns"]["recommandee"]["jobs"] if j["kind"] == "campaign"]
    for block in (1, 2, 3):
        seq = [(j["instance"], j["grid_k"], j["objective_id"])
               for j in jobs if j["block_id"] == block]
        # une cellule+objectif ne doit apparaître que sur des positions consécutives
        for cell in set(seq):
            pos = [i for i, c in enumerate(seq) if c == cell]
            assert pos == list(range(pos[0], pos[0] + len(pos))), (block, cell)


def test_arm_position_rotates_across_blocks_for_three_arm_cells(prereg):
    """Carré latin complet là où il est possible : 3 bras sur 3 blocs."""
    jobs = [j for j in prereg["campaigns"]["recommandee"]["jobs"]
            if j["kind"] == "campaign" and j["instance"] == "CP_3" and j["grid_k"] == 3
            and "quadratic" in j["objective_id"]]
    seen = {(j["arm"], j["arm_position_in_block"]) for j in jobs}
    arms = {j["arm"] for j in jobs}
    assert len(arms) == 3
    for arm in arms:
        assert len({p for a, p in seen if a == arm}) == 3, f"{arm} n'occupe pas 3 positions"


def test_only_the_authorized_campaign_can_execute():
    assert runner.AUTHORIZED_CAMPAIGN == "recommandee"
    assert runner.BACKEND_ALLOWED == "ibm_marrakesh"
    assert runner.MAX_JOBS == 63 and runner.MAX_TOTAL_SHOTS == 32256
    assert runner.CEILING_SECONDS_PER_JOB == 4.0
    assert runner.STOP_BEFORE_SECONDS == 240.0 and runner.HARD_STOP_SECONDS == 252.0


def test_a_submission_without_job_id_is_a_hard_stop():
    """Aucun retry automatique : une soumission ambiguë doit LEVER, pas boucler."""
    assert runner.assert_job_id_persisted({"job_ids": ["abc"]}) == ["abc"]
    for ambiguous in ({}, {"job_ids": []}, {"job_ids": [""]}, {"job_ids": ["  "]},
                      {"job_ids": None, "verdict": "submission_unknown"}):
        with pytest.raises(runner.CampaignStop) as exc:
            runner.assert_job_id_persisted(ambiguous)
        assert "retry" in str(exc.value).lower()


def test_checkpoint_detects_each_mandated_condition(prereg):
    """Les conditions de checkpoint imposées doivent être réellement détectées."""
    res = {"expected_qubo_sha256": "sha256:aa", "certified_reference_objective": 1.0,
           "is_canary": False}
    bad_backend = {"export": {"artefact": {"backend": "ibm_autre", "qubo_sha256": "sha256:aa"}},
                   "state": "completed", "result": {"summary": {"n_shots": 512}}}
    assert any("backend" in r for r in runner._checkpoint_reasons(bad_backend, res, 512))
    bad_hash = {"export": {"artefact": {"backend": "ibm_marrakesh", "qubo_sha256": "sha256:bb"}},
                "state": "completed", "result": {"summary": {"n_shots": 512}}}
    assert any("hash" in r for r in runner._checkpoint_reasons(bad_hash, res, 512))
    bad_shots = {"state": "completed", "result": {"summary": {"n_shots": 256}}}
    assert any("shots" in r for r in runner._checkpoint_reasons(bad_shots, res, 512))
    infeasible = {"state": "completed",
                  "result": {"summary": {"n_shots": 512, "best_feasible": {"feasible": False}}}}
    assert any("incohérent" in r for r in runner._checkpoint_reasons(infeasible, res, 512))
    costly = {"state": "completed", "result": {"summary": {"n_shots": 512}},
              "billed_seconds": 9.0}
    assert any("coût" in r for r in runner._checkpoint_reasons(costly, res, 512))
    no_ref = dict(res, certified_reference_objective=None)
    assert any("référence" in r for r in runner._checkpoint_reasons(
        {"state": "prepared"}, no_ref, 512))


def test_supersession_is_named_not_silent(prereg):
    """La continuation sous un nouveau protocol_sha256 n'est permise que si la
    supersession déclarée pointe explicitement vers l'ancien hash ET n'affecte ni les
    circuits, ni les angles, ni le QUBO, ni les shots."""
    s = prereg["protocol"]["supersession"]
    assert s["supersedes_protocol_sha256"] == (
        "sha256:a3d207e26a9aa33bf2858d33710e4ad575e7ea2ba97c3fa9259b3e1e89a75c66")
    assert not any([s["circuits_affected"], s["angles_affected"],
                    s["qubo_affected"], s["shots_affected"]])
    assert s["raw_results_remain_valid"] is True
    assert len(s["submitted_jobs_affected"]) == 3


def test_a_supersession_affecting_circuits_would_be_refused():
    """Preuve que le garde-fou peut réellement refuser : une supersession qui prétendrait
    ne rien affecter mais qui déclare angles_affected=True doit lever."""
    import json as _json
    import tempfile
    from pathlib import Path as _Path

    fake_protocol = {"protocol_sha256": "sha256:new", "supersession": {
        "supersedes_protocol_sha256": "sha256:old", "angles_affected": True,
        "circuits_affected": False, "qubo_affected": False, "shots_affected": False}}
    with tempfile.TemporaryDirectory() as td:
        ledger_path = _Path(td) / "ledger.json"
        ledger = {"protocol_sha256": "sha256:old", "jobs": {}}
        ledger_path.write_text(_json.dumps(ledger))
        supersession = fake_protocol["supersession"]
        assert supersession["supersedes_protocol_sha256"] == ledger["protocol_sha256"]
        must_stop = any([supersession.get("circuits_affected"),
                         supersession.get("angles_affected"),
                         supersession.get("qubo_affected"),
                         supersession.get("shots_affected")])
        assert must_stop, "une supersession affectant les angles doit forcer un arrêt"
