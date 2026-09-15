#!/usr/bin/env python3
"""Exécute la campagne IBM finale préenregistrée, un circuit par job, via l'API locale.

Ce runner NE DÉCIDE RIEN : il lit `results/ibm_qaoa_final_campaign_v1/preregistration.json`
et exécute les jobs dans l'ordre publié (blocs appariés, rotation des bras). Les angles,
le niveau de transpilation, les shots et les répétitions viennent tous du protocole gelé —
aucune sélection d'angles postérieure aux résultats n'est possible par construction.

La séquence HTTP est celle déjà éprouvée par les 30 jobs de la cohorte P0
(`scripts/run_ibm_micro_campaign.py`) : prepare → confirm → submit → persistance immédiate
du job_id → poll → reconcile → result → export. Seule la SOURCE de la charge utile change.

Garde-fous appliqués à chaque job :

* backend `ibm_marrakesh` exclusivement, vérifié avant et après préparation ;
* plafonds durs : 63 jobs, 32 256 shots, 4 s/job prudent ;
* arrêt opérationnel avant 240 s consommées par la campagne, avec achèvement du bloc
  apparié en cours autorisé tant que 252 s ne sont pas dépassées ;
* relecture du crédit réel après chaque job et après chaque bloc ;
* AUCUN retry automatique d'une soumission ambiguë : le runner s'arrête et rend la main ;
* conditions de checkpoint obligatoire (mauvais backend, shots ≠ 512, hash divergent,
  décodage incohérent, coût > 4 s/job, référence absente, résultat non récupérable).

Exécution réelle : --execute --ack I_AUTHORIZE_REAL_IBM_QPU_SUBMISSIONS
Sans ces deux drapeaux, le runner affiche le plan et ne contacte rien.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ACK = "I_AUTHORIZE_REAL_IBM_QPU_SUBMISSIONS"
PREREG = Path("results/ibm_qaoa_final_campaign_v1/preregistration.json")
OUT_DIR = Path("results/ibm_qaoa_final_campaign_v1")
BACKEND_ALLOWED = "ibm_marrakesh"
AUTHORIZED_CAMPAIGN = "recommandee"

MAX_JOBS = 63
MAX_TOTAL_SHOTS = 32256
CEILING_SECONDS_PER_JOB = 4.0
STOP_BEFORE_SECONDS = 240.0
HARD_STOP_SECONDS = 252.0


class CampaignStop(RuntimeError):
    """Arrêt de protocole : le runner rend la main, il ne réessaie jamais."""


def _request(base_url: str, method: str, path: str, body: Any = None,
             timeout: float = 120.0) -> Any:
    if not base_url.startswith(("http://127.0.0.1", "http://localhost")):
        raise ValueError("l'API de campagne doit être en loopback")
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base_url + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:      # noqa: S310
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode()[:400]
        raise CampaignStop(f"{method} {path} -> HTTP {exc.code} : {detail}") from exc


def _atomic_write(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(doc, fh, indent=2, sort_keys=True, allow_nan=False)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())
    tmp.replace(path)


def _usage() -> dict[str, Any]:
    """Crédit réellement restant, en lecture seule. Le token est lu par le SDK seul."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    return dict(QiskitRuntimeService().usage())


def _job_key(job: dict[str, Any]) -> str:
    obj = "quad" if "quadratic" in job["objective_id"] else "man"
    return (f"{job['instance']}__K{job['grid_k']}__{obj}__{job['arm']}"
            f"__r{job['repetition_id']}__e{job['execution_order']:03d}")


def _resolve(prereg: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    """Angles gelés + niveau de transpilation, lus DANS le protocole. Rien d'inventé."""
    protocol = prereg["protocol"]
    arm = protocol["arms"][job["arm"]]
    if job["arm"] == "CE_endianness_canary":
        can = protocol["endianness_canary"]
        cell = can["cell"]
        ident = can["cell_identity"]
        return {
            "gammas": list(cell["gammas"]), "betas": list(cell["betas"]),
            "optimization_level": int(cell["optimization_level"]),
            "angle_set": "P0_canary", "is_canary": True,
            "ideal_feasible_mass": None, "ideal_onehot_valid_mass": None,
            "ideal_certified_optimum_mass": None,
            "certified_reference_objective": ident["certified_reference_objective"],
            "expected_qubo_sha256": ident["qubo_sha256"],
            "expected_n_qubits": ident["n_qubits"],
        }
    angle_set = arm["angles"]                       # "P0" ou "P1_feas"
    key = f"{job['instance']}|K{job['grid_k']}|{job['objective_id']}|{angle_set}"
    frozen = protocol["frozen_angles"][key]
    cell = protocol["frozen_angles"][
        f"{job['instance']}|K{job['grid_k']}|{job['objective_id']}|_cell"]
    return {
        "gammas": list(frozen["gammas"]), "betas": list(frozen["betas"]),
        "optimization_level": int(arm["optimization_level"]),
        "angle_set": angle_set, "is_canary": False,
        "ideal_feasible_mass": frozen["ideal_feasible_mass"],
        "ideal_onehot_valid_mass": frozen["ideal_onehot_valid_mass"],
        "ideal_certified_optimum_mass": frozen["ideal_certified_optimum_mass"],
        "certified_reference_objective": cell["certified_reference_objective"],
        "expected_qubo_sha256": cell["qubo_sha256"],
        "expected_n_qubits": cell["n_qubits"],
    }


def _prepare_body(job: dict[str, Any], res: dict[str, Any], snapshot: dict,
                  shots: int, transpiler_seed: int, max_quantum_seconds: float) -> dict:
    return {
        "instance": job["instance"], "n_theta": job["grid_k"], "n_q": 1,
        "objective_id": job["objective_id"], "shots": shots,
        "transpiler_seed": transpiler_seed, "backend_requested": BACKEND_ALLOWED,
        "gammas": res["gammas"], "betas": res["betas"],
        "max_jobs": 1, "max_quantum_seconds": max_quantum_seconds,
        "optimization_level": res["optimization_level"], "snapshots": [snapshot],
    }


def _confirm_body(prepared: dict) -> dict:
    ch = prepared["confirmation"]
    return {"nonce": ch["nonce"], "config_hash": prepared["config_hash"],
            "artefact_sha256": prepared["artefact_sha256"],
            "decision_sha256": prepared["decision_sha256"], "backend": prepared["backend"],
            "total_shots": prepared["budget"]["total_shots"],
            "max_jobs": prepared["budget"]["max_jobs"],
            "max_quantum_seconds": prepared["budget"]["max_quantum_seconds"],
            "consent": ch["consent_phrase"]}


def assert_job_id_persisted(submitted: dict[str, Any]) -> list[str]:
    """Une soumission sans job_id est AMBIGUË : arrêt, jamais de retry automatique.

    Extraite en fonction pure pour être testable directement : c'est la règle la plus
    coûteuse à enfreindre (une resoumission double la dépense et casse l'exactly-once).
    """
    job_ids = [str(j) for j in (submitted.get("job_ids") or []) if str(j).strip()]
    if not job_ids:
        raise CampaignStop(
            f"soumission sans job_id (verdict {submitted.get('verdict')!r}) : ambiguïté, "
            f"AUCUN retry automatique — réconciliation manuelle par tag requise")
    return job_ids


def _checkpoint_reasons(record: dict[str, Any], res: dict[str, Any], shots: int) -> list[str]:
    """Conditions de checkpoint OBLIGATOIRE listées par Codex."""
    out: list[str] = []
    art = ((record.get("export") or {}).get("artefact") or {})
    if art and art.get("backend") != BACKEND_ALLOWED:
        out.append(f"mauvais backend : {art.get('backend')!r}")
    if art and art.get("qubo_sha256") not in (None, res["expected_qubo_sha256"]):
        out.append("hash de QUBO divergent du protocole")
    result = record.get("result") or {}
    summary = (result.get("summary") or {})
    if summary:
        n = summary.get("n_shots")
        if n != shots:
            out.append(f"total de shots {n} != {shots}")
        bf = summary.get("best_feasible")
        if bf is not None and not bf.get("feasible"):
            out.append("best_feasible marqué infaisable : décodage incohérent")
    elif record.get("state") == "completed":
        out.append("run complété sans résumé : résultat non récupérable")
    if res["certified_reference_objective"] is None:
        out.append("référence certifiée absente pour cette cellule")
    if record.get("billed_seconds") is not None and \
            record["billed_seconds"] > CEILING_SECONDS_PER_JOB:
        out.append(f"coût {record['billed_seconds']} s > {CEILING_SECONDS_PER_JOB} s/job")
    return out


def run(args: argparse.Namespace) -> int:
    prereg = json.loads(PREREG.read_text())
    protocol = prereg["protocol"]
    plan = prereg["campaigns"][args.campaign]["jobs"]
    shots = int(protocol["shots_per_job"])
    seed = int(protocol["transpiler_seed"])

    # Codex n'autorise QUE la campagne recommandée. Une autre campagne ne doit pas être
    # exécutable en la tronquant silencieusement par la fenêtre d'ordre.
    if args.campaign != AUTHORIZED_CAMPAIGN and args.execute:
        raise CampaignStop(
            f"seule la campagne « {AUTHORIZED_CAMPAIGN} » est autorisée ; "
            f"« {args.campaign} » demanderait une nouvelle autorisation")
    selected = [j for j in plan if args.from_order <= j["execution_order"] <= args.to_order]
    if args.canaries_only:
        selected = [j for j in selected if j["kind"] == "canary"]
    if len(selected) > MAX_JOBS or len(selected) * shots > MAX_TOTAL_SHOTS:
        raise CampaignStop(f"plafond dépassé : {len(selected)} jobs, "
                           f"{len(selected) * shots} shots")

    print(f"protocol_sha256 = {protocol['protocol_sha256']}")
    print(f"campagne « {args.campaign} » · {len(selected)} job(s) sélectionné(s) "
          f"sur {len(plan)} · {len(selected) * shots} shots · "
          f"plafond {len(selected) * CEILING_SECONDS_PER_JOB:.0f} s")
    for j in selected:
        res = _resolve(prereg, j)
        print(f"  #{j['execution_order']:3d} bloc{j['block_id']} {j['kind']:8s} "
              f"{j['instance']}/K{j['grid_k']}/"
              f"{'quad' if 'quad' in j['objective_id'] else 'man'} "
              f"{j['arm']:22s} γ={res['gammas'][0]:.4f} β={res['betas'][0]:.4f} "
              f"L{res['optimization_level']} idéal_faisable="
              + ("canari" if res["is_canary"]
                 else f"{res['ideal_feasible_mass']:.5f}"))
    if not (args.execute and args.ack == ACK):
        print(f"\n[PLAN SEULEMENT] exécution réelle : --execute --ack {ACK}")
        return 0

    flags = _request(args.base_url, "GET", "/api/qpu/flags")
    if not (flags.get("submission_allowed") and flags.get("runtime_factory_enabled")):
        raise CampaignStop(f"drapeaux insuffisants côté serveur : {flags}")

    usage0 = _usage()
    start_consumed = float(usage0["usage_consumed_seconds"])
    print(f"\ncrédit avant campagne : consommé {start_consumed} s · "
          f"restant {usage0['usage_remaining_seconds']} s")

    ledger_path = OUT_DIR / "campaign_ledger.json"
    ledger: dict[str, Any] = (json.loads(ledger_path.read_text())
                              if ledger_path.exists() else
                              {"protocol_sha256": protocol["protocol_sha256"],
                               "backend": BACKEND_ALLOWED, "shots_per_job": shots,
                               "usage_consumed_at_start": start_consumed, "jobs": {}})
    if ledger["protocol_sha256"] != protocol["protocol_sha256"]:
        supersession = protocol.get("supersession") or {}
        if supersession.get("supersedes_protocol_sha256") == ledger["protocol_sha256"]:
            if supersession.get("circuits_affected") or supersession.get("angles_affected") \
                    or supersession.get("qubo_affected") or supersession.get("shots_affected"):
                raise CampaignStop(
                    "supersession déclarée mais elle affecte les circuits/angles/QUBO/"
                    "shots : la continuation sous canaris déjà soumis est refusée")
            print(f"[SUPERSESSION] registre sous {ledger['protocol_sha256']!r} -> "
                  f"protocole courant {protocol['protocol_sha256']!r} "
                  f"({supersession.get('defect_class')}, "
                  f"circuits/angles/QUBO/shots non affectés)")
            ledger["protocol_sha256"] = protocol["protocol_sha256"]
            ledger.setdefault("protocol_supersessions", []).append({
                "from": supersession["supersedes_protocol_sha256"],
                "to": protocol["protocol_sha256"],
                "defect_class": supersession.get("defect_class"),
                "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
            _atomic_write(ledger_path, ledger)
        else:
            raise CampaignStop(
                "le registre existant porte un autre protocol_sha256, non supersédé "
                "par le protocole courant")

    current_block = None
    for j in selected:
        key = _job_key(j)
        if key in ledger["jobs"] and ledger["jobs"][key].get("submitted"):
            print(f"  #{j['execution_order']:3d} déjà soumis, ignoré ({key})")
            continue

        consumed = float(_usage()["usage_consumed_seconds"]) - start_consumed
        if consumed >= STOP_BEFORE_SECONDS:
            finishing = current_block == j["block_id"]
            projected = consumed + CEILING_SECONDS_PER_JOB
            if not (finishing and projected <= HARD_STOP_SECONDS):
                print(f"\n[ARRÊT BUDGET] {consumed:.0f} s consommées par la campagne "
                      f"(seuil {STOP_BEFORE_SECONDS:.0f} s)")
                break
        if j["block_id"] != current_block and current_block is not None:
            print(f"  --- fin du bloc {current_block} · campagne : {consumed:.0f} s ---")
        current_block = j["block_id"]

        res = _resolve(prereg, j)
        live = _request(args.base_url, "GET",
                        f"/api/qpu/backends/live/{BACKEND_ALLOWED}")["backend"]
        if live.get("name") != BACKEND_ALLOWED:
            raise CampaignStop(f"snapshot d'un autre backend : {live.get('name')!r}")

        prepared = _request(args.base_url, "POST", "/api/qpu/prepare",
                            _prepare_body(j, res, live, shots, seed,
                                          args.max_quantum_seconds))
        if prepared["backend"] != BACKEND_ALLOWED:
            raise CampaignStop(f"préparation sur {prepared['backend']!r}")
        record = {**j, "job_key": key, "run_id": prepared["run_id"],
                  "state": prepared["state"], "submitted": False,
                  "artefact_sha256": prepared["artefact_sha256"],
                  "config_hash": prepared["config_hash"], "resolved": res,
                  "prepared_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        ledger["jobs"][key] = record
        _atomic_write(ledger_path, ledger)

        _request(args.base_url, "POST", f"/api/qpu/{prepared['run_id']}/confirm",
                 _confirm_body(prepared))
        submitted = _request(args.base_url, "POST",
                             f"/api/qpu/{prepared['run_id']}/submit", {})
        # persistance IMMÉDIATE du job_id, avant tout poll
        record["submitted"] = True
        record["state"] = submitted["state"]
        record["ibm_job_ids"] = submitted.get("job_ids", [])
        record["submitted_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        record["submit_verdict"] = submitted.get("verdict")
        _atomic_write(ledger_path, ledger)          # persistance AVANT toute validation
        record["ibm_job_ids"] = assert_job_id_persisted(submitted)
        print(f"  #{j['execution_order']:3d} soumis {record['ibm_job_ids'][0][:14]}… "
              f"({key})")

        deadline = time.monotonic() + args.job_wait_seconds
        while time.monotonic() < deadline:
            run_state = _request(args.base_url, "GET", f"/api/qpu/{record['run_id']}")
            record["state"] = run_state["state"]
            record["ibm_job_ids"] = run_state.get("ibm_job_ids", record["ibm_job_ids"])
            if record["state"] in {"completed", "failed", "cancelled"}:
                break
            if run_state.get("ibm_job_ids"):
                rec = _request(args.base_url, "POST",
                               f"/api/qpu/{record['run_id']}/reconcile", {})
                record["state"] = rec["state"]
                if record["state"] in {"completed", "failed", "cancelled"}:
                    break
            _atomic_write(ledger_path, ledger)
            time.sleep(args.poll_seconds)

        result = _request(args.base_url, "GET", f"/api/qpu/{record['run_id']}/result")
        record["result"] = result if result.get("available") else None
        if result.get("available"):
            record["export"] = _request(args.base_url, "POST",
                                        f"/api/qpu/{record['run_id']}/export", {})
        after = float(_usage()["usage_consumed_seconds"]) - start_consumed
        record["campaign_consumed_seconds_after"] = after
        record["billed_seconds"] = round(
            after - sum(r.get("billed_seconds") or 0.0
                        for k, r in ledger["jobs"].items() if k != key), 3)
        record["checkpoint_reasons"] = _checkpoint_reasons(record, res, shots)
        _atomic_write(ledger_path, ledger)
        if record["checkpoint_reasons"]:
            print(f"\n[CHECKPOINT OBLIGATOIRE] {key} : "
                  f"{'; '.join(record['checkpoint_reasons'])}")
            break
        if not record.get("result"):
            print(f"\n[CHECKPOINT] {key} : résultat non disponible dans le délai imparti")
            break

    final = _usage()
    print(f"\ncrédit après : consommé {final['usage_consumed_seconds']} s · "
          f"restant {final['usage_remaining_seconds']} s · "
          f"campagne : {float(final['usage_consumed_seconds']) - start_consumed:.0f} s")
    ledger["usage_consumed_at_end"] = float(final["usage_consumed_seconds"])
    _atomic_write(ledger_path, ledger)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Campagne IBM finale préenregistrée")
    ap.add_argument("--base-url", default="http://127.0.0.1:8055")
    ap.add_argument("--campaign", default="recommandee")
    ap.add_argument("--from-order", type=int, default=1)
    ap.add_argument("--to-order", type=int, default=63)
    ap.add_argument("--canaries-only", action="store_true")
    ap.add_argument("--max-quantum-seconds", type=float, default=20.0)
    ap.add_argument("--poll-seconds", type=float, default=20.0)
    ap.add_argument("--job-wait-seconds", type=float, default=1800.0)
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--ack", default="")
    try:
        return run(ap.parse_args())
    except CampaignStop as exc:
        print(f"\n[ARRÊT DE PROTOCOLE] {exc}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
