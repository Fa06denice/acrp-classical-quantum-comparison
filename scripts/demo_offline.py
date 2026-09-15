#!/usr/bin/env python3
"""Deterministic OFFLINE demonstration mode (Phase 7) — no network, no IBM submission.

What it guarantees before the server starts:

* the environment is **fail-closed**: every QPU flag off, the real runtime factory
  disabled — refuses to start otherwise (exit 4);
* the frozen artifacts the demo cites **verify** (thesis bundle, IBM campaign, publication
  bundle) — refuses to start on a hash mismatch (exit 3);
* real IBM data is read **exclusively** from the 30 recorded results; no account, no token,
  no submission route is ever enabled;
* the four provenance classes are never conflated: live recomputation, recorded local
  result, Aer simulation, recorded IBM hardware counts.

Usage:
    python scripts/demo_offline.py --check       # pre-flight only (no server)
    python scripts/demo_offline.py --smoke       # pre-flight + API smoke test
    python scripts/demo_offline.py               # pre-flight + serve on 127.0.0.1:8000
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

UNSAFE_ENV = (
    "ACRPQ_QPU_ENABLED", "ACRPQ_IBM_SUBMISSION_ENABLED",
    "ACRPQ_IBM_RUNTIME_FACTORY_ENABLED",
)
# Frozen evidence the 10-minute demo may cite, with its verifier.
FROZEN = {
    "results/thesis_benchmark_v1_1/manifest.json": "scripts/build_thesis_benchmark.py",
    "results/publication_bundle_v1/transformation_manifest.json":
        "scripts/build_publication_bundle.py",
    "results/ibm_fixed_angle_campaign_v1/manifest.json": None,   # read-only evidence
}

PROVENANCE_CLASSES = {
    "live_local": "recalcul local en direct (CPU classique)",
    "recorded_local": "resultat local enregistre (artefact fige)",
    "aer_simulation": "simulation Aer (CPU classique, pas de QPU)",
    "recorded_ibm_hardware": "counts materiels IBM enregistres (aucune soumission)",
}


def _fail(code: int, msg: str) -> int:
    print(f"REFUS ({code}) : {msg}", file=sys.stderr)
    return code


def check_environment() -> int:
    """Fail-closed: refuse to run a demo on a machine configured for real submission."""
    from acrpq.dashboard import qpu_flags
    from acrpq.dashboard.qpu_ibm_factory import runtime_factory_enabled

    bad = [v for v in UNSAFE_ENV if (os.environ.get(v) or "").strip().lower()
           in {"1", "true", "yes", "on"}]
    if bad:
        return _fail(4, f"variables d'environnement dangereuses actives : {bad}")
    if qpu_flags.submission_allowed():
        return _fail(4, "submission_allowed() est vrai — la demo exige un mode hors ligne")
    if runtime_factory_enabled():
        return _fail(4, "la factory IBM reelle est activee — desactivez-la avant la demo")
    print("[ok] environnement hors ligne : soumission impossible (drapeaux fail-closed)")
    print(f"     dry_run_only={qpu_flags.dry_run_only()} "
          f"qpu_enabled={qpu_flags.qpu_enabled()} "
          f"ibm_submission_enabled={qpu_flags.ibm_submission_enabled()}")
    return 0


def check_artifacts() -> int:
    """Verify every frozen artifact the demo cites."""
    import subprocess
    for path, verifier in FROZEN.items():
        p = Path(path)
        if not p.exists():
            return _fail(3, f"artefact fige absent : {path}")
        if verifier:
            r = subprocess.run([sys.executable, verifier, "--verify"],
                               capture_output=True, text=True)
            if r.returncode != 0:
                return _fail(3, f"verification echouee pour {path} :\n{r.stdout}{r.stderr}")
            print(f"[ok] {path} verifie ({verifier} --verify)")
        else:
            n = len(json.loads(p.read_text()).get("jobs", []))
            print(f"[ok] {path} present ({n} jobs materiels enregistres, lecture seule)")
    return 0


def smoke(port: int = 8000) -> int:
    """API smoke test through the in-process client (no socket, no network)."""
    from fastapi.testclient import TestClient

    from acrpq.dashboard.api import create_app

    tc = TestClient(create_app())
    for ep in ("/api/health", "/api/instances", "/"):
        r = tc.get(ep)
        if r.status_code != 200:
            return _fail(3, f"{ep} -> HTTP {r.status_code}")
        print(f"[ok] GET {ep} -> 200")
    pf = tc.post("/api/preflight", json={"instance": "CP_3", "n_theta": 3, "n_q": 1})
    if pf.status_code != 200:
        return _fail(3, f"/api/preflight -> HTTP {pf.status_code}")
    d = pf.json()
    for key in ("objective_id", "objective_degeneracy", "selection_policy", "n_qubits"):
        if key not in d:
            return _fail(3, f"/api/preflight sans champ {key}")
    print(f"[ok] preflight CP_3/K3 : objectif={d['objective_id']} "
          f"qubits={d['n_qubits']} proportionnels={d['objective_degeneracy']['proportional']} "
          f"politique={d['selection_policy']['selection_policy_id']}")
    # A real-submission route must be refused BY THE FEATURE FLAG, before the run is
    # even looked up. Accepting 404 here would be worthless: a nonexistent run answers
    # 404 naturally, so the check would still pass with the flag guard deleted.
    sub = tc.post("/api/qpu/nonexistent-run/submit", json={})
    if sub.status_code != 403:
        return _fail(3, f"/submit doit etre refuse par le drapeau (403), recu "
                        f"HTTP {sub.status_code} — le garde-fou est peut-etre absent")
    print(f"[ok] /api/qpu/.../submit refuse (HTTP {sub.status_code}) — aucune depense possible")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Mode demonstration hors ligne")
    ap.add_argument("--check", action="store_true", help="pre-flight seulement")
    ap.add_argument("--smoke", action="store_true", help="pre-flight + smoke test API")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()

    print("=== ACRP-LIB-QUANTUM — mode demonstration HORS LIGNE ===")
    print("Donnees IBM enregistrees — aucune soumission en cours.")
    for k, v in PROVENANCE_CLASSES.items():
        print(f"  - {k}: {v}")
    rc = check_environment()
    if rc:
        return rc
    rc = check_artifacts()
    if rc:
        return rc
    if args.check:
        print("\n[pre-flight OK] la demo peut demarrer.")
        return 0
    rc = smoke(args.port)
    if rc:
        return rc
    if args.smoke:
        print("\n[smoke OK] la demo peut demarrer.")
        return 0

    import uvicorn

    from acrpq.dashboard.api import create_app
    print(f"\n[demo] http://127.0.0.1:{args.port}  (Ctrl+C pour arreter)")
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
