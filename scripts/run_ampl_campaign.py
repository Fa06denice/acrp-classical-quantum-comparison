#!/usr/bin/env python
"""Full AMPL/Gurobi campaign under `ampl_gurobi_supervised_v3` — SEQUENTIAL, resume-safe.

Rules (reviewer D): one instance at a time, no parallelism, bounded timeout per
instance (the protocol's 120 s, enforced by the runner's process-group kill), atomic
write after each instance, idempotent resume, never overwrite a validated artifact
without verification, keep timeout/error/unavailable as honest results, never anchor an
uncertified incumbent, stop on licence error or corruption, do NOT stop the whole
campaign on a scientific timeout, log progress + duration, and check licence expiry
(2026-09-15) before launch. Discrete K3/K5/K7 references are recorded per instance;
when a grid exceeds the exact-search cap it is an explicit non-proven status with NO
official delta. No quantum is re-run and no historical result is modified.

Usage:  python scripts/run_ampl_campaign.py [--instances CP_3 CP_4 ...] [--family CP]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from acrpq.classical.ampl_campaign import (
    build_instance_artifact,
    check_license_not_expired,
    discretisation_costs,
    git_info,
    load_instance,
)
from acrpq.classical.ampl_protocol import OFFICIAL_PROTOCOL as PROTOCOL
from acrpq.classical.ampl_protocol import accept_official_anchor
from acrpq.io.loader import InstanceLoader

LICENSE_EXPIRY = "2026-09-15"
OUT_DIR = Path("results") / "ampl_gurobi_campaign_v3"
HARD_STOP_KINDS = {"license_error"}


def _sha256_file(p: Path) -> str:
    return "sha256:" + hashlib.sha256(p.read_bytes()).hexdigest()


def _atomic_write_json(path: Path, obj: dict) -> str:
    """Write JSON atomically (temp + fsync + os.replace); return the file's SHA-256."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    data = json.dumps(obj, indent=2, default=str)
    with tmp.open("w", encoding="utf-8") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    return _sha256_file(path)


def _valid_existing(path: Path) -> dict | None:
    """Return a parseable, protocol-matching artifact for resume, else None."""
    if not path.is_file():
        return None
    try:
        art = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if art.get("protocol_id") == PROTOCOL.protocol_id and art.get("status"):
        return art
    return None


def _hard_stop(art: dict) -> str | None:
    if art.get("status_kind") in HARD_STOP_KINDS:
        return art["status_kind"]
    return None


def _family(name: str) -> str:
    import re
    m = re.match(r"^([A-Za-z_]+?)_?\d", name)
    return m.group(1) if m else name


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instances", nargs="*", default=None)
    ap.add_argument("--family", default="CP")
    args = ap.parse_args(argv)

    ok, today = check_license_not_expired(LICENSE_EXPIRY)
    print(f"licence expiry {LICENSE_EXPIRY} | today {today} | usable={ok}")
    if not ok:
        print("ABORT: licence expired or expiring today — not launching.")
        return 2

    loader = InstanceLoader()
    if args.instances:
        instances = args.instances
    else:
        instances = sorted(
            (n for n in loader.list_names() if _family(n) == args.family),
            key=lambda s: (len(s), s))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    git = git_info()
    print(f"campaign: {PROTOCOL.protocol_id} | {len(instances)} instances of family "
          f"{args.family} | git {git['git_commit']} "
          f"(scientific_code_dirty={git['scientific_code_dirty']}, "
          f"repository_dirty={git['repository_dirty']})")

    manifest_rows: list[dict[str, Any]] = []
    stopped = None
    for i, name in enumerate(instances, 1):
        path = OUT_DIR / f"{name}.json"
        existing = _valid_existing(path)
        if existing is not None:
            print(f"[{i}/{len(instances)}] {name}: resume — already present, verified, skipped")
            art = existing
            sha = _sha256_file(path)
        else:
            t0 = time.perf_counter()
            try:
                inst, dat_text, source = load_instance(name, loader)
            except Exception as exc:  # instance not loadable — honest, continue
                print(f"[{i}/{len(instances)}] {name}: UNAVAILABLE ({exc}) — recorded, continue")
                art = {"protocol_id": PROTOCOL.protocol_id, "instance": name,
                       "status": "unavailable", "status_kind": "instance_load_error",
                       "is_official_anchor": False, "error": str(exc)[:200]}
                sha = _atomic_write_json(path, art)
                manifest_rows.append({"instance": name, "sha256": sha, "status": "unavailable",
                                      "is_anchor": False})
                continue
            art = build_instance_artifact(inst, dat_text, source, git, PROTOCOL)
            art["discretisation_costs"] = discretisation_costs(art)
            sha = _atomic_write_json(path, art)
            # corruption guard: re-read and re-hash
            reread = json.loads(path.read_text(encoding="utf-8"))
            if _sha256_file(path) != sha or reread.get("instance") != name:
                stopped = {"instance": name, "reason": "artifact_corruption"}
                print(f"[{i}/{len(instances)}] {name}: CORRUPTION detected — STOP")
                break
            dur = time.perf_counter() - t0
            m = art.get("residual_measurement") or {}
            print(f"[{i}/{len(instances)}] {name}: status={art['status']} "
                  f"anchor={art['is_official_anchor']} gap={art.get('optimality_gap')} "
                  f"residual={art.get('max_model_constraint_residual')} "
                  f"meas_matches={m.get('solution_matches_primary')} {dur:.1f}s")

        manifest_rows.append({"instance": name, "sha256": sha,
                              "status": art.get("status"),
                              "is_anchor": bool(art.get("is_official_anchor"))})
        reason = _hard_stop(art)
        if reason:
            stopped = {"instance": name, "reason": reason}
            print(f"[{i}/{len(instances)}] {name}: HARD STOP: {reason}")
            break

    _write_outputs(instances, manifest_rows, git, today, stopped)
    print(f"\ncampaign artifacts -> {OUT_DIR}/  (manifest + summary CSV/JSON)")
    if stopped:
        print(f"STOPPED EARLY: {stopped}")
    return 0


def _write_outputs(instances, manifest_rows, git, today, stopped) -> None:
    import csv
    import statistics

    arts = []
    for row in manifest_rows:
        p = OUT_DIR / f"{row['instance']}.json"
        if p.is_file():
            try:
                arts.append(json.loads(p.read_text(encoding="utf-8")))
            except ValueError:
                pass

    # counts + anchor/excluded lists + distributions
    counts: dict[str, int] = {"optimal": 0, "timeout": 0, "error": 0, "unavailable": 0,
                              "unknown": 0, "infeasible": 0, "non_anchor": 0, "anchor": 0}
    anchors: list[str] = []
    excluded: list[dict[str, Any]] = []
    gaps: list[float] = []
    times: list[float] = []
    residuals: list[float] = []
    margins: list[float] = []
    summary_rows: list[dict[str, Any]] = []
    for a in arts:
        st = a.get("status", "unknown")
        counts[st] = counts.get(st, 0) + 1
        is_anchor = bool(a.get("is_official_anchor"))
        counts["anchor" if is_anchor else "non_anchor"] += 1
        if is_anchor:
            anchors.append(a["instance"])
            for lst, key in ((gaps, "optimality_gap"), (residuals, "max_model_constraint_residual"),
                             (margins, "minimum_safety_margin")):
                v = a.get(key)
                if isinstance(v, (int, float)):
                    lst.append(v)
        else:
            failing = [k for k, v in (a.get("anchor_criteria") or {}).items() if not v]
            excluded.append({"instance": a["instance"], "status": st,
                             "status_kind": a.get("status_kind"),
                             "failing_criteria": failing or None})
        if isinstance(a.get("wall_time_s"), (int, float)):
            times.append(a["wall_time_s"])
        dc = a.get("discretisation_costs") or {}
        summary_rows.append({
            "instance": a.get("instance"), "n": a.get("n"), "status": st,
            "is_official_anchor": is_anchor,
            "optimality_gap": a.get("optimality_gap"),
            "max_model_constraint_residual": a.get("max_model_constraint_residual"),
            "minimum_safety_margin": a.get("minimum_safety_margin"),
            "rescored_objective": a.get("rescored_objective"),
            "discretisation_cost_K3_q1": (dc.get("discretisation_cost_K3_q1") or {}).get("value"),
            "discretisation_cost_K5_q1": (dc.get("discretisation_cost_K5_q1") or {}).get("value"),
            "discretisation_cost_K7_q1": (dc.get("discretisation_cost_K7_q1") or {}).get("value"),
            "wall_time_s": a.get("wall_time_s"),
        })

    def _dist(xs):
        xs = [x for x in xs if isinstance(x, (int, float))]
        if not xs:
            return {"n": 0}
        return {"n": len(xs), "min": min(xs), "max": max(xs),
                "mean": statistics.fmean(xs),
                "median": statistics.median(xs)}

    manifest = {
        "protocol_id": PROTOCOL.protocol_id, "protocol": PROTOCOL.as_dict(),
        "family_size": len(instances), "recorded_utc": now_iso(), "today": today,
        "git_commit": git["git_commit"],
        "scientific_code_dirty": git["scientific_code_dirty"],
        "repository_dirty": git["repository_dirty"],
        "stopped_early": stopped,
        "counts": counts,
        "official_anchors": anchors,
        "excluded_results": excluded,
        "distributions": {"optimality_gap": _dist(gaps), "wall_time_s": _dist(times),
                          "mp_residual": _dist(residuals), "minimum_safety_margin": _dist(margins)},
        "artifact_hashes": {r["instance"]: r["sha256"] for r in manifest_rows if "sha256" in r},
    }
    _atomic_write_json(OUT_DIR / "campaign_manifest.json", manifest)
    _atomic_write_json(OUT_DIR / "campaign_summary.json", {"rows": summary_rows})

    cols = list(summary_rows[0].keys()) if summary_rows else []
    tmp = OUT_DIR / "campaign_summary.csv.tmp"
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, OUT_DIR / "campaign_summary.csv")

    # cross-check against the validated mini-sweep CP_3/4/5 (never overwrite them)
    _cross_check_mini_sweep(arts)
    # gate: prove every stored anchor is accepted by the fail-closed guard
    for a in arts:
        if a.get("is_official_anchor"):
            accept_official_anchor(a, OUT_DIR)  # raises if any anchor is not truly official


def _cross_check_mini_sweep(arts) -> None:
    mini = Path("results") / PROTOCOL.protocol_id
    for a in arts:
        name = a.get("instance")
        mp = mini / f"{name}.json"
        if not mp.is_file():
            continue
        ref = json.loads(mp.read_text(encoding="utf-8"))
        if bool(ref.get("is_official_anchor")) != bool(a.get("is_official_anchor")):
            print(f"  WARNING: {name} anchor differs from mini-sweep "
                  f"({ref.get('is_official_anchor')} vs {a.get('is_official_anchor')})")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


if __name__ == "__main__":
    raise SystemExit(main())
