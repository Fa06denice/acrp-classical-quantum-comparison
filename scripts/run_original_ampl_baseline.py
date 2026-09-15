#!/usr/bin/env python
"""Reproducible original-AMPL baseline campaign.

Runs the four ACRP objective levels (original_ampl, pyomo_minlp, discrete_reference
and the approximate legs) over the WHOLE-instance scenarios and writes the
scientific delta artifact to a dedicated, versioned directory. It NEVER:

* fabricates an AMPL number — if AMPL/Couenne are absent every original_ampl leg is
  recorded ``unavailable`` and the continuous deltas stay ``None`` with a note;
* overwrites the QPU/IBM campaign results — output goes under
  ``results/original_ampl_baseline/<label>/`` (default label from --label), a path
  disjoint from ``campaign_summary.json`` and the QPU run store.

Only whole-instance scenarios are eligible: the historical NC_ACRP model runs the
verbatim whole-instance .dat, so subset scenarios (Q2/Q2h/QR3) are skipped with an
explicit note rather than fed a reconstructed .dat.

Usage:
    python scripts/run_original_ampl_baseline.py --label run1 [--no-quantum]
    python scripts/run_original_ampl_baseline.py --scenarios Q4 Q4f --out results
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from acrpq.benchmark.baseline import compute_baseline, write_baseline_csv, write_baseline_json
from acrpq.benchmark.runner import BenchmarkRunner
from acrpq.benchmark.scenarios import list_scenarios


def _whole_instance_scenarios() -> list[str]:
    return [sc.id for sc in list_scenarios() if not sc.subset]


def _sanitize_label(label: str) -> str:
    keep = "".join(c if (c.isalnum() or c in "-_.") else "_" for c in label)
    return keep or "run"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios", nargs="*", default=None,
                    help="scenario ids (default: all whole-instance scenarios)")
    ap.add_argument("--out", default="results", help="base results dir")
    ap.add_argument("--label", default="latest", help="subdir label under the baseline dir")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--maxiter", type=int, default=100)
    ap.add_argument("--max-qubits", type=int, default=24)
    ap.add_argument("--no-quantum", action="store_true", help="continuous+discrete legs only")
    args = ap.parse_args(argv)

    scenarios = args.scenarios or _whole_instance_scenarios()
    if not scenarios:
        print("no whole-instance scenarios available", file=sys.stderr)
        return 1

    out_dir = Path(args.out) / "original_ampl_baseline" / _sanitize_label(args.label)
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = BenchmarkRunner(out_dir=str(out_dir), seed=args.seed)
    comparisons = []
    for sid in scenarios:
        recs = runner.compare(
            sid,
            reps=args.reps,
            maxiter=args.maxiter,
            with_quantum=not args.no_quantum,
            with_original_ampl=True,
            with_pyomo_minlp=True,
            max_qubits=args.max_qubits,
            write=True,  # per-scenario benchmark_* + baseline_* under out_dir
        )
        bl = compute_baseline(recs)
        if bl is not None:
            comparisons.append(bl)

    if comparisons:
        write_baseline_csv(comparisons, out_dir / "baseline_all.csv")
        write_baseline_json(comparisons, out_dir / "baseline_all.json")

    manifest = {
        "scenarios": scenarios,
        "seed": args.seed,
        "with_quantum": not args.no_quantum,
        "n_comparisons": len(comparisons),
        "original_ampl_proven_optimum": sum(
            1 for c in comparisons
            if c.legs.get("original_ampl") and c.legs["original_ampl"].is_optimal_anchor
        ),
        "original_ampl_feasible_incumbent": sum(
            1 for c in comparisons
            if c.legs.get("original_ampl")
            and c.legs["original_ampl"].is_incumbent
            and not c.legs["original_ampl"].is_optimal_anchor
        ),
        "original_ampl_unavailable": sum(
            1 for c in comparisons
            if c.legs.get("original_ampl") and c.legs["original_ampl"].status == "unavailable"
        ),
        "out_dir": str(out_dir),
    }
    (out_dir / "campaign_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    print(f"wrote {len(comparisons)} baseline comparisons to {out_dir}/")
    print(f"  original_ampl proven optimum: {manifest['original_ampl_proven_optimum']}"
          f" | feasible incumbent: {manifest['original_ampl_feasible_incumbent']}"
          f" | unavailable: {manifest['original_ampl_unavailable']}")
    if manifest["original_ampl_proven_optimum"] == 0:
        print("  NOTE: no proven continuous optimum (AMPL/Couenne absent or only"
              " feasible). Official continuous deltas are honestly None.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
