#!/usr/bin/env python
"""Compact the experimental adaptive artefacts WITHOUT scientific loss (Phase 11).

* JSON files under results/experimental_adaptive_grid_v1 are rewritten in canonical compact
  form (sort_keys, no indentation). Content parity is asserted by reloading both versions.
* ``decoded.json`` expansions of the chains are deleted: they are reconstructible bit for bit
  from ``raw_counts.json`` + the round record (see scripts/experimental_reconstruct_decoded.py;
  the reconstruction hash is enforced by ChainStore.verify(deep=True)).
* ``chain.lock`` files are removed from the pack (runtime locks, not data).
* Manifests that list per-file sha256 are regenerated. A COMPACT_PACK.json records sizes
  before/after, files removed/kept, and the reconstruction command.

Never touches anything outside results/experimental_adaptive_grid_v1.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "experimental_adaptive_grid_v1"
if len(sys.argv) > 1:  # optional: compact another experimental results directory
    BASE = (ROOT / sys.argv[1]).resolve()
    if not str(BASE).startswith(str((ROOT / "results").resolve())) or "experimental" not in BASE.name:
        raise SystemExit("refusing to compact a non-experimental directory")


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _lines(p: Path) -> int:
    return p.read_text().count("\n") + 1


def main() -> int:
    if not BASE.exists():
        print("nothing to compact")
        return 1
    files_before = [p for p in BASE.rglob("*") if p.is_file()]
    size_before = sum(p.stat().st_size for p in files_before)
    lines_before = sum(_lines(p) for p in files_before if p.suffix in (".json", ".csv", ".md"))
    removed: list[str] = []
    compacted: list[str] = []
    # 1) delete reconstructible decoded.json + lock files
    for p in sorted(BASE.rglob("decoded.json")):
        rec = json.loads((p.parent / "round.json").read_text())
        doc = json.loads(p.read_text())
        doc = {k: v for k, v in doc.items() if k not in ("schema", "experimental", "official_benchmark")}
        h = hashlib.sha256(json.dumps(doc, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if h != rec["decoded_result_hash"]:
            print(f"REFUSE: {p} does not match its recorded hash; not deleting")
            return 2
        p.unlink()
        removed.append(str(p.relative_to(ROOT)))
    for p in sorted(BASE.rglob("chain.lock")):
        p.unlink()
        removed.append(str(p.relative_to(ROOT)))
    # 2) compact every JSON with parity check
    for p in sorted(BASE.rglob("*.json")):
        if p.name in ("manifest.json",):
            continue
        obj = json.loads(p.read_text())
        text = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
        if json.loads(text) != obj:
            print(f"REFUSE: parity failure for {p}")
            return 3
        p.write_text(text)
        compacted.append(str(p.relative_to(ROOT)))
    # 3) regenerate manifests' per-file hashes
    for man in sorted(BASE.rglob("manifest.json")):
        m = json.loads(man.read_text())
        if "runs" in m and isinstance(m["runs"], list):
            m["runs"] = [{"path": r["path"], "sha256": _sha(man.parent / r["path"])} for r in m["runs"]]
        if "files" in m and isinstance(m["files"], dict):
            m["files"] = {q.name: _sha(q) for q in man.parent.iterdir() if q.is_file() and q.name != "manifest.json"}
        if "outputs" in m and isinstance(m["outputs"], dict):
            for name in list(m["outputs"]):
                q = man.parent / name
                if q.exists():
                    m["outputs"][name] = {"sha256": _sha(q), "bytes": q.stat().st_size}
        m["compact_pack"] = True
        man.write_text(json.dumps(m, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    files_after = [p for p in BASE.rglob("*") if p.is_file()]
    size_after = sum(p.stat().st_size for p in files_after)
    lines_after = sum(_lines(p) for p in files_after if p.suffix in (".json", ".csv", ".md"))
    report = {
        "schema": "acrpq-experimental-compact-pack/1", "experimental": True, "official_benchmark": False,
        "real_qpu": False, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "bytes_before": size_before, "bytes_after": size_after, "text_lines_before": lines_before,
        "text_lines_after": lines_after, "files_before": len(files_before), "files_after": len(files_after),
        "removed": removed, "compacted_json": compacted,
        "kept": ["raw_counts.json (unique aggregated counts, qiskit order)", "round.json (per-round summary, centres, deltas, hashes, ISA report)",
                 "preregistration.json / chain_manifest.json (protocol, provenance)", "summary/benchmark CSV+JSON", "moe_reports/*.md", "references"],
        "reconstruction_command": "PYTHONPATH=src python scripts/experimental_reconstruct_decoded.py results/experimental_adaptive_grid_v1/chains/<chain> --write",
        "parity_proof": "every JSON reloaded equal before/after compaction; decoded.json hashes matched decoded_result_hash before deletion and reconstruction is bit-identical (tested)",
    }
    (BASE / "COMPACT_PACK.json").write_text(json.dumps(report, sort_keys=True, indent=1) + "\n")
    print(json.dumps({k: report[k] for k in ("bytes_before", "bytes_after", "text_lines_before", "text_lines_after", "files_before", "files_after")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
