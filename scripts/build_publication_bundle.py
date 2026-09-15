#!/usr/bin/env python3
"""Emit a publication-safe copy of the result artifacts (MoE condition 3c / Phase 4).

47 committed result files embed the developer's absolute paths (``.code_dir``,
``provenance.ampl_binary``) inside **hashed** content, so scrubbing them in place would
invalidate every published hash. This builder therefore:

* leaves every historical source file **byte-identical** (it only reads them);
* writes cleaned copies under ``results/publication_bundle_v1/``;
* replaces private absolute paths by portable markers, touching **no scientific value**;
* records a transformation manifest binding each output to its source by SHA-256, with
  the cleaning rules, the fields modified and the substitution counts;
* validates that the only differences are non-scientific provenance fields;
* scans the output for ``/Users/``-style paths, e-mail addresses, tokens,
  ``Authorization``/``Bearer``, API keys, licence files and venv paths.

The bundle is **scientifically equivalent** to the sources and linked to them by manifest.
It is NOT byte-identical, and this script never claims that.

Run:  python scripts/build_publication_bundle.py [--verify]
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

BUNDLE_SCHEMA = "acrpq-publication-bundle/1"
OUT = Path("results/publication_bundle_v1")
SRC_ROOT = Path("results")

# Bundles known to embed host-local paths (MoE Expert F, CRITICAL).
SOURCE_DIRS = (
    "ampl_gurobi_campaign_v3", "ampl_gurobi_supervised_v1", "ampl_gurobi_supervised_v2",
    "ampl_gurobi_supervised_v3", "ampl_gurobi_timeout_recovery", "original_ampl_diag",
    "original_ampl_real",
)

# Only these keys may be rewritten: pure provenance/location, never a measurement.
CLEANABLE_KEYS = ("code_dir", "ampl_binary", "data_dir", "output_dir", "repo_root",
                  "venv", "python_executable", "artifact_path", "json_path", "csv_path")

_ABS_UNIX = re.compile(r"/(?:Users|home|private|var|opt)/[^\s\"']*")
_ABS_WIN = re.compile(r"[A-Za-z]:\\\\[^\s\"']*")
_FORBIDDEN = (
    ("absolute_unix_path", re.compile(r"(?:^|[\"\s:=\[(])/(?:Users|home|private)/")),
    ("email", re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")),
    ("token", re.compile(r"(?i)QISKIT_IBM_TOKEN|api[_-]?key|apikey")),
    ("authorization", re.compile(r"(?i)\bauthorization\b\s*[:=]|\bBearer\s")),
    ("secret_material", re.compile(r"ghp_[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16}|-----BEGIN")),
    ("licence_file", re.compile(r"(?i)[^\s\"']*\.lic\b|ampl\.lic|gurobi\.lic")),
    ("venv_path", re.compile(r"(?:^|[\"\s:=])[^\s\"']*/\.venv/")),
)

PORTABLE = "<portable>"


class BundleError(RuntimeError):
    pass


def _sha(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _clean_scalar(value: Any) -> tuple[Any, int]:
    """Replace a private absolute path by a portable marker. Returns (value, n_subs)."""
    if not isinstance(value, str):
        return value, 0
    new, n1 = _ABS_UNIX.subn(PORTABLE, value)
    new, n2 = _ABS_WIN.subn(PORTABLE, new)
    return new, n1 + n2


def _clean(obj: Any, path: str, changed: list[dict]) -> Any:
    """Recursively clean ONLY the allow-listed provenance keys."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            p = f"{path}.{k}" if path else k
            if k in CLEANABLE_KEYS:
                nv, n = _clean_scalar(v)
                if n:
                    changed.append({"field": p, "substitutions": n})
                out[k] = nv
            else:
                out[k] = _clean(v, p, changed)
        return out
    if isinstance(obj, list):
        return [_clean(v, f"{path}[{i}]", changed) for i, v in enumerate(obj)]
    return obj


def _strip_paths_for_diff(obj: Any) -> Any:
    """Project a document with every cleanable field removed, to prove the scientific
    content is untouched."""
    if isinstance(obj, dict):
        return {k: _strip_paths_for_diff(v) for k, v in obj.items() if k not in CLEANABLE_KEYS}
    if isinstance(obj, list):
        return [_strip_paths_for_diff(v) for v in obj]
    return obj


def _numeric_leaves(obj: Any, out: list[float] | None = None) -> list[float]:
    """Every numeric leaf, in document order.

    This is the guard that can actually FIRE. Comparing the two documents with the
    cleanable keys stripped cannot detect "cleaning touched a scientific value" when the
    cleaning rule itself is wrong, because both projections use the SAME key list: widen
    CLEANABLE_KEYS by mistake and the offending key disappears from both sides. A
    measurement is always a number, so an unchanged numeric sequence is a check that no
    key list can silently satisfy.
    """
    out = [] if out is None else out
    if isinstance(obj, bool):
        return out
    if isinstance(obj, (int, float)):
        out.append(float(obj))
    elif isinstance(obj, dict):
        for k in sorted(obj):
            _numeric_leaves(obj[k], out)
    elif isinstance(obj, list):
        for v in obj:
            _numeric_leaves(v, out)
    return out


def _scan_forbidden(text: str, where: str) -> None:
    for name, rx in _FORBIDDEN:
        m = rx.search(text)
        if m:
            raise BundleError(f"{where}: forbidden {name} -> {m.group(0)[:60]!r}")


def build() -> dict:
    records: list[dict] = []
    files: dict[str, str] = {}
    for d in SOURCE_DIRS:
        src_dir = SRC_ROOT / d
        if not src_dir.is_dir():
            continue
        for src in sorted(src_dir.rglob("*.json")):
            raw = src.read_bytes()
            doc = json.loads(raw)
            changed: list[dict] = []
            cleaned = _clean(doc, "", changed)
            # PROOF 1: nothing outside the cleanable provenance keys differs.
            if _strip_paths_for_diff(doc) != _strip_paths_for_diff(cleaned):
                raise BundleError(f"{src}: cleaning altered scientific content")
            # PROOF 2 (fires even if the key list itself is wrong): no measurement moved.
            if _numeric_leaves(doc) != _numeric_leaves(cleaned):
                raise BundleError(f"{src}: cleaning altered a numeric value")
            text = json.dumps(cleaned, indent=2, sort_keys=True, allow_nan=False) + "\n"
            _scan_forbidden(text, str(src.relative_to(SRC_ROOT)))
            rel = f"{d}/{src.relative_to(src_dir)}"
            files[rel] = text
            records.append({
                "source_path": str(src.as_posix()),
                "source_sha256": _sha(raw),
                "output_path": f"{OUT.as_posix()}/{rel}",
                "output_sha256": _sha(text.encode()),
                "fields_modified": [c["field"] for c in changed],
                "substitutions": sum(c["substitutions"] for c in changed),
            })
    if not records:
        raise BundleError("no source artifact found")

    from acrpq.classical.ampl_campaign import git_info
    git = git_info()
    manifest = {
        "schema": BUNDLE_SCHEMA,
        "relationship_to_sources": (
            "scientifically equivalent, NOT byte-identical: only host-local provenance "
            "paths were replaced by the portable marker; every source file is left "
            "byte-identical and is bound to its output by sha256 below"
        ),
        "cleaning_rules": {
            "cleanable_keys": list(CLEANABLE_KEYS),
            "replacement": PORTABLE,
            "patterns": ["/(Users|home|private|var|opt)/...", "X:\\\\..."],
            "scientific_values_modified": 0,
        },
        "source_commit": git["git_commit"],
        "scientific_code_dirty": git["scientific_code_dirty"],
        "repository_dirty": git["repository_dirty"],
        "n_files": len(records),
        "total_substitutions": sum(r["substitutions"] for r in records),
        "files": records,
    }
    manifest["manifest_sha256"] = _sha(
        json.dumps({k: v for k, v in manifest.items() if k != "manifest_sha256"},
                   sort_keys=True).encode())
    files["transformation_manifest.json"] = (
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return {"manifest": manifest, "files": files}


_README = """# Publication bundle (results/publication_bundle_v1)

Publication-safe copies of the AMPL/Gurobi result artifacts. The historical sources under
`results/` are **byte-identical and untouched**; their published hashes remain valid.

These copies differ from the sources in **non-scientific provenance fields only**: private
absolute paths (`code_dir`, `ampl_binary`, ...) are replaced by the marker `<portable>`.
Every output is bound to its source by SHA-256 in `transformation_manifest.json`, together
with the cleaning rules, the fields modified and the substitution counts.

**This bundle is scientifically equivalent to the sources. It is NOT byte-identical.**
Cite the sources for hash verification, this bundle for publication.
"""


def _write(bundle: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for rel, text in bundle["files"].items():
        p = OUT / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    (OUT / "README.md").write_text(_README)


def _verify() -> int:
    mp = OUT / "transformation_manifest.json"
    if not mp.exists():
        print("MISSING transformation_manifest.json")
        return 2
    manifest = json.loads(mp.read_text())
    ok = True
    for rec in manifest["files"]:
        src = Path(rec["source_path"])
        out = Path(rec["output_path"])
        if not src.exists() or _sha(src.read_bytes()) != rec["source_sha256"]:
            print(f"SOURCE CHANGED: {src}")
            ok = False
        if not out.exists() or _sha(out.read_bytes()) != rec["output_sha256"]:
            print(f"OUTPUT CHANGED: {out}")
            ok = False
    recon = _sha(json.dumps({k: v for k, v in manifest.items() if k != "manifest_sha256"},
                            sort_keys=True).encode())
    if recon != manifest["manifest_sha256"]:
        print("MANIFEST TAMPERED")
        ok = False
    print("OK publication bundle verified" if ok else "VERIFICATION FAILED")
    return 0 if ok else 2


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--verify":
        return _verify()
    b = build()
    _write(b)
    m = b["manifest"]
    print(f"[publication] {m['n_files']} files, {m['total_substitutions']} path substitutions")
    print(f"  manifest_sha256={m['manifest_sha256'][:24]}…  -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
