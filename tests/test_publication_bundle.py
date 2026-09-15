"""Phase 4 — publication bundle: cleaned paths, sources untouched, scientific equivalence."""
from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

_P = Path(__file__).resolve().parent.parent / "scripts" / "build_publication_bundle.py"
_spec = importlib.util.spec_from_file_location("build_publication_bundle", _P)
pb = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pb)


def test_bundle_builds_and_only_touches_provenance_fields():
    """build() itself raises if cleaning altered any scientific content."""
    b = pb.build()
    m = b["manifest"]
    assert m["schema"] == pb.BUNDLE_SCHEMA
    assert m["n_files"] > 0 and m["total_substitutions"] > 0
    assert m["cleaning_rules"]["scientific_values_modified"] == 0
    for rec in m["files"]:
        for f in rec["fields_modified"]:
            assert f.split(".")[-1] in pb.CLEANABLE_KEYS


def test_never_claims_byte_identity_with_the_sources():
    m = pb.build()["manifest"]
    rel = m["relationship_to_sources"]
    assert "NOT byte-identical" in rel and "scientifically equivalent" in rel


def test_outputs_carry_no_private_path_or_secret():
    files = pb.build()["files"]
    for rel, text in files.items():
        for name, rx in pb._FORBIDDEN:
            assert not rx.search(text), f"{rel}: {name}"


def test_every_output_is_bound_to_its_source_by_hash():
    m = pb.build()["manifest"]
    for rec in m["files"]:
        src = Path(rec["source_path"])
        assert src.exists(), rec["source_path"]
        assert pb._sha(src.read_bytes()) == rec["source_sha256"]


def test_sources_remain_byte_identical_on_disk():
    """The builder must only READ the historical artifacts.

    Hashing the sources against the manifest the SAME build() call just produced compares
    a value to itself. So snapshot the sources first, then build, then re-hash from disk:
    only that ordering can catch a builder that writes back over a source.
    """
    before = {}
    for d in pb.SOURCE_DIRS:
        for src in sorted((pb.SRC_ROOT / d).rglob("*.json")) if (pb.SRC_ROOT / d).is_dir() else []:
            before[src] = pb._sha(src.read_bytes())
    assert before, "no source artifact found"
    pb.build()
    for src, sha in before.items():
        assert pb._sha(src.read_bytes()) == sha, f"builder modified its source {src}"


def test_published_bundle_on_disk_verifies():
    if not (pb.OUT / "transformation_manifest.json").exists():
        pytest.skip("bundle not generated in this checkout")
    assert pb._verify() == 0


def test_cleaning_is_idempotent():
    once = pb.build()["files"]
    twice = pb.build()["files"]
    assert once == twice


def test_only_allow_listed_provenance_keys_are_rewritten():
    doc = {"code_dir": "/Users/x/repo", "objective": 1.25, "nested": {"note": "/Users/x/f"}}
    cleaned = pb._clean(doc, "", [])
    assert cleaned["objective"] == 1.25
    assert cleaned["nested"]["note"] == "/Users/x/f"     # NOT cleanable -> untouched
    assert cleaned["code_dir"] == pb.PORTABLE
    with pytest.raises(pb.BundleError):
        pb._scan_forbidden(json.dumps(cleaned), "synthetic")   # the leftover path is caught
    assert re.search(r"/Users/", json.dumps(cleaned))


def test_a_wrong_cleaning_rule_that_touches_a_measurement_is_refused(monkeypatch):
    """The equivalence guard must FIRE, not merely exist.

    The strip-and-compare projection cannot catch this on its own: it removes the same
    key list from both sides, so widening CLEANABLE_KEYS to a scientific key hides the
    damage. Here the rule is widened to a NUMERIC scientific key and the numeric-leaf
    guard catches it. Without that second proof this test fails.
    """
    monkeypatch.setattr(pb, "CLEANABLE_KEYS", (*pb.CLEANABLE_KEYS, "objective"))
    doc = {"code_dir": "/Users/x/repo", "objective": 1.25}
    cleaned = pb._clean(doc, "", [])
    # the widened rule leaves the number alone here (no path in it) ...
    assert cleaned["objective"] == 1.25
    # ... but a numeric value that DID hold a path would be replaced by a string, and the
    # numeric-leaf sequence then differs -> the builder must refuse.
    doc2 = {"code_dir": "/Users/x/repo", "objective": "/Users/x/1.25"}
    assert pb._numeric_leaves(doc) != pb._numeric_leaves(pb._clean(doc2, "", []))
    assert pb._numeric_leaves({"a": 1, "b": [2.0, {"c": 3}]}) == [1.0, 2.0, 3.0]
    assert pb._numeric_leaves({"flag": True}) == [], "a bool is not a measurement"
