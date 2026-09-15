"""Tests for the zip-aware instance loader."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from acrpq.io.loader import circle_default_coords
from acrpq.model import Family, PI

DATA = Path(__file__).resolve().parent / "data"


def test_discovers_all_families(loader):
    names = loader.list_names()
    # 742 unique stems + 300 RCP_FL5 aliases (@FL5) = 1042 addressable names
    assert len(names) > 1000
    assert "CP_3" in names
    assert "CP_4" in names
    fams = {loader.load(n).family for n in ["CP_3", "FP_4", "GP_4", "RCP_10_1"]}
    assert Family.CP in fams and Family.RCP in fams


def test_fl5_addressable(loader):
    # both flight-level variants of the same stem must be reachable
    fl3 = loader.load("RCP_50_1")
    fl5 = loader.load("RCP_50_1", flight_levels=5)
    assert fl3.nf == 3
    assert fl5.nf == 5
    assert fl5.name.endswith("@FL5")
    assert loader.load("RCP_50_1@FL5").nf == 5
    with pytest.raises(ValueError, match="@FL5"):
        loader.load("RCP_50_1@FL5", flight_levels=3)


def test_cp4_explicit_coordinates(cp4):
    assert cp4.n == 4
    assert cp4.x0[0] == 2.0  # from the .dat, not reconstructed
    assert cp4.y0[0] == 0.0  # stored as "-0.00" -> 0.0


def test_circle_default_matches_mod_formula():
    # x0[i] = -radius*cos((i-1)*2*PI/n + PI); for i=1, n=3 -> -2*cos(PI) ~ 2.
    # NB: PI is the .mod literal 3.141592 (not math.pi), so sin(PI) != 0 exactly
    # and y0 carries a ~1.3e-6 artifact. This is intentional fidelity to AMPL.
    xs, ys = circle_default_coords(3, 2.0)
    assert math.isclose(xs[0], 2.0, abs_tol=1e-6)
    assert math.isclose(ys[0], 0.0, abs_tol=1e-4)


def test_fl_instance_reconstructs_coords(loader):
    # an RCP_FL instance omits x0/y0; the loader must materialise them
    fl_names = [n for n in loader.list_names(Family.RCP_FL)]
    assert fl_names, "expected RCP_FL instances"
    inst = loader.load(fl_names[0])
    assert inst.has_flight_levels
    assert len(inst.x0) == inst.n
    assert len(inst.l0) == inst.n


def test_theta0_normalization(cp4):
    # cap values >= PI are shifted by -2*PI (NC_ACRP.mod line 16)
    for c, t in zip(cp4.cap, cp4.theta0):
        expected = c - 2 * PI if c >= PI else c
        assert math.isclose(t, expected, abs_tol=1e-9)


def test_subset(cp4):
    sub = cp4.subset([1, 3])
    assert sub.n == 2
    assert sub.x0 == (cp4.x0[0], cp4.x0[2])


def test_load_loose_file(loader):
    inst = loader.load_file(DATA / "cp3_spaces.dat")
    assert inst.n == 3
    assert inst.family == Family.CP


def test_load_fl_loose_file(loader):
    inst = loader.load_file(DATA / "fl2_tabs.dat")
    assert inst.n == 2
    assert inst.has_flight_levels
    assert inst.nf == 2


def test_loader_rejects_silently_truncated_or_partial_data(loader, tmp_path):
    invalid_n = tmp_path / "invalid_n.dat"
    invalid_n.write_text(
        "param n := 1.5;\nparam d := 0.05;\n"
        "param v0 :=\n1 5\n;\nparam cap :=\n1 0\n;\n"
    )
    with pytest.raises(Exception, match="positive integer"):
        loader.load_file(invalid_n)

    partial_coords = tmp_path / "partial_coords.dat"
    partial_coords.write_text(
        "param n := 1;\nparam d := 0.05;\n"
        "param v0 :=\n1 5\n;\nparam cap :=\n1 0\n;\n"
        "param x0 :=\n1 2\n;\n"
    )
    with pytest.raises(Exception, match="both be present"):
        loader.load_file(partial_coords)
