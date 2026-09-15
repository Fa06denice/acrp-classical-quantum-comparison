"""Campaign helpers — licence expiry, git clarity, grid-gated discretisation costs."""

from __future__ import annotations

from datetime import datetime, timezone

from acrpq.classical.ampl_campaign import (
    check_license_not_expired,
    discretisation_costs,
    git_info,
)


def test_license_expiry_gate() -> None:
    before = datetime(2026, 8, 16, tzinfo=timezone.utc)
    on = datetime(2026, 9, 15, tzinfo=timezone.utc)
    after = datetime(2026, 9, 16, tzinfo=timezone.utc)
    assert check_license_not_expired("2026-09-15", now=before) == (True, "2026-08-16")
    assert check_license_not_expired("2026-09-15", now=on)[0] is False   # expiry day => not ok
    assert check_license_not_expired("2026-09-15", now=after)[0] is False


def test_git_info_reports_scientific_and_repository_separately() -> None:
    g = git_info()
    assert set(g) >= {"git_commit", "scientific_code_dirty", "scientific_code_paths",
                      "repository_dirty"}
    assert "src" in g["scientific_code_paths"] and "Code" in g["scientific_code_paths"]


def _artifact(*, anchored, discrete):
    return {"is_official_anchor": anchored, "rescored_objective": 6.25e-4,
            "discrete_references": {"K3_q1": discrete}}


def _disc(*, proven, obj, status="optimal", within=True):
    return {"grid_key": "K3_q1", "objective": obj, "exhaustive_optimality_proven": proven,
            "status": status, "within_search_cap": within, "search_space_size": 81,
            "mode": "exact" if within else "heuristic"}


def test_official_discretisation_delta_only_for_anchor_and_proven_grid() -> None:
    ok = discretisation_costs(_artifact(anchored=True, discrete=_disc(proven=True, obj=0.41)))
    c = ok["discretisation_cost_K3_q1"]
    assert c["official"] is True
    assert abs(c["value"] - (0.41 - 6.25e-4)) < 1e-12
    assert c["grid_key"] == "K3_q1"


def test_no_official_delta_when_not_anchor() -> None:
    c = discretisation_costs(_artifact(anchored=False, discrete=_disc(proven=True, obj=0.41)))
    assert c["discretisation_cost_K3_q1"]["official"] is False
    assert c["discretisation_cost_K3_q1"]["value"] is None


def test_no_official_delta_when_grid_over_cap_not_proven() -> None:
    disc = _disc(proven=False, obj=0.41, status="feasible", within=False)
    c = discretisation_costs(_artifact(anchored=True, discrete=disc))
    assert c["discretisation_cost_K3_q1"]["official"] is False
    assert "not an exhaustively-proven optimum" in c["discretisation_cost_K3_q1"]["reason"]
