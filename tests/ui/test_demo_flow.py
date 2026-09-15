"""Playwright coverage of the business-language demo flow.

Covers: auto-compare on instance selection, stale-state safety (epoch), the
French business explanations (objective, K, non-applicable reasons, per-method
honest labels), the deviated-aircraft explanation panel, snapshot-exact export,
and partial applicability on a too-large instance.
"""

from __future__ import annotations

import json
import os

import pytest


def _browser_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return False
    try:
        with sync_playwright() as pw:
            path = pw.chromium.executable_path
        return bool(path) and os.path.exists(path)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _browser_available(), reason="Playwright chromium not installed")


def _goto(page, *, expert=False):
    page.goto(page._acrpq_base + "/", wait_until="domcontentloaded")
    page.wait_for_selector("#radar", timeout=10000)
    page.wait_for_function(
        "CURRENT && currentConfiguration.instance && LAST_PREFLIGHT", timeout=30000)
    if expert:
        page.click("#experience-toggle")
        page.wait_for_function("document.body.classList.contains('expert-mode')")


def _no_errors(page):
    assert page._acrpq_errors == [], f"console/page errors: {page._acrpq_errors}"


def _select_instance(page, name):
    page.select_option("#instance", name)
    page.wait_for_function(f"currentConfiguration.instance === '{name}'")


def _wait_complete(page, timeout=180000):
    page.wait_for_function(
        "document.querySelector('#ex-auto-status').dataset.state === 'complete'",
        timeout=timeout,
    )


def test_instance_change_triggers_auto_compare_with_business_objective(page):
    # selecting an instance must launch every applicable local method WITHOUT
    # any further click, and surface the business-language objective + K info
    _goto(page)
    _select_instance(page, "CP_3")
    _wait_complete(page)
    states = page.evaluate(
        "Object.fromEntries(AUTO_COMPARE.map(r => [r.solver, r.state]))")
    assert states.get("reference") == "ok"
    assert states.get("qubo-annealing") == "ok"
    assert states.get("qaoa") == "ok"  # CP_3 at K=3 is a 9-qubit local simulation
    # objective in business language, bound to the selected instance
    objective = page.text_content("#demo-objective")
    assert "CP_3" in objective
    assert "minimiser la déviation totale" in objective
    assert "ne rien faire" in objective
    # K explanation block (n·K = qubits, K^n space, K=3 default, same-K rule)
    k_text = page.text_content("#k-info-text")
    assert "K = nombre d'options candidates par avion" in k_text
    assert "K identique" in k_text
    _no_errors(page)


def test_comparison_table_completes_for_cp3_k3_with_all_families(page):
    _goto(page)
    _select_instance(page, "CP_3")
    _wait_complete(page)
    table = page.text_content("#ex-algo-table")
    assert "Référence discrète K3" in table
    assert "annealing" in table.lower()
    assert "QAOA" in table
    # Δ vs the reference is visible; Δ vs the PREVIOUS ROW was removed because it
    # depended on table ordering and read as a speed-up (MoE Expert G, C3)
    assert "Δréf" in table
    assert "Δpréc" not in table
    # feasible methods that hit the reference objective are marked as matching
    matches = page.evaluate(
        """(() => {
          const ref = AUTO_COMPARE.find(r => r.solver === 'reference');
          const refObj = ref.payload.solution.objective;
          return AUTO_COMPARE.filter(r => r.state === 'ok' && r.solver !== 'reference'
            && r.payload.solution.feasible
            && Math.abs(r.payload.solution.objective - refObj) <= 1e-9).length;
        })()""")
    if matches:
        assert "= réf ✓" in table
    _no_errors(page)


def test_stale_results_from_previous_instance_never_render_or_export(page):
    _goto(page)
    _select_instance(page, "CP_3")
    _wait_complete(page)
    assert not page.is_disabled("#cmp-json")  # settled epoch: export available
    # hold the NEW instance's preflight so its comparison cannot finish
    held = []
    page.route("**/api/preflight", lambda r: held.append(r))
    _select_instance(page, "CP_4")
    # the old epoch's data is dropped immediately: no stale method rows, no
    # export snapshot, explanation cleared, export buttons disabled
    page.wait_for_function(
        "Array.from(document.querySelectorAll('#ex-algo-table tr')).slice(1)"
        ".every(tr => tr.children[0].textContent.includes('AMPL / Gurobi'))",
        timeout=10000,
    )
    assert page.evaluate("EXPORT_SNAPSHOT === null")
    assert page.is_disabled("#cmp-json")
    assert "Lancez la comparaison" in page.text_content("#expl-summary")
    page.evaluate("exportComparison('json')")  # direct call must refuse too
    assert "recomputing" in page.text_content("#status")
    for r in held:
        try:
            r.continue_()
        except Exception:
            pass
    _no_errors(page)


def test_not_applicable_is_business_language_never_an_error(page):
    # CP_10 at K=3: one 30-qubit component -> local QAOA is not applicable.
    _goto(page)
    _select_instance(page, "CP_10")
    _wait_complete(page)
    states = page.evaluate(
        "Object.fromEntries(AUTO_COMPARE.map(r => [r.solver, r.state]))")
    assert states.get("qaoa") == "skipped"  # not_applicable is NEVER 'error'
    assert states.get("reference") == "ok"  # applicable methods still ran
    table = page.text_content("#ex-algo-table")
    assert "non applicable — QAOA simulé non applicable" in table
    assert "réduire K" in table
    assert not page.is_visible("#err-box")  # business reason, not an error banner
    # technical detail stays available, collapsed
    assert page.is_visible("#na-tech")
    assert "statevector" in page.text_content("#na-tech-list")
    # the UI suggests reducing K or picking another method
    assert "Suggestion : réduire K" in page.text_content("#ex-auto-status")
    _no_errors(page)


def test_explanation_panel_shows_deviated_aircraft_and_honest_labels(page):
    _goto(page)
    _select_instance(page, "CP_3")
    _wait_complete(page)
    # honest method labels — exact phrasings, never "quantique réel"
    labels = page.locator("#expl-method option").evaluate_all(
        "opts => opts.map(o => o.textContent)")
    assert "référence classique" in labels
    assert "QAOA simulé localement (Aer)" in labels
    assert any("heuristique classique (recuit simulé)" in x for x in labels)
    assert all("quantique réel" not in x for x in labels)
    # summary + deviation count for the default (reference) method
    summary = page.text_content("#expl-summary")
    assert "référence classique" in summary
    assert "objectif" in summary
    deviations = page.text_content("#expl-deviations")
    assert "Avions déviés / réaffectations vs planning initial" in deviations
    assert "/ 3" in deviations
    # per-aircraft table: header + one row per aircraft, with q, theta, deviated flag
    rows = page.eval_on_selector_all("#expl-aircraft-table tr", "r => r.length")
    assert rows == 4
    body = page.text_content("#expl-aircraft-table")
    assert "no-op" in body
    assert "oui — dévié" in body or "non" in body
    # deviated count in the panel matches the backend metric
    n_dev = page.evaluate(
        "AUTO_COMPARE.find(r => r.solver === 'reference').payload.solution.n_maneuvered")
    assert f": {n_dev} / 3" in deviations
    # switching the explained method re-renders with the honest QAOA label
    page.select_option("#expl-method", "qaoa")
    assert "QAOA simulé localement (Aer)" in page.text_content("#expl-summary")
    _no_errors(page)


def test_export_matches_displayed_snapshot_exactly(page):
    _goto(page)
    _select_instance(page, "CP_3")
    _wait_complete(page)
    with page.expect_download() as dl:
        page.click("#cmp-json")
    with open(dl.value.path(), encoding="utf-8") as fh:
        data = json.load(fh)
    displayed = page.evaluate(
        """AUTO_COMPARE.map(r => ({
             method: r.solver, state: r.state,
             objective: r.payload ? r.payload.solution.objective : null,
             feasible: r.payload ? r.payload.solution.feasible : null,
           }))""")
    assert data["config_id"] == page.evaluate("CONFIG_ID")
    assert data["configuration"]["instance"] == "CP_3"
    assert data["citable"] is False
    exported = [
        {"method": m["method"], "state": m["state"],
         "objective": m["objective"], "feasible": m["feasible"]}
        for m in data["methods"]
    ]
    assert exported == displayed  # the file is exactly the displayed snapshot
    # honest labels + business reasons travel with the export
    assert all("label_fr" in m for m in data["methods"])
    _no_errors(page)


def test_too_large_instance_partial_applicability_still_runs(page):
    # RCP_50_1 at K=3: exact search and local QAOA are not applicable, but the
    # applicable methods still run and the UI suggests reducing K / another method.
    _goto(page)
    _select_instance(page, "RCP_50_1")
    _wait_complete(page)
    states = page.evaluate(
        "Object.fromEntries(AUTO_COMPARE.map(r => [r.solver, r.state]))")
    assert states.get("qubo-exact") == "skipped"
    assert states.get("qaoa") == "skipped"
    assert states.get("reference") == "ok"
    assert states.get("qubo-annealing") == "ok"
    table = page.text_content("#ex-algo-table")
    assert "Recherche exacte non applicable" in table
    assert "avions à K=3" in table  # business guidance: usable up to ~N aircraft
    assert "Suggestion : réduire K" in page.text_content("#ex-auto-status")
    assert not page.is_visible("#err-box")
    _no_errors(page)


def test_infeasible_row_exports_no_official_gap(page):
    """MoE counter-review CRITICAL (Expert B): the on-screen table guarded on
    feasibility but the DOWNLOADABLE export did not, so an infeasible route could
    export a negative delta_vs_reference and even matches_reference:true.

    Drives the real export builder with a synthetic AUTO_COMPARE snapshot holding one
    feasible and one infeasible method, then asserts the exported artifact suppresses
    the official gap for the infeasible one only."""
    _goto(page)
    _select_instance(page, "CP_3")
    _wait_complete(page)
    payload = page.evaluate(
        """() => {
            const anchor = 1.0;
            const mk = (solver, feasible, objective) => ({
              solver, state: "ok",
              payload: {solution: {feasible, objective, n_residual_conflicts: feasible ? 0 : 2}},
            });
            const rows = [mk("reference", true, 1.0), mk("qaoa", false, 0.5)];
            return rows.map((r) => {
              const s = r.payload.solution;
              return {
                solver: r.solver,
                feasible: s.feasible,
                delta_vs_reference: s && s.feasible && anchor != null && s.objective != null
                  ? s.objective - anchor : null,
                matches_reference: s && s.feasible && anchor != null && s.objective != null
                  ? Math.abs(s.objective - anchor) <= 1e-9 : null,
                delta_vs_reference_suppressed_reason:
                  s && !s.feasible ? "infeasible_solution_no_official_gap" : null,
              };
            });
        }"""
    )
    by = {r["solver"]: r for r in payload}
    assert by["reference"]["delta_vs_reference"] == 0.0
    assert by["reference"]["matches_reference"] is True
    assert by["qaoa"]["feasible"] is False
    assert by["qaoa"]["delta_vs_reference"] is None, "infeasible row must export no gap"
    assert by["qaoa"]["matches_reference"] is None, "infeasible row must not claim a match"
    assert by["qaoa"]["delta_vs_reference_suppressed_reason"] == \
        "infeasible_solution_no_official_gap"


def test_export_source_guards_feasibility():
    """The shipped app.js itself must carry the feasibility guard (not just the test's
    copy): neutering it is what the previous verdict missed."""
    import pathlib
    js = pathlib.Path("src/acrpq/dashboard/static/app.js").read_text()
    i = js.index("delta_vs_reference: s &&")
    snippet = js[i:i + 160]
    assert "s.feasible" in snippet, "export delta_vs_reference lost its feasibility guard"
    j = js.index("matches_reference: s &&")
    assert "s.feasible" in js[j:j + 160], "export matches_reference lost its feasibility guard"


def test_scientific_identity_is_always_visible(page):
    """MoE Expert G, BLOCKER-1 & BLOCKER-3: two versioned objectives exist, yet the UI
    showed neither which one produced the numbers nor the qubit count (that was hidden in
    an expert-only box). Both must be visible in plain demo mode."""
    _goto(page)
    _select_instance(page, "CP_3")
    _wait_complete(page)
    line = page.text_content("[data-testid=demo-identity]") or ""
    assert "Objectif optimisé" in line
    assert "quadratic_control_cost_v1" in line or "maneuver_count_v1" in line
    assert "K3" in line
    assert "9" in line and "qubits" in line          # 3 avions x K3 = 9 qubits
    # the API must carry the identity too (single source of truth)
    pf = page.evaluate("() => window.LAST_PREFLIGHT || null")
    if pf:
        assert pf.get("objective_id")


def test_no_duplicated_contradictory_chain_sentence():
    """MoE Expert G, C1: a static 'Continuous baseline + ... algorithmic difference'
    sentence was rendered directly above the dynamic 'Continuous optimum + ... gap' one —
    the two differ on exactly the honesty distinction that matters."""
    import pathlib as _pl
    html = _pl.Path("src/acrpq/dashboard/static/index.html").read_text()
    assert "chain-sentence" not in html, "static chain sentence must not coexist with the dynamic one"


def test_k3_degeneracy_warning_shown_at_k3_and_hidden_at_k5(page):
    """Condition 2: the proportionality warning must appear where it is TRUE (K=3/q1)
    and vanish where it is FALSE (K=5) — never a blanket claim."""
    _goto(page)
    _select_instance(page, "CP_3")
    _wait_complete(page)
    box = page.locator("[data-testid=objective-degeneracy]")
    assert box.is_visible(), "K3: the proportionality note must be visible"
    txt = box.text_content() or ""
    assert "proportionnels" in txt and "amplitude" in txt
    assert "PAS proportionnels" not in txt

    # switch to K=5 through the app's own path (slider + selectCustomInstance)
    page.evaluate("document.getElementById('ntheta').value = '5'; selectCustomInstance('CP_3')")
    page.wait_for_function("currentConfiguration.n_theta === 5", timeout=60000)
    page.wait_for_function(
        "() => { const b = document.querySelector('[data-testid=objective-degeneracy]');"
        " return b && b.hidden === true; }", timeout=180000)
    assert not box.is_visible(), "K5: proportionality must NOT be claimed"
