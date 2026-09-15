"""Playwright browser coverage of the scientific-simulator dashboard.

Each test asserts no page/console errors and a concrete behaviour. Snapshots are
avoided (they would depend on host-variable wall-times); assertions are on DOM
state, not pixels.
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


pytestmark = pytest.mark.skipif(not _browser_available(), reason="Playwright chromium not installed")


def _goto(page, *, expert=True):
    # The dashboard intentionally starts preview/preflight/job polling during
    # initialisation, so the page may never have a 500 ms "network idle" window
    # on a slower CI runner. Wait for the application state we actually require.
    page.goto(page._acrpq_base + "/", wait_until="domcontentloaded")
    page.wait_for_selector("#radar", timeout=10000)
    page.wait_for_function(
        "CURRENT && currentConfiguration.instance && LAST_PREFLIGHT",
        timeout=30000,
    )
    if expert:
        page.click("#experience-toggle")
        page.wait_for_function("document.body.classList.contains('expert-mode')")


def _no_errors(page):
    assert page._acrpq_errors == [], f"console/page errors: {page._acrpq_errors}"


def test_loads_offline(page):
    # block every non-localhost request -> proves the dashboard needs no CDN
    ext = []

    def route(r):
        u = r.request.url
        (r.continue_() if ("127.0.0.1" in u or "localhost" in u) else (ext.append(u), r.abort()))

    page.context.route("**/*", route)
    _goto(page)
    page.click('[data-tab="benchmark"]')
    page.wait_for_timeout(1000)
    assert ext == [], f"external requests: {ext}"
    _no_errors(page)


def test_demo_mode_is_the_clear_default_and_reveals_results(page):
    _goto(page, expert=False)
    # minimal default: instance + K (recommended K3) + one Run comparison button
    assert page.input_value("#ntheta") == "3"
    assert page.get_attribute("#experience-toggle", "aria-pressed") == "false"
    assert page.is_visible('[data-testid="demo-badge"]')
    assert page.is_visible("#demo-results-btn")  # the single primary action
    assert "Run comparison" in page.text_content("#demo-results-btn")
    assert page.is_visible("#ntheta")            # K is selectable in the default view
    assert page.is_hidden("#expert-controls")
    # every disclosed section is behind MORE DETAILS
    for tab in ("qubo", "benchmark", "explain", "qpuval"):
        assert page.is_hidden(f'[data-tab="{tab}"]')
    # Run comparison stays on the radar screen and reveals the unified table there
    page.click("#demo-results-btn")
    page.wait_for_function(
        "document.querySelector('#ex-auto-status').dataset.state === 'complete'",
        timeout=120000,
    )
    assert "active" in (page.get_attribute("#tab-radar", "class") or "")
    assert page.is_visible("#ex-algo-table")     # comparison lives on the main screen
    assert page.is_hidden("#ex-bit-table")       # expert-only detail stays hidden
    _no_errors(page)


def test_more_details_reveals_disclosed_sections_without_losing_features(page):
    # progressive disclosure: MORE DETAILS reveals Explain / Scientific / Evidence /
    # IBM QPU. Every previously-available section is still reachable — nothing removed.
    _goto(page, expert=False)
    for tab in ("explain", "qubo", "benchmark", "qpuval"):
        assert page.is_hidden(f'[data-tab="{tab}"]')
    page.click("#experience-toggle")
    page.wait_for_function("document.body.classList.contains('expert-mode')")
    for tab in ("explain", "qubo", "benchmark", "qpuval"):
        assert page.is_visible(f'[data-tab="{tab}"]')
    # the IBM QPU section is present and fail-closed (no real submission possible)
    page.click('[data-tab="qpuval"]')
    page.wait_for_selector("#tab-qpuval.active", timeout=5000)
    assert page.is_visible('[data-testid="qv-disabled-note"]') or \
        "dry-run" in (page.text_content("#qv-badge-mode") or "").lower()
    # leaving details returns to the minimal radar screen (no stranding on a hidden tab)
    page.click("#experience-toggle")
    page.wait_for_function("!document.body.classList.contains('expert-mode')")
    assert "active" in (page.get_attribute("#tab-radar", "class") or "")
    _no_errors(page)


def test_instances_are_grouped_by_algorithm_applicability(page):
    _goto(page, expert=False)
    page.wait_for_function("document.querySelectorAll('#instance optgroup').length === 2")
    labels = page.locator("#instance optgroup").evaluate_all(
        "groups => groups.map(group => group.label)"
    )
    assert labels[0].startswith("ALL ALGORITHMS (")
    assert labels[1].startswith("LIMITED / SOME SKIPPED (")
    assert page.locator('#instance optgroup[label^="ALL"] option[value="CP_4"]').count() == 1
    assert page.locator('#instance optgroup[label^="LIMITED"] option[value="CP_10"]').count() == 1
    assert "K=3" in page.text_content("#instance-classification-note")
    _no_errors(page)


def test_scenario_syncs_radar_and_qubo(page):
    _goto(page)
    page.select_option("#scenario", "Q2")  # CP_3 subset [1,2]
    page.wait_for_timeout(700)
    assert page.text_content("#m-n") == "2"
    page.click('[data-tab="qubo"]')
    page.wait_for_timeout(700)
    assert page.text_content("#qb-qubits") == "10"  # 2 aircraft x K=5
    _no_errors(page)


def test_reference_solve_and_timeline(page):
    _goto(page)
    page.select_option("#instance", "CP_4")
    page.wait_for_timeout(400)
    page.click("#solve-btn")
    page.wait_for_timeout(1500)
    assert page.text_content("#m-conf") == "0"
    assert page.text_content("#m-resolved") == "6 / 6"
    # timeline advances then pauses
    page.click("#sim-play")
    page.wait_for_timeout(600)
    page.click("#sim-play")
    _no_errors(page)


def test_scientific_profile_reports_seed_dispersion(page):
    _goto(page)
    page.select_option("#instance", "CP_4")
    page.wait_for_timeout(300)
    page.select_option("#profile", "scientific")
    assert page.is_visible("#seeds-row")
    page.fill("#seeds", "1,2,3")
    page.click("#solve-btn")
    page.wait_for_function(
        "document.querySelector('#status').textContent.includes('scientific run complete')",
        timeout=30000,
    )
    assert page.is_visible("#sci-box")
    assert "3/3" in page.text_content("#sci-feas")  # reference is feasible on CP_4
    assert page.text_content("#m-profile") == "scientific ✓"
    _no_errors(page)


def test_save_scientific_run_persists_and_reports_receipt(page):
    _goto(page)
    page.select_option("#instance", "CP_4")
    page.wait_for_timeout(300)
    page.select_option("#profile", "scientific")
    assert page.is_visible("#save-run-btn")
    page.fill("#seeds", "1,2")
    page.click("#save-run-btn")
    page.wait_for_function(
        "document.querySelector('#status').textContent.includes('SAVED')",
        timeout=30000,
    )
    assert "saved:" in page.text_content("#save-run-note")
    _no_errors(page)


def test_cancel_all_marks_running_routes_cancelled(page):
    _goto(page)
    # Drive the cancel path deterministically: seed a run with a running + a
    # queued route, then cancel. (A network-timed cancel would be racy.)
    page.evaluate(
        """async () => {
            AUTO_COMPARE = [
                { solver: 'reference', label: 'Ref', state: 'running' },
                { solver: 'qaoa', label: 'QAOA', state: 'pending' },
            ];
            document.getElementById('ex-cancel-all').classList.remove('hidden');
            await cancelAutoComparison();
        }"""
    )
    assert "cancelled" in page.text_content("#ex-auto-status")
    assert page.is_hidden("#ex-cancel-all")
    assert "cancelled" in page.text_content("#ex-algo-table")
    _no_errors(page)


def test_cancel_all_issues_a_real_backend_job_delete(page):
    # Prove cancellation hits the backend: a real job is submitted, its status is
    # held "running" so the run stays cancellable, and Cancel-all must issue a
    # real DELETE /api/jobs/<id> (not merely abandon a browser fetch).
    deletes = []
    page.on(
        "request",
        lambda r: deletes.append(r.url)
        if r.method == "DELETE" and "/api/jobs/" in r.url else None,
    )

    # Hold only the status GET as "running"; let submit (POST) and cancel (DELETE)
    # reach the real backend. "**/api/jobs/*" matches /api/jobs/<id> but not
    # /api/jobs/<id>/result (extra segment) nor the POST to /api/jobs.
    running = json.dumps({
        "id": "held", "state": "running", "cancel_requested": False,
        "elapsed_s": 1, "queued_for_s": 0, "error": None, "has_result": False,
    })

    def hold(route):
        if route.request.method == "GET":
            route.fulfill(status=200, content_type="application/json", body=running)
        else:
            route.continue_()

    _goto(page)
    page.route("**/api/jobs/*", hold)
    page.select_option("#scenario", "Q2")  # schedules the auto-comparison
    # the comparison panel (and its Cancel-all button) live on the radar screen
    page.wait_for_selector("#ex-cancel-all:not(.hidden)", timeout=30000)
    page.click("#ex-cancel-all")
    page.wait_for_function(
        "document.querySelector('#ex-auto-status').textContent.includes('cancelled')",
        timeout=15000,
    )
    assert deletes, "Cancel-all must DELETE the backend job, not just abort the fetch"
    assert page.is_hidden("#ex-cancel-all")
    _no_errors(page)


def test_manual_button_disabled_for_non_applicable_method(page):
    # CP_20 at the K=3 demo default: 3^20 one-hot states exceed the exact cap,
    # and the grid-limit clamp cannot silently move K (3 is already the minimum).
    _goto(page)
    page.select_option("#instance", "CP_20")
    page.wait_for_function(
        "LAST_PREFLIGHT && LAST_PREFLIGHT.instance === 'CP_20'", timeout=30000
    )
    page.select_option("#solver", "qubo-exact")
    # non-applicable: button disabled and the backend reason (with formula) shown
    assert page.is_disabled("#solve-btn")
    assert "non applicable" in page.text_content("#status") or "not applicable" in page.text_content("#status")
    assert "3^20" in page.text_content("#status")
    # switching to an always-applicable route re-enables the button
    page.select_option("#solver", "qubo-annealing")
    assert page.is_enabled("#solve-btn")
    _no_errors(page)


def test_export_full_comparison_json_is_not_citable(page):
    _goto(page)
    page.select_option("#scenario", "Q2")
    page.wait_for_function(
        "document.querySelector('#ex-auto-status').dataset.state === 'complete'",
        timeout=120000,
    )
    with page.expect_download() as dl:
        page.click("#cmp-json")  # export button lives with the comparison on the radar screen
    with open(dl.value.path(), encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["schema"] == "acrpq-comparison/1"
    assert data["citable"] is False  # the interactive comparison is never citable
    assert len(data["methods"]) == 5  # every local route is captured
    reference = next(method for method in data["methods"] if method["method"] == "reference")
    assert reference["wall_time_delta_vs_reference_s"] == pytest.approx(0.0)
    assert any("optimality gap" in c for c in data["caveats"])
    # the export carries the launch snapshot (config-id + instance), not live config
    assert data["config_id"] is not None
    assert isinstance(data["configuration"]["instance"], str) and data["configuration"]["instance"]
    _no_errors(page)


def test_comparison_html_export_escapes_backend_strings(page):
    _goto(page)
    # a backend-provided reason must never inject markup into the HTML export
    html_out = page.evaluate(
        """() => {
            AUTO_COMPARE = [{
                solver: 'qaoa', label: 'QAOA', family: 'sim', state: 'error',
                reason: '<img src=x onerror=alert(1)>', payload: null,
            }];
            return _comparisonHTML(_comparisonExport());
        }"""
    )
    assert "<img src=x" not in html_out
    assert "&lt;img src=x" in html_out
    _no_errors(page)


def test_switching_instance_clears_results_and_blocks_stale_export(page):
    _goto(page)
    # the auto-comparison runs on scenario select regardless of the active tab
    page.select_option("#scenario", "Q2")
    page.wait_for_function(
        "document.querySelector('#ex-auto-status').dataset.state === 'complete'",
        timeout=120000,
    )
    assert page.eval_on_selector_all("#ex-algo-table tr", "r => r.length") == 6
    # switch instance (from the default tab, sidebar visible) with the new preflight
    # held so the new comparison cannot finish
    held = []
    page.route("**/api/preflight", lambda r: held.append(r))
    page.select_option("#instance", "CP_5")
    # the old instance's algorithm/discrete rows are dropped immediately; the only data
    # row that may remain is the NEW whole-instance Gurobi baseline (pre-computed,
    # loaded simultaneously) — never a stale discrete/method row from the old config.
    page.wait_for_function(
        "Array.from(document.querySelectorAll('#ex-algo-table tr')).slice(1)"
        ".every(tr => tr.children[0].textContent.includes('AMPL / Gurobi'))",
        timeout=10000,
    )
    # exporting now must refuse — the finished comparison no longer matches config:
    # the buttons are disabled, and even a direct call refuses with an explanation
    assert page.is_disabled("#cmp-json")
    assert page.is_disabled("#cmp-csv")
    assert page.is_disabled("#cmp-html")
    page.evaluate("exportComparison('json')")  # bypass the disabled button on purpose
    assert "recomputing" in page.text_content("#status")
    for r in held:
        try:
            r.continue_()
        except Exception:
            pass
    _no_errors(page)


def test_manual_button_ignores_previous_instance_preflight(page):
    _goto(page)
    # CP_20 at the K=3 demo default: qubo-exact over 3^20 states is too large ->
    # not applicable (and K=3 is the slider minimum, so no silent clamping).
    page.select_option("#instance", "CP_20")
    page.wait_for_function(
        "LAST_PREFLIGHT && LAST_PREFLIGHT.instance === 'CP_20'", timeout=30000
    )
    page.select_option("#solver", "qubo-exact")
    assert page.is_disabled("#solve-btn")
    # switch to a compatible instance with its preflight held: the button must not
    # stay disabled on the *previous* instance's (stale) preflight
    held = []
    page.route("**/api/preflight", lambda r: held.append(r))
    page.select_option("#instance", "CP_4")
    page.wait_for_function(
        "document.querySelector('#solve-btn').disabled === false", timeout=10000
    )
    for r in held:
        try:
            r.continue_()
        except Exception:
            pass
    _no_errors(page)


def test_preview_out_of_order_renders_only_latest_config(page):
    _goto(page)  # loads CP_4 by default; set the route afterwards
    held = []

    def handler(route):
        body = route.request.post_data_json or {}
        if body.get("instance") == "CP_5":
            held.append(route)  # hold instance A's preview
        else:
            route.continue_()

    page.route("**/api/preview", handler)
    page.select_option("#instance", "CP_5")  # A: held
    page.wait_for_timeout(300)
    page.select_option("#instance", "CP_4")  # B: resolves first
    page.wait_for_function(
        "document.querySelector('#m-name').textContent === 'CP_4'", timeout=15000
    )
    for r in held:  # release A's late response — it must be ignored
        try:
            r.continue_()
        except Exception:
            pass
    page.wait_for_timeout(600)
    assert page.text_content("#m-name") == "CP_4"
    _no_errors(page)


def test_before_after_and_maneuver_table(page):
    _goto(page)
    page.select_option("#instance", "CP_4")
    page.wait_for_timeout(300)
    page.click("#solve-btn")
    page.wait_for_timeout(1500)
    assert page.is_visible("#compare-box")
    rows = page.eval_on_selector_all("#man-table tr[data-id]", "e=>e.length")
    assert rows == 4
    page.click("#ba-before")
    # "Before" shows only the initial (red) trajectories
    assert page.is_checked("#tg-initial")
    assert not page.is_checked("#tg-continuous")
    assert not page.is_checked("#tg-discrete")
    _no_errors(page)


def test_explain_tab_traces_resources_energy_and_bits(page):
    _goto(page)
    page.select_option("#instance", "CP_4")
    page.wait_for_timeout(300)
    page.click("#solve-btn")
    page.wait_for_timeout(1500)
    page.click("#explain-run-btn")
    page.wait_for_timeout(700)
    assert "qubits" in page.text_content("#ex-raw-q")
    assert "Identity verified" in page.text_content("#ex-energy-check")
    assert page.text_content("#ex-onehot") == "yes ✓"
    assert page.eval_on_selector_all("#ex-bit-table tr", "rows => rows.length") == 5
    # proof status is shown and, for the exact reference solve, is the optimum
    assert "exact" in page.text_content("#ex-proof").lower()
    assert page.text_content("#ex-dom-ac").startswith("A")
    _no_errors(page)


def test_instance_selection_runs_all_safe_local_algorithms(page):
    _goto(page)
    page.select_option("#scenario", "Q2")  # 2 aircraft / 10 qubits: every local route fits
    page.click('[data-tab="explain"]')
    page.wait_for_function(
        "document.querySelector('#ex-auto-status').dataset.state === 'complete'",
        timeout=120000,
    )
    rows = page.eval_on_selector_all("#ex-algo-table tr", "rows => rows.length")
    assert rows == 6  # header + five local algorithms
    assert "0 en échec" in page.text_content("#ex-auto-status")
    _no_errors(page)


def test_qaoa_over_capacity_disables_button_instead_of_submitting(page):
    # CP_10 at default K=3 is a single 30-qubit component: local QAOA does not fit. The
    # preflight-driven guard must disable the manual button (with the reason) and
    # never submit a solve — replacing the old post-hoc capacity error banner.
    _goto(page)
    page.select_option("#instance", "CP_10")
    page.wait_for_function(
        "LAST_PREFLIGHT && LAST_PREFLIGHT.instance === 'CP_10'", timeout=30000
    )
    page.select_option("#solver", "qaoa")
    assert page.is_disabled("#solve-btn")
    assert "non applicable" in page.text_content("#status") or "not applicable" in page.text_content("#status")
    assert "ceiling" in page.text_content("#status")
    assert not page.is_visible("#err-box")  # nothing was submitted


def test_presentation_mode(page):
    _goto(page)
    page.click("#present-btn")
    page.wait_for_timeout(400)
    assert page.eval_on_selector("body", "b=>b.classList.contains('presenting')")
    page.click("#ps-exit")
    page.wait_for_timeout(200)
    assert page.eval_on_selector("body", "b=>!b.classList.contains('presenting')")
    _no_errors(page)


def test_responsive_no_hscroll(page):
    for w, h in [(1440, 900), (1280, 720), (768, 1024), (390, 844)]:
        page.set_viewport_size({"width": w, "height": h})
        _goto(page)
        hscroll = page.evaluate(
            "document.documentElement.scrollWidth > document.documentElement.clientWidth + 2"
        )
        assert not hscroll, f"horizontal scroll at {w}x{h}"


# ---- Original AMPL / Gurobi baseline in the unified RESULTS chain ---- #
def _select_instance(page, name):
    page.select_option("#instance", name)
    page.wait_for_function(f"currentConfiguration.instance === '{name}'")


def _wait_autorun(page):
    # the auto-comparison has finished when every route is terminal (not pending/running)
    page.wait_for_function(
        "AUTO_COMPARE.length > 0 && AUTO_COMPARE.every(r => "
        "['ok','skipped','error','cancelled'].includes(r.state))", timeout=90000)


def _chain_rows(page):
    # returns [[cellText,...], ...] of the unified #ex-algo-table
    return page.evaluate(
        "Array.from(document.querySelectorAll('#ex-algo-table tr')).slice(1)"
        ".map(tr => Array.from(tr.children).map(td => td.textContent))")


def test_cp4_k3_unified_chain_gurobi_discrete_and_methods(page):
    _goto(page)  # default CP_4, K3
    page.wait_for_function("AMPL_GUROBI_STATE === 'ready'", timeout=30000)
    _select_instance(page, "CP_4")
    _wait_autorun(page)
    rows = _chain_rows(page)
    methods = [r[0] for r in rows]
    # one unified table: Gurobi continuous + discrete K3 + executed algorithms
    assert any("AMPL / Gurobi" in m for m in methods)
    assert any("Référence discrète K3" in m for m in methods)
    assert any("QUBO" in m or "annealing" in m or "QAOA" in m for m in methods)
    # NO K5/K7 rows in the main table (sensitivity lives elsewhere)
    assert not any("K5" in m or "K7" in m for m in methods)
    # decomposition line present: continuous + discretisation + algorithmic = objective
    decomp = page.text_content("#ex-chain-decomp")
    assert "discretisation" in decomp and "=" in decomp
    # the Gurobi row carries the certified guarantee and never claims "exact"
    grow = next(r for r in rows if "AMPL / Gurobi" in r[0])
    assert "certified" in grow[4].lower()          # guarantee cell
    assert "exact" not in " ".join(grow).lower()
    control_rows = page.eval_on_selector_all("#control-diff-table tr", "rows => rows.length")
    assert control_rows == 5  # header + one continuous/discrete control row per aircraft
    controls = page.text_content("#control-diff-table")
    assert "Gurobi continu" in controls and "Discrète K3" in controls
    assert page.is_visible("#control-diff")
    for layer in ("initial", "continuous", "discrete"):
        assert page.is_visible(f"#tg-{layer}") and page.is_checked(f"#tg-{layer}")
    page.uncheck("#tg-continuous")
    assert page.evaluate("SIM.toggles.continuous") is False
    page.select_option("#tg-motion", "continuous")
    assert page.evaluate("SIM.toggles.motion") == "continuous"
    page.select_option("#tg-motion", "initial")
    assert page.evaluate("SIM.toggles.motion") == "initial"
    _no_errors(page)


def test_grid_slider_max_tracks_selected_instance(page):
    _goto(page, expert=False)
    page.wait_for_function("document.querySelector('#k-limit-note').textContent.includes('technical max: K7')")
    assert "technical max: K7" in page.text_content("#k-limit-note")
    page.select_option("#instance", "CP_3")
    page.wait_for_function("document.querySelector('#ntheta').max === '9'")
    assert "technical max: K9" in page.text_content("#k-limit-note")
    page.select_option("#instance", "CP_5")
    page.wait_for_function("document.querySelector('#ntheta').max === '5'")
    assert "technical max: K5" in page.text_content("#k-limit-note")
    _no_errors(page)


def test_cp4_k3_delta_arithmetic_is_consistent(page):
    _goto(page)
    page.wait_for_function("AMPL_GUROBI_STATE === 'ready'", timeout=30000)
    _select_instance(page, "CP_4")
    _wait_autorun(page)
    # Pull structured numbers straight from the JS state (avoids parsing formatted cells)
    data = page.evaluate("""() => {
      const g = AMPL_GUROBI.instances.find(x => x.instance === currentConfiguration.instance);
      const ref = AUTO_COMPARE.find(r => r.solver==='reference' && r.state==='ok');
      const disc = ref && ref.payload.solution.objective;
      const algo = AUTO_COMPARE.filter(r => r.solver!=='reference' && r.state==='ok'
                     && r.payload.solution.feasible)
                   .map(r => ({label:r.label, obj:r.payload.solution.objective}));
      return {g: g.original_ampl_gurobi_objective, official: g.is_official_anchor, disc, algo};
    }""")
    disc_cost = data["disc"] - data["g"]           # discretisation cost = discrete − Gurobi
    for a in data["algo"]:
        alg = a["obj"] - data["disc"]              # algorithmic gap = method − discrete
        total = a["obj"] - data["g"]               # total continuous gap = method − Gurobi
        assert abs(total - (disc_cost + alg)) < 1e-9   # total ≈ discretisation + algorithmic
    _no_errors(page)


def test_cp10_k3_gurobi_differences_are_provisional(page):
    _goto(page)
    page.wait_for_function("AMPL_GUROBI_STATE === 'ready'", timeout=30000)
    _select_instance(page, "CP_10")
    _wait_autorun(page)
    rows = _chain_rows(page)
    grow = next(r for r in rows if "AMPL / Gurobi" in r[0])
    assert "Provisional" in grow[-1] or "provisoire" in grow[4].lower()  # status/guarantee
    # provisional Gurobi -> method "Δ vs continuous Gurobi" is provisional / N/A, never official
    method_rows = [r for r in rows if "grid K10" in r[1] and "Discrete" not in r[0]]
    for r in method_rows:
        assert r[7] in ("provisoire", "N/A", "—")   # Δ vs continuous Gurobi column
    # the provisional Gurobi row never advertises a certified optimum
    assert "optimum" not in grow[-1].lower()
    _no_errors(page)


def test_switch_cp4_to_cp10_invalidates_the_whole_table(page):
    _goto(page)
    page.wait_for_function("AMPL_GUROBI_STATE === 'ready'", timeout=30000)
    _select_instance(page, "CP_4")
    _wait_autorun(page)
    assert any("Référence discrète K3" in r[0] for r in _chain_rows(page))
    _select_instance(page, "CP_10")
    # the table must not keep CP_4's certified anchor while CP_10 recomputes: the
    # Gurobi row's status flips to provisional (CP_10 is a timeout instance)
    page.wait_for_function(
        "Array.from(document.querySelectorAll('#ex-algo-table tr'))"
        ".some(tr => tr.children[0] && tr.children[0].textContent.includes('AMPL / Gurobi')"
        " && /provisoire|provisional/i.test(tr.children[tr.children.length-1].textContent))",
        timeout=30000)
    grow = next((r for r in _chain_rows(page) if "AMPL / Gurobi" in r[0]), None)
    if grow:
        assert "Référence numérique certifiée" not in grow[-1]
    _no_errors(page)


def test_gurobi_baseline_row_readable_on_narrow_viewport(page):
    # the Gurobi baseline now lives as a row in the unified table (the standalone
    # panel was removed); it must still be present and cause no horizontal scroll
    page.set_viewport_size({"width": 390, "height": 844})
    _goto(page, expert=False)
    page.wait_for_function("AMPL_GUROBI_STATE === 'ready'", timeout=30000)
    _select_instance(page, "CP_4")
    _wait_autorun(page)
    assert any("AMPL / Gurobi" in r[0] for r in _chain_rows(page))
    hscroll = page.evaluate(
        "document.documentElement.scrollWidth > document.documentElement.clientWidth + 2")
    assert not hscroll
    _no_errors(page)
