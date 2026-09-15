"""Playwright coverage of the Phase-8 QPU Validation workspace.

Uses the QPU-enabled server (``qpu_page``) for the prepare/confirm journey and the
default disabled server (``page``) for the disabled case. Never submits a real job.
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


pytestmark = pytest.mark.skipif(not _browser_available(),
                                reason="Playwright chromium not installed")

CONSENT = "I understand this may submit a real, billable IBM Quantum job"


def _open_qpu(pg, *, instance="CP_3"):
    pg.goto(pg._acrpq_base + "/", wait_until="domcontentloaded")
    pg.wait_for_selector("#radar", timeout=10000)
    # wait until init() has run (config populated + experience mode applied)
    pg.wait_for_function(
        "typeof CURRENT !== 'undefined' && currentConfiguration && currentConfiguration.instance"
        " && $('experience-toggle').getAttribute('aria-pressed') !== null", timeout=30000)
    # reveal expert-only tabs, then open the QPU validation tab
    if pg.get_attribute("#experience-toggle", "aria-pressed") == "false":
        pg.click("#experience-toggle")
    pg.wait_for_selector('.tab[data-tab="qpuval"]', state="visible", timeout=10000)
    pg.click('.tab[data-tab="qpuval"]')
    pg.wait_for_function("$('qv-badge-mode') && $('qv-badge-mode').textContent !== '—'", timeout=15000)
    pg.select_option("#qv-instance", instance)
    return pg


def _no_errors(pg):
    assert pg._acrpq_errors == [], f"page/console errors: {pg._acrpq_errors}"


def _assess_dryrun_prepare(pg):
    pg.click("#qv-assess-btn")
    pg.wait_for_function("QV_DECISION_E === QV_EPOCH", timeout=15000)
    pg.wait_for_function("QV_BASELINES_E === QV_EPOCH", timeout=120000)
    pg.click("#qv-dryrun-btn")
    pg.wait_for_function("QV_DRYRUN_E === QV_EPOCH", timeout=15000)
    pg.click("#qv-prepare-btn")
    pg.wait_for_function("QV_RUN_ID !== null && QV_PREPARED_E === QV_EPOCH", timeout=20000)


# --------------------------------------------------------------------------- #
def test_qpu_disabled_when_flags_off(page):
    _open_qpu(page)
    assert page.is_visible("#qv-disabled-note")
    assert "DISABLED" in page.text_content("#qv-badge-mode")
    # prepare is refused client-side (disabled) when the workflow is off
    assert page.is_disabled("#qv-prepare-btn")
    _no_errors(page)


def test_qpu_backend_ranking_and_fake_marked(qpu_page):
    _open_qpu(qpu_page)
    qpu_page.click("#qv-assess-btn")
    qpu_page.wait_for_function("QV_DECISION_E === QV_EPOCH", timeout=15000)
    qpu_page.wait_for_function("QV_BASELINES_E === QV_EPOCH", timeout=120000)
    # ranking table has a row for the fake backend; selection matches
    assert "fake_lagos" in qpu_page.text_content("#qv-ranking-table")
    assert qpu_page.text_content("#qv-selected-backend").strip() == "fake_lagos"
    # the selected badge is explicitly a fake backend
    assert qpu_page.get_attribute("#qv-badge-selected", "data-kind") == "fake-backend"
    comparison = qpu_page.text_content("#qv-comparison-table")
    assert "Discrete reference" in comparison and "QAOA Aer" in comparison
    assert qpu_page.locator("#qv-consent-input").count() == 0
    assert not qpu_page.is_disabled("#qv-dryrun-btn")
    _no_errors(qpu_page)


def test_qpu_dry_run_never_submittable(qpu_page):
    _open_qpu(qpu_page)
    qpu_page.click("#qv-assess-btn")
    qpu_page.wait_for_function("QV_DECISION_E === QV_EPOCH", timeout=15000)
    qpu_page.wait_for_function("QV_BASELINES_E === QV_EPOCH", timeout=120000)
    qpu_page.click("#qv-dryrun-btn")
    qpu_page.wait_for_function("QV_DRYRUN_E === QV_EPOCH", timeout=15000)
    assert qpu_page.text_content("#qv-dryrun-submittable").strip() == "false"
    assert "NO GATEWAY CALLED" in qpu_page.text_content("#qv-badge-dryrun")
    _no_errors(qpu_page)


def test_qpu_prepare_confirm_journey(qpu_page):
    _open_qpu(qpu_page)
    _assess_dryrun_prepare(qpu_page)
    assert qpu_page.text_content("#qv-run-id").strip() not in ("", "—")
    # confirm with the exact consent phrase -> awaiting_confirmation
    qpu_page.click("#qv-confirm-btn")
    qpu_page.wait_for_function(
        "$('qv-run-state').textContent === 'awaiting_confirmation'", timeout=15000)
    # events table is a real, non-empty projection
    assert "transition" in qpu_page.text_content("#qv-events-table")
    _no_errors(qpu_page)


def test_qpu_submit_button_hidden_and_disabled(qpu_page):
    _open_qpu(qpu_page)
    _assess_dryrun_prepare(qpu_page)
    qpu_page.click("#qv-confirm-btn")
    qpu_page.wait_for_function(
        "$('qv-run-state').textContent === 'awaiting_confirmation'", timeout=15000)
    # submission is withheld by the flags: the button stays hidden AND disabled
    assert qpu_page.is_hidden("#qv-submit-btn")
    assert qpu_page.get_attribute("#qv-submit-btn", "disabled") is not None
    _no_errors(qpu_page)


def test_qpu_stale_state_on_config_change(qpu_page):
    _open_qpu(qpu_page)
    _assess_dryrun_prepare(qpu_page)
    # change an angle after prepare -> derived state invalidated, stale banner shown
    qpu_page.fill("#qv-gamma", "0.9")
    qpu_page.dispatch_event("#qv-gamma", "change")
    assert qpu_page.is_visible("#qv-stale-banner")
    assert qpu_page.is_disabled("#qv-confirm-btn")     # cannot confirm a stale prepare
    _no_errors(qpu_page)


def test_qpu_export_downloads_verifiable_json(qpu_page):
    _open_qpu(qpu_page)
    _assess_dryrun_prepare(qpu_page)
    with qpu_page.expect_download() as dl:
        qpu_page.click("#qv-export-json")
    path = dl.value.path()
    export = json.loads(open(path).read())
    assert export["run"]["local_run_id"] == qpu_page.text_content("#qv-run-id").strip()
    assert export["artefact"] is not None and export["plan"] is not None
    _no_errors(qpu_page)
