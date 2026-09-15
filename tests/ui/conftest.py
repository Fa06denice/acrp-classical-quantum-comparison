"""Fixtures for the Playwright browser tests.

Launches the dashboard server in a subprocess on a free port and provides a
Chromium page. The whole module is skipped when Playwright or its Chromium
browser is not installed, so the default (browser-less) test run stays green;
the CI "ui" job installs the browser and runs these.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time

import pytest


def _browser_available() -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        return False
    try:
        with sync_playwright() as pw:
            path = pw.chromium.executable_path
        import os

        return bool(path) and os.path.exists(path)
    except Exception:
        return False


BROWSER = _browser_available()
pytestmark = pytest.mark.skipif(not BROWSER, reason="Playwright chromium not installed")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture(scope="session")
def server(tmp_path_factory):
    port = _free_port()
    # keep scientific-run saves out of the repo working tree
    env = {**os.environ, "ACRPQ_RUNS_DIR": str(tmp_path_factory.mktemp("runs"))}
    proc = subprocess.Popen(
        [sys.executable, "-c",
         f"from acrpq.dashboard.api import run; run(host='127.0.0.1', port={port})"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(120):
            try:
                socket.create_connection(("127.0.0.1", port), 0.3).close()
                break
            except OSError:
                time.sleep(0.5)
        else:
            raise RuntimeError("dashboard server did not start")
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


def _start_server(tmp_path_factory, extra_env):
    port = _free_port()
    env = {**os.environ, "ACRPQ_RUNS_DIR": str(tmp_path_factory.mktemp("runs")), **extra_env}
    proc = subprocess.Popen(
        [sys.executable, "-c",
         f"from acrpq.dashboard.api import run; run(host='127.0.0.1', port={port})"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env,
    )
    base = f"http://127.0.0.1:{port}"
    for _ in range(120):
        try:
            socket.create_connection(("127.0.0.1", port), 0.3).close()
            return proc, base
        except OSError:
            time.sleep(0.5)
    proc.terminate()
    raise RuntimeError("dashboard server did not start")


@pytest.fixture(scope="session")
def qpu_server(tmp_path_factory):
    """A dashboard server with the QPU workflow ENABLED (prepare/confirm allowed)."""
    proc, base = _start_server(tmp_path_factory, {"ACRPQ_QPU_ENABLED": "true"})
    try:
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()


def _page(server):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        pg = browser.new_page()
        errors: list[str] = []
        pg.on("pageerror", lambda e: errors.append(str(e)))
        pg.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        pg._acrpq_errors = errors  # type: ignore[attr-defined]
        pg._acrpq_base = server  # type: ignore[attr-defined]
        yield pg
        browser.close()


@pytest.fixture()
def page(server):
    yield from _page(server)


@pytest.fixture()
def qpu_page(qpu_server):
    yield from _page(qpu_server)
