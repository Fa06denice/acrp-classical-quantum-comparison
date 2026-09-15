"""Shared test fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from acrpq.io.loader import InstanceLoader

# Locate the repo root (the dir holding Code/ and Data/) from the tests dir.
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(__file__).resolve().parent / "data"


@pytest.fixture(scope="session")
def loader() -> InstanceLoader:
    return InstanceLoader(data_root=REPO_ROOT)


@pytest.fixture(scope="session")
def cp3(loader):
    return loader.load("CP_3")


@pytest.fixture(scope="session")
def cp4(loader):
    return loader.load("CP_4")
