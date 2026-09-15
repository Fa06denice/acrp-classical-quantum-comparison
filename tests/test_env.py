"""Tests for the .env loader (pure stdlib)."""

from __future__ import annotations

import os

from acrpq.io.env import find_dotenv, load_dotenv, parse_dotenv


def test_parse_basic():
    text = (
        "# comment\n"
        "KEY=value\n"
        'QUOTED="a b c"\n'
        "export EXPORTED=x\n"
        "EMPTY=\n"
        "ignored line without equals\n"
    )
    parsed = parse_dotenv(text)
    assert parsed["KEY"] == "value"
    assert parsed["QUOTED"] == "a b c"
    assert parsed["EXPORTED"] == "x"
    assert parsed["EMPTY"] == ""
    assert "ignored line without equals" not in parsed


def test_load_does_not_override_real_env(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("ACRPQ_TEST_KEY=from_file\n")
    monkeypatch.setenv("ACRPQ_TEST_KEY", "from_env")
    applied = load_dotenv(env_file)
    assert "ACRPQ_TEST_KEY" not in applied  # real env wins
    assert os.environ["ACRPQ_TEST_KEY"] == "from_env"


def test_load_sets_missing_key(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("ACRPQ_TEST_KEY2=from_file\n")
    monkeypatch.delenv("ACRPQ_TEST_KEY2", raising=False)
    applied = load_dotenv(env_file)
    assert applied["ACRPQ_TEST_KEY2"] == "from_file"
    assert os.environ["ACRPQ_TEST_KEY2"] == "from_file"


def test_find_dotenv_upward(tmp_path):
    (tmp_path / ".env").write_text("X=1\n")
    sub = tmp_path / "a" / "b"
    sub.mkdir(parents=True)
    found = find_dotenv(sub)
    assert found == tmp_path / ".env"
