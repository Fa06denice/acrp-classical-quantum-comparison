"""Tests for the AMPL .dat parser/serializer."""

from __future__ import annotations

from pathlib import Path

import pytest

from acrpq.exceptions import ParseError
from acrpq.io.ampl_dat import parse_ampl_dat, serialize_ampl_dat

DATA = Path(__file__).resolve().parent / "data"


def test_parse_crlf_spaces():
    data = parse_ampl_dat((DATA / "cp3_spaces.dat").read_text())
    assert data.scalars["n"] == 3
    assert data.scalars["d"] == 0.05
    assert data.indexed["v0"] == {1: 5.0, 2: 5.0, 3: 5.0}
    assert set(data.indexed["x0"]) == {1, 2, 3}


def test_parse_tabs_and_levels():
    data = parse_ampl_dat((DATA / "fl2_tabs.dat").read_text())
    assert data.scalars["nf"] == 2
    assert data.indexed["l0"] == {1: 1.0, 2: 2.0}
    assert "x0" not in data.indexed  # FL instances omit coordinates


def test_comment_styles():
    for header in ["# Circle Problem", "#Flow Problem", "# Random Circle Problem;"]:
        text = f"{header}\nparam n := 1;\nparam d := 0.05;\nparam v0 :=\n1 5\n;\n"
        data = parse_ampl_dat(text)
        assert data.scalars["n"] == 1


def test_inline_terminator_and_trailing_semicolon():
    text = "param n := 2;\nparam d := 0.05;\nparam v0 :=\n1 5\n2 6;\n"
    data = parse_ampl_dat(text)
    assert data.indexed["v0"] == {1: 5.0, 2: 6.0}


def test_roundtrip():
    text = (DATA / "cp3_spaces.dat").read_text()
    data = parse_ampl_dat(text)
    reparsed = parse_ampl_dat(serialize_ampl_dat(data, header="Circle Problem"))
    assert reparsed.scalars["n"] == data.scalars["n"]
    assert reparsed.indexed["cap"] == data.indexed["cap"]


def test_unterminated_block_raises():
    with pytest.raises(ParseError):
        parse_ampl_dat("param v0 :=\n1 5\n")


def test_garbage_line_raises():
    with pytest.raises(ParseError):
        parse_ampl_dat("this is not ampl\n")
