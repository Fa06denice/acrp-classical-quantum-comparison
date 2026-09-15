"""Minimal AMPL ``.dat`` reader/writer for ACRP instances.

The library instances use a small, regular subset of AMPL data syntax::

    # comment line
    param d := 0.05;
    param n := 4;
    param v0 :=
    1 5.00
    2 5.00
    ;
    param cap :=
    1 3.14159
    ...
    ;

Observed quirks across the 742 shipped instances, all handled here:

* CRLF line terminators.
* Comment first line in several styles (``# Circle Problem``,
  ``#Flow Problem``, ``# Random Circle Problem;``) — any ``#``-prefixed line is
  ignored.
* Index/value separated by either spaces or tabs.
* Scalar params (``d``, ``n``, ``radius``, ``nf``) and indexed params
  (``v0``, ``cap``, ``x0``, ``y0``, ``l0``).
* Some families (RCP_FL) omit ``x0``/``y0`` entirely.

This is intentionally a focused parser, not a general AMPL implementation; it
raises :class:`ParseError` on anything it does not understand rather than
guessing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..exceptions import ParseError

# A scalar assignment: "param d := 0.05 ;"
_SCALAR_RE = re.compile(r"^param\s+([A-Za-z_]\w*)\s*:=\s*([^;]+);?\s*$")
# Start of an indexed block: "param v0 :=" (value comes on following lines)
_INDEXED_START_RE = re.compile(r"^param\s+([A-Za-z_]\w*)\s*:=\s*$")


@dataclass(slots=True)
class AmplData:
    """Raw parsed contents of an AMPL ``.dat`` file."""

    scalars: dict[str, float] = field(default_factory=dict)
    indexed: dict[str, dict[int, float]] = field(default_factory=dict)

    def require_scalar(self, name: str) -> float:
        if name not in self.scalars:
            raise ParseError(f"missing required scalar param '{name}'")
        return self.scalars[name]

    def require_indexed(self, name: str) -> dict[int, float]:
        if name not in self.indexed:
            raise ParseError(f"missing required indexed param '{name}'")
        return self.indexed[name]


def _strip_comment(line: str) -> str:
    """Drop a trailing/inline ``#`` comment and surrounding whitespace."""
    hatch = line.find("#")
    if hatch != -1:
        line = line[:hatch]
    return line.strip()


def parse_ampl_dat(text: str) -> AmplData:
    """Parse AMPL ``.dat`` text into an :class:`AmplData`.

    Tolerant of CRLF/CR/LF, tabs vs spaces, comments and trailing semicolons.
    """
    data = AmplData()
    # Normalise line endings; splitlines handles CRLF, CR and LF uniformly.
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    current: str | None = None  # name of the indexed block being filled, if any
    for raw in lines:
        line = _strip_comment(raw)
        if not line:
            continue

        if current is not None:
            # Inside an indexed block: either the terminating ';' or "idx val".
            if line.startswith(";"):
                current = None
                continue
            # A line may be "idx val" possibly followed by ';'
            ends_block = line.endswith(";")
            body = line[:-1].strip() if ends_block else line
            if body:
                parts = body.split()
                if len(parts) != 2:
                    raise ParseError(
                        f"expected 'index value' in param '{current}', got: {body!r}"
                    )
                try:
                    idx = int(parts[0])
                    val = float(parts[1])
                except ValueError as exc:
                    raise ParseError(
                        f"bad index/value in param '{current}': {body!r}"
                    ) from exc
                data.indexed.setdefault(current, {})[idx] = val
            if ends_block:
                current = None
            continue

        m = _INDEXED_START_RE.match(line)
        if m:
            current = m.group(1)
            data.indexed.setdefault(current, {})
            continue

        m = _SCALAR_RE.match(line)
        if m:
            name, value = m.group(1), m.group(2).strip()
            try:
                data.scalars[name] = float(value)
            except ValueError as exc:
                raise ParseError(f"non-numeric scalar param '{name}': {value!r}") from exc
            continue

        raise ParseError(f"unrecognised line: {line!r}")

    if current is not None:
        raise ParseError(f"unterminated indexed param block '{current}' (missing ';')")
    return data


def serialize_ampl_dat(data: AmplData, *, header: str | None = None) -> str:
    """Serialise :class:`AmplData` back to AMPL ``.dat`` text (LF endings).

    Round-trips the subset produced by :func:`parse_ampl_dat`. Useful for
    writing reduced/synthetic instances.
    """
    out: list[str] = []
    if header:
        out.append(f"# {header}")
    for name, value in data.scalars.items():
        # render integers without a trailing ".0" for n / nf
        if name in {"n", "nf"} and float(value).is_integer():
            out.append(f"param {name} := {int(value)};")
        else:
            out.append(f"param {name} := {value};")
    for name, mapping in data.indexed.items():
        out.append(f"param {name} :=")
        for idx in sorted(mapping):
            v = mapping[idx]
            if name in {"l0"} and float(v).is_integer():
                out.append(f"{idx} {int(v)}")
            else:
                out.append(f"{idx} {v}")
        out.append(";")
    return "\n".join(out) + "\n"
