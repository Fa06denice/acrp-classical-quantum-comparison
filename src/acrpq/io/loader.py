"""Zip-aware ACRP instance loader.

The shipped instances live inside zip archives under ``Data/`` at the
repository root:

* ``Data/CP_Instances/CP_Instances.zip``     -> family CP
* ``Data/FP_Instances/FP_Instances.zip``     -> family FP
* ``Data/GP_Instances/GP_Instances.zip``     -> family GP
* ``Data/RCP_Instances/RCP_Instances.zip``   -> family RCP (2D)
* ``Data/RCP_Instances/RCP_FL_3_Instances.zip`` -> family RCP_FL (3 levels)
* ``Data/RCP_Instances/RCP_FL_5_Instances.zip`` -> family RCP_FL (5 levels)

Members are stored under a nested ``<FAM>_Instances/`` directory inside each
archive. Instances are addressed by their stem name (``"CP_4"``,
``"RCP_10_1"``); the loader maps the name to the right archive.

The data root is resolved in this order: an explicit constructor argument; the
``ACRPQ_DATA_ROOT`` environment variable; otherwise an upward search from this
file for the directory that contains both ``Code/`` and ``Data/``.

For instances that omit ``x0``/``y0`` (the RCP_FL family), coordinates are
reconstructed from the ``NC_ACRP.mod`` circle defaults (lines 17-18) using the
``PI = 3.141592`` literal so the geometry matches AMPL exactly.
"""

from __future__ import annotations

import math
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

from ..exceptions import InstanceError, ParseError
from ..model import DEFAULT_W, PI, Family, Instance
from .ampl_dat import AmplData, parse_ampl_dat

# Archive registry: family -> list of (relative zip path, label).
_ARCHIVES: list[tuple[Family, str, str]] = [
    (Family.CP, "Data/CP_Instances/CP_Instances.zip", "CP"),
    (Family.FP, "Data/FP_Instances/FP_Instances.zip", "FP"),
    (Family.GP, "Data/GP_Instances/GP_Instances.zip", "GP"),
    (Family.RCP, "Data/RCP_Instances/RCP_Instances.zip", "RCP"),
    (Family.RCP_FL, "Data/RCP_Instances/RCP_FL_3_Instances.zip", "RCP_FL3"),
    (Family.RCP_FL, "Data/RCP_Instances/RCP_FL_5_Instances.zip", "RCP_FL5"),
]


def _default_data_root() -> Path:
    env = os.environ.get("ACRPQ_DATA_ROOT")
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "Code").is_dir() and (parent / "Data").is_dir():
            return parent
    # Fall back to current working directory; load() will error clearly if wrong.
    return Path.cwd()


def _family_from_name(name: str) -> Family | None:
    """Infer the family from a name like ``CP_4``; return ``None`` if unknown."""
    head = name.split("_", 1)[0].upper()
    try:
        return Family(head)
    except ValueError:
        return None


def circle_default_coords(n: int, radius: float) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Reconstruct ``x0``/``y0`` from the ``NC_ACRP.mod`` circle defaults.

    ``x0[i] = -radius * cos((i-1) * 2*PI/n + PI)`` and analogously for ``y0``
    (lines 17-18), with ``PI`` the ``.mod`` literal.
    """
    xs: list[float] = []
    ys: list[float] = []
    for i in range(1, n + 1):
        ang = (i - 1) * 2 * PI / n + PI
        xs.append(-radius * math.cos(ang))
        ys.append(-radius * math.sin(ang))
    return tuple(xs), tuple(ys)


def build_instance(
    name: str,
    data: AmplData,
    *,
    family: Family | None = None,
    w: float = DEFAULT_W,
    source_path: str | None = None,
) -> Instance:
    """Assemble an :class:`Instance` from parsed AMPL data, filling defaults."""
    # Family precedence: explicit arg, then inferred from name, then a sensible
    # default (CP) for ad-hoc user files; an "nf" param below promotes to RCP_FL.
    fam = family or _family_from_name(name) or Family.CP
    raw_n = data.require_scalar("n")
    if not math.isfinite(raw_n) or not raw_n.is_integer() or raw_n < 1:
        raise ParseError(f"param 'n' must be a positive integer, got {raw_n!r}")
    n = int(raw_n)
    d = data.require_scalar("d")
    radius = data.scalars.get("radius", 2.0)
    if not math.isfinite(d) or d < 0:
        raise ParseError(f"param 'd' must be finite and >= 0, got {d!r}")
    if not math.isfinite(radius) or radius < 0:
        raise ParseError(f"param 'radius' must be finite and >= 0, got {radius!r}")
    if not math.isfinite(w) or not 0.0 <= w <= 1.0:
        raise ParseError(f"objective weight must be finite and in [0, 1], got {w!r}")

    def vec(param: str) -> tuple[float, ...]:
        mapping = data.require_indexed(param)
        missing = [i for i in range(1, n + 1) if i not in mapping]
        if missing:
            raise ParseError(f"param '{param}' missing indices {missing} (n={n})")
        extra = sorted(set(mapping) - set(range(1, n + 1)))
        if extra:
            raise ParseError(f"param '{param}' has out-of-range indices {extra} (n={n})")
        values = tuple(mapping[i] for i in range(1, n + 1))
        if not all(math.isfinite(value) for value in values):
            raise ParseError(f"param '{param}' contains a non-finite value")
        return values

    v0 = vec("v0")
    if any(speed <= 0 for speed in v0):
        raise ParseError("param 'v0' values must all be > 0")
    cap = vec("cap")

    has_x0 = "x0" in data.indexed
    has_y0 = "y0" in data.indexed
    if has_x0 != has_y0:
        raise ParseError("params 'x0' and 'y0' must either both be present or both be absent")
    if has_x0:
        x0 = vec("x0")
        y0 = vec("y0")
    else:
        # RCP_FL omits coordinates -> reconstruct circle defaults.
        x0, y0 = circle_default_coords(n, radius)

    nf: int | None = None
    l0: tuple[int, ...] = ()
    if "nf" in data.scalars:
        raw_nf = data.scalars["nf"]
        if not math.isfinite(raw_nf) or not raw_nf.is_integer() or raw_nf < 1:
            raise ParseError(f"param 'nf' must be a positive integer, got {raw_nf!r}")
        nf = int(raw_nf)
        raw_l0 = vec("l0")
        if any(not level.is_integer() or not 1 <= level <= nf for level in raw_l0):
            raise ParseError(f"param 'l0' values must be integer levels in 1..{nf}")
        l0 = tuple(int(level) for level in raw_l0)
        fam = Family.RCP_FL
    elif "l0" in data.indexed:
        raise ParseError("param 'l0' requires scalar param 'nf'")

    return Instance(
        name=name,
        family=fam,
        n=n,
        d=d,
        radius=radius,
        v0=v0,
        cap=cap,
        x0=x0,
        y0=y0,
        w=w,
        nf=nf,
        l0=l0,
        level=data.scalars.get("level", 0.0),
        source_path=source_path,
    )


@dataclass
class InstanceLoader:
    """Loads ACRP instances by name from the ``Data/`` archives, or from files."""

    data_root: Path | None = None
    w: float = DEFAULT_W

    def __post_init__(self) -> None:
        self.data_root = Path(self.data_root) if self.data_root else _default_data_root()

    # -- discovery ------------------------------------------------------- #
    def _archive_path(self, rel: str) -> Path:
        return self.data_root / rel  # type: ignore[operator]

    def _index(self) -> dict[str, tuple[Family, Path, str]]:
        """Map instance stem -> (family, archive path, member name)."""
        index: dict[str, tuple[Family, Path, str]] = {}
        for fam, rel, label in _ARCHIVES:
            path = self._archive_path(rel)
            if not path.exists():
                continue
            with zipfile.ZipFile(path) as zf:
                for member in zf.namelist():
                    if not member.endswith(".dat"):
                        continue
                    stem = Path(member).stem
                    # RCP_FL3 and RCP_FL5 archives share identical stem names.
                    # FL3 keeps the bare stem (the default); FL5 is also exposed
                    # under a suffixed alias "<stem>@FL5" so it stays reachable.
                    if label == "RCP_FL5" and stem in index:
                        key = f"{stem}@FL5"
                    else:
                        key = stem
                    index.setdefault(key, (fam, path, member))
        return index

    def list_names(self, family: Family | str | None = None) -> list[str]:
        """Sorted instance names, optionally filtered by family.

        RCP_FL5 instances appear under a ``<stem>@FL5`` alias (the bare stem
        resolves to the 3-level variant).
        """
        fam = Family(family) if isinstance(family, str) else family
        names = [
            name
            for name, (f, _p, _m) in self._index().items()
            if fam is None or f == fam
        ]
        return sorted(names, key=_natural_key)

    # -- loading --------------------------------------------------------- #
    def load(
        self, name: str, *, w: float | None = None, flight_levels: int | None = None
    ) -> Instance:
        """Load instance ``name`` (e.g. ``"CP_4"``) from its archive.

        For random-circle flight-level instances, ``flight_levels=5`` selects
        the 5-level archive (equivalent to the ``"<stem>@FL5"`` alias);
        ``flight_levels=3`` or the bare name selects the 3-level variant.
        """
        if flight_levels not in (None, 3, 5):
            raise ValueError("flight_levels must be 3, 5, or None")
        if flight_levels == 3 and name.endswith("@FL5"):
            raise ValueError("an @FL5 alias cannot be requested with flight_levels=3")
        index = self._index()
        lookup = name
        if flight_levels == 5 and not name.endswith("@FL5"):
            lookup = f"{name}@FL5"
        entry = index.get(lookup)
        if entry is None:
            raise InstanceError(
                f"instance {name!r} not found under {self.data_root}. "
                f"Use list_names() to see available instances."
            )
        fam, path, member = entry
        with zipfile.ZipFile(path) as zf:
            try:
                text = zf.read(member).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ParseError(f"instance {name!r} is not valid UTF-8") from exc
        data = parse_ampl_dat(text)
        return build_instance(
            lookup,
            data,
            family=fam,
            w=self.w if w is None else w,
            source_path=f"{path.name}::{member}",
        )

    def read_dat_text(
        self, name: str, *, flight_levels: int | None = None
    ) -> tuple[str, str]:
        """Return the ORIGINAL ``.dat`` text (verbatim) + its source path.

        Resolution goes through the fixed archive index (same as :meth:`load`), so
        a name containing ``/`` or ``..`` is simply not found — there is no path
        interpolation and no traversal. Used by the original-AMPL runner to feed
        the exact instance bytes to the historical model (no reconstruction).
        """
        if flight_levels not in (None, 3, 5):
            raise ValueError("flight_levels must be 3, 5, or None")
        index = self._index()
        lookup = name
        if flight_levels == 5 and not name.endswith("@FL5"):
            lookup = f"{name}@FL5"
        entry = index.get(lookup)
        if entry is None:
            raise InstanceError(
                f"instance {name!r} not found under {self.data_root}. "
                f"Use list_names() to see available instances."
            )
        _fam, path, member = entry
        with zipfile.ZipFile(path) as zf:
            try:
                text = zf.read(member).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ParseError(f"instance {name!r} is not valid UTF-8") from exc
        # parse for validation only (never guess); the returned text stays verbatim
        parse_ampl_dat(text)
        return text, f"{path.name}::{member}"

    def load_file(self, path: str | Path, *, w: float | None = None) -> Instance:
        """Load an instance from a loose ``.dat`` file (user-provided)."""
        p = Path(path)
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ParseError(f"instance file {p} is not valid UTF-8") from exc
        data = parse_ampl_dat(text)
        return build_instance(
            p.stem,
            data,
            w=self.w if w is None else w,
            source_path=str(p),
        )


def _natural_key(name: str) -> tuple:
    """Sort key so CP_2 < CP_10 (numeric-aware)."""
    parts = name.replace("-", "_").split("_")
    key: list = []
    for p in parts:
        key.append((0, int(p)) if p.isdigit() else (1, p))
    return tuple(key)


def discover_instances(
    data_root: Path | None = None, family: Family | str | None = None
) -> list[str]:
    """Convenience wrapper returning all instance names."""
    return InstanceLoader(data_root=data_root).list_names(family)
