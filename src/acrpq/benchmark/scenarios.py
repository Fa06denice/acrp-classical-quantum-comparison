"""Standardised reduced benchmark scenarios.

The quantum layer is constrained by qubit budget (``n * K`` qubits). These
curated scenarios stay within a simulator-friendly budget (<= ~24 qubits) and
give a progressive complexity ladder for the thesis: from a 2-aircraft toy up
to a few aircraft with denser conflicts.

A :class:`Scenario` references a real library instance (loaded from the
``Data/`` archives), optionally sub-samples aircraft, and fixes the maneuver
grid resolution. Sub-sampling lets larger families contribute small,
quantum-tractable cases without inventing synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..model import DEFAULT_W


@dataclass(frozen=True)
class Scenario:
    """A named, reproducible benchmark case."""

    id: str
    instance_name: str
    description: str
    n_theta: int = 5
    n_q: int = 1
    subset: tuple[int, ...] | None = None  # 1-based aircraft ids to keep
    w: float = DEFAULT_W
    tags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def grid_kwargs(self) -> dict[str, int]:
        return {"n_theta": self.n_theta, "n_q": self.n_q}

    def expected_qubits(self) -> int:
        n = len(self.subset) if self.subset else _instance_n_hint(self.instance_name)
        return n * (self.n_theta * self.n_q)


# Rough n hint by name so expected_qubits works without loading (best effort).
def _instance_n_hint(name: str) -> int:
    parts = name.split("_")
    for p in parts[1:]:
        if p.isdigit():
            return int(p)
    return 0


REDUCED_SCENARIOS: dict[str, Scenario] = {
    s.id: s
    for s in [
        Scenario(
            id="Q2",
            instance_name="CP_3",
            subset=(1, 2),
            n_theta=5,
            description="2 aircraft, head-on subset of the circle problem (simplest case).",
            tags=("2-aircraft", "easy"),
        ),
        Scenario(
            id="Q2h",
            instance_name="CP_4",
            subset=(1, 3),
            n_theta=7,
            description="2 aircraft, diametrically opposed, finer maneuver grid.",
            tags=("2-aircraft", "options"),
        ),
        Scenario(
            id="Q3",
            instance_name="CP_3",
            n_theta=5,
            description="3 aircraft converging on the circle centre (3 crossing conflicts).",
            tags=("3-aircraft",),
        ),
        Scenario(
            id="Q3f",
            instance_name="CP_3",
            n_theta=3,
            description="3 aircraft, coarse 3-heading grid (fast smoke, 9 qubits).",
            tags=("3-aircraft", "fast"),
        ),
        Scenario(
            id="Q4",
            instance_name="CP_4",
            n_theta=5,
            description="4 aircraft converging on the circle centre (6 dense conflicts).",
            tags=("4-aircraft", "dense"),
        ),
        Scenario(
            id="Q4f",
            instance_name="CP_4",
            n_theta=3,
            description="4 aircraft, coarse 3-heading grid (12 qubits).",
            tags=("4-aircraft", "fast"),
        ),
        Scenario(
            id="QR3",
            instance_name="RCP_10_1",
            subset=(1, 4),
            n_theta=5,
            description="2 conflicting aircraft drawn from a random circle instance (sparse).",
            tags=("random", "sparse"),
        ),
    ]
}


def get_scenario(scenario_id: str) -> Scenario:
    try:
        return REDUCED_SCENARIOS[scenario_id]
    except KeyError as exc:
        raise KeyError(
            f"unknown scenario {scenario_id!r}; available: {sorted(REDUCED_SCENARIOS)}"
        ) from exc


def list_scenarios() -> list[Scenario]:
    return list(REDUCED_SCENARIOS.values())
