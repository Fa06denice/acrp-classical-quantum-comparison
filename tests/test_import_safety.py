"""The pure-Python core must import with ZERO third-party dependencies.

Run under any interpreter (the default test env has no numpy/qiskit/pyomo). If
any of these modules grows an eager heavy import, this test fails.
"""

from __future__ import annotations

import importlib

PURE_MODULES = [
    "acrpq",
    "acrpq.model",
    "acrpq.geometry",
    "acrpq.exceptions",
    "acrpq._optional",
    "acrpq.io",
    "acrpq.io.ampl_dat",
    "acrpq.io.loader",
    "acrpq.discretize",
    "acrpq.classical",
    "acrpq.classical.base",
    "acrpq.classical.reference",
    "acrpq.quantum.qubo",
    "acrpq.quantum.ising",
    "acrpq.quantum.decode",
    "acrpq.quantum.qubo_solvers",
    "acrpq.quantum.memguard",
    "acrpq.quantum.dwave_solver",
    "acrpq.benchmark.three_way",
    "acrpq.benchmark.scaling",
    "acrpq.benchmark.metrics",
    "acrpq.benchmark.scenarios",
    "acrpq.benchmark.export",
    "acrpq.benchmark.runner",
    "acrpq.cli.main",
]


HEAVY_DEPS = {
    "numpy",
    "scipy",
    "qiskit",
    "qiskit_aer",
    "qiskit_optimization",
    "qiskit_ibm_runtime",
    "pyomo",
    "amplpy",
    "matplotlib",
    "dimod",
    "dwave",
    "minorminer",
}


def test_pure_modules_import():
    for name in PURE_MODULES:
        importlib.import_module(name)


def test_pure_imports_do_not_pull_heavy_deps():
    """Importing the core must not import any heavy optional dependency.

    This guards the zero-dependency contract even in an environment (like the
    validation venv) where the heavy libraries *are* installed: a regression to
    a top-level ``import numpy`` in a core module would leak into sys.modules
    and fail here.
    """
    import os
    import subprocess
    import sys

    code = (
        "import importlib, sys\n"
        f"for m in {PURE_MODULES!r}:\n"
        "    importlib.import_module(m)\n"
        f"leaked = sorted({HEAVY_DEPS!r} & set(sys.modules))\n"
        "print(','.join(leaked))\n"
    )
    # Run in a fresh interpreter so we measure a clean import, not the test
    # process which has already imported qiskit/etc. for other tests. Pass the
    # current sys.path through PYTHONPATH so the child can find acrpq whether it
    # is installed or run from src/.
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    leaked = out.stdout.strip()
    assert leaked == "", f"core import leaked heavy deps: {leaked}"


def test_top_level_lazy_attrs():
    import acrpq

    assert acrpq.__version__
    # lazy re-exports resolve without importing the quantum stack
    assert acrpq.InstanceLoader is not None
    assert acrpq.DiscreteReferenceSolver is not None
    assert acrpq.build_qubo is not None


def test_version_capture_does_not_import_optional_packages():
    import sys

    from acrpq._optional import versions

    optional = {"matplotlib", "pyomo", "qiskit", "qiskit_aer", "qiskit_ibm_runtime"}
    before = optional & set(sys.modules)
    resolved = versions()
    after = optional & set(sys.modules)
    assert after == before
    assert "qiskit" in resolved


def test_unknown_attr_raises():
    import acrpq

    try:
        _ = acrpq.NoSuchThing
    except AttributeError:
        return
    raise AssertionError("expected AttributeError for unknown attribute")
