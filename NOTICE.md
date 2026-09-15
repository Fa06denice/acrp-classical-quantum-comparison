# Third-party notices

This project depends on third-party packages that are **not vendored** here — they are
installed separately via pip and remain under their own licenses:

- Qiskit, qiskit-ibm-runtime, qiskit-aer, qiskit-optimization — Apache License 2.0
- dwave-system, dwave-samplers, dimod, minorminer — Apache License 2.0
- NumPy, Matplotlib — BSD-style licenses
- FastAPI, uvicorn — MIT License

Two optional extras require **separately licensed, proprietary software** the user must
obtain themselves — they are never bundled or redistributed by this repository:

- `amplpy` (AMPL Python API) — requires an AMPL license.
- Gurobi (via Pyomo) — requires a Gurobi license.
