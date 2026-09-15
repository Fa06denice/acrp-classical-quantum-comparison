#!/usr/bin/env bash
set -euo pipefail

repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
cd "$repo_root"

fail=0
check_file() {
  if [[ -r "$1" ]]; then
    printf 'OK   %s\n' "$1"
  else
    printf 'FAIL %s is missing or unreadable\n' "$1" >&2
    fail=1
  fi
}

printf 'ACRP-Q demo preflight\n'
printf '=====================\n'

demo_port="${ACRPQ_DEMO_PORT:-8000}"

if [[ -x "$repo_root/.venv/bin/python" ]]; then
  python_cmd="$repo_root/.venv/bin/python"
else
  python_cmd="$(command -v python3 || command -v python || true)"
fi
if [[ -x "$repo_root/.venv/bin/acrpq" ]]; then
  acrpq_cmd="$repo_root/.venv/bin/acrpq"
else
  acrpq_cmd="$(command -v acrpq || true)"
fi

[[ -n "$acrpq_cmd" ]] || {
  printf 'FAIL acrpq is not installed in the active environment\n' >&2
  fail=1
}
[[ -n "$python_cmd" ]] || {
  printf 'FAIL python is not available\n' >&2
  fail=1
}

check_file results/campaign_summary.json
check_file results/three_way.json
check_file docs/TRACEABILITY.md
check_file src/acrpq/dashboard/static/vendor/chart.umd.min.js

if command -v lsof >/dev/null 2>&1 && lsof -nP -iTCP:"$demo_port" -sTCP:LISTEN >/dev/null 2>&1; then
  printf 'WARN port %s is already in use; stop that process or set ACRPQ_DEMO_PORT\n' \
    "$demo_port" >&2
else
  printf 'OK   port %s is available\n' "$demo_port"
fi

if [[ -n "$python_cmd" ]]; then
"$python_cmd" - <<'PY' || fail=1
import importlib.util
import json
import sys

required = ("fastapi", "uvicorn", "httpx")
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    print("FAIL missing Python packages: " + ", ".join(missing))
    raise SystemExit(1)

from acrpq.dashboard.qpu_flags import flags_snapshot

safe_env = {
    "ACRPQ_QPU_ENABLED": "true",
    "ACRPQ_QPU_DRY_RUN": "true",
    "ACRPQ_IBM_SUBMISSION_ENABLED": "false",
}
snapshot = flags_snapshot(env=safe_env.get)
if snapshot["submission_allowed"]:
    print("FAIL safe demo profile unexpectedly permits real submission")
    raise SystemExit(1)
print("OK   safe QPU profile " + json.dumps(snapshot, sort_keys=True))
print("OK   Python " + sys.version.split()[0])
PY
fi

if git diff --quiet -- Code src/acrpq/classical results; then
  printf 'OK   protected classical/results paths have no working-tree changes\n'
else
  printf 'WARN protected classical/results paths contain working-tree changes\n' >&2
fi

if [[ "$fail" -ne 0 ]]; then
  printf '\nPreflight FAILED. Fix the items above before the meeting.\n' >&2
  exit 1
fi

printf '\nPreflight PASSED. Start with: ./scripts/demo_supervisor.sh\n'
