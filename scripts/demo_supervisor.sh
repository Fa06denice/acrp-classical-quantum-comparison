#!/usr/bin/env bash
set -euo pipefail

# Safe, local-only supervisor demo. These assignments deliberately override any
# IBM/QPU flags inherited from the operator's shell.
export ACRPQ_QPU_ENABLED=true
export ACRPQ_QPU_DRY_RUN=true
export ACRPQ_IBM_SUBMISSION_ENABLED=false
export ACRPQ_IBM_RUNTIME_FACTORY_ENABLED=false
export ACRPQ_FULL_QAOA_SUBMISSION_ENABLED=false

host="127.0.0.1"
port="${ACRPQ_DEMO_PORT:-8000}"
repo_root="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"

if [[ -x "$repo_root/.venv/bin/acrpq" ]]; then
  acrpq_cmd="$repo_root/.venv/bin/acrpq"
elif command -v acrpq >/dev/null 2>&1; then
  acrpq_cmd="$(command -v acrpq)"
else
  printf 'error: acrpq is not installed; run: python3 -m pip install -e ".[quantum,dashboard]"\n' >&2
  exit 1
fi

printf 'ACRP-Q supervisor demo: http://%s:%s\n' "$host" "$port"
printf 'Safety: local dry-run only; IBM submission and runtime factory disabled.\n'
cd "$repo_root"
exec "$acrpq_cmd" dashboard --host "$host" --port "$port" --results results
