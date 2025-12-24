#!/usr/bin/env bash
set -euo pipefail

# Loads env vars from the project root `.env` file (if present) and exports them.
# Usage: source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"

script_dir="$(
  cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1
  pwd
)"
project_root="$(
  cd -- "$script_dir/.." >/dev/null 2>&1
  pwd
)"

env_file="${ENV_FILE:-$project_root/.env}"
if [[ -f "$env_file" ]]; then
  set -a
  # shellcheck source=/dev/null
  . "$env_file"
  set +a
else
  echo "Warning: .env not found at: $env_file (copy .env.example -> .env)" >&2
fi

