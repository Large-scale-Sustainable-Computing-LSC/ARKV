#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if ! command -v conda >/dev/null 2>&1; then
  echo "Error: conda not found in PATH. Please install Miniconda/Anaconda and try again." >&2
  exit 1
fi

usage() {
  cat <<'USAGE'
Usage: scripts/setup_env.sh [--file <env_spec>] [--name <env_name>]

Creates/updates a conda environment from an env spec (YAML), then installs this repo
in editable mode via: pip install -e .

Defaults:
  --file: environment.yml (if present), otherwise env.txt (if present)
  --name: parsed from the env spec's "name:" field
USAGE
}

ENV_FILE=""
ENV_NAME_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --file)
      ENV_FILE="${2:-}"; shift 2 ;;
    --name)
      ENV_NAME_OVERRIDE="${2:-}"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Error: unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$ENV_FILE" ]]; then
  if [[ -f "environment.yml" ]]; then
    ENV_FILE="environment.yml"
  elif [[ -f "env.txt" ]]; then
    ENV_FILE="env.txt"
  else
    echo "Error: no env spec found. Expected environment.yml or env.txt in $ROOT_DIR" >&2
    exit 1
  fi
fi

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Error: env spec file not found: $ENV_FILE" >&2
  exit 1
fi

# Some conda exports include a 'prefix:' line, which can make env creation non-portable.
TMP_ENV_FILE="$(mktemp -t arkv-env-XXXXXX.yml)"
cleanup() { rm -f "$TMP_ENV_FILE"; }
trap cleanup EXIT

grep -v '^prefix:' "$ENV_FILE" > "$TMP_ENV_FILE"

ENV_NAME_PARSED="$(grep -E '^name:\s*' "$TMP_ENV_FILE" | head -n 1 | awk '{print $2}')"
ENV_NAME="${ENV_NAME_OVERRIDE:-$ENV_NAME_PARSED}"
if [[ -z "${ENV_NAME}" ]]; then
  echo "Error: could not determine conda env name (use --name to specify one)" >&2
  exit 1
fi

echo "[setup] Creating/updating conda env: ${ENV_NAME}"
conda env update -n "$ENV_NAME" -f "$TMP_ENV_FILE" --prune || conda env create -n "$ENV_NAME" -f "$TMP_ENV_FILE"

echo "[setup] Installing ARKV in editable mode (pip -e)"
conda run -n "$ENV_NAME" python -m pip install --upgrade pip
conda run -n "$ENV_NAME" python -m pip install -e .

echo "[setup] Done. Activate with: conda activate ${ENV_NAME}"
