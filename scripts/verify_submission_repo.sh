#!/usr/bin/env bash
set -euo pipefail

echo "== Checking required files =="
for path in README.md SUBMISSION.md pyproject.toml src/nmai/cli.py configs/experiments/yolo_baseline.yaml; do
  if [[ ! -f "$path" ]]; then
    echo "Missing: $path" >&2
    exit 1
  fi
  echo "OK: $path"
done

echo "== Checking CLI entrypoint =="
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
elif [[ -x "../.venv/bin/python" ]]; then
  PYTHON_BIN="../.venv/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  PYTHON_BIN="python3"
fi
"$PYTHON_BIN" -m nmai.cli --help >/dev/null

echo "== Checking for tracked generated artifacts =="
if git ls-files --error-unmatch data >/dev/null 2>&1; then
  echo "Warning: data/ is tracked. Consider removing generated files from git history." >&2
fi
if git ls-files --error-unmatch artifacts >/dev/null 2>&1; then
  echo "Warning: artifacts/ is tracked. Consider removing generated files from git history." >&2
fi
if git ls-files --error-unmatch reports >/dev/null 2>&1; then
  echo "Warning: reports/ is tracked. Consider removing generated files from git history." >&2
fi

echo "== Git status =="
git status --short

echo "Repository verification completed."
