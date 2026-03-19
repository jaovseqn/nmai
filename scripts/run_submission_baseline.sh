#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${NMAI_TRAIN_ANN:-}" || -z "${NMAI_TRAIN_IMG_DIR:-}" || -z "${NMAI_CATALOG_ROOT:-}" ]]; then
  echo "Please set NMAI_TRAIN_ANN, NMAI_TRAIN_IMG_DIR, and NMAI_CATALOG_ROOT." >&2
  exit 1
fi

python -m nmai.cli audit --annotations "$NMAI_TRAIN_ANN" --out reports/audit.json
python -m nmai.cli split --annotations "$NMAI_TRAIN_ANN" --num-folds 5 --seed 42 --out reports/folds.json
python -m nmai.cli catalog-manifest --catalog-root "$NMAI_CATALOG_ROOT" --out reports/catalog_manifest.json

python -m nmai.cli run-fold-suite \
  --annotations "$NMAI_TRAIN_ANN" \
  --images-dir "$NMAI_TRAIN_IMG_DIR" \
  --split-json reports/folds.json \
  --experiment-config configs/experiments/yolo_baseline.yaml \
  --export-root data/yolo_folds \
  --project-dir artifacts/folds \
  --reports-root reports/suite \
  --folds "${NMAI_FOLDS:-0}" \
  --tile-size "${NMAI_TILE_SIZE:-1280}" \
  --overlap "${NMAI_OVERLAP:-256}" \
  --conf "${NMAI_CONF:-0.2}" \
  --iou-thr "${NMAI_IOU_THR:-0.55}"

echo "Submission baseline run completed."
echo "Summary: reports/suite/suite_summary.json"
