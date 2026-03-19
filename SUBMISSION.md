# Submission Guide

This repository is structured to satisfy code-verification requirements for competition submission:

- Contains training code, inference code, and custom tooling used to produce submissions.
- Avoids hardcoded prediction outputs or precomputed responses in tracked source files.
- Uses reproducible CLI commands and explicit configuration.

## 1) Repository Visibility

Before submitting, ensure the repository is public and the latest commit is pushed.

```bash
git push origin main
```

## 2) Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .[train,eval]
```

## 3) Dataset Paths

Set paths to your local dataset copy.

```bash
export NMAI_TRAIN_ANN="../train/annotations.json"
export NMAI_TRAIN_IMG_DIR="../train/images"
export NMAI_CATALOG_ROOT="../NM_NGD_product_images"
```

## 4) Prepare Metadata and Splits

```bash
nmai audit --annotations "$NMAI_TRAIN_ANN" --out reports/audit.json
nmai split --annotations "$NMAI_TRAIN_ANN" --num-folds 5 --seed 42 --out reports/folds.json
nmai catalog-manifest --catalog-root "$NMAI_CATALOG_ROOT" --out reports/catalog_manifest.json
```

## 5) Baseline Training + Evaluation

Quick single-fold run:

```bash
nmai run-fold-suite \
  --annotations "$NMAI_TRAIN_ANN" \
  --images-dir "$NMAI_TRAIN_IMG_DIR" \
  --split-json reports/folds.json \
  --experiment-config configs/experiments/yolo_baseline.yaml \
  --export-root data/yolo_folds \
  --project-dir artifacts/folds \
  --reports-root reports/suite \
  --folds 0
```

The aggregate output is written to:

- `reports/suite/suite_summary.json`

## 6) Inference Example

```bash
nmai predict-yolo-tiles \
  --model-path artifacts/folds/fold_0/fold_0/weights/best.pt \
  --input-path "$NMAI_TRAIN_IMG_DIR" \
  --out artifacts/predictions/tiled_predictions.json \
  --tile-size 1280 \
  --overlap 256 \
  --conf 0.2 \
  --iou-thr 0.55
```

## 7) Verification Checklist

- Repository is public.
- All code used for final scores is committed.
- No hidden/private dependency is required for core training/inference flow.
- A clean clone can execute setup + baseline commands from this document.

## 8) Integrity Notes

- The repository contains source code and configuration only.
- Generated artifacts (`data/`, `artifacts/`, `reports/`) are ignored by default.
- The code paths are parameterized and avoid machine-specific hardcoding.
