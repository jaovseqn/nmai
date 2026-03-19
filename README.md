# nmai

Utilities and baseline tooling for the Norwegian AI Championship 2026 grocery shelf detection task.

## What is implemented

- COCO dataset audit for class balance, object density, and box scale.
- Deterministic multi-fold split generation for local cross-validation.
- COCO to YOLO export for fast baseline training.
- Baseline experiment configs for dense shelf detection.
- Config-driven YOLO training entrypoint.
- Tiled YOLO inference with weighted-box fusion fallback.
- Catalog manifest generation for the retrieval branch.
- COCO mAP evaluation for both full-frame and tiled inference.
- Fold-orchestrated training command for reproducible cross-validation.
- One-command fold suite orchestration for export + train + eval + aggregate summary.

## Repository layout

```text
nmai/
	configs/
		dataset_paths.example.yaml
		experiments/yolo_baseline.yaml
	src/nmai/
		audit.py
		cli.py
		coco_utils.py
		splits.py
		yolo_export.py
```

## Install

```bash
cd nmai
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

If you want the YOLO training stack as well:

```bash
pip install -e .[train]
```

If you also want COCO metric evaluation:

```bash
pip install -e .[eval]
```

## Commands

Set your dataset roots once (adjust these paths to your machine):

```bash
export NMAI_TRAIN_ANN="../train/annotations.json"
export NMAI_TRAIN_IMG_DIR="../train/images"
export NMAI_CATALOG_ROOT="../NM_NGD_product_images"
```

Create an audit report:

```bash
nmai audit \
	--annotations "$NMAI_TRAIN_ANN" \
	--out reports/audit.json
```

Create deterministic cross-validation folds:

```bash
nmai split \
	--annotations "$NMAI_TRAIN_ANN" \
	--num-folds 5 \
	--seed 42 \
	--out reports/folds.json

nmai catalog-manifest \
	--catalog-root "$NMAI_CATALOG_ROOT" \
	--out reports/catalog_manifest.json
```

Export the dataset to YOLO layout:

```bash
nmai export-yolo \
	--annotations "$NMAI_TRAIN_ANN" \
	--images-dir "$NMAI_TRAIN_IMG_DIR" \
	--out-dir data/yolo \
	--split-json reports/folds.json \
	--val-fold 0

nmai train-yolo \
	--experiment-config configs/experiments/yolo_baseline.yaml \
	--data-yaml data/yolo/data.yaml \
	--project-dir artifacts

nmai train-yolo-folds \
	--annotations "$NMAI_TRAIN_ANN" \
	--images-dir "$NMAI_TRAIN_IMG_DIR" \
	--split-json reports/folds.json \
	--experiment-config configs/experiments/yolo_baseline.yaml \
	--export-root data/yolo_folds \
	--project-dir artifacts/folds \
	--folds 0,1

nmai predict-yolo-tiles \
	--model-path artifacts/yolo_dense_shelf_baseline/weights/best.pt \
	--input-path "$NMAI_TRAIN_IMG_DIR" \
	--out artifacts/predictions/tiled_val.json \
	--tile-size 1280 \
	--overlap 256 \
	--conf 0.2 \
	--iou-thr 0.55

nmai eval-coco \
	--annotations "$NMAI_TRAIN_ANN" \
	--images-dir "$NMAI_TRAIN_IMG_DIR" \
	--model-path artifacts/yolo_dense_shelf_baseline/weights/best.pt \
	--split-json reports/folds.json \
	--val-fold 0 \
	--tiled \
	--tile-size 1280 \
	--overlap 256 \
	--out reports/eval_fold0_tiled.json

nmai run-fold-suite \
	--annotations "$NMAI_TRAIN_ANN" \
	--images-dir "$NMAI_TRAIN_IMG_DIR" \
	--split-json reports/folds.json \
	--experiment-config configs/experiments/yolo_baseline.yaml \
	--export-root data/yolo_folds \
	--project-dir artifacts/folds \
	--reports-root reports/suite \
	--folds 0,1
```

## Submission Notes

For competition submission packaging and verification steps, see [SUBMISSION.md](SUBMISSION.md).

## Suggested next experiments

1. Run the audit and inspect the tail classes plus object-size histogram.
2. Export a first YOLO dataset and train a high-resolution dense-shelf baseline.
3. Add tiled validation and weighted boxes fusion before adding more model complexity.
4. Build a catalog-image retrieval branch after the detector baseline is stable.

The evaluation command writes both the final metric summary and the raw detection file next to the output path, so experiments are reproducible and debuggable.

The fold suite command writes a single aggregate summary file at `reports/suite/suite_summary.json` and includes per-fold train/eval details for quick leaderboard comparisons.

The tiled inference command writes a JSON file with per-image summaries plus fused predictions in both `xyxy` and `xywh` formats.

The catalog manifest command writes a normalized index over the product bank so later retrieval code can consume a stable inventory rather than crawling directories at runtime.

## Current dataset observations

The first audit run on the provided dataset produced these high-value signals:

- 248 shelf images
- 22,731 annotations
- 356 categories
- 241 of 248 images contain more than 31 objects
- 1,126 tiny boxes and 18,395 small boxes
- 811 annotations touch an image edge

This validates the original plan: dense-scene handling and tiled inference are likely to matter more than swapping among standard detector families too early.
