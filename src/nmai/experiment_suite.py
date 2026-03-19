from __future__ import annotations

import json
from pathlib import Path

from .eval_coco import run as run_coco_eval
from .fold_pipeline import _resolve_fold_indices
from .train_yolo import run as run_yolo_training
from .yolo_export import run as run_yolo_export


def _resolve_checkpoint_path(
    train_output: dict | None,
    fold_project_dir: Path,
    checkpoint_preference: str = "best",
) -> str | None:
    if train_output:
        if checkpoint_preference == "best" and train_output.get("best_weights"):
            return str(train_output["best_weights"])
        if checkpoint_preference == "last" and train_output.get("last_weights"):
            return str(train_output["last_weights"])

    pattern = "**/weights/best.pt" if checkpoint_preference == "best" else "**/weights/last.pt"
    candidates = sorted(
        fold_project_dir.glob(pattern),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return str(candidates[0]) if candidates else None


def run(
    annotation_path: str | Path,
    images_dir: str | Path,
    split_json: str | Path,
    experiment_config_path: str | Path,
    export_root: str | Path,
    project_dir: str | Path,
    reports_root: str | Path,
    folds: str | None = None,
    checkpoint_preference: str = "best",
    train: bool = True,
    eval_full: bool = True,
    eval_tiled: bool = True,
    tile_size: int = 1280,
    overlap: int = 256,
    conf: float = 0.2,
    iou_thr: float = 0.55,
    imgsz: int | None = None,
    device: str | int | None = None,
    max_det: int = 300,
) -> dict:
    export_root = Path(export_root)
    project_dir = Path(project_dir)
    reports_root = Path(reports_root)
    reports_root.mkdir(parents=True, exist_ok=True)

    fold_indices = _resolve_fold_indices(split_json, folds)
    summary_runs: list[dict] = []

    for fold_index in fold_indices:
        fold_data_dir = export_root / f"fold_{fold_index}"
        run_yolo_export(
            annotation_path=annotation_path,
            images_dir=images_dir,
            output_dir=fold_data_dir,
            split_json=split_json,
            val_fold=fold_index,
        )

        fold_project_dir = project_dir / f"fold_{fold_index}"
        train_output = None
        if train:
            train_output = run_yolo_training(
                experiment_config_path=experiment_config_path,
                data_yaml_path=fold_data_dir / "data.yaml",
                project_dir=fold_project_dir,
                run_name_override=f"fold_{fold_index}",
            )

        checkpoint_path = _resolve_checkpoint_path(train_output, fold_project_dir, checkpoint_preference)

        fold_summary: dict = {
            "fold": fold_index,
            "data_yaml": str(fold_data_dir / "data.yaml"),
            "project_dir": str(fold_project_dir),
            "checkpoint_path": checkpoint_path,
            "train_output": train_output,
            "eval_full": None,
            "eval_tiled": None,
            "warnings": [],
        }

        if checkpoint_path is None:
            fold_summary["warnings"].append("No checkpoint found for this fold. Evaluation skipped.")
            summary_runs.append(fold_summary)
            continue

        if eval_full:
            try:
                full_eval_path = reports_root / f"eval_fold{fold_index}_full.json"
                full_eval = run_coco_eval(
                    annotation_path=annotation_path,
                    images_dir=images_dir,
                    model_path=checkpoint_path,
                    output_path=full_eval_path,
                    split_json=split_json,
                    val_fold=fold_index,
                    tiled=False,
                    conf=conf,
                    iou_thr=iou_thr,
                    imgsz=imgsz,
                    device=device,
                    max_det=max_det,
                )
                fold_summary["eval_full"] = {
                    "path": str(full_eval_path),
                    "metrics": full_eval["metrics"],
                }
            except RuntimeError as exc:
                fold_summary["warnings"].append(str(exc))

        if eval_tiled:
            try:
                tiled_eval_path = reports_root / f"eval_fold{fold_index}_tiled.json"
                tiled_eval = run_coco_eval(
                    annotation_path=annotation_path,
                    images_dir=images_dir,
                    model_path=checkpoint_path,
                    output_path=tiled_eval_path,
                    split_json=split_json,
                    val_fold=fold_index,
                    tiled=True,
                    tile_size=tile_size,
                    overlap=overlap,
                    conf=conf,
                    iou_thr=iou_thr,
                    imgsz=imgsz,
                    device=device,
                    max_det=max_det,
                )
                fold_summary["eval_tiled"] = {
                    "path": str(tiled_eval_path),
                    "metrics": tiled_eval["metrics"],
                }
            except RuntimeError as exc:
                fold_summary["warnings"].append(str(exc))

        summary_runs.append(fold_summary)

    def _mean(values: list[float]) -> float | None:
        if not values:
            return None
        return float(sum(values) / len(values))

    full_map = [
        run_info["eval_full"]["metrics"]["mAP_50_95"]
        for run_info in summary_runs
        if run_info.get("eval_full") is not None
    ]
    tiled_map = [
        run_info["eval_tiled"]["metrics"]["mAP_50_95"]
        for run_info in summary_runs
        if run_info.get("eval_tiled") is not None
    ]

    aggregate = {
        "annotation_path": str(annotation_path),
        "images_dir": str(images_dir),
        "split_json": str(split_json),
        "experiment_config": str(experiment_config_path),
        "checkpoint_preference": checkpoint_preference,
        "train_enabled": train,
        "eval_full_enabled": eval_full,
        "eval_tiled_enabled": eval_tiled,
        "tile_size": tile_size if eval_tiled else None,
        "overlap": overlap if eval_tiled else None,
        "folds": fold_indices,
        "mean_map_50_95_full": _mean(full_map),
        "mean_map_50_95_tiled": _mean(tiled_map),
        "runs": summary_runs,
    }

    summary_path = reports_root / "suite_summary.json"
    summary_path.write_text(json.dumps(aggregate, indent=2, ensure_ascii=False), encoding="utf-8")
    aggregate["summary_path"] = str(summary_path)
    return aggregate
