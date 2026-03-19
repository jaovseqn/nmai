from __future__ import annotations

import json
from pathlib import Path

from .train_yolo import run as run_yolo_training
from .yolo_export import run as run_yolo_export


def _resolve_fold_indices(split_json_path: str | Path, folds: str | None) -> list[int]:
    split_data = json.loads(Path(split_json_path).read_text(encoding="utf-8"))
    if folds:
        return [int(value.strip()) for value in folds.split(",") if value.strip()]
    return sorted(int(fold_info["fold"]) for fold_info in split_data["folds"])


def run(
    annotation_path: str | Path,
    images_dir: str | Path,
    split_json: str | Path,
    experiment_config_path: str | Path,
    export_root: str | Path,
    project_dir: str | Path,
    folds: str | None = None,
) -> dict:
    fold_indices = _resolve_fold_indices(split_json, folds)

    export_root = Path(export_root)
    project_dir = Path(project_dir)

    runs: list[dict] = []
    for fold_index in fold_indices:
        fold_data_dir = export_root / f"fold_{fold_index}"
        run_yolo_export(
            annotation_path=annotation_path,
            images_dir=images_dir,
            output_dir=fold_data_dir,
            split_json=split_json,
            val_fold=fold_index,
        )

        data_yaml_path = fold_data_dir / "data.yaml"
        fold_project_dir = project_dir / f"fold_{fold_index}"
        run_yolo_training(
            experiment_config_path=experiment_config_path,
            data_yaml_path=data_yaml_path,
            project_dir=fold_project_dir,
        )

        runs.append(
            {
                "fold": fold_index,
                "data_yaml": str(data_yaml_path),
                "project_dir": str(fold_project_dir),
            }
        )

    return {
        "annotation_path": str(annotation_path),
        "images_dir": str(images_dir),
        "split_json": str(split_json),
        "experiment_config": str(experiment_config_path),
        "runs": runs,
    }
