from __future__ import annotations

from pathlib import Path

from .config import load_yaml


def run(
    experiment_config_path: str | Path,
    data_yaml_path: str | Path,
    project_dir: str | Path = "artifacts",
    run_name_override: str | None = None,
) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "ultralytics is not installed. Install training extras with `pip install -e .[train]`."
        ) from exc

    experiment_config = load_yaml(experiment_config_path)
    model_name = experiment_config.pop("model")
    run_name = run_name_override or experiment_config.pop("name", Path(experiment_config_path).stem)
    experiment_config.pop("notes", None)

    model = YOLO(model_name)
    result = model.train(data=str(data_yaml_path), project=str(project_dir), name=run_name, **experiment_config)

    save_dir = None
    if hasattr(model, "trainer") and getattr(model.trainer, "save_dir", None) is not None:
        save_dir = Path(model.trainer.save_dir)
    elif hasattr(result, "save_dir") and getattr(result, "save_dir") is not None:
        save_dir = Path(result.save_dir)

    best_weights = None
    last_weights = None
    if save_dir is not None:
        best_path = save_dir / "weights" / "best.pt"
        last_path = save_dir / "weights" / "last.pt"
        best_weights = str(best_path) if best_path.exists() else None
        last_weights = str(last_path) if last_path.exists() else None

    return {
        "run_name": run_name,
        "save_dir": str(save_dir) if save_dir is not None else None,
        "best_weights": best_weights,
        "last_weights": last_weights,
    }
