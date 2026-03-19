from __future__ import annotations

import argparse
import json

from . import __version__
from . import audit, splits, yolo_export
from .catalog import run as run_catalog_manifest
from .eval_coco import run as run_coco_eval
from .experiment_suite import run as run_experiment_suite
from .fold_pipeline import run as run_fold_pipeline
from .tiled_inference import run as run_tiled_inference
from .train_yolo import run as run_yolo_training


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nmai")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", required=True)

    audit_parser = subparsers.add_parser("audit", help="Audit COCO annotations")
    audit_parser.add_argument("--annotations", required=True)
    audit_parser.add_argument("--out")

    split_parser = subparsers.add_parser("split", help="Create deterministic folds")
    split_parser.add_argument("--annotations", required=True)
    split_parser.add_argument("--num-folds", type=int, default=5)
    split_parser.add_argument("--seed", type=int, default=42)
    split_parser.add_argument("--out")

    catalog_parser = subparsers.add_parser("catalog-manifest", help="Index catalog metadata and images")
    catalog_parser.add_argument("--catalog-root", required=True)
    catalog_parser.add_argument("--out")

    export_parser = subparsers.add_parser("export-yolo", help="Export COCO data to YOLO format")
    export_parser.add_argument("--annotations", required=True)
    export_parser.add_argument("--images-dir", required=True)
    export_parser.add_argument("--out-dir", required=True)
    export_parser.add_argument("--split-json")
    export_parser.add_argument("--val-fold", type=int, default=0)

    train_parser = subparsers.add_parser("train-yolo", help="Train a YOLO baseline from config")
    train_parser.add_argument("--experiment-config", required=True)
    train_parser.add_argument("--data-yaml", required=True)
    train_parser.add_argument("--project-dir", default="artifacts")

    train_folds_parser = subparsers.add_parser(
        "train-yolo-folds",
        help="Export and train YOLO across one or more folds",
    )
    train_folds_parser.add_argument("--annotations", required=True)
    train_folds_parser.add_argument("--images-dir", required=True)
    train_folds_parser.add_argument("--split-json", required=True)
    train_folds_parser.add_argument("--experiment-config", required=True)
    train_folds_parser.add_argument("--export-root", default="data/yolo_folds")
    train_folds_parser.add_argument("--project-dir", default="artifacts/folds")
    train_folds_parser.add_argument(
        "--folds",
        help="Comma-separated fold indices, e.g. 0,1,2. Default is all folds.",
    )

    eval_parser = subparsers.add_parser("eval-coco", help="Evaluate YOLO predictions with COCO metrics")
    eval_parser.add_argument("--annotations", required=True)
    eval_parser.add_argument("--images-dir", required=True)
    eval_parser.add_argument("--model-path", required=True)
    eval_parser.add_argument("--out", required=True)
    eval_parser.add_argument("--split-json")
    eval_parser.add_argument("--val-fold", type=int)
    eval_parser.add_argument("--tiled", action="store_true")
    eval_parser.add_argument("--tile-size", type=int, default=1280)
    eval_parser.add_argument("--overlap", type=int, default=256)
    eval_parser.add_argument("--conf", type=float, default=0.2)
    eval_parser.add_argument("--iou-thr", type=float, default=0.55)
    eval_parser.add_argument("--imgsz", type=int)
    eval_parser.add_argument("--device")
    eval_parser.add_argument("--max-det", type=int, default=300)

    suite_parser = subparsers.add_parser(
        "run-fold-suite",
        help="Run export, optional training, and optional evaluation across folds",
    )
    suite_parser.add_argument("--annotations", required=True)
    suite_parser.add_argument("--images-dir", required=True)
    suite_parser.add_argument("--split-json", required=True)
    suite_parser.add_argument("--experiment-config", required=True)
    suite_parser.add_argument("--export-root", default="data/yolo_folds")
    suite_parser.add_argument("--project-dir", default="artifacts/folds")
    suite_parser.add_argument("--reports-root", default="reports/suite")
    suite_parser.add_argument("--folds", help="Comma-separated fold indices. Default is all folds.")
    suite_parser.add_argument("--checkpoint-preference", choices=["best", "last"], default="best")
    suite_parser.add_argument("--skip-train", action="store_true")
    suite_parser.add_argument("--skip-eval-full", action="store_true")
    suite_parser.add_argument("--skip-eval-tiled", action="store_true")
    suite_parser.add_argument("--tile-size", type=int, default=1280)
    suite_parser.add_argument("--overlap", type=int, default=256)
    suite_parser.add_argument("--conf", type=float, default=0.2)
    suite_parser.add_argument("--iou-thr", type=float, default=0.55)
    suite_parser.add_argument("--imgsz", type=int)
    suite_parser.add_argument("--device")
    suite_parser.add_argument("--max-det", type=int, default=300)

    predict_parser = subparsers.add_parser(
        "predict-yolo-tiles",
        help="Run tiled YOLO inference with box fusion over one image or a directory",
    )
    predict_parser.add_argument("--model-path", required=True)
    predict_parser.add_argument("--input-path", required=True)
    predict_parser.add_argument("--out", required=True)
    predict_parser.add_argument("--tile-size", type=int, default=1280)
    predict_parser.add_argument("--overlap", type=int, default=256)
    predict_parser.add_argument("--conf", type=float, default=0.2)
    predict_parser.add_argument("--iou-thr", type=float, default=0.55)
    predict_parser.add_argument("--imgsz", type=int)
    predict_parser.add_argument("--device")
    predict_parser.add_argument("--max-det", type=int, default=300)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "audit":
        report = audit.run(args.annotations, args.out)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    if args.command == "split":
        result = splits.run(args.annotations, args.num_folds, args.seed, args.out)
        summary = {
            "num_folds": result["num_folds"],
            "fold_sizes": [fold["num_images"] for fold in result["folds"]],
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    if args.command == "catalog-manifest":
        manifest = run_catalog_manifest(args.catalog_root, args.out)
        print(
            json.dumps(
                {
                    "total_products": manifest["total_products"],
                    "products_with_images": manifest["products_with_images"],
                    "products_without_images": manifest["products_without_images"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    if args.command == "export-yolo":
        yolo_export.run(
            annotation_path=args.annotations,
            images_dir=args.images_dir,
            output_dir=args.out_dir,
            split_json=args.split_json,
            val_fold=args.val_fold,
        )
        print("YOLO export completed.")
        return

    if args.command == "train-yolo":
        output = run_yolo_training(
            experiment_config_path=args.experiment_config,
            data_yaml_path=args.data_yaml,
            project_dir=args.project_dir,
        )
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return

    if args.command == "train-yolo-folds":
        output = run_fold_pipeline(
            annotation_path=args.annotations,
            images_dir=args.images_dir,
            split_json=args.split_json,
            experiment_config_path=args.experiment_config,
            export_root=args.export_root,
            project_dir=args.project_dir,
            folds=args.folds,
        )
        print(
            json.dumps(
                {
                    "num_runs": len(output["runs"]),
                    "folds": [run_info["fold"] for run_info in output["runs"]],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    if args.command == "eval-coco":
        output = run_coco_eval(
            annotation_path=args.annotations,
            images_dir=args.images_dir,
            model_path=args.model_path,
            output_path=args.out,
            split_json=args.split_json,
            val_fold=args.val_fold,
            tiled=args.tiled,
            tile_size=args.tile_size,
            overlap=args.overlap,
            conf=args.conf,
            iou_thr=args.iou_thr,
            imgsz=args.imgsz,
            device=args.device,
            max_det=args.max_det,
        )
        print(json.dumps(output["metrics"], indent=2, ensure_ascii=False))
        return

    if args.command == "run-fold-suite":
        output = run_experiment_suite(
            annotation_path=args.annotations,
            images_dir=args.images_dir,
            split_json=args.split_json,
            experiment_config_path=args.experiment_config,
            export_root=args.export_root,
            project_dir=args.project_dir,
            reports_root=args.reports_root,
            folds=args.folds,
            checkpoint_preference=args.checkpoint_preference,
            train=not args.skip_train,
            eval_full=not args.skip_eval_full,
            eval_tiled=not args.skip_eval_tiled,
            tile_size=args.tile_size,
            overlap=args.overlap,
            conf=args.conf,
            iou_thr=args.iou_thr,
            imgsz=args.imgsz,
            device=args.device,
            max_det=args.max_det,
        )
        print(
            json.dumps(
                {
                    "summary_path": output["summary_path"],
                    "folds": output["folds"],
                    "mean_map_50_95_full": output["mean_map_50_95_full"],
                    "mean_map_50_95_tiled": output["mean_map_50_95_tiled"],
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    if args.command == "predict-yolo-tiles":
        output = run_tiled_inference(
            model_path=args.model_path,
            input_path=args.input_path,
            output_path=args.out,
            tile_size=args.tile_size,
            overlap=args.overlap,
            conf=args.conf,
            iou_thr=args.iou_thr,
            imgsz=args.imgsz,
            device=args.device,
            max_det=args.max_det,
        )
        print(
            json.dumps(
                {
                    "num_images": output["num_images"],
                    "num_predictions": output["num_predictions"],
                    "output_path": args.out,
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return


if __name__ == "__main__":
    main()
