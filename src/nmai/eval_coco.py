from __future__ import annotations

import json
from pathlib import Path

from .coco_utils import build_indexes, load_coco
from .tiled_inference import predict_tiled_image


def _require_eval_dependencies() -> tuple[object, object, object]:
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "COCO evaluation requires ultralytics and pycocotools. Install with `pip install -e .[train,eval]`."
        ) from exc

    return COCO, COCOeval, YOLO


def _load_val_image_ids(split_json: str | Path | None, val_fold: int | None, all_image_ids: list[int]) -> set[int]:
    if split_json is None:
        return set(all_image_ids)

    split_data = json.loads(Path(split_json).read_text(encoding="utf-8"))
    image_to_fold = split_data.get("image_to_fold", {})
    if val_fold is None:
        raise ValueError("val_fold must be provided when split_json is provided.")

    return {
        int(image_id)
        for image_id, fold_index in image_to_fold.items()
        if int(fold_index) == int(val_fold)
    }


def run(
    annotation_path: str | Path,
    images_dir: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    split_json: str | Path | None = None,
    val_fold: int | None = None,
    tiled: bool = False,
    tile_size: int = 1280,
    overlap: int = 256,
    conf: float = 0.2,
    iou_thr: float = 0.55,
    imgsz: int | None = None,
    device: str | int | None = None,
    max_det: int = 300,
) -> dict:
    COCO, COCOeval, YOLO = _require_eval_dependencies()

    images_dir = Path(images_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    coco = load_coco(annotation_path)
    images, categories, _ = build_indexes(coco)

    val_image_ids = _load_val_image_ids(split_json, val_fold, list(images.keys()))
    val_images = [images[image_id] for image_id in sorted(val_image_ids)]

    sorted_category_ids = sorted(categories)
    yolo_to_coco = {index: category_id for index, category_id in enumerate(sorted_category_ids)}

    model = YOLO(str(model_path))
    detections: list[dict] = []

    for image in val_images:
        image_path = images_dir / image.file_name
        if not image_path.exists():
            continue

        if tiled:
            tiled_output = predict_tiled_image(
                model=model,
                image_path=image_path,
                tile_size=tile_size,
                overlap=overlap,
                conf=conf,
                iou_thr=iou_thr,
                imgsz=imgsz,
                device=device,
                max_det=max_det,
            )
            for prediction in tiled_output["predictions"]:
                yolo_id = int(prediction["category_id"])
                if yolo_id not in yolo_to_coco:
                    continue
                x, y, width, height = prediction["bbox_xywh"]
                detections.append(
                    {
                        "image_id": int(image.id),
                        "category_id": int(yolo_to_coco[yolo_id]),
                        "bbox": [float(x), float(y), float(width), float(height)],
                        "score": float(prediction["score"]),
                    }
                )
        else:
            results = model.predict(
                source=str(image_path),
                conf=conf,
                iou=iou_thr,
                imgsz=imgsz,
                device=device,
                max_det=max_det,
                verbose=False,
            )
            result = results[0]
            if result.boxes is None or len(result.boxes) == 0:
                continue

            xyxy = result.boxes.xyxy.cpu().tolist()
            scores = result.boxes.conf.cpu().tolist()
            labels = result.boxes.cls.cpu().tolist()

            for box, score, yolo_label in zip(xyxy, scores, labels):
                yolo_id = int(yolo_label)
                if yolo_id not in yolo_to_coco:
                    continue

                x1, y1, x2, y2 = box
                detections.append(
                    {
                        "image_id": int(image.id),
                        "category_id": int(yolo_to_coco[yolo_id]),
                        "bbox": [
                            float(x1),
                            float(y1),
                            float(max(0.0, x2 - x1)),
                            float(max(0.0, y2 - y1)),
                        ],
                        "score": float(score),
                    }
                )

    coco_gt = COCO(str(annotation_path))
    eval_image_ids = [int(image.id) for image in val_images]

    if detections:
        detections_path = output_path.with_suffix(".detections.json")
        detections_path.write_text(json.dumps(detections, ensure_ascii=False), encoding="utf-8")
        coco_dt = coco_gt.loadRes(str(detections_path))
        evaluator = COCOeval(coco_gt, coco_dt, iouType="bbox")
        evaluator.params.imgIds = eval_image_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        stats = [float(value) for value in evaluator.stats.tolist()]
    else:
        stats = [0.0] * 12

    summary = {
        "annotation_path": str(annotation_path),
        "images_dir": str(images_dir),
        "model_path": str(model_path),
        "split_json": str(split_json) if split_json is not None else None,
        "val_fold": val_fold,
        "tiled": tiled,
        "tile_size": tile_size if tiled else None,
        "overlap": overlap if tiled else None,
        "num_eval_images": len(eval_image_ids),
        "num_detections": len(detections),
        "metrics": {
            "mAP_50_95": stats[0],
            "mAP_50": stats[1],
            "mAP_75": stats[2],
            "mAP_small": stats[3],
            "mAP_medium": stats[4],
            "mAP_large": stats[5],
            "AR_1": stats[6],
            "AR_10": stats[7],
            "AR_100": stats[8],
            "AR_small": stats[9],
            "AR_medium": stats[10],
            "AR_large": stats[11],
        },
    }
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
