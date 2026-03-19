from __future__ import annotations

import json
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _require_runtime_dependencies() -> tuple[object, object, object, object]:
    try:
        import numpy as np
        from PIL import Image
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Tiled inference requires pillow and ultralytics. Install with `pip install -e .[train]`."
        ) from exc

    try:
        from ensemble_boxes import weighted_boxes_fusion
    except ImportError:
        weighted_boxes_fusion = None

    return np, Image, YOLO, weighted_boxes_fusion


def _iter_images(input_path: str | Path) -> list[Path]:
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    return sorted(
        path for path in input_path.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _tile_starts(full_size: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size >= full_size:
        return [0]

    stride = max(1, tile_size - overlap)
    starts = list(range(0, max(1, full_size - tile_size + 1), stride))
    final_start = full_size - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def _generate_tiles(width: int, height: int, tile_size: int, overlap: int) -> list[tuple[int, int, int, int]]:
    x_starts = _tile_starts(width, tile_size, overlap)
    y_starts = _tile_starts(height, tile_size, overlap)
    return [
        (x0, y0, min(width, x0 + tile_size), min(height, y0 + tile_size))
        for y0 in y_starts
        for x0 in x_starts
    ]


def _clip_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> tuple[float, float, float, float] | None:
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    x2 = max(0.0, min(float(width), x2))
    y2 = max(0.0, min(float(height), y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter_area
    return inter_area / denom if denom > 0.0 else 0.0


def _nms_fallback(boxes: list[list[float]], scores: list[float], labels: list[int], iou_thr: float) -> tuple[list[list[float]], list[float], list[int]]:
    kept_boxes: list[list[float]] = []
    kept_scores: list[float] = []
    kept_labels: list[int] = []

    indices = sorted(range(len(boxes)), key=lambda idx: scores[idx], reverse=True)
    for index in indices:
        candidate_box = boxes[index]
        candidate_label = labels[index]
        if any(candidate_label == existing_label and _iou(candidate_box, existing_box) >= iou_thr for existing_box, existing_label in zip(kept_boxes, kept_labels)):
            continue
        kept_boxes.append(candidate_box)
        kept_scores.append(scores[index])
        kept_labels.append(candidate_label)

    return kept_boxes, kept_scores, kept_labels


def _fuse_predictions(
    model_boxes: list[list[list[float]]],
    model_scores: list[list[float]],
    model_labels: list[list[int]],
    iou_thr: float,
    skip_box_thr: float,
    weighted_boxes_fusion: object | None,
) -> tuple[list[list[float]], list[float], list[int]]:
    flat_boxes = [box for boxes in model_boxes for box in boxes]
    flat_scores = [score for scores in model_scores for score in scores]
    flat_labels = [label for labels in model_labels for label in labels]

    if not flat_boxes:
        return [], [], []

    if weighted_boxes_fusion is None:
        return _nms_fallback(flat_boxes, flat_scores, flat_labels, iou_thr)

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        model_boxes,
        model_scores,
        model_labels,
        weights=None,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
    )
    return fused_boxes, fused_scores.tolist(), fused_labels.astype(int).tolist()


def predict_tiled_image(
    model: object,
    image_path: str | Path,
    tile_size: int = 1280,
    overlap: int = 256,
    conf: float = 0.2,
    iou_thr: float = 0.55,
    imgsz: int | None = None,
    device: str | int | None = None,
    max_det: int = 300,
    weighted_boxes_fusion: object | None = None,
    np: object | None = None,
    Image: object | None = None,
) -> dict:
    if np is None or Image is None:
        np, Image, _, default_wbf = _require_runtime_dependencies()
        if weighted_boxes_fusion is None:
            weighted_boxes_fusion = default_wbf

    image_path = Path(image_path)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    tiles = _generate_tiles(width, height, tile_size, overlap)

    model_boxes: list[list[list[float]]] = []
    model_scores: list[list[float]] = []
    model_labels: list[list[int]] = []

    for x0, y0, x1, y1 in tiles:
        tile_array = np.array(image.crop((x0, y0, x1, y1)))
        results = model.predict(
            source=tile_array,
            conf=conf,
            imgsz=imgsz or tile_size,
            iou=iou_thr,
            max_det=max_det,
            device=device,
            verbose=False,
        )
        result = results[0]

        tile_boxes: list[list[float]] = []
        tile_scores: list[float] = []
        tile_labels: list[int] = []

        if result.boxes is not None and len(result.boxes) > 0:
            xyxy = result.boxes.xyxy.cpu().tolist()
            scores = result.boxes.conf.cpu().tolist()
            labels = result.boxes.cls.cpu().tolist()

            for box, score, label in zip(xyxy, scores, labels):
                local_x1, local_y1, local_x2, local_y2 = box
                clipped_box = _clip_box(
                    x0 + local_x1,
                    y0 + local_y1,
                    x0 + local_x2,
                    y0 + local_y2,
                    width,
                    height,
                )
                if clipped_box is None:
                    continue
                global_x1, global_y1, global_x2, global_y2 = clipped_box
                tile_boxes.append(
                    [
                        global_x1 / width,
                        global_y1 / height,
                        global_x2 / width,
                        global_y2 / height,
                    ]
                )
                tile_scores.append(float(score))
                tile_labels.append(int(label))

        model_boxes.append(tile_boxes)
        model_scores.append(tile_scores)
        model_labels.append(tile_labels)

    fused_boxes, fused_scores, fused_labels = _fuse_predictions(
        model_boxes,
        model_scores,
        model_labels,
        iou_thr=iou_thr,
        skip_box_thr=conf,
        weighted_boxes_fusion=weighted_boxes_fusion,
    )

    predictions: list[dict] = []
    for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
        x1_norm, y1_norm, x2_norm, y2_norm = box
        x1 = x1_norm * width
        y1 = y1_norm * height
        x2 = x2_norm * width
        y2 = y2_norm * height
        predictions.append(
            {
                "category_id": int(label),
                "score": float(score),
                "bbox_xyxy": [x1, y1, x2, y2],
                "bbox_xywh": [x1, y1, x2 - x1, y2 - y1],
            }
        )

    return {
        "image_path": str(image_path),
        "width": width,
        "height": height,
        "tile_count": len(tiles),
        "prediction_count": len(predictions),
        "predictions": predictions,
    }


def run(
    model_path: str | Path,
    input_path: str | Path,
    output_path: str | Path,
    tile_size: int = 1280,
    overlap: int = 256,
    conf: float = 0.2,
    iou_thr: float = 0.55,
    imgsz: int | None = None,
    device: str | int | None = None,
    max_det: int = 300,
) -> dict:
    np, Image, YOLO, weighted_boxes_fusion = _require_runtime_dependencies()

    image_paths = _iter_images(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    class_names = model.names
    predictions: list[dict] = []
    image_summaries: list[dict] = []

    for image_path in image_paths:
        image_output = predict_tiled_image(
            model=model,
            image_path=image_path,
            tile_size=tile_size,
            overlap=overlap,
            conf=conf,
            iou_thr=iou_thr,
            imgsz=imgsz,
            device=device,
            max_det=max_det,
            weighted_boxes_fusion=weighted_boxes_fusion,
            np=np,
            Image=Image,
        )

        width = int(image_output["width"])
        height = int(image_output["height"])
        image_summaries.append(
            {
                "image_path": image_output["image_path"],
                "width": width,
                "height": height,
                "tile_count": image_output["tile_count"],
                "prediction_count": image_output["prediction_count"],
            }
        )

        for prediction in image_output["predictions"]:
            x1, y1, x2, y2 = prediction["bbox_xyxy"]
            label = int(prediction["category_id"])
            score = float(prediction["score"])
            predictions.append(
                {
                    "image_path": str(image_path),
                    "category_id": label,
                    "category_name": str(class_names[int(label)]),
                    "score": score,
                    "bbox_xyxy": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
                    "bbox_xywh": [round(x1, 3), round(y1, 3), round(x2 - x1, 3), round(y2 - y1, 3)],
                }
            )

    output = {
        "model_path": str(model_path),
        "input_path": str(input_path),
        "tile_size": tile_size,
        "overlap": overlap,
        "confidence_threshold": conf,
        "iou_threshold": iou_thr,
        "num_images": len(image_paths),
        "num_predictions": len(predictions),
        "images": image_summaries,
        "predictions": predictions,
    }
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    return output
