from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .coco_utils import build_indexes, load_coco


def _bucket_bbox(area_ratio: float) -> str:
    if area_ratio < 0.001:
        return "tiny"
    if area_ratio < 0.01:
        return "small"
    if area_ratio < 0.05:
        return "medium"
    return "large"


def run(annotation_path: str | Path, output_path: str | Path | None = None) -> dict:
    coco = load_coco(annotation_path)
    images, categories, annotations_by_image = build_indexes(coco)

    class_counts = Counter()
    box_size_counts = Counter()
    image_density = Counter()
    edge_touch_count = 0

    for image_id, annotations in annotations_by_image.items():
        image = images[image_id]
        image_density[len(annotations)] += 1

        for annotation in annotations:
            category_id = int(annotation["category_id"])
            class_counts[category_id] += 1

            x, y, width, height = annotation["bbox"]
            area_ratio = (width * height) / max(1, image.width * image.height)
            box_size_counts[_bucket_bbox(area_ratio)] += 1

            near_left = x <= 1
            near_top = y <= 1
            near_right = x + width >= image.width - 1
            near_bottom = y + height >= image.height - 1
            if near_left or near_top or near_right or near_bottom:
                edge_touch_count += 1

    class_summary = [
        {
            "category_id": category_id,
            "name": categories[category_id].name,
            "count": class_counts[category_id],
        }
        for category_id in sorted(categories)
    ]
    class_summary.sort(key=lambda item: item["count"], reverse=True)

    report = {
        "num_images": len(images),
        "num_annotations": len(coco["annotations"]),
        "num_categories": len(categories),
        "images_without_annotations": sum(1 for image_id in images if image_id not in annotations_by_image),
        "edge_touch_annotations": edge_touch_count,
        "bbox_size_buckets": dict(box_size_counts),
        "dense_image_histogram": {
            "1-5": sum(count for objects, count in image_density.items() if 1 <= objects <= 5),
            "6-15": sum(count for objects, count in image_density.items() if 6 <= objects <= 15),
            "16-30": sum(count for objects, count in image_density.items() if 16 <= objects <= 30),
            "31+": sum(count for objects, count in image_density.items() if objects >= 31),
        },
        "top_classes": class_summary[:25],
        "tail_classes": [item for item in class_summary if item["count"] <= 10],
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    return report
