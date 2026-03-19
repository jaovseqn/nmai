from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from .coco_utils import build_indexes, load_coco


def run(
    annotation_path: str | Path,
    num_folds: int = 5,
    seed: int = 42,
    output_path: str | Path | None = None,
) -> dict:
    coco = load_coco(annotation_path)
    images, categories, annotations_by_image = build_indexes(coco)

    rng = random.Random(seed)

    image_label_counts: dict[int, Counter] = {}
    for image_id in images:
        label_counter = Counter()
        for annotation in annotations_by_image.get(image_id, []):
            label_counter[int(annotation["category_id"])] += 1
        image_label_counts[image_id] = label_counter

    image_ids = list(images.keys())
    rng.shuffle(image_ids)
    image_ids.sort(key=lambda image_id: sum(image_label_counts[image_id].values()), reverse=True)

    fold_class_counts = [Counter() for _ in range(num_folds)]
    fold_sizes = [0 for _ in range(num_folds)]
    assignments: dict[int, int] = {}

    def fold_score(fold_index: int, image_id: int) -> tuple[int, int]:
        label_counter = image_label_counts[image_id]
        overlap = sum(fold_class_counts[fold_index][class_id] for class_id in label_counter)
        return overlap, fold_sizes[fold_index]

    for image_id in image_ids:
        best_fold = min(range(num_folds), key=lambda fold_index: fold_score(fold_index, image_id))
        assignments[image_id] = best_fold
        fold_class_counts[best_fold].update(image_label_counts[image_id])
        fold_sizes[best_fold] += 1

    result = {
        "seed": seed,
        "num_folds": num_folds,
        "folds": [
            {
                "fold": fold_index,
                "num_images": sum(1 for image_id in assignments if assignments[image_id] == fold_index),
                "num_annotations": sum(
                    len(annotations_by_image.get(image_id, []))
                    for image_id in assignments
                    if assignments[image_id] == fold_index
                ),
                "image_ids": sorted(
                    image_id for image_id in assignments if assignments[image_id] == fold_index
                ),
                "class_counts": {
                    categories[class_id].name: count
                    for class_id, count in sorted(fold_class_counts[fold_index].items())
                },
            }
            for fold_index in range(num_folds)
        ],
        "image_to_fold": {str(image_id): fold_index for image_id, fold_index in assignments.items()},
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    return result
