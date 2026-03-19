from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CocoImage:
    id: int
    file_name: str
    width: int
    height: int


@dataclass(frozen=True)
class CocoCategory:
    id: int
    name: str


def load_coco(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    required_keys = {"images", "annotations", "categories"}
    missing_keys = required_keys - set(data.keys())
    if missing_keys:
        raise ValueError(f"Invalid COCO file. Missing keys: {sorted(missing_keys)}")

    return data


def build_indexes(coco: dict) -> tuple[dict[int, CocoImage], dict[int, CocoCategory], dict[int, list[dict]]]:
    images = {
        int(image["id"]): CocoImage(
            id=int(image["id"]),
            file_name=str(image["file_name"]),
            width=int(image["width"]),
            height=int(image["height"]),
        )
        for image in coco["images"]
    }
    categories = {
        int(category["id"]): CocoCategory(id=int(category["id"]), name=str(category["name"]))
        for category in coco["categories"]
    }

    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    return images, categories, annotations_by_image
