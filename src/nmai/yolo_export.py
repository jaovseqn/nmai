from __future__ import annotations

import json
import shutil
from collections import defaultdict
from pathlib import Path

from .coco_utils import build_indexes, load_coco


def _safe_link_or_copy(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        if target.exists() or target.is_symlink():
            target.unlink()
        target.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, target)


def _to_yolo_bbox(x: float, y: float, width: float, height: float, image_width: int, image_height: int) -> tuple[float, float, float, float]:
    center_x = (x + width / 2.0) / image_width
    center_y = (y + height / 2.0) / image_height
    norm_width = width / image_width
    norm_height = height / image_height
    return center_x, center_y, norm_width, norm_height


def run(
    annotation_path: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    split_json: str | Path | None = None,
    val_fold: int = 0,
) -> None:
    coco = load_coco(annotation_path)
    images, categories, annotations_by_image = build_indexes(coco)

    images_dir = Path(images_dir)
    output_dir = Path(output_dir)

    category_ids = sorted(categories.keys())
    yolo_class_map = {category_id: index for index, category_id in enumerate(category_ids)}

    image_to_subset = defaultdict(lambda: "train")
    if split_json is not None:
        split_data = json.loads(Path(split_json).read_text(encoding="utf-8"))
        for image_id, fold_index in split_data["image_to_fold"].items():
            image_to_subset[int(image_id)] = "val" if int(fold_index) == val_fold else "train"

    for subset in ("train", "val"):
        (output_dir / "images" / subset).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / subset).mkdir(parents=True, exist_ok=True)

    for image_id, image in images.items():
        subset = image_to_subset[image_id]
        source_image = images_dir / image.file_name
        target_image = output_dir / "images" / subset / Path(image.file_name).name
        _safe_link_or_copy(source_image, target_image)

        label_file = output_dir / "labels" / subset / f"{Path(image.file_name).stem}.txt"
        label_lines: list[str] = []
        for annotation in annotations_by_image.get(image_id, []):
            x, y, width, height = annotation["bbox"]
            center_x, center_y, norm_width, norm_height = _to_yolo_bbox(
                x,
                y,
                width,
                height,
                image.width,
                image.height,
            )
            class_id = yolo_class_map[int(annotation["category_id"])]
            label_lines.append(
                f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}"
            )

        label_file.write_text("\n".join(label_lines), encoding="utf-8")

    names_yaml = "".join(
        f"  {index}: {categories[category_id].name}\n"
        for index, category_id in enumerate(category_ids)
    )
    data_yaml = (
        f"path: {output_dir.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"{names_yaml}"
    )
    (output_dir / "data.yaml").write_text(data_yaml, encoding="utf-8")
