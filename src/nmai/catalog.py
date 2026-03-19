from __future__ import annotations

import json
import unicodedata
from pathlib import Path


def normalize_product_name(name: str) -> str:
    text = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    text = text.lower().replace("&", " and ")
    allowed = []
    for char in text:
        allowed.append(char if char.isalnum() else " ")
    return " ".join("".join(allowed).split())


def build_manifest(catalog_root: str | Path) -> dict:
    catalog_root = Path(catalog_root)
    metadata_path = catalog_root / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    products = metadata.get("products", [])
    products_by_code = {str(product["product_code"]): product for product in products}
    product_dirs = {path.name: path for path in catalog_root.iterdir() if path.is_dir()}

    entries: list[dict] = []
    all_product_codes = sorted(set(products_by_code) | set(product_dirs))

    for product_code in all_product_codes:
        product_dir = product_dirs.get(product_code)
        metadata_entry = products_by_code.get(product_code, {})
        image_paths = []
        if product_dir is not None:
            image_paths = sorted(
                str(path)
                for path in product_dir.iterdir()
                if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
            )

        if product_code in products_by_code and product_dir is not None:
            source = "metadata_and_directory"
        elif product_code in products_by_code:
            source = "metadata_only"
        else:
            source = "directory_only"

        entries.append(
            {
                "product_code": product_code,
                "product_name": metadata_entry.get("product_name", ""),
                "normalized_name": normalize_product_name(metadata_entry.get("product_name", "")),
                "annotation_count": int(metadata_entry.get("annotation_count", 0)),
                "corrected_count": int(metadata_entry.get("corrected_count", 0)),
                "has_images": bool(image_paths),
                "image_count": len(image_paths),
                "image_types": metadata_entry.get("image_types", []),
                "image_paths": image_paths,
                "source": source,
            }
        )

    manifest = {
        "catalog_root": str(catalog_root),
        "metadata_path": str(metadata_path),
        "metadata_total_products": int(metadata.get("total_products", len(products))),
        "total_products": len(entries),
        "products_with_images": sum(1 for entry in entries if entry["has_images"]),
        "products_without_images": sum(1 for entry in entries if not entry["has_images"]),
        "products_with_metadata": sum(1 for entry in entries if entry["source"] != "directory_only"),
        "products_with_directories": sum(1 for entry in entries if entry["source"] != "metadata_only"),
        "metadata_only_products": sum(1 for entry in entries if entry["source"] == "metadata_only"),
        "directory_only_products": sum(1 for entry in entries if entry["source"] == "directory_only"),
        "products": entries,
    }
    return manifest


def run(catalog_root: str | Path, output_path: str | Path | None = None) -> dict:
    manifest = build_manifest(catalog_root)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return manifest
