from __future__ import annotations
import json
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from PIL import Image

def yolo_to_coco(
    images_root: str | Path,
    labels_root: str | Path,
    output_dir: str | Path,
    categories: Optional[List[str]] = None,
    image_exts: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".webp"),
    rounding: int = 2,
    strict_mirror: bool = False,
) -> Path:
    """
    Convierte anotaciones YOLO (class cx cy w h normalizados) a un único COCO JSON (instances),
    preservando subcarpetas mediante file_name relativo y enlazando cada annotation con image_id.
    """
    images_root = Path(images_root)
    labels_root = Path(labels_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / "annotations.json"

    # Categorías por defecto (IDs COCO 1..K)
    if categories is None:
        categories = ["flowering", "growing_g", "growing_w", "nearly_m", "mature"]
    coco_categories = [{"id": i + 1, "name": name, "supercategory": "strawberry"}
                       for i, name in enumerate(categories)]

    # Indexar posibles .txt si no hay estructura espejo perfecta
    all_label_paths = list(labels_root.rglob("*.txt"))
    labels_by_stem = {}
    for lp in all_label_paths:
        labels_by_stem.setdefault(lp.stem, []).append(lp)

    def find_label_for(img_path: Path) -> Optional[Path]:
        # Caso 1: estructura espejo: labels_root/<subdirs>/<name>.txt
        rel = img_path.relative_to(images_root)
        candidate = labels_root / rel.with_suffix(".txt")
        if candidate.exists():
            return candidate
        if strict_mirror:
            return None
        # Caso 2: buscar por nombre (si es único)
        cands = labels_by_stem.get(img_path.stem, [])
        if len(cands) == 1:
            return cands[0]
        return None

    # Recolectar imágenes
    image_paths = []
    for ext in image_exts:
        image_paths.extend(sorted(images_root.rglob(f"*{ext}")))

    coco_images: List[Dict[str, Any]] = []
    coco_annotations: List[Dict[str, Any]] = []
    img_id = 1
    ann_id = 1

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    for img_path in image_paths:
        # Ruta relativa para COCO
        rel_file = img_path.relative_to(images_root).as_posix()

        with Image.open(img_path) as im:
            W, H = im.size

        coco_images.append({
            "id": img_id,
            "file_name": rel_file,  # ruta relativa
            "width": W,
            "height": H,
        })

        label_path = find_label_for(img_path)
        if label_path and label_path.exists():
            lines = [ln.strip() for ln in label_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
            for ln in lines:
                parts = ln.split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:5])

                # YOLO -> COCO xywh en píxeles
                px_w = w * W
                px_h = h * H
                px_x = (cx - w / 2.0) * W
                px_y = (cy - h / 2.0) * H

                x1 = clamp(px_x, 0, W)
                y1 = clamp(px_y, 0, H)
                x2 = clamp(px_x + px_w, 0, W)
                y2 = clamp(px_y + px_h, 0, H)
                bw = max(0.0, x2 - x1)
                bh = max(0.0, y2 - y1)

                coco_annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": cls + 1,  # COCO 1..K
                    "bbox": [round(x1, rounding), round(y1, rounding),
                             round(bw, rounding), round(bh, rounding)],
                    "area": round(bw * bh, rounding),
                    "iscrowd": 0,
                    "segmentation": [],
                })
                ann_id += 1

        img_id += 1

    coco_dict = {
        "info": {"version": "1.0", "description": "YOLO->COCO (recursive)"},
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": coco_categories,
    }

    out_json.write_text(json.dumps(coco_dict, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_json


yolo_to_coco(
     images_root="src/data/raw/images",        
     labels_root="src/data/raw/labels",        
     output_dir="src/data/processed/coco_out"
)
