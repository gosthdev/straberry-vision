"""Run quick evaluations of a trained SGSNet checkpoint over single images or folders.

Usage examples
--------------
Evaluate a single file and show detections on screen::
    python -m test.evaluate_trained_model --checkpoint src/data/processed/models/best_model.pth --image test/files/fresa1.webp

Process all images inside a folder saving annotated copies::
    python -m test.evaluate_trained_model --checkpoint src/data/processed/models/best_model.pth --folder test/files --save-images
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch

# Import model components and utilities
def _ensure_project_root_on_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))


_ensure_project_root_on_path()

from src.core.model import (
    Config,
    SGSNet,
    non_max_suppression,
    load_data,
)

# Valores predeterminados para ejecución directa (por ejemplo, botón "Run" en VS Code)
DEFAULT_RUN_CONFIG = {
    "checkpoint": Path("src/data/processed/models/best_model.pth"),
    "image": None,
    "folder": Path("test/files"),
    "include_test_set": False,
    "fallback_to_test": True,
    "max_test_images": 25,
    "conf": 0.4,
    "save_images": True,
    "output_dir": Path("test/outputs"),
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class InferenceResult:
    image_path: Path
    detections: List[Tuple[str, float, Tuple[int, int, int, int]]]
    annotated_image: np.ndarray


def load_model(checkpoint_path: Path) -> SGSNet:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = Config.DEVICE
    model = SGSNet(Config.NUM_CLASSES).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    print(f"✓ Checkpoint loaded: {checkpoint_path}")
    return model


def preprocess_image(image_path: Path) -> Tuple[torch.Tensor, np.ndarray, Tuple[int, int]]:
    """Read, resize and normalise an image exactly like validation preprocessing."""
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")

    original_h, original_w = image_bgr.shape[:2]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized_rgb = cv2.resize(image_rgb, (Config.IMAGE_SIZE, Config.IMAGE_SIZE), interpolation=cv2.INTER_LINEAR)
    resized_rgb = resized_rgb.astype(np.float32) / 255.0

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    normalised = (resized_rgb - mean) / std

    tensor = torch.from_numpy(normalised.transpose(2, 0, 1)).unsqueeze(0)
    tensor = tensor.to(Config.DEVICE)

    return tensor, image_bgr, (original_h, original_w)


def detection_to_pixels(
    boxes: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    original_shape: Tuple[int, int],
) -> List[Tuple[str, float, Tuple[int, int, int, int]]]:
    """Convert normalised (cx, cy, w, h) to absolute pixel boxes on the original image."""
    original_h, original_w = original_shape
    results: List[Tuple[str, float, Tuple[int, int, int, int]]] = []

    for box, label, score in zip(boxes, labels, scores):
        cx, cy, w, h = box
        x1 = int((cx - w / 2) * original_w)
        y1 = int((cy - h / 2) * original_h)
        x2 = int((cx + w / 2) * original_w)
        y2 = int((cy + h / 2) * original_h)

        x1 = max(0, min(original_w - 1, x1))
        y1 = max(0, min(original_h - 1, y1))
        x2 = max(0, min(original_w - 1, x2))
        y2 = max(0, min(original_h - 1, y2))

        class_idx = int(label)
        class_name = Config.CLASS_NAMES[class_idx] if 0 <= class_idx < len(Config.CLASS_NAMES) else f"cls_{class_idx}"
        results.append((class_name, float(score), (x1, y1, x2, y2)))

    return results


def annotate_image(image_bgr: np.ndarray, detections: List[Tuple[str, float, Tuple[int, int, int, int]]]) -> np.ndarray:
    annotated = image_bgr.copy()
    colour_palette = [
        (0, 0, 255),      # flowering -> red
        (0, 255, 0),      # growing_g -> green
        (255, 255, 0),    # growing_w -> cyan
        (0, 255, 255),    # nearly_m -> yellow
        (255, 0, 255),    # mature -> magenta
    ]

    for class_name, score, (x1, y1, x2, y2) in detections:
        colour = colour_palette[Config.CLASS_NAMES.index(class_name) % len(colour_palette)] if class_name in Config.CLASS_NAMES else (200, 200, 200)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

        label = f"{class_name}: {score:.2f}"
        (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, max(0, y1 - h - baseline - 2)), (x1 + w, y1), colour, thickness=-1)
        cv2.putText(annotated, label, (x1, max(10, y1 - baseline - 2)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    return annotated


def run_inference_on_image(
    model: SGSNet,
    image_path: Path,
    conf_threshold: float,
    save_images: bool,
    output_dir: Path | None,
) -> InferenceResult:
    tensor, original_image_bgr, original_shape = preprocess_image(image_path)

    with torch.no_grad():
        predictions = model(tensor)

    batch_detections = non_max_suppression(
        predictions,
        conf_threshold=conf_threshold,
        iou_threshold=Config.IOU_THRESHOLD,
    )[0]

    boxes = batch_detections[0].cpu().numpy()
    labels = batch_detections[1].cpu().numpy()
    scores = batch_detections[2].cpu().numpy()

    detections = detection_to_pixels(boxes, labels, scores, original_shape)
    annotated = annotate_image(original_image_bgr, detections)

    print(f"\n{image_path.name}")
    if detections:
        for cls_name, cls_score, (x1, y1, x2, y2) in detections:
            print(f"  → {cls_name:>10} | score={cls_score:.3f} | box=({x1},{y1})-({x2},{y2})")
    else:
        print("  (sin detecciones por encima del umbral)")

    if save_images and output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / f"annotated_{image_path.stem}.png"
        cv2.imwrite(str(out_path), annotated)
        print(f"  ✓ Imagen anotada guardada en {out_path}")

    return InferenceResult(image_path=image_path, detections=detections, annotated_image=annotated)


def collect_image_paths(target: Path | None) -> List[Path]:
    if target is None:
        return []

    if target.is_file():
        return [target]

    if not target.exists():
        print(f"⚠️  Carpeta no encontrada: {target}")
        return []

    collected: List[Path] = []
    for ext in IMAGE_EXTENSIONS:
        collected.extend(sorted(target.glob(f"*{ext}")))

    if not collected:
        print(f"⚠️  Carpeta sin imágenes compatibles: {target}")

    return collected


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained SGSNet checkpoint on images.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Ruta al archivo .pth del modelo entrenado.")
    parser.add_argument("--image", type=Path, help="Ruta a una imagen individual para evaluar.")
    parser.add_argument("--folder", type=Path, help="Ruta a una carpeta con imágenes a evaluar.")
    parser.add_argument("--test-set", action="store_true", help="Recorrer el split de test definido en el dataset COCO.")
    parser.add_argument("--fallback-test", action="store_true", help="Usar imágenes del split test si la carpeta está vacía.")
    parser.add_argument("--max-test-images", type=int, default=25, help="Número máximo de imágenes tomadas del split test.")
    parser.add_argument("--conf", type=float, default=0.4, help="Umbral de confianza para considerar detecciones.")
    parser.add_argument("--save-images", action="store_true", help="Guardar copias anotadas de las imágenes procesadas.")
    parser.add_argument("--output-dir", type=Path, default=Path("test/outputs"), help="Directorio donde guardar anotaciones si --save-images está activo.")
    return parser.parse_args(argv)


def gather_test_set_paths(max_images: int) -> List[Path]:
    print("\nRecolectando imágenes del split de test definido en Config...")
    coco_data, _, _, test_ids = load_data()
    id_to_info = {img["id"]: img for img in coco_data["images"]}

    image_root = Path(Config.IMAGES_PATH)
    collected: List[Path] = []

    for img_id in test_ids:
        info = id_to_info.get(img_id)
        if not info:
            continue
        candidate = image_root / info["file_name"]
        if candidate.exists():
            collected.append(candidate)
        if max_images and len(collected) >= max_images:
            break

    if not collected:
        print("⚠️  No se encontraron imágenes del split test en la ruta configurada.")
    else:
        print(f"✓ Encontradas {len(collected)} imágenes de test")

    return collected


def main(argv: List[str] | None = None) -> None:
    args = parse_args(argv or sys.argv[1:])

    if not args.image and not args.folder and not args.test_set and not args.fallback_test:
        raise SystemExit("Debe proporcionar --image o --folder para evaluar.")

    model = load_model(args.checkpoint)

    targets: List[Path] = []
    if args.image:
        targets.append(args.image)

    folder_paths = collect_image_paths(args.folder)
    targets.extend(folder_paths)

    should_use_test = args.test_set or (args.fallback_test and not folder_paths)
    if should_use_test:
        targets.extend(gather_test_set_paths(args.max_test_images))

    # Quitar duplicados preservando el orden
    targets = list(dict.fromkeys(targets))

    if not targets:
        raise SystemExit("No se encontraron imágenes con las extensiones permitidas.")

    print(f"\nProcesando {len(targets)} imagen(es) con umbral {args.conf:.2f}\n")

    for path in targets:
        run_inference_on_image(
            model=model,
            image_path=path,
            conf_threshold=args.conf,
            save_images=args.save_images,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # Ejecución rápida sin argumentos externos (útil desde VS Code "Run" o depurador)
        config = DEFAULT_RUN_CONFIG
        print(
            "Ejecutando evaluate_trained_model con la configuración predeterminada:\n"
            f"  checkpoint: {config['checkpoint']}\n"
            f"  folder:     {config['folder']}\n"
            f"  image:      {config['image']}\n"
            f"  include test set: {config['include_test_set']}\n"
            f"  fallback to test: {config['fallback_to_test']}\n"
            f"  conf:       {config['conf']}\n"
            f"  save_img?:  {config['save_images']}\n"
        )

        run_args = [
            "--checkpoint", str(config["checkpoint"]),
            "--conf", str(config["conf"]),
        ]

        if config["image"]:
            run_args.extend(["--image", str(config["image"])])
        if config["folder"]:
            run_args.extend(["--folder", str(config["folder"])])
        if config.get("include_test_set"):
            run_args.append("--test-set")
            run_args.extend(["--max-test-images", str(config["max_test_images"])])
        elif config.get("fallback_to_test"):
            run_args.append("--fallback-test")
            run_args.extend(["--max-test-images", str(config["max_test_images"])])
        if config["save_images"]:
            run_args.append("--save-images")
            run_args.extend(["--output-dir", str(config["output_dir"])])

        main(run_args)
    else:
        main()
