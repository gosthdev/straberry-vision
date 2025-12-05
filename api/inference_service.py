import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Optional

import cv2
import numpy as np
import torch

API_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(API_DIR)
SRC_PATH = os.path.join(PROJECT_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.append(SRC_PATH)

from core.model import Config, SGSNet, get_transforms, non_max_suppression


device = Config.DEVICE

_MODEL: Optional[SGSNet] = None
_VAL_TRANSFORM = None


def _load_model() -> Tuple[SGSNet, Any]:
    """
    Carga el modelo y las transformaciones de validación una sola vez.
    Devuelve (model, val_transform).
    """
    global _MODEL, _VAL_TRANSFORM

    if _MODEL is not None and _VAL_TRANSFORM is not None:
        return _MODEL, _VAL_TRANSFORM

    checkpoint_path = os.path.join(
        SRC_PATH, "data", "processed", "models", "best_model.pth"
    )
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"No se encontró el checkpoint del modelo en: {checkpoint_path}"
        )

    model = SGSNet(Config.NUM_CLASSES).to(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    _, val_transform = get_transforms()

    _MODEL = model
    _VAL_TRANSFORM = val_transform
    return model, val_transform


CLASS_COLORS: Dict[str, Tuple[int, int, int]] = {
    "flowering": (255, 0, 255),   # magenta
    "growing_g": (0, 255, 0),     # verde
    "growing_w": (0, 255, 255),   # cian / amarillo
    "nearly_m": (0, 165, 255),    # naranja
    "mature": (0, 0, 255),        # rojo
}


def _postprocess(
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    orig_w: int,
    orig_h: int,
) -> List[Dict[str, Any]]:
    """
    Convierte las cajas normalizadas (cx, cy, w, h) al sistema de píxeles
    de la imagen original.
    """
    detections: List[Dict[str, Any]] = []

    for box, label, score in zip(boxes, labels, scores):
        cx, cy, bw, bh = box.tolist()

        x1 = int((cx - bw / 2) * orig_w)
        y1 = int((cy - bh / 2) * orig_h)
        x2 = int((cx + bw / 2) * orig_w)
        y2 = int((cy + bh / 2) * orig_h)

        x1 = max(0, min(orig_w - 1, x1))
        y1 = max(0, min(orig_h - 1, y1))
        x2 = max(0, min(orig_w - 1, x2))
        y2 = max(0, min(orig_h - 1, y2))

        class_idx = int(label.item())
        if class_idx < 0 or class_idx >= len(Config.CLASS_NAMES):
            class_name = f"class_{class_idx}"
        else:
            class_name = Config.CLASS_NAMES[class_idx]

        detections.append(
            {
                "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                "class_index": class_idx,
                "class_name": class_name,
                "score": float(score.item()),
            }
        )

    return detections


def _annotate_image(
    image_bgr: np.ndarray,
    detections: List[Dict[str, Any]],
) -> np.ndarray:
    """
    Dibuja bounding boxes y etiquetas sobre una copia de la imagen original.
    """
    annotated = image_bgr.copy()

    for det in detections:
        x1 = det["bbox"]["x1"]
        y1 = det["bbox"]["y1"]
        x2 = det["bbox"]["x2"]
        y2 = det["bbox"]["y2"]
        class_name = det["class_name"]
        score = det["score"]

        colour = CLASS_COLORS.get(class_name, (200, 200, 200))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)

        label_text = f"{class_name}: {score:.2f}"
        (w, h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        y1_text = max(0, y1 - h - baseline - 2)

        cv2.rectangle(
            annotated,
            (x1, y1_text),
            (x1 + w, y1_text + h + baseline),
            colour,
            thickness=-1,
        )
        cv2.putText(
            annotated,
            label_text,
            (x1, y1_text + h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            lineType=cv2.LINE_AA,
        )

    return annotated


def infer_from_bytes(
    image_bytes: bytes,
    conf_threshold: float = 0.4,
) -> Dict[str, Any]:
    """
    Pipeline completo de inferencia:
      - decodifica bytes -> imagen OpenCV
      - aplica transformaciones
      - ejecuta modelo + NMS
      - convierte resultados a píxeles
      - dibuja bounding boxes
      - genera resumen por clase

    Devuelve un diccionario listo para serializar a JSON.
    """
    model, val_transform = _load_model()

    nparr = np.frombuffer(image_bytes, np.uint8)
    original_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if original_image is None:
        raise ValueError("No se pudo decodificar la imagen recibida.")

    orig_h, orig_w = original_image.shape[:2]

    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    transformed = val_transform(image=image_rgb, bboxes=[], class_labels=[])
    image_tensor = transformed["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(image_tensor)

    batch_detections = non_max_suppression(
        predictions,
        conf_threshold=conf_threshold,
        iou_threshold=Config.IOU_THRESHOLD,
    )

    if not batch_detections or batch_detections[0] is None:
        detections: List[Dict[str, Any]] = []
    else:
        boxes, labels, scores = batch_detections[0]
        detections = _postprocess(boxes, labels, scores, orig_w, orig_h)

    # Ordenar por score (mayor a menor)
    detections.sort(key=lambda d: d["score"], reverse=True)
    MAX_PER_CLASS = 10
    by_class = {}
    filtered = []
    for det in detections:
        name = det["class_name"]
        by_class.setdefault(name, 0)
        if by_class[name] < MAX_PER_CLASS:
            filtered.append(det)
            by_class[name] += 1
    detections = filtered

    # Resumen por clase
    summary_by_class: Dict[str, Dict[str, float]] = defaultdict(
        lambda: {"count": 0, "best_score": 0.0}
    )
    for det in detections:
        name = det["class_name"]
        summary_by_class[name]["count"] += 1
        summary_by_class[name]["best_score"] = max(
            summary_by_class[name]["best_score"], det["score"]
        )

    if detections:
        max_conf = max(d["score"] for d in detections)
        mean_conf = float(sum(d["score"] for d in detections) / len(detections))
    else:
        max_conf = 0.0
        mean_conf = 0.0

    annotated_image = _annotate_image(original_image, detections)

    return {
        "image_size": {"width": orig_w, "height": orig_h},
        "num_detections": len(detections),
        "max_confidence": float(max_conf),
        "mean_confidence": float(mean_conf),
        "detections": detections,
        "summary_by_class": dict(summary_by_class),
        "annotated_image_bgr": annotated_image,
    }
