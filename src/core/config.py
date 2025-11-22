"""
Configuración del modelo SGSNet
"""
import json
import os
from pathlib import Path

import numpy as np
import torch


class Config:
    # Rutas
    IMAGES_PATH = 'src/data/raw/images'
    ANNOTATIONS_PATH = 'src/data/processed/coco_out/annotations.json'
    OUTPUT_PATH = 'src/data/processed/models/'

    # Clases
    CLASS_NAMES = ['flowering', 'growing_g', 'growing_w', 'nearly_m', 'mature']
    NUM_CLASSES = len(CLASS_NAMES)

    # Hiperparámetros
    IMAGE_SIZE = 416
    BATCH_SIZE = 16
    EPOCHS = 70
    LEARNING_RATE = 0.0015  # Reducido para mejor convergencia
    WEIGHT_DECAY = 0.0005
    EARLY_STOP_PATIENCE = 7
    SAVE_EVERY = 3

    # Anchors y strides (se recalculan automáticamente al cargar las anotaciones)
    STRIDES = [8, 16, 32]
    NUM_SCALES = len(STRIDES)
    ANCHORS_PER_SCALE = 3
    ANCHORS = torch.tensor(
        [
            [[0.05, 0.05], [0.07, 0.07], [0.09, 0.09]],
            [[0.12, 0.12], [0.16, 0.16], [0.20, 0.20]],
            [[0.28, 0.28], [0.35, 0.35], [0.45, 0.45]],
        ],
        dtype=torch.float32,
    )

    # Thresholds
    CONF_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.4

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # DEBUG: Visualización durante entrenamiento
    VISUALIZE_TRAINING = True  # CAMBIAR A False PARA DESACTIVAR
    VIS_INTERVAL = 5  # Visualizar cada N épocas


# Crear directorio de salida
os.makedirs(Config.OUTPUT_PATH, exist_ok=True)


def _bbox_wh_iou(boxes: np.ndarray, anchors: np.ndarray) -> np.ndarray:
    """Compute IoU between width/height pairs."""
    boxes = boxes[:, None, :]
    anchors = anchors[None, :, :]
    min_wh = np.minimum(boxes, anchors)
    inter = min_wh[..., 0] * min_wh[..., 1]
    box_area = boxes[..., 0] * boxes[..., 1]
    anchor_area = anchors[..., 0] * anchors[..., 1]
    union = box_area + anchor_area - inter + 1e-12
    return inter / union


def _compute_kmeans_anchors(num_clusters: int) -> torch.Tensor | None:
    annotations_path = Path(Config.ANNOTATIONS_PATH)
    if not annotations_path.exists():
        return None

    data = json.loads(annotations_path.read_text(encoding="utf-8"))
    if not data.get("annotations"):
        return None

    image_map = {img["id"]: (img["width"], img["height"]) for img in data.get("images", [])}
    wh_boxes = []
    for ann in data["annotations"]:
        img_wh = image_map.get(ann.get("image_id"))
        if not img_wh:
            continue
        img_w, img_h = img_wh
        if img_w <= 0 or img_h <= 0:
            continue
        w, h = ann.get("bbox", [0, 0, 0, 0])[2:]
        if w <= 1 or h <= 1:
            continue
        wh_boxes.append([w / img_w, h / img_h])

    if len(wh_boxes) < num_clusters:
        return None

    boxes = np.asarray(wh_boxes, dtype=np.float32)
    anchors = boxes[np.random.choice(len(boxes), num_clusters, replace=False)]

    for _ in range(25):
        ious = _bbox_wh_iou(boxes, anchors)
        assignments = np.argmax(ious, axis=1)
        new_anchors = []
        for idx in range(num_clusters):
            cluster = boxes[assignments == idx]
            if cluster.size == 0:
                new_anchors.append(anchors[idx])
            else:
                new_anchors.append(cluster.mean(axis=0))
        anchors = np.stack(new_anchors, axis=0)

    anchors = anchors[np.argsort(anchors[:, 0] * anchors[:, 1])]
    try:
        anchors = anchors.reshape(Config.NUM_SCALES, Config.ANCHORS_PER_SCALE, 2)
    except ValueError:
        return None

    return torch.tensor(anchors, dtype=torch.float32)


_dynamic_anchors = _compute_kmeans_anchors(Config.NUM_SCALES * Config.ANCHORS_PER_SCALE)
if _dynamic_anchors is not None:
    Config.ANCHORS = _dynamic_anchors
    Config.ANCHORS_PER_SCALE = Config.ANCHORS.shape[1]
