from typing import Iterable, Sequence

import numpy as np
import torch

from .config import Config
from .loss import calculate_iou


class Metrics:
    """
    Clase para calcular y mantener métricas de evaluación
    Incluye: Loss, Precision, Recall, F1-Score
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """Reinicia todas las métricas"""
        self.losses = []
        self.obj_losses = []
        self.bbox_losses = []
        self.class_losses = []
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0

    def update(self, loss, loss_dict, predictions=None, target_boxes=None, target_labels=None):
        """
        Actualiza las métricas con nuevos datos
        
        Args:
            loss: Pérdida total
            loss_dict: Diccionario con componentes de pérdida
            predictions: Predicciones del modelo (opcional)
            target_boxes: Cajas ground truth (opcional)
            target_labels: Etiquetas ground truth (opcional)
        """
        self.losses.append(loss)
        self.obj_losses.append(loss_dict.get('obj', 0))
        self.bbox_losses.append(loss_dict.get('bbox', 0))
        self.class_losses.append(loss_dict.get('class', 0))

        if predictions is not None and target_boxes is not None:
            # Aplicar NMS antes de calcular métricas
            batch_detections = non_max_suppression(
                predictions, 
                conf_threshold=0.3,
                iou_threshold=0.4
            )
            
            for batch_idx in range(len(target_boxes)):
                gt_boxes = target_boxes[batch_idx].cpu().numpy()
                gt_labels = target_labels[batch_idx].cpu().numpy()
                
                pred_boxes, pred_labels, pred_scores = batch_detections[batch_idx]
                pred_boxes = pred_boxes.cpu().numpy()
                pred_labels = pred_labels.cpu().numpy()
                
                matched_gt = set()
                
                # Para cada predicción, buscar el mejor match
                for pred_box, pred_label in zip(pred_boxes, pred_labels):
                    best_iou = 0
                    best_gt_idx = -1
                    
                    for gt_idx, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
                        if gt_idx in matched_gt:
                            continue
                        iou = calculate_iou(pred_box, gt_box)
                        if iou > best_iou and pred_label == gt_label:
                            best_iou = iou
                            best_gt_idx = gt_idx
                    
                    if best_iou >= Config.IOU_THRESHOLD:
                        self.true_positives += 1
                        matched_gt.add(best_gt_idx)
                    else:
                        self.false_positives += 1
                
                self.false_negatives += len(gt_boxes) - len(matched_gt)

    def get_metrics(self):
        """
        Calcula y retorna todas las métricas
        
        Returns:
            dict: Diccionario con todas las métricas calculadas
        """
        if not self.losses:
            return {
                'loss': 0.0, 'obj_loss': 0.0, 'bbox_loss': 0.0, 'class_loss': 0.0,
                'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0
            }

        precision = self.true_positives / (self.true_positives + self.false_positives + 1e-16)
        recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-16)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-16)

        return {
            'loss': np.mean(self.losses),
            'obj_loss': np.mean(self.obj_losses),
            'bbox_loss': np.mean(self.bbox_losses),
            'class_loss': np.mean(self.class_losses),
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
        }


def non_max_suppression(
    predictions: Sequence[torch.Tensor] | torch.Tensor,
    conf_threshold: float = 0.3,
    iou_threshold: float = 0.4,
):
    """
    Non-Maximum Suppression para eliminar detecciones duplicadas
    
    Args:
        predictions: Tensor o secuencia de tensores [B, C, H, W] por escala
        conf_threshold: Umbral de confianza mínimo
        iou_threshold: Umbral de IoU para considerar duplicados
    
    Returns:
        Lista de detecciones por imagen: [(boxes, labels, scores), ...]
    """
    if isinstance(predictions, torch.Tensor):
        predictions = [predictions]

    batch_size = predictions[0].shape[0]
    anchors = Config.ANCHORS.to(predictions[0].device)
    batch_detections = []

    for b in range(batch_size):
        detections = []

        for scale_idx, pred_scale in enumerate(predictions):
            num_anchors = anchors.shape[1]
            H, W = pred_scale.shape[2], pred_scale.shape[3]
            anchor_scale = anchors[scale_idx].cpu().numpy()

            pred = (
                pred_scale[b]
                .view(num_anchors, Config.NUM_CLASSES + 5, H, W)
                .permute(0, 2, 3, 1)
                .detach()
                .cpu()
                .numpy()
            )

            for anchor_idx in range(num_anchors):
                anchor_w, anchor_h = anchor_scale[anchor_idx]

                for gy in range(H):
                    for gx in range(W):
                        raw = pred[anchor_idx, gy, gx]
                        obj_conf = 1.0 / (1.0 + np.exp(-raw[0]))

                        if obj_conf < conf_threshold:
                            continue

                        class_scores = 1.0 / (1.0 + np.exp(-raw[5:]))
                        class_idx = int(np.argmax(class_scores))
                        class_conf = float(class_scores[class_idx])
                        final_conf = obj_conf * class_conf

                        if final_conf < conf_threshold:
                            continue

                        dx = 1.0 / (1.0 + np.exp(-raw[1]))
                        dy = 1.0 / (1.0 + np.exp(-raw[2]))
                        dw = raw[3]
                        dh = raw[4]

                        cx = (gx + dx) / W
                        cy = (gy + dy) / H
                        w = float(anchor_w * np.exp(dw))
                        h = float(anchor_h * np.exp(dh))

                        detections.append(
                            {
                                "box": [cx, cy, w, h],
                                "class": class_idx,
                                "score": final_conf,
                            }
                        )

        filtered_detections = []
        for class_id in range(Config.NUM_CLASSES):
            class_dets = [d for d in detections if d["class"] == class_id]
            if not class_dets:
                continue

            class_dets.sort(key=lambda x: x["score"], reverse=True)
            keep = []
            while class_dets:
                best = class_dets.pop(0)
                keep.append(best)
                class_dets = [
                    det
                    for det in class_dets
                    if calculate_iou(best["box"], det["box"]) < iou_threshold
                ]

            filtered_detections.extend(keep)

        if filtered_detections:
            boxes = torch.tensor([d["box"] for d in filtered_detections])
            labels = torch.tensor([d["class"] for d in filtered_detections])
            scores = torch.tensor([d["score"] for d in filtered_detections])
        else:
            boxes = torch.zeros((0, 4))
            labels = torch.zeros((0,), dtype=torch.long)
            scores = torch.zeros((0,))

        batch_detections.append((boxes, labels, scores))

    return batch_detections
