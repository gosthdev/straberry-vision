"""Funciones de pérdida para SGSNet mejorado."""
from __future__ import annotations

from typing import Iterable, Sequence

import torch
import torch.nn as nn

from .config import Config


class SGSNetLoss(nn.Module):
    """Pérdida multiescala similar a YOLO para entrenamiento de SGSNet."""

    def __init__(self, num_classes: int, anchors: torch.Tensor) -> None:
        super().__init__()
        if anchors.dim() == 2:
            anchors = anchors.view(1, -1, 2)
        self.num_classes = num_classes
        self.anchors = anchors.to(Config.DEVICE)
        self.strides = Config.STRIDES
        self.num_scales = len(self.strides)
        self.anchors_per_scale = anchors.shape[1]

        self.bce_obj = nn.BCEWithLogitsLoss(reduction="none")
        self.bce_cls = nn.BCEWithLogitsLoss(reduction="none")
        self.mse_box = nn.MSELoss(reduction="none")

    def forward(
        self,
        predictions: Sequence[torch.Tensor] | torch.Tensor,
        targets_boxes: Iterable[torch.Tensor],
        targets_labels: Iterable[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if isinstance(predictions, torch.Tensor):
            predictions = [predictions]

        obj_losses = []
        bbox_losses = []
        cls_losses = []

        for scale_idx, pred_scale in enumerate(predictions):
            anchor_set = self.anchors[scale_idx].to(pred_scale.device)
            obj, bbox, cls = self._scale_loss(pred_scale, targets_boxes, targets_labels, anchor_set)
            obj_losses.append(obj)
            bbox_losses.append(bbox)
            cls_losses.append(cls)

        total_obj = torch.stack(obj_losses).mean()
        total_bbox = torch.stack(bbox_losses).mean()
        total_cls = torch.stack(cls_losses).mean()

        loss = 2.0 * total_obj + 5.0 * total_bbox + 2.0 * total_cls

        return loss, {
            "obj": total_obj.item(),
            "bbox": total_bbox.item(),
            "class": total_cls.item(),
        }

    def _scale_loss(
        self,
        pred_scale: torch.Tensor,
        targets_boxes: Iterable[torch.Tensor],
        targets_labels: Iterable[torch.Tensor],
        anchors: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, _, H, W = pred_scale.shape
        device = pred_scale.device
        num_anchors = anchors.shape[0]

        pred_scale = pred_scale.view(B, num_anchors, self.num_classes + 5, H, W)
        pred_scale = pred_scale.permute(0, 1, 3, 4, 2).contiguous()

        pred_obj = pred_scale[..., 0]
        pred_bbox = pred_scale[..., 1:5]
        pred_cls = pred_scale[..., 5:]

        obj_losses = []
        bbox_losses = []
        cls_losses = []

        for b, (gt_boxes, gt_labels) in enumerate(zip(targets_boxes, targets_labels)):
            gt_boxes = gt_boxes.to(device)
            gt_labels = gt_labels.to(device)

            target_obj = torch.zeros((num_anchors, H, W), device=device)
            target_bbox = torch.zeros((num_anchors, H, W, 4), device=device)
            target_cls = torch.zeros((num_anchors, H, W, self.num_classes), device=device)
            obj_mask = torch.zeros((num_anchors, H, W), device=device)

            if gt_boxes.numel() > 0:
                self._assign_targets(gt_boxes, gt_labels, anchors, target_obj, target_bbox, target_cls, obj_mask, H, W)

            pos_weight = (obj_mask == 0).sum() / (obj_mask.sum() + 1e-9)
            obj_loss = self.bce_obj(pred_obj[b], target_obj)
            obj_loss = torch.where(obj_mask > 0, obj_loss * pos_weight, obj_loss)
            obj_losses.append(obj_loss.mean())

            if obj_mask.sum() > 0:
                pos_mask = obj_mask > 0
                bbox_loss = self.mse_box(pred_bbox[b][pos_mask], target_bbox[pos_mask]).mean()
                cls_loss = self.bce_cls(pred_cls[b][pos_mask], target_cls[pos_mask]).mean()
            else:
                bbox_loss = torch.zeros((), device=device)
                cls_loss = torch.zeros((), device=device)

            bbox_losses.append(bbox_loss)
            cls_losses.append(cls_loss)

        return torch.stack(obj_losses).mean(), torch.stack(bbox_losses).mean(), torch.stack(cls_losses).mean()

    def _assign_targets(
        self,
        gt_boxes: torch.Tensor,
        gt_labels: torch.Tensor,
        anchors: torch.Tensor,
        target_obj: torch.Tensor,
        target_bbox: torch.Tensor,
        target_cls: torch.Tensor,
        obj_mask: torch.Tensor,
        H: int,
        W: int,
    ) -> None:
        device = target_obj.device
        num_anchors = anchors.shape[0]

        gt_boxes = gt_boxes.clamp_(min=1e-4, max=1 - 1e-4)
        anchor_wh = anchors.to(device)

        wh = gt_boxes[:, 2:4]
        inter = torch.minimum(wh[:, None, :], anchor_wh[None, :, :]).prod(-1)
        union = wh[:, None, :].prod(-1) + anchor_wh[None, :, :].prod(-1) - inter + 1e-9
        best_anchor = torch.argmax(inter / union, dim=1)

        for idx, anchor_idx in enumerate(best_anchor.tolist()):
            cx, cy, w, h = gt_boxes[idx]
            gx = min(max(int(cx.item() * W), 0), W - 1)
            gy = min(max(int(cy.item() * H), 0), H - 1)

            obj_mask[anchor_idx, gy, gx] = 1.0
            target_obj[anchor_idx, gy, gx] = 1.0

            target_bbox[anchor_idx, gy, gx, 0] = cx * W - gx
            target_bbox[anchor_idx, gy, gx, 1] = cy * H - gy
            target_bbox[anchor_idx, gy, gx, 2] = torch.log(w / anchor_wh[anchor_idx, 0] + 1e-9)
            target_bbox[anchor_idx, gy, gx, 3] = torch.log(h / anchor_wh[anchor_idx, 1] + 1e-9)

            label_idx = int(gt_labels[idx].item())
            if 0 <= label_idx < self.num_classes:
                target_cls[anchor_idx, gy, gx, label_idx] = 1.0


def calculate_iou(box1, box2):
    """
    Calcula IoU (Intersection over Union) entre dos cajas
    
    Args:
        box1: Primera caja en formato [cx, cy, w, h]
        box2: Segunda caja en formato [cx, cy, w, h]
        
    Returns:
        iou: Valor de IoU entre 0 y 1
    """
    # Convertir de centro+tamaño a esquinas
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2

    # Calcular intersección
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    
    # Calcular áreas individuales
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    # Calcular unión
    union_area = box1_area + box2_area - inter_area

    return inter_area / (union_area + 1e-16)
