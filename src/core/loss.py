"""
Funciones de pérdida para SGSNet
"""
import torch
import torch.nn as nn

from .config import Config


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class SGSNetLoss(nn.Module):
    """
    Función de pérdida personalizada para SGSNet
    Combina pérdidas de objectness, bounding box y clasificación
    """
    def __init__(self, num_classes, anchors):
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors.to(Config.DEVICE)
        self.bce_obj = nn.BCEWithLogitsLoss(reduction='none')
        # Usar Focal Loss para clasificación para manejar desbalance de clases y ejemplos difíciles
        self.focal_cls = FocalLoss(alpha=0.25, gamma=2.0, reduction='none')
        self.mse_box = nn.MSELoss(reduction='none')

    def forward(self, predictions, targets_boxes, targets_labels):
        """
        Calcula la pérdida total
        
        Args:
            predictions: Predicciones del modelo [B, C, H, W]
            targets_boxes: Lista de cajas ground truth por imagen
            targets_labels: Lista de etiquetas por imagen
            
        Returns:
            total_loss: Pérdida total
            loss_dict: Diccionario con componentes individuales
        """
        B = predictions.shape[0]
        H = W = Config.IMAGE_SIZE // 32
        num_anchors = 3

        # Reformatear predicciones
        predictions = predictions.view(B, num_anchors, 5 + self.num_classes, H, W)
        predictions = predictions.permute(0, 1, 3, 4, 2).contiguous()

        # Extraer componentes
        pred_obj = predictions[..., 0]
        pred_bbox = predictions[..., 1:5]
        pred_cls = predictions[..., 5:]

        total_obj_loss = 0
        total_bbox_loss = 0
        total_cls_loss = 0
        total_samples = 0

        for b in range(B):
            gt_boxes = targets_boxes[b]
            gt_labels = targets_labels[b]

            # Inicializar targets
            target_obj = torch.zeros((num_anchors, H, W), device=predictions.device)
            target_bbox = torch.zeros((num_anchors, H, W, 4), device=predictions.device)
            target_cls = torch.zeros((num_anchors, H, W, self.num_classes), device=predictions.device)
            obj_mask = torch.zeros((num_anchors, H, W), device=predictions.device)

            if len(gt_boxes) == 0:
                # Solo penalizar falsos positivos
                obj_loss = self.bce_obj(pred_obj[b], target_obj).mean()
                total_obj_loss += obj_loss
                total_samples += 1
                continue

            # Asignar cada GT box al mejor anchor
            for gt_box, gt_label in zip(gt_boxes, gt_labels):
                cx, cy, w, h = gt_box

                if not (0 < cx < 1 and 0 < cy < 1 and w > 0 and h > 0):
                    continue

                # Posición en la grilla
                gx = int(cx * W)
                gy = int(cy * H)
                gx = min(max(gx, 0), W - 1)
                gy = min(max(gy, 0), H - 1)

                # Encontrar mejor anchor por IoU
                best_anchor_idx = 0
                best_iou = 0
                for anchor_idx in range(num_anchors):
                    anchor_w, anchor_h = self.anchors[anchor_idx]
                    iou = min(w, anchor_w) * min(h, anchor_h) / (w * h + anchor_w * anchor_h - min(w, anchor_w) * min(h, anchor_h))
                    if iou > best_iou:
                        best_iou = iou
                        best_anchor_idx = anchor_idx

                # Asignar target al mejor anchor
                target_obj[best_anchor_idx, gy, gx] = 1.0
                obj_mask[best_anchor_idx, gy, gx] = 1.0

                # Offsets relativos a la celda
                target_bbox[best_anchor_idx, gy, gx, 0] = cx * W - gx
                target_bbox[best_anchor_idx, gy, gx, 1] = cy * H - gy
                target_bbox[best_anchor_idx, gy, gx, 2] = torch.log(w / self.anchors[best_anchor_idx, 0] + 1e-16)
                target_bbox[best_anchor_idx, gy, gx, 3] = torch.log(h / self.anchors[best_anchor_idx, 1] + 1e-16)

                # Clase
                label_idx = int(gt_label)
                if 0 <= label_idx < self.num_classes:
                    target_cls[best_anchor_idx, gy, gx, label_idx] = 1.0

            # Calcular pérdidas
            # Objectness loss con balanceo
            # Limitamos el peso positivo para evitar demasiados falsos positivos (mejorar precisión)
            neg_count = (obj_mask == 0).sum()
            pos_count = obj_mask.sum() + 1e-16
            # Reducimos drásticamente el peso positivo para forzar precisión
            pos_weight = torch.clamp(neg_count / pos_count, max=1.0)
            
            obj_loss = self.bce_obj(pred_obj[b], target_obj)
            obj_loss = torch.where(obj_mask > 0, obj_loss * pos_weight, obj_loss)
            total_obj_loss += obj_loss.mean()

            # BBox y Class loss solo en posiciones positivas
            if obj_mask.sum() > 0:
                pos_mask = obj_mask > 0
                pred_bbox_pos = pred_bbox[b][pos_mask]
                target_bbox_pos = target_bbox[pos_mask]
                bbox_loss = self.mse_box(pred_bbox_pos, target_bbox_pos).mean()
                total_bbox_loss += bbox_loss

                pred_cls_pos = pred_cls[b][pos_mask]
                target_cls_pos = target_cls[pos_mask]
                # Usar Focal Loss para clasificación
                cls_loss = self.focal_cls(pred_cls_pos, target_cls_pos).mean()
                total_cls_loss += cls_loss

            total_samples += 1

        # Promediar pérdidas
        total_obj_loss = total_obj_loss / total_samples
        total_bbox_loss = total_bbox_loss / total_samples if total_bbox_loss != 0 else torch.tensor(0.0, device=predictions.device)
        total_cls_loss = total_cls_loss / total_samples if total_cls_loss != 0 else torch.tensor(0.0, device=predictions.device)

        # Pesos ajustados (importancia relativa)
        total_loss = 2.0 * total_obj_loss + 5.0 * total_bbox_loss + 2.0 * total_cls_loss

        return total_loss, {
            'obj': total_obj_loss.item(),
            'bbox': total_bbox_loss.item(),
            'class': total_cls_loss.item()
        }


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
