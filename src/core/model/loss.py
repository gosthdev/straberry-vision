"""
Funciones de pérdida para SGSNet v2 - Multi-escala con CIoU
"""
import torch
import torch.nn as nn
import math

from .config import Config


class FocalLoss(nn.Module):
    """Focal Loss para manejar desbalance de clases - Estable para mixed precision"""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        # Trabajar en float32 para estabilidad
        inputs = inputs.float()
        targets = targets.float()
        
        bce_loss = self.bce(inputs, targets)
        # Clamp para evitar inestabilidad numérica
        bce_loss = bce_loss.clamp(max=100)
        pt = torch.exp(-bce_loss).clamp(min=1e-8, max=1-1e-8)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CIoULoss(nn.Module):
    """
    Complete IoU Loss para mejor localización de bounding boxes
    CIoU = IoU - (rho^2(b, b_gt) / c^2) - alpha*v
    
    donde:
    - rho es la distancia euclidiana entre centros
    - c es la diagonal del rectángulo envolvente más pequeño
    - v mide la consistencia de aspect ratio
    - alpha es un parámetro de trade-off
    """
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction
        
    def forward(self, pred_boxes, target_boxes, eps=1e-6):
        """
        Args:
            pred_boxes: [N, 4] tensor con (cx, cy, w, h) predichos
            target_boxes: [N, 4] tensor con (cx, cy, w, h) ground truth
            
        Returns:
            ciou_loss: Pérdida CIoU
        """
        # Trabajar en float32 para estabilidad numérica
        pred_boxes = pred_boxes.float()
        target_boxes = target_boxes.float()
        
        # Extraer componentes
        pred_cx, pred_cy, pred_w, pred_h = pred_boxes.unbind(-1)
        target_cx, target_cy, target_w, target_h = target_boxes.unbind(-1)
        
        # Clamp para evitar valores extremos
        pred_w = pred_w.clamp(min=eps)
        pred_h = pred_h.clamp(min=eps)
        target_w = target_w.clamp(min=eps)
        target_h = target_h.clamp(min=eps)
        
        # Convertir a esquinas (x1, y1, x2, y2)
        pred_x1 = pred_cx - pred_w / 2
        pred_y1 = pred_cy - pred_h / 2
        pred_x2 = pred_cx + pred_w / 2
        pred_y2 = pred_cy + pred_h / 2
        
        target_x1 = target_cx - target_w / 2
        target_y1 = target_cy - target_h / 2
        target_x2 = target_cx + target_w / 2
        target_y2 = target_cy + target_h / 2
        
        # Calcular intersección
        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Calcular áreas
        pred_area = pred_w * pred_h
        target_area = target_w * target_h
        union_area = pred_area + target_area - inter_area + eps
        
        # IoU
        iou = inter_area / union_area
        iou = iou.clamp(min=0, max=1)
        
        # Calcular rectángulo envolvente (enclosing box)
        enclose_x1 = torch.min(pred_x1, target_x1)
        enclose_y1 = torch.min(pred_y1, target_y1)
        enclose_x2 = torch.max(pred_x2, target_x2)
        enclose_y2 = torch.max(pred_y2, target_y2)
        
        # Diagonal al cuadrado del rectángulo envolvente
        c2 = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + eps
        
        # Distancia al cuadrado entre centros
        rho2 = (pred_cx - target_cx) ** 2 + (pred_cy - target_cy) ** 2
        
        # Término de aspect ratio (con clamps adicionales)
        v = (4 / (math.pi ** 2)) * torch.pow(
            torch.atan(target_w / target_h) - torch.atan(pred_w / pred_h), 2
        )
        
        with torch.no_grad():
            alpha = v / (1 - iou + v + eps)
        
        # CIoU
        ciou = iou - (rho2 / c2) - alpha * v
        ciou = ciou.clamp(min=-1, max=1)
        
        # Loss = 1 - CIoU
        loss = 1 - ciou
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class SGSNetLoss(nn.Module):
    """
    Función de pérdida multi-escala para SGSNet v2
    Combina pérdidas de objectness, bounding box (CIoU) y clasificación (Focal)
    para múltiples escalas de detección
    """
    def __init__(self, num_classes, anchors_list):
        """
        Args:
            num_classes: Número de clases
            anchors_list: Lista de tensores de anchors por escala [anchors_p3, anchors_p4, anchors_p5]
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_scales = len(anchors_list)
        
        # Mover anchors a device y guardar como lista
        self.anchors_list = [a.to(Config.DEVICE) for a in anchors_list]
        
        # Pérdidas
        self.bce_obj = nn.BCEWithLogitsLoss(reduction='none')
        self.focal_cls = FocalLoss(alpha=0.25, gamma=2.0, reduction='none')
        self.ciou_loss = CIoULoss(reduction='none')
        
        # Pesos por escala (más peso a escalas más pequeñas donde hay más celdas)
        self.scale_weights = [4.0, 1.0, 0.4]  # P3, P4, P5
        
    def forward(self, predictions_list, targets_boxes, targets_labels):
        """
        Calcula la pérdida total para todas las escalas
        
        Args:
            predictions_list: Lista de predicciones por escala [pred_p3, pred_p4, pred_p5]
            targets_boxes: Lista de cajas ground truth por imagen
            targets_labels: Lista de etiquetas por imagen
            
        Returns:
            total_loss: Pérdida total
            loss_dict: Diccionario con componentes individuales
        """
        # Convertir predicciones a float32 para estabilidad numérica
        predictions_list = [p.float() for p in predictions_list]
        
        total_obj_loss = 0
        total_bbox_loss = 0
        total_cls_loss = 0
        
        B = predictions_list[0].shape[0]
        
        # Procesar cada escala
        for scale_idx, predictions in enumerate(predictions_list):
            anchors = self.anchors_list[scale_idx]
            scale_weight = self.scale_weights[scale_idx]
            grid_size = predictions.shape[-1]  # H = W
            num_anchors = anchors.shape[0]
            
            # Reformatear predicciones: [B, A*(5+C), H, W] -> [B, A, H, W, 5+C]
            predictions = predictions.view(B, num_anchors, 5 + self.num_classes, grid_size, grid_size)
            predictions = predictions.permute(0, 1, 3, 4, 2).contiguous()
            
            # Extraer componentes
            pred_obj = predictions[..., 0]
            pred_bbox = predictions[..., 1:5]
            pred_cls = predictions[..., 5:]
            
            # Procesar cada imagen del batch
            for b in range(B):
                gt_boxes = targets_boxes[b]
                gt_labels = targets_labels[b]
                
                # Inicializar targets para esta escala
                target_obj = torch.zeros((num_anchors, grid_size, grid_size), device=predictions.device)
                target_bbox = torch.zeros((num_anchors, grid_size, grid_size, 4), device=predictions.device)
                target_cls = torch.zeros((num_anchors, grid_size, grid_size, self.num_classes), device=predictions.device)
                obj_mask = torch.zeros((num_anchors, grid_size, grid_size), device=predictions.device)
                
                if len(gt_boxes) == 0:
                    # Solo penalizar falsos positivos
                    obj_loss = self.bce_obj(pred_obj[b], target_obj).mean()
                    total_obj_loss += obj_loss * scale_weight
                    continue
                
                # Asignar cada GT box a los mejores anchors de esta escala
                for gt_box, gt_label in zip(gt_boxes, gt_labels):
                    cx, cy, w, h = gt_box
                    
                    if not (0 < cx < 1 and 0 < cy < 1 and w > 0 and h > 0):
                        continue
                    
                    # Posición en la grilla de esta escala
                    gx = int(cx * grid_size)
                    gy = int(cy * grid_size)
                    gx = min(max(gx, 0), grid_size - 1)
                    gy = min(max(gy, 0), grid_size - 1)
                    
                    # Encontrar mejor anchor por IoU
                    best_anchor_idx = 0
                    best_iou = -1  # Empezar en -1 para siempre asignar
                    
                    for anchor_idx in range(num_anchors):
                        anchor_w, anchor_h = anchors[anchor_idx]
                        # IoU simplificado (asumiendo mismo centro)
                        inter_w = min(w, anchor_w.item())
                        inter_h = min(h, anchor_h.item())
                        inter_area = inter_w * inter_h
                        union_area = w * h + anchor_w.item() * anchor_h.item() - inter_area
                        iou = inter_area / (union_area + 1e-16)
                        
                        if iou > best_iou:
                            best_iou = iou
                            best_anchor_idx = anchor_idx
                    
                    # Asignar siempre al mejor anchor de esta escala
                    # El IoU bajo significa que la escala no es ideal para este objeto,
                    # pero aún así debe detectarse
                    if best_iou >= 0:  # Siempre asignar
                        target_obj[best_anchor_idx, gy, gx] = 1.0
                        obj_mask[best_anchor_idx, gy, gx] = 1.0
                        
                        # Offsets relativos a la celda
                        target_bbox[best_anchor_idx, gy, gx, 0] = cx * grid_size - gx  # offset x
                        target_bbox[best_anchor_idx, gy, gx, 1] = cy * grid_size - gy  # offset y
                        target_bbox[best_anchor_idx, gy, gx, 2] = w  # ancho directo
                        target_bbox[best_anchor_idx, gy, gx, 3] = h  # alto directo
                        
                        # Clase
                        label_idx = int(gt_label)
                        if 0 <= label_idx < self.num_classes:
                            target_cls[best_anchor_idx, gy, gx, label_idx] = 1.0
                
                # Calcular pérdidas para esta imagen y escala
                # Objectness loss con balanceo
                neg_count = (obj_mask == 0).sum()
                pos_count = obj_mask.sum() + 1e-16
                pos_weight = torch.clamp(neg_count / pos_count, max=15.0)
                
                obj_loss = self.bce_obj(pred_obj[b], target_obj)
                obj_loss = torch.where(obj_mask > 0, obj_loss * pos_weight, obj_loss)
                total_obj_loss += obj_loss.mean() * scale_weight
                
                # BBox y Class loss solo en posiciones positivas
                if obj_mask.sum() > 0:
                    pos_mask = obj_mask > 0
                    
                    # BBox loss con CIoU
                    pred_bbox_pos = pred_bbox[b][pos_mask]
                    target_bbox_pos = target_bbox[pos_mask]
                    
                    # Decodificar predicciones para CIoU
                    # pred_bbox_pos contiene [offset_x, offset_y, w, h]
                    # Necesitamos reconstruir cx, cy, w, h
                    pos_indices = torch.nonzero(pos_mask, as_tuple=False)
                    
                    pred_boxes_decoded = []
                    target_boxes_decoded = []
                    
                    for i, (a_idx, gy_idx, gx_idx) in enumerate(pos_indices):
                        # Decodificar predicción
                        pred_offset_x = torch.sigmoid(pred_bbox_pos[i, 0])
                        pred_offset_y = torch.sigmoid(pred_bbox_pos[i, 1])
                        pred_w = torch.sigmoid(pred_bbox_pos[i, 2])  # Sigmoid para mantener en [0,1]
                        pred_h = torch.sigmoid(pred_bbox_pos[i, 3])
                        
                        pred_cx = (gx_idx.float() + pred_offset_x) / grid_size
                        pred_cy = (gy_idx.float() + pred_offset_y) / grid_size
                        
                        pred_boxes_decoded.append(torch.stack([pred_cx, pred_cy, pred_w, pred_h]))
                        
                        # Target
                        target_offset_x = target_bbox_pos[i, 0]
                        target_offset_y = target_bbox_pos[i, 1]
                        target_w = target_bbox_pos[i, 2]
                        target_h = target_bbox_pos[i, 3]
                        
                        target_cx = (gx_idx.float() + target_offset_x) / grid_size
                        target_cy = (gy_idx.float() + target_offset_y) / grid_size
                        
                        target_boxes_decoded.append(torch.stack([target_cx, target_cy, target_w, target_h]))
                    
                    if pred_boxes_decoded:
                        pred_boxes_tensor = torch.stack(pred_boxes_decoded)
                        target_boxes_tensor = torch.stack(target_boxes_decoded)
                        
                        bbox_loss = self.ciou_loss(pred_boxes_tensor, target_boxes_tensor).mean()
                        total_bbox_loss += bbox_loss * scale_weight
                    
                    # Classification loss
                    pred_cls_pos = pred_cls[b][pos_mask]
                    target_cls_pos = target_cls[pos_mask]
                    cls_loss = self.focal_cls(pred_cls_pos, target_cls_pos).mean()
                    total_cls_loss += cls_loss * scale_weight
        
        # Promediar pérdidas
        num_samples = B * self.num_scales
        total_obj_loss = total_obj_loss / num_samples
        total_bbox_loss = total_bbox_loss / num_samples if total_bbox_loss != 0 else torch.tensor(0.0, device=predictions_list[0].device)
        total_cls_loss = total_cls_loss / num_samples if total_cls_loss != 0 else torch.tensor(0.0, device=predictions_list[0].device)
        
        # Pesos finales (importancia relativa)
        total_loss = 1.0 * total_obj_loss + 5.0 * total_bbox_loss + 1.0 * total_cls_loss
        
        return total_loss, {
            'obj': total_obj_loss.item() if torch.is_tensor(total_obj_loss) else total_obj_loss,
            'bbox': total_bbox_loss.item() if torch.is_tensor(total_bbox_loss) else total_bbox_loss,
            'class': total_cls_loss.item() if torch.is_tensor(total_cls_loss) else total_cls_loss
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
