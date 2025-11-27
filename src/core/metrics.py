"""
Métricas y evaluación del modelo
"""
import torch
import numpy as np

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
                conf_threshold=Config.CONF_THRESHOLD,
                iou_threshold=Config.IOU_THRESHOLD
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


def non_max_suppression(predictions, conf_threshold=0.3, iou_threshold=0.4):
    """
    Non-Maximum Suppression para eliminar detecciones duplicadas
    
    Args:
        predictions: Tensor [B, C, H, W] con predicciones del modelo
        conf_threshold: Umbral de confianza mínimo
        iou_threshold: Umbral de IoU para considerar duplicados
    
    Returns:
        Lista de detecciones por imagen: [(boxes, labels, scores), ...]
    """
    B, C, H, W = predictions.shape
    predictions = predictions.view(B, 3, 5 + Config.NUM_CLASSES, H, W)
    predictions = predictions.permute(0, 1, 3, 4, 2).contiguous()
    
    batch_detections = []
    
    for b in range(B):
        detections = []
        
        # Extraer todas las detecciones con confianza > threshold
        for anchor_idx in range(3):
            for gy in range(H):
                for gx in range(W):
                    obj_conf = torch.sigmoid(predictions[b, anchor_idx, gy, gx, 0]).item()
                    
                    if obj_conf > conf_threshold:
                        dx = torch.sigmoid(predictions[b, anchor_idx, gy, gx, 1]).item()
                        dy = torch.sigmoid(predictions[b, anchor_idx, gy, gx, 2]).item()
                        dw = predictions[b, anchor_idx, gy, gx, 3].item()
                        dh = predictions[b, anchor_idx, gy, gx, 4].item()
                        
                        cx = (gx + dx) / W
                        cy = (gy + dy) / H
                        w = Config.ANCHORS[anchor_idx, 0].item() * np.exp(dw)
                        h = Config.ANCHORS[anchor_idx, 1].item() * np.exp(dh)
                        
                        class_scores = torch.sigmoid(predictions[b, anchor_idx, gy, gx, 5:])
                        class_conf, class_idx = torch.max(class_scores, dim=0)
                        
                        final_conf = obj_conf * class_conf.item()
                        
                        detections.append({
                            'box': [cx, cy, w, h],
                            'class': class_idx.item(),
                            'score': final_conf
                        })
        
        # Aplicar NMS por clase
        filtered_detections = []
        for class_id in range(Config.NUM_CLASSES):
            class_dets = [d for d in detections if d['class'] == class_id]
            
            if not class_dets:
                continue
            
            # Ordenar por score descendente
            class_dets.sort(key=lambda x: x['score'], reverse=True)
            
            keep = []
            while class_dets:
                best = class_dets.pop(0)
                keep.append(best)
                
                # Eliminar detecciones con IoU > threshold
                class_dets = [
                    det for det in class_dets
                    if calculate_iou(best['box'], det['box']) < iou_threshold
                ]
            
            filtered_detections.extend(keep)
        
        # Convertir a formato estándar
        if filtered_detections:
            boxes = torch.tensor([d['box'] for d in filtered_detections])
            labels = torch.tensor([d['class'] for d in filtered_detections])
            scores = torch.tensor([d['score'] for d in filtered_detections])
        else:
            boxes = torch.zeros((0, 4))
            labels = torch.zeros((0,), dtype=torch.long)
            scores = torch.zeros((0,))
        
        batch_detections.append((boxes, labels, scores))
    
    return batch_detections
