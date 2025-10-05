"""
SGSNet para Detección de Estados de Madurez de Fresas
Versión Corregida con Mejoras Críticas
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn.functional as F
import os
import glob
import time
import random
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACIÓN
# ============================================================================
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
    BATCH_SIZE = 8
    EPOCHS = 70
    LEARNING_RATE = 0.001  # Reducido para mejor convergencia
    WEIGHT_DECAY = 0.0005
    EARLY_STOP_PATIENCE = 10
    SAVE_EVERY = 5

    # Anchors definidos explícitamente (pequeño, mediano, grande)
    # Basados en análisis típico de fresas en imágenes
    ANCHORS = torch.tensor([
        [0.05, 0.05],  # Anchor pequeño
        [0.10, 0.10],  # Anchor mediano
        [0.15, 0.15]   # Anchor grande
    ])

    # Thresholds
    CONF_THRESHOLD = 0.3
    IOU_THRESHOLD = 0.4

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # DEBUG: Visualización durante entrenamiento
    VISUALIZE_TRAINING = True  # CAMBIAR A False PARA DESACTIVAR
    VIS_INTERVAL = 5  # Visualizar cada N épocas

os.makedirs(Config.OUTPUT_PATH, exist_ok=True)

# ============================================================================
# ARQUITECTURA SGSNET MEJORADA
# ============================================================================
class SGSNet(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = 3

        # Backbone: MobileNetV3-Small
        mobilenet = models.mobilenet_v3_small(pretrained=True)
        self.backbone = mobilenet.features

        # Feature Pyramid Network (FPN) - Multi-escala
        self.fpn_conv1 = nn.Sequential(
            nn.Conv2d(576, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        
        self.fpn_conv2 = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Detection heads mejorados
        self.detection_head = self._make_detection_head(128)

        self._initialize_weights()

    def _make_detection_head(self, in_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, self.num_anchors * (5 + self.num_classes), 1)
        )

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.backbone(x)
        fpn_out1 = self.fpn_conv1(features)
        fpn_out2 = self.fpn_conv2(fpn_out1)
        predictions = self.detection_head(fpn_out2)
        return predictions

# ============================================================================
# DATASET (SIN CAMBIOS CRÍTICOS)
# ============================================================================
class StrawberryDataset(Dataset):
    def __init__(self, coco_data, image_dir, image_ids, transform=None):
        self.coco_data = coco_data
        self.image_dir = Path(image_dir)
        self.image_ids = image_ids
        self.transform = transform

        self.cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
        self.name_to_label = {name: i for i, name in enumerate(Config.CLASS_NAMES)}

        self.annotations = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        self.valid_ids = [img_id for img_id in image_ids
                         if img_id in self.annotations and len(self.annotations[img_id]) > 0]

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        img_id = self.valid_ids[idx]
        img_info = next(img for img in self.coco_data['images'] if img['id'] == img_id)
        img_path = self.image_dir / img_info['file_name']

        if not img_path.exists():
            for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                test_path = img_path.with_suffix(ext)
                if test_path.exists():
                    img_path = test_path
                    break

        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"No se pudo cargar: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        anns = self.annotations[img_id]
        bboxes = []
        class_labels = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            if w > 5 and h > 5 and x >= 0 and y >= 0:
                bboxes.append([x, y, w, h])
                cat_name = self.cat_id_to_name[ann['category_id']]
                class_labels.append(self.name_to_label.get(cat_name, 0))

        if not bboxes:
            bboxes = [[10, 10, 50, 50]]
            class_labels = [0]

        transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
        image = transformed['image']
        boxes = transformed['bboxes']
        labels = transformed['class_labels']

        normalized_boxes = []
        valid_labels = []
        for bbox, label in zip(boxes, labels):
            x_min, y_min, w, h = bbox
            cx = (x_min + w/2) / Config.IMAGE_SIZE
            cy = (y_min + h/2) / Config.IMAGE_SIZE
            nw = w / Config.IMAGE_SIZE
            nh = h / Config.IMAGE_SIZE

            if 0 <= cx <= 1 and 0 <= cy <= 1 and nw > 0.01 and nh > 0.01:
                normalized_boxes.append([cx, cy, nw, nh])
                valid_labels.append(label)

        if not normalized_boxes:
            normalized_boxes = [[0.5, 0.5, 0.1, 0.1]]
            valid_labels = [0]

        return image, torch.FloatTensor(normalized_boxes), torch.LongTensor(valid_labels)

def get_transforms():
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.3),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='coco', min_area=100, min_visibility=0.3, label_fields=['class_labels']))

    val_transform = A.Compose([
        A.Resize(Config.IMAGE_SIZE, Config.IMAGE_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='coco', min_area=0, min_visibility=0.0, label_fields=['class_labels']))

    return train_transform, val_transform

def collate_fn(batch):
    images, boxes, labels = zip(*batch)
    images = torch.stack(images, 0)
    return images, boxes, labels

# ============================================================================
# LOSS MEJORADA CON ANCHORS EXPLÍCITOS
# ============================================================================
class SGSNetLoss(nn.Module):
    def __init__(self, num_classes, anchors):
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors.to(Config.DEVICE)
        self.bce_obj = nn.BCEWithLogitsLoss(reduction='none')
        self.bce_cls = nn.BCEWithLogitsLoss(reduction='none')
        self.mse_box = nn.MSELoss(reduction='none')

    def forward(self, predictions, targets_boxes, targets_labels):
        B = predictions.shape[0]
        H = W = Config.IMAGE_SIZE // 32
        num_anchors = 3

        predictions = predictions.view(B, num_anchors, 5 + self.num_classes, H, W)
        predictions = predictions.permute(0, 1, 3, 4, 2).contiguous()

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

            for gt_box, gt_label in zip(gt_boxes, gt_labels):
                cx, cy, w, h = gt_box

                if not (0 < cx < 1 and 0 < cy < 1 and w > 0 and h > 0):
                    continue

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

                label_idx = int(gt_label)
                if 0 <= label_idx < self.num_classes:
                    target_cls[best_anchor_idx, gy, gx, label_idx] = 1.0

            # Objectness loss con balanceo
            pos_weight = (obj_mask == 0).sum() / (obj_mask.sum() + 1e-16)
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
                cls_loss = self.bce_cls(pred_cls_pos, target_cls_pos).mean()
                total_cls_loss += cls_loss

            total_samples += 1

        total_obj_loss = total_obj_loss / total_samples
        total_bbox_loss = total_bbox_loss / total_samples if total_bbox_loss != 0 else torch.tensor(0.0, device=predictions.device)
        total_cls_loss = total_cls_loss / total_samples if total_cls_loss != 0 else torch.tensor(0.0, device=predictions.device)

        # Pesos ajustados
        total_loss = 2.0 * total_obj_loss + 5.0 * total_bbox_loss + 2.0 * total_cls_loss   #WEIGHT AJUSTADOS ES LA IMPORTANCIA

        return total_loss, {
            'obj': total_obj_loss.item(),
            'bbox': total_bbox_loss.item(),
            'class': total_cls_loss.item()
        }

# ============================================================================
# MÉTRICAS CORREGIDAS CON IOU
# ============================================================================
def calculate_iou(box1, box2):
    """Calcula IoU entre dos boxes en formato [cx, cy, w, h]"""
    x1_min = box1[0] - box1[2] / 2
    y1_min = box1[1] - box1[3] / 2
    x1_max = box1[0] + box1[2] / 2
    y1_max = box1[1] + box1[3] / 2

    x2_min = box2[0] - box2[2] / 2
    y2_min = box2[1] - box2[3] / 2
    x2_max = box2[0] + box2[2] / 2
    y2_max = box2[1] + box2[3] / 2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area

    return inter_area / (union_area + 1e-16)

class Metrics:
    """Métricas con NMS integrado para evitar contar duplicados"""
    
    def __init__(self):
        self.reset()

    def reset(self):
        self.losses = []
        self.obj_losses = []
        self.bbox_losses = []
        self.class_losses = []
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0

    def update(self, loss, loss_dict, predictions=None, target_boxes=None, target_labels=None):
        self.losses.append(loss)
        self.obj_losses.append(loss_dict.get('obj', 0))
        self.bbox_losses.append(loss_dict.get('bbox', 0))
        self.class_losses.append(loss_dict.get('class', 0))

        if predictions is not None and target_boxes is not None:
            # CAMBIO CRÍTICO: Aplicar NMS antes de calcular métricas
            batch_detections = non_max_suppression(
                predictions, 
                conf_threshold=0.3,  # Más bajo que Config.CONF_THRESHOLD para métricas
                iou_threshold=0.4    # Eliminar cajas con IoU > 0.4
            )
            
            for batch_idx in range(len(target_boxes)):
                gt_boxes = target_boxes[batch_idx].cpu().numpy()
                gt_labels = target_labels[batch_idx].cpu().numpy()
                
                pred_boxes, pred_labels, pred_scores = batch_detections[batch_idx]
                pred_boxes = pred_boxes.cpu().numpy()
                pred_labels = pred_labels.cpu().numpy()
                
                matched_gt = set()
                
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
        if not self.losses:
            return {'loss': 0.0, 'obj_loss': 0.0, 'bbox_loss': 0.0, 'class_loss': 0.0,
                   'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0}

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
        predictions: tensor [B, C, H, W] con predicciones del modelo
        conf_threshold: umbral de confianza mínimo
        iou_threshold: umbral de IoU para considerar duplicados
    
    Returns:
        Lista de detecciones por imagen: [boxes, labels, scores]
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

# ============================================================================
# VISUALIZACIÓN DURANTE ENTRENAMIENTO (NUEVO - FÁCIL DE ELIMINAR)
# ============================================================================
def visualize_predictions(model, val_loader, epoch, save_path):
    """Visualización con NMS aplicado"""
    if not Config.VISUALIZE_TRAINING:
        return
    
    model.eval()
    with torch.no_grad():
        try:
            total_batches = len(val_loader)
        except TypeError:
            total_batches = None

        if total_batches:
            random_batch_idx = random.randrange(total_batches)
            selected_batch = None
            for idx, batch in enumerate(val_loader):
                if idx == random_batch_idx:
                    selected_batch = batch
                    break
            if selected_batch is None:
                selected_batch = next(iter(val_loader))
        else:
            selected_batch = next(iter(val_loader))

        images, target_boxes, target_labels = selected_batch
        images = images.to(Config.DEVICE)
        predictions = model(images)
        
        # Aplicar NMS
        batch_detections = non_max_suppression(
            predictions,
            conf_threshold=0.3,
            iou_threshold=0.4
        )
        
        num_display = min(4, images.shape[0])
        fig, axes = plt.subplots(2, num_display, figsize=(20, 10))
        if num_display == 1:
            axes = np.array(axes).reshape(2, 1)
        
        colors = ['red', 'blue', 'cyan', 'yellow', 'magenta']
        
        for idx in range(num_display):
            img = images[idx].cpu().permute(1, 2, 0).numpy()
            img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
            img = np.clip(img, 0, 1)
            
            # Ground truth
            ax_gt = axes[0, idx] if num_display > 1 else axes[0]
            ax_gt.imshow(img)
            ax_gt.set_title(f'Epoch {epoch} - GT')
            ax_gt.axis('off')
            
            for gt_box, gt_label in zip(target_boxes[idx], target_labels[idx]):
                cx, cy, w, h = gt_box.cpu().numpy()
                x1 = int((cx - w/2) * Config.IMAGE_SIZE)
                y1 = int((cy - h/2) * Config.IMAGE_SIZE)
                x2 = int((cx + w/2) * Config.IMAGE_SIZE)
                y2 = int((cy + h/2) * Config.IMAGE_SIZE)
                
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='green', linewidth=2)
                ax_gt.add_patch(rect)
                ax_gt.text(x1, y1-5, Config.CLASS_NAMES[gt_label], color='green', fontsize=8, weight='bold')
            
            # Predictions con NMS
            ax_pred = axes[1, idx] if num_display > 1 else axes[1]
            ax_pred.imshow(img)
            
            pred_boxes, pred_labels, pred_scores = batch_detections[idx]
            ax_pred.set_title(f'Epoch {epoch} - Pred (NMS: {len(pred_boxes)} boxes)')
            ax_pred.axis('off')
            
            for pred_box, pred_label, pred_score in zip(pred_boxes, pred_labels, pred_scores):
                cx, cy, w, h = pred_box.cpu().numpy()
                x1 = int((cx - w/2) * Config.IMAGE_SIZE)
                y1 = int((cy - h/2) * Config.IMAGE_SIZE)
                x2 = int((cx + w/2) * Config.IMAGE_SIZE)
                y2 = int((cy + h/2) * Config.IMAGE_SIZE)
                
                color = colors[int(pred_label) % len(colors)]
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color=color, linewidth=2)
                ax_pred.add_patch(rect)
                ax_pred.text(x1, y1-5, f'{Config.CLASS_NAMES[int(pred_label)]}:{pred_score:.2f}', 
                           color=color, fontsize=8, weight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/predictions_epoch_{epoch}.png', dpi=100)
        plt.close()
        print(f"  → Visualización guardada: predictions_epoch_{epoch}.png")

# ============================================================================
# ENTRENAMIENTO (CON VISUALIZACIÓN INTEGRADA)
# ============================================================================
def train_epoch(model, train_loader, optimizer, criterion, scaler):
    """
    Entrenar una época con flush buffer para evitar congelamiento
    """
    model.train()
    metrics = Metrics()

    progress_bar = tqdm(train_loader, desc="Training", 
                       mininterval=1.0,  # Actualizar cada 1 segundo
                       maxinterval=2.0)  # Máximo 2 segundos
    
    for images, target_boxes, target_labels in progress_bar:
        images = images.to(Config.DEVICE)

        with torch.cuda.amp.autocast():
            predictions = model(images)
            loss, loss_dict = criterion(predictions, target_boxes, target_labels)

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()

        metrics.update(loss.item(), loss_dict, predictions, target_boxes, target_labels)
        
        # Actualizar barra con formato compacto
        progress_bar.set_postfix({
            'L': f'{loss.item():.3f}',
            'O': f'{loss_dict["obj"]:.3f}',
            'B': f'{loss_dict["bbox"]:.3f}',
            'C': f'{loss_dict["class"]:.3f}'
        }, refresh=True)

    return metrics.get_metrics()

def validate_epoch(model, val_loader, criterion):
    """Validación con NMS integrado"""
    model.eval()
    metrics = Metrics()

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validating",
                          mininterval=1.0,
                          maxinterval=2.0)
        
        for images, target_boxes, target_labels in progress_bar:
            images = images.to(Config.DEVICE)
            predictions = model(images)
            loss, loss_dict = criterion(predictions, target_boxes, target_labels)

            metrics.update(loss.item(), loss_dict, predictions, target_boxes, target_labels)
            
            progress_bar.set_postfix({
                'L': f'{loss.item():.3f}',
                'O': f'{loss_dict["obj"]:.3f}',
                'B': f'{loss_dict["bbox"]:.3f}',
                'C': f'{loss_dict["class"]:.3f}'
            }, refresh=True)

    return metrics.get_metrics()

# ============================================================================
# CHECKPOINTS Y EARLY STOPPING
# ============================================================================
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return self.should_stop

def save_checkpoint(epoch, model, optimizer, scheduler, early_stopping,
                   train_metrics, val_metrics, history, is_best=False):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'early_stopping_counter': early_stopping.counter,
        'early_stopping_best_loss': early_stopping.best_loss,
        'train_loss': train_metrics['loss'],
        'val_loss': val_metrics['loss'],
        'history': history,
        'config': {
            'learning_rate': Config.LEARNING_RATE,
            'batch_size': Config.BATCH_SIZE,
            'num_classes': Config.NUM_CLASSES
        }
    }

    if is_best:
        best_path = Config.OUTPUT_PATH + '/best_model.pth'
        torch.save(checkpoint, best_path)
        print(f"✓ Mejor modelo guardado: {best_path}")
    else:
        checkpoint_path = Config.OUTPUT_PATH + f'/checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint guardado: {checkpoint_path}")

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint no encontrado: {checkpoint_path}")
        return 0, None, None

    print(f"Cargando checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✓ Modelo restaurado desde época {checkpoint['epoch']}")

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    early_stopping_state = {
        'counter': checkpoint.get('early_stopping_counter', 0),
        'best_loss': checkpoint.get('early_stopping_best_loss', float('inf'))
    }

    history = checkpoint.get('history', None)
    start_epoch = checkpoint['epoch'] + 1

    return start_epoch, history, early_stopping_state

def find_latest_checkpoint(checkpoint_dir):
    pattern = os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth')
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    def get_epoch_num(path):
        basename = os.path.basename(path)
        try:
            return int(basename.split('_')[-1].replace('.pth', ''))
        except:
            return 0
    return max(checkpoints, key=get_epoch_num)

# ============================================================================
# DATA LOADING
# ============================================================================
def load_data():
    print("Cargando dataset...")
    with open(Config.ANNOTATIONS_PATH, 'r') as f:
        coco_data = json.load(f)

    print(f"✓ Imágenes: {len(coco_data['images'])}")
    print(f"✓ Anotaciones: {len(coco_data['annotations'])}")

    image_ids = [img['id'] for img in coco_data['images']]
    train_ids, temp_ids = train_test_split(image_ids, test_size=0.3, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    print(f"✓ Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")
    return coco_data, train_ids, val_ids, test_ids

def create_dataloaders(coco_data, train_ids, val_ids, test_ids):
    train_transform, val_transform = get_transforms()
    
    train_dataset = StrawberryDataset(coco_data, Config.IMAGES_PATH, train_ids, train_transform)
    val_dataset = StrawberryDataset(coco_data, Config.IMAGES_PATH, val_ids, val_transform)
    test_dataset = StrawberryDataset(coco_data, Config.IMAGES_PATH, test_ids, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
                             collate_fn=collate_fn, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                           collate_fn=collate_fn, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader

# ============================================================================
# FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# ============================================================================
def train_sgsnet(resume_from_checkpoint=None):
    print("="*60)
    print("STRAWBERRY DETECTION - SGSNET MEJORADO")
    print("="*60)
    print(f"Clases: {Config.CLASS_NAMES}")
    print(f"Device: {Config.DEVICE}\n")

    # Cargar datos
    coco_data, train_ids, val_ids, test_ids = load_data()
    train_loader, val_loader, test_loader = create_dataloaders(coco_data, train_ids, val_ids, test_ids)

    # Modelo
    model = SGSNet(Config.NUM_CLASSES).to(Config.DEVICE)
    print(f"✓ Modelo creado (parámetros: {sum(p.numel() for p in model.parameters())/1e6:.2f}M)\n")

    # Optimizador y pérdida
    optimizer = optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=Config.WEIGHT_DECAY)
    criterion = SGSNetLoss(Config.NUM_CLASSES, Config.ANCHORS)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler()
    early_stopping = EarlyStopping(patience=Config.EARLY_STOP_PATIENCE)

    # Tracking
    start_epoch = 1
    history = {
        'train_loss': [], 'val_loss': [],
        'train_obj': [], 'val_obj': [],
        'train_bbox': [], 'val_bbox': [],
        'train_class': [], 'val_class': [],
        'train_precision': [], 'val_precision': [],
        'train_recall': [], 'val_recall': [],
        'train_f1': [], 'val_f1': [],
    }
    best_val_loss = float('inf')

    # Cargar checkpoint si existe
    if resume_from_checkpoint:
        if resume_from_checkpoint == 'latest':
            checkpoint_path = find_latest_checkpoint(Config.OUTPUT_PATH)
        else:
            checkpoint_path = resume_from_checkpoint

        if checkpoint_path and os.path.exists(checkpoint_path):
            start_epoch, loaded_history, es_state = load_checkpoint(
                checkpoint_path, model, optimizer, scheduler
            )
            if loaded_history:
                history = loaded_history
                best_val_loss = min(history['val_loss']) if history['val_loss'] else float('inf')
            if es_state:
                early_stopping.counter = es_state['counter']
                early_stopping.best_loss = es_state['best_loss']
            print(f"✓ Reanudando desde época {start_epoch}\n")

    print(f"Entrenando hasta época {Config.EPOCHS}...\n")
    start_time = time.time()

    for epoch in range(start_epoch, Config.EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"ÉPOCA {epoch}/{Config.EPOCHS}")
        print(f"{'='*60}")

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, scaler)
        history['train_loss'].append(train_metrics['loss'])
        history['train_obj'].append(train_metrics['obj_loss'])
        history['train_bbox'].append(train_metrics['bbox_loss'])
        history['train_class'].append(train_metrics['class_loss'])
        history['train_precision'].append(train_metrics['precision'])
        history['train_recall'].append(train_metrics['recall'])
        history['train_f1'].append(train_metrics['f1_score'])

        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion)
        history['val_loss'].append(val_metrics['loss'])
        history['val_obj'].append(val_metrics['obj_loss'])
        history['val_bbox'].append(val_metrics['bbox_loss'])
        history['val_class'].append(val_metrics['class_loss'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1_score'])

        scheduler.step()

        # VISUALIZACIÓN (NUEVO - Comentar estas 2 líneas para eliminar)
        if epoch % Config.VIS_INTERVAL == 0:
            visualize_predictions(model, val_loader, epoch, Config.OUTPUT_PATH)

        # Resumen
        print(f"\n{'─'*60}")
        print(f"{'Métrica':<20} {'Train':>12} {'Val':>12} {'Δ':>8}")
        print(f"{'─'*60}")
        print(f"{'Loss Total':<20} {train_metrics['loss']:>12.4f} {val_metrics['loss']:>12.4f} {(val_metrics['loss']-train_metrics['loss']):>8.4f}")
        print(f"{'  - Objectness':<20} {train_metrics['obj_loss']:>12.4f} {val_metrics['obj_loss']:>12.4f}")
        print(f"{'  - BBox':<20} {train_metrics['bbox_loss']:>12.4f} {val_metrics['bbox_loss']:>12.4f}")
        print(f"{'  - Class':<20} {train_metrics['class_loss']:>12.4f} {val_metrics['class_loss']:>12.4f}")
        print(f"{'─'*60}")
        print(f"{'Precision':<20} {train_metrics['precision']:>12.4f} {val_metrics['precision']:>12.4f}")
        print(f"{'Recall':<20} {train_metrics['recall']:>12.4f} {val_metrics['recall']:>12.4f}")
        print(f"{'F1 Score':<20} {train_metrics['f1_score']:>12.4f} {val_metrics['f1_score']:>12.4f}")
        print(f"{'─'*60}")

        # Guardar mejor modelo
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
            print(f"\n🎯 NUEVO MEJOR MODELO (val_loss: {best_val_loss:.4f})")
            save_checkpoint(epoch, model, optimizer, scheduler, early_stopping,
                          train_metrics, val_metrics, history, is_best=True)
        
        if epoch % Config.SAVE_EVERY == 0:
            save_checkpoint(epoch, model, optimizer, scheduler, early_stopping,
                          train_metrics, val_metrics, history, is_best=False)

        # Early stopping
        if early_stopping(val_metrics['loss']):
            print(f"\n⚠️  Early stopping activado en época {epoch}")
            save_checkpoint(epoch, model, optimizer, scheduler, early_stopping,
                          train_metrics, val_metrics, history, is_best=False)
            break

    # Test evaluation
    print("\n" + "="*60)
    print("EVALUACIÓN FINAL EN TEST SET")
    print("="*60)
    test_metrics = validate_epoch(model, test_loader, criterion)
    print(f"\n{'Métrica':<20} {'Valor':>12}")
    print(f"{'─'*40}")
    print(f"{'Loss Total':<20} {test_metrics['loss']:>12.4f}")
    print(f"{'  - Objectness':<20} {test_metrics['obj_loss']:>12.4f}")
    print(f"{'  - BBox':<20} {test_metrics['bbox_loss']:>12.4f}")
    print(f"{'  - Class':<20} {test_metrics['class_loss']:>12.4f}")
    print(f"{'─'*40}")
    print(f"{'Precision':<20} {test_metrics['precision']:>12.4f}")
    print(f"{'Recall':<20} {test_metrics['recall']:>12.4f}")
    print(f"{'F1 Score':<20} {test_metrics['f1_score']:>12.4f}")
    print(f"{'─'*40}")

    # Guardar resultados
    total_time = time.time() - start_time
    print(f"\n✓ Entrenamiento completado en {total_time/60:.1f} minutos")
    print(f"✓ Mejor val_loss: {best_val_loss:.4f}")

    with open(Config.OUTPUT_PATH + '/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    plot_results(history)
    return model

# ============================================================================
# VISUALIZACIÓN DE RESULTADOS
# ============================================================================
def plot_results(history):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Total')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(history['train_precision'], label='Train', linewidth=2)
    axes[0, 1].plot(history['val_precision'], label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(history['train_recall'], label='Train', linewidth=2)
    axes[0, 2].plot(history['val_recall'], label='Val', linewidth=2)
    axes[0, 2].set_xlabel('Época')
    axes[0, 2].set_ylabel('Recall')
    axes[0, 2].set_title('Recall')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(history['train_f1'], label='Train', linewidth=2)
    axes[1, 0].plot(history['val_f1'], label='Val', linewidth=2)
    axes[1, 0].set_xlabel('Época')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(history['train_bbox'], label='Train', linewidth=2)
    axes[1, 1].plot(history['val_bbox'], label='Val', linewidth=2)
    axes[1, 1].set_xlabel('Época')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Bounding Box Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(history['train_class'], label='Train', linewidth=2)
    axes[1, 2].plot(history['val_class'], label='Val', linewidth=2)
    axes[1, 2].set_xlabel('Época')
    axes[1, 2].set_ylabel('Loss')
    axes[1, 2].set_title('Classification Loss')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Config.OUTPUT_PATH + '/training_curves.png', dpi=150, bbox_inches='tight')
    print(f"\nGráficas guardadas: {Config.OUTPUT_PATH}/training_curves.png")
    plt.show()

#verificar si el modelo funciona en una carpeta de imágenes

def test_model_on_image(model_path, image_path, conf_threshold=0.2, save_output=True):
    """
    Función completa para probar el modelo en una imagen con visualización

    Args:
        model_path: ruta al checkpoint del modelo
        image_path: ruta a la imagen a testear
        conf_threshold: umbral de confianza para mostrar detecciones (default 0.5)
        save_output: si guardar la imagen con detecciones

    Returns:
        dict con métricas y visualización
    """
    print("="*60)
    print("TESTING MODELO EN IMAGEN")
    print("="*60)

    # 1. CARGAR MODELO
    print(f"\n[1/5] Cargando modelo desde: {model_path}")
    model = SGSNet(Config.NUM_CLASSES).to(Config.DEVICE)
    checkpoint = torch.load(model_path, map_location=Config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ Modelo cargado exitosamente")

    # 2. CARGAR IMAGEN ORIGINAL
    print(f"\n[2/5] Cargando imagen: {image_path}")
    if not os.path.exists(image_path):
        print(f"❌ ERROR: Imagen no encontrada")
        return None

    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"❌ ERROR: No se pudo leer la imagen")
        return None

    orig_h, orig_w = original_image.shape[:2]
    print(f"✓ Dimensiones originales: {orig_w}x{orig_h} pixels")

    # Convertir BGR (OpenCV) a RGB
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # 3. PREPROCESAMIENTO (CRÍTICO - DEBE SER IDÉNTICO AL ENTRENAMIENTO)
    print(f"\n[3/5] Preprocesando imagen...")
    print(f"  → Redimensionar: {orig_w}x{orig_h} → {Config.IMAGE_SIZE}x{Config.IMAGE_SIZE}")
    print(f"  → Normalizar: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")
    print(f"  → Convertir a tensor PyTorch")

    # Usar la misma transformación que en validación
    _, val_transform = get_transforms()

    # IMPORTANTE: Albumentations necesita bboxes vacías
    transformed = val_transform(image=image_rgb, bboxes=[], class_labels=[])
    image_tensor = transformed['image'].unsqueeze(0).to(Config.DEVICE)

    print(f"✓ Tensor shape: {image_tensor.shape}")

    # 4. INFERENCIA
    print(f"\n[4/5] Ejecutando inferencia...")
    with torch.no_grad():
        predictions = model(image_tensor)

    # Procesar predicciones
    B, C, H, W = predictions.shape
    num_anchors = 3
    pred_reshaped = predictions.view(B, num_anchors, 5 + Config.NUM_CLASSES, H, W)
    pred_reshaped = pred_reshaped.permute(0, 1, 3, 4, 2).contiguous()

    # Extraer componentes
    obj_scores = torch.sigmoid(pred_reshaped[0, :, :, :, 0])  # [anchors, H, W]
    bbox_preds = pred_reshaped[0, :, :, :, 1:5]              # [anchors, H, W, 4]
    class_preds = torch.sigmoid(pred_reshaped[0, :, :, :, 5:])  # [anchors, H, W, num_classes]

    # Encontrar detecciones con alta confianza
    detections = []
    for anchor_idx in range(num_anchors):
        for gy in range(H):
            for gx in range(W):
                obj_conf = obj_scores[anchor_idx, gy, gx].item()

                if obj_conf > conf_threshold:
                    # Decodificar bbox
                    dx = bbox_preds[anchor_idx, gy, gx, 0].item()
                    dy = bbox_preds[anchor_idx, gy, gx, 1].item()
                    dw = bbox_preds[anchor_idx, gy, gx, 2].item()
                    dh = bbox_preds[anchor_idx, gy, gx, 3].item()

                    # Convertir a coordenadas absolutas
                    cx = (gx + dx) / W
                    cy = (gy + dy) / H
                    w = dw
                    h = dh

                    # Obtener clase con mayor confianza
                    class_scores = class_preds[anchor_idx, gy, gx]
                    class_conf, class_idx = torch.max(class_scores, dim=0)

                    detections.append({
                        'bbox': [cx, cy, w, h],
                        'obj_conf': obj_conf,
                        'class_idx': class_idx.item(),
                        'class_conf': class_conf.item(),
                        'class_name': Config.CLASS_NAMES[class_idx.item()]
                    })

    print(f"✓ Detecciones encontradas: {len(detections)}")

    # Métricas
    max_conf = obj_scores.max().item()
    mean_conf = obj_scores.mean().item()
    high_conf_count = (obj_scores > conf_threshold).sum().item()

    print(f"\nMÉTRICAS:")
    print(f"  Confianza máxima: {max_conf:.4f}")
    print(f"  Confianza promedio: {mean_conf:.4f}")
    print(f"  Predicciones > {conf_threshold}: {high_conf_count}")

    # 5. VISUALIZACIÓN
    print(f"\n[5/5] Generando visualización...")

    # Crear copia para dibujar
    vis_image = image_rgb.copy()

    # Colores para cada clase (RGB)
    colors = [
        (255, 0, 0),      # flowering - Rojo
        (0, 255, 0),      # growing_g - Verde
        (0, 255, 255),    # growing_w - Cyan
        (255, 255, 0),    # nearly_m - Amarillo
        (255, 0, 255)     # mature - Magenta
    ]

    for det in detections:
        # Convertir coordenadas normalizadas a pixeles originales
        cx, cy, w, h = det['bbox']

        # Escalar al tamaño de la imagen redimensionada
        x1 = int((cx - w/2) * Config.IMAGE_SIZE)
        y1 = int((cy - h/2) * Config.IMAGE_SIZE)
        x2 = int((cx + w/2) * Config.IMAGE_SIZE)
        y2 = int((cy + h/2) * Config.IMAGE_SIZE)

        # Dibujar bbox
        color = colors[det['class_idx']]
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

        # Texto con clase y confianza
        label = f"{det['class_name']}: {det['obj_conf']:.2f}"

        # Fondo para el texto
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis_image, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
        cv2.putText(vis_image, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Mostrar resultado
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Imagen original
    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Imagen Original\n{orig_w}x{orig_h} px', fontsize=12)
    axes[0].axis('off')

    # Imagen con detecciones
    axes[1].imshow(vis_image)
    axes[1].set_title(f'Detecciones: {len(detections)}\nConf > {conf_threshold}', fontsize=12)
    axes[1].axis('off')

    # Leyenda de clases
    legend_text = "CLASES:\n"
    for idx, (name, color) in enumerate(zip(Config.CLASS_NAMES, colors)):
        legend_text += f"  {name}\n"

    plt.figtext(0.02, 0.02, legend_text, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Guardar resultado
    """
    if save_output:
        output_path = image_path.replace('.jpg', '_detections.jpg').replace('.png', '_detections.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✓ Imagen guardada: {output_path}")
    """

    plt.show()

    print("\n" + "="*60)
    print("DETECCIONES POR CLASE:")
    print("="*60)
    for class_name in Config.CLASS_NAMES:
        count = sum(1 for d in detections if d['class_name'] == class_name)
        if count > 0:
            avg_conf = np.mean([d['obj_conf'] for d in detections if d['class_name'] == class_name])
            print(f"  {class_name:<15}: {count:>3} detecciones (conf promedio: {avg_conf:.3f})")

    return {
        'detections': detections,
        'num_detections': len(detections),
        'max_confidence': max_conf,
        'mean_confidence': mean_conf,
        'image_shape': (orig_h, orig_w),
        'visualization': vis_image
    }


# Función adicional para testear múltiples imágenes
def test_model_on_folder(model_path, folder_path, conf_threshold=0.5, max_images=10):
    """
    Testear el modelo en múltiples imágenes de una carpeta
    """
    print(f"Testeando modelo en carpeta: {folder_path}")

    # Buscar imágenes
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.webp']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(folder_path, ext)))

    image_paths = image_paths[:max_images]

    if not image_paths:
        print("No se encontraron imágenes")
        return

    print(f"Encontradas {len(image_paths)} imágenes\n")

    results = []
    for img_path in image_paths:
        print(f"\nProcesando: {os.path.basename(img_path)}")
        result = test_model_on_image(model_path, img_path, conf_threshold, save_output=True)
        if result:
            results.append(result)

    # Resumen
    print("\n" + "="*60)
    print("RESUMEN GENERAL")
    print("="*60)
    total_detections = sum(r['num_detections'] for r in results)
    avg_confidence = np.mean([r['max_confidence'] for r in results])

    print(f"Imágenes procesadas: {len(results)}")
    print(f"Detecciones totales: {total_detections}")
    print(f"Confianza promedio: {avg_confidence:.4f}")

    return results

# ============================================================================
# MAIN
# ============================================================================
def main(resume=True):
    print("\n🍓 STRAWBERRY MATURITY DETECTION - SGSNET")
    print("="*60)

    if not os.path.exists(Config.IMAGES_PATH):
        print(f"❌ Imágenes no encontradas: {Config.IMAGES_PATH}")
        return

    if not os.path.exists(Config.ANNOTATIONS_PATH):
        print(f"❌ Anotaciones no encontradas: {Config.ANNOTATIONS_PATH}")
        return

    print(f"✓ Configuración verificada")
    print(f"  Imágenes: {Config.IMAGES_PATH}")
    print(f"  Anotaciones: {Config.ANNOTATIONS_PATH}")
    print(f"  Salida: {Config.OUTPUT_PATH}")
    print(f"  Visualización: {'ACTIVADA' if Config.VISUALIZE_TRAINING else 'DESACTIVADA'}\n")

    checkpoint_to_resume = 'latest' if resume else None
    model = train_sgsnet(resume_from_checkpoint=checkpoint_to_resume)

    print("\n" + "="*60)
    print("✓ ENTRENAMIENTO COMPLETADO")
    print(f"✓ Modelos guardados en: {Config.OUTPUT_PATH}")
    print("="*60)

if __name__ == "__main__":
    main(resume=True)

    """
    result = test_model_on_folder(
         model_path='src/data/processed/models/best_model.pth',
         folder_path='test/files/',
         conf_threshold=0.4,
         max_images=10
    )

    """

