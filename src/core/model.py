"""
SGSNet para Detección de Estados de Madurez de Fresas
Versión porfavor ya entrena
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
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
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

#Configuración

class Config:
    # Rutas
    IMAGES_PATH = '/data/raw/images' #cambiar
    ANNOTATIONS_PATH = '/data/processed/annotations.json' #cambiar
    OUTPUT_PATH = '/src/core//models/'

    # Clases
    CLASS_NAMES = ['flowering', 'growing_g', 'growing_w', 'nearly_m', 'mature'] # las clases que detectara el modelo
    NUM_CLASSES = len(CLASS_NAMES)

    # Hiperparámetros
    IMAGE_SIZE = 416 #tamaño de la imagen
    BATCH_SIZE = 8 #conjunto de imagenes por asi decirlo
    EPOCHS = 70 #N° de epocas
    LEARNING_RATE = 0.001
    EARLY_STOP_PATIENCE = 7 #paciencia es lo que me falta
    SAVE_EVERY = 5 #desde aca modificas cada cuantas epocas se hace un checkpoint

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') #esto basicamente es para ver que servicio esta activo cpu o gpu

os.makedirs(Config.OUTPUT_PATH, exist_ok=True)

#Arquitectura SGSNET
# CORREGIR O SOMETER A INVESTIGACIÓN ELABORAR DOCUMENTACIÓN SU FUNCIONAMIENTO LO HIZO GPT YO ME RENDI
class ConvBNAct(nn.Module):
    """Bloque Conv + BatchNorm + Activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=None, groups=1, act=True):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                             stride, padding, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class UniversalInvertedBottleneck(nn.Module):
    """UIB - Universal Inverted Bottleneck (MobileNetV4 inspired)"""
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=4):
        super().__init__()
        self.stride = stride
        self.use_residual = (stride == 1 and in_channels == out_channels)

        hidden_dim = int(in_channels * expand_ratio)

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_channels, hidden_dim, kernel_size=1))

        layers.extend([
            ConvBNAct(hidden_dim, hidden_dim, kernel_size=3,
                     stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        return self.conv(x)


class GrowthNetBackbone(nn.Module):
    """GrowthNet - Backbone ligero inspirado en MobileNetV4"""
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            ConvBNAct(3, 16, kernel_size=3, stride=2),
            ConvBNAct(16, 32, kernel_size=3, stride=1)
        )

        self.stage1 = nn.Sequential(
            UniversalInvertedBottleneck(32, 32, stride=2, expand_ratio=4),
            UniversalInvertedBottleneck(32, 64, stride=1, expand_ratio=4)
        )

        self.stage2 = nn.Sequential(
            UniversalInvertedBottleneck(64, 64, stride=2, expand_ratio=4),
            UniversalInvertedBottleneck(64, 128, stride=1, expand_ratio=4),
            UniversalInvertedBottleneck(128, 128, stride=1, expand_ratio=4)
        )

        self.stage3 = nn.Sequential(
            UniversalInvertedBottleneck(128, 128, stride=2, expand_ratio=6),
            UniversalInvertedBottleneck(128, 256, stride=1, expand_ratio=6),
            UniversalInvertedBottleneck(256, 256, stride=1, expand_ratio=6)
        )

        self.stage4 = nn.Sequential(
            UniversalInvertedBottleneck(256, 256, stride=2, expand_ratio=6),
            UniversalInvertedBottleneck(256, 512, stride=1, expand_ratio=6)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)
        p4 = self.stage3(p3)
        p5 = self.stage4(p4)
        return p3, p4, p5


class SimpleUpsampling(nn.Module):
    """Upsampling adaptativo simplificado"""
    def __init__(self, in_channels, scale_factor=2):
        super().__init__()
        self.scale_factor = scale_factor
        self.conv = ConvBNAct(in_channels, in_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv(x)
        return F.interpolate(x, scale_factor=self.scale_factor,
                           mode='bilinear', align_corners=False)


class FeatureFusionBlock(nn.Module):
    """Bloque de fusión de características"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.reduce = ConvBNAct(in_channels, out_channels, kernel_size=1)
        self.fuse = nn.Sequential(
            ConvBNAct(out_channels, out_channels, kernel_size=3),
            ConvBNAct(out_channels, out_channels, kernel_size=3)
        )

    def forward(self, x):
        x = self.reduce(x)
        return self.fuse(x)


class SGSNeck(nn.Module):
    """Neck para fusión multi-escala"""
    def __init__(self):
        super().__init__()
        self.up_p5_to_p4 = SimpleUpsampling(512, scale_factor=2)
        self.up_p4_to_p3 = SimpleUpsampling(256, scale_factor=2)
        self.fuse_p4 = FeatureFusionBlock(768, 256)
        self.fuse_p3 = FeatureFusionBlock(384, 128)
        self.out_p3 = ConvBNAct(128, 128, kernel_size=3)
        self.out_p4 = ConvBNAct(256, 256, kernel_size=3)
        self.out_p5 = ConvBNAct(512, 512, kernel_size=3)

    def forward(self, p3, p4, p5):
        p5_up = self.up_p5_to_p4(p5)
        p4_fused = self.fuse_p4(torch.cat([p4, p5_up], dim=1))
        p4_up = self.up_p4_to_p3(p4_fused)
        p3_fused = self.fuse_p3(torch.cat([p3, p4_up], dim=1))

        out_p3 = self.out_p3(p3_fused)
        out_p4 = self.out_p4(p4_fused)
        out_p5 = self.out_p5(p5)

        return out_p3, out_p4, out_p5


class DetectionHead(nn.Module):
    """Head de detección multi-escala"""
    def __init__(self, in_channels, num_classes, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.num_outputs = num_anchors * (5 + num_classes)

        self.conv = nn.Sequential(
            ConvBNAct(in_channels, in_channels, kernel_size=3),
            ConvBNAct(in_channels, in_channels, kernel_size=3),
            nn.Conv2d(in_channels, self.num_outputs, kernel_size=1)
        )

    def forward(self, x):
        return self.conv(x)


class SGSNet(nn.Module):
    """SGSNet - Strawberry Growth Stage Detection Network"""
    def __init__(self, num_classes=5):
        super().__init__()
        self.num_classes = num_classes

        self.backbone = GrowthNetBackbone()
        self.neck = SGSNeck()
        self.head_p3 = DetectionHead(128, num_classes, num_anchors=3)
        self.head_p4 = DetectionHead(256, num_classes, num_anchors=3)
        self.head_p5 = DetectionHead(512, num_classes, num_anchors=3)

        self._initialize_weights()

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
        p3, p4, p5 = self.backbone(x)
        p3_out, p4_out, p5_out = self.neck(p3, p4, p5)

        pred_p3 = self.head_p3(p3_out)
        pred_p4 = self.head_p4(p4_out)
        pred_p5 = self.head_p5(p5_out)

        return {
            'p3': pred_p3,
            'p4': pred_p4,
            'p5': pred_p5
        }

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total': total,
            'trainable': total,
            'millions': total / 1e6
        }


#Dataset

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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        anns = self.annotations[img_id]
        bboxes = []
        class_labels = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            if w > 0 and h > 0 and x >= 0 and y >= 0:
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
        for bbox in boxes:
            x_min, y_min, w, h = bbox
            cx = (x_min + w/2) / Config.IMAGE_SIZE
            cy = (y_min + h/2) / Config.IMAGE_SIZE
            nw = w / Config.IMAGE_SIZE
            nh = h / Config.IMAGE_SIZE
            normalized_boxes.append([cx, cy, nw, nh])

        if not normalized_boxes:
            normalized_boxes = [[0.5, 0.5, 0.1, 0.1]]
            labels = [0]

        # RETORNAR normalized_boxes, no boxes
        return image, torch.FloatTensor(normalized_boxes), torch.LongTensor(labels)

#Augments transforma las imagenes a la hora de entrenar y validar
def get_transforms():
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=15, p=0.4, border_mode=cv2.BORDER_CONSTANT),
        A.ShiftScaleRotate(shift_limit=0.08, scale_limit=0.15, rotate_limit=10, p=0.4),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.6),
        A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(format='coco', min_area=100, min_visibility=0.3, label_fields=['class_labels']))

    val_transform = A.Compose([
        A.Resize(height=Config.IMAGE_SIZE, width=Config.IMAGE_SIZE, p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], p=1.0),
        ToTensorV2(p=1.0),
    ], bbox_params=A.BboxParams(format='coco', min_area=0, min_visibility=0.0, label_fields=['class_labels']))

    return train_transform, val_transform


def collate_fn(batch):
    images, boxes, labels = zip(*batch)
    images = torch.stack(images, 0)
    return images, boxes, labels


# Funcion loss este cuidado con tocar su trabajo es a la hora de hacer predicciones sacar las loss en objetos, boundig box y clasificación, segun estas 3 se ajustan los hiperparametros

class SGSNetLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
        self.bce_cls = nn.BCEWithLogitsLoss(reduction='none')

    def forward_single_scale(self, predictions, targets_boxes, targets_labels,
                            grid_size, stride):
        B = predictions.shape[0]
        H, W = grid_size
        num_anchors = 3

        # Reshape: [B, anchors*(5+C), H, W] -> [B, anchors, H, W, 5+C]
        predictions = predictions.view(B, num_anchors, 5 + self.num_classes, H, W)
        predictions = predictions.permute(0, 1, 3, 4, 2).contiguous()

        pred_obj = predictions[..., 0]
        pred_bbox = predictions[..., 1:5]
        pred_cls = predictions[..., 5:]

        total_obj_loss = 0
        total_bbox_loss = 0
        total_cls_loss = 0
        num_positive = 0

        for b in range(B):
            gt_boxes = targets_boxes[b]
            gt_labels = targets_labels[b]

            if len(gt_boxes) == 0:
                target_obj = torch.zeros_like(pred_obj[b])
                total_obj_loss += self.bce_loss(pred_obj[b], target_obj).mean()
                continue

            target_obj = torch.zeros((num_anchors, H, W), device=predictions.device)
            target_bbox = torch.zeros((num_anchors, H, W, 4), device=predictions.device)
            target_cls = torch.zeros((num_anchors, H, W, self.num_classes), device=predictions.device)

            for gt_box, gt_label in zip(gt_boxes, gt_labels):
                cx, cy, w, h = gt_box

                gx = int(cx * W)
                gy = int(cy * H)
                gx = min(gx, W - 1)
                gy = min(gy, H - 1)

                anchor_idx = 0
                target_obj[anchor_idx, gy, gx] = 1.0
                target_bbox[anchor_idx, gy, gx, 0] = cx * W - gx
                target_bbox[anchor_idx, gy, gx, 1] = cy * H - gy
                target_bbox[anchor_idx, gy, gx, 2] = w
                target_bbox[anchor_idx, gy, gx, 3] = h
                target_cls[anchor_idx, gy, gx, int(gt_label)] = 1.0

                num_positive += 1

            obj_loss = self.bce_loss(pred_obj[b], target_obj).mean()
            total_obj_loss += obj_loss

            pos_mask = target_obj > 0.5
            if pos_mask.sum() > 0:
                pred_bbox_pos = pred_bbox[b][pos_mask]
                target_bbox_pos = target_bbox[pos_mask]
                bbox_loss = F.mse_loss(pred_bbox_pos, target_bbox_pos)
                total_bbox_loss += bbox_loss

                pred_cls_pos = pred_cls[b][pos_mask]
                target_cls_pos = target_cls[pos_mask]
                cls_loss = self.bce_cls(pred_cls_pos, target_cls_pos).mean()
                total_cls_loss += cls_loss

        total_obj_loss = total_obj_loss / B
        total_bbox_loss = total_bbox_loss / max(num_positive, 1)
        total_cls_loss = total_cls_loss / max(num_positive, 1)

        return total_obj_loss, total_bbox_loss, total_cls_loss

    def forward(self, predictions, targets_boxes, targets_labels):
        """
        predictions: dict con 'p3', 'p4', 'p5'
        """
        loss_p3_obj, loss_p3_bbox, loss_p3_cls = self.forward_single_scale(
            predictions['p3'], targets_boxes, targets_labels,
            grid_size=(52, 52), stride=8
        )

        loss_p4_obj, loss_p4_bbox, loss_p4_cls = self.forward_single_scale(
            predictions['p4'], targets_boxes, targets_labels,
            grid_size=(26, 26), stride=16
        )

        loss_p5_obj, loss_p5_bbox, loss_p5_cls = self.forward_single_scale(
            predictions['p5'], targets_boxes, targets_labels,
            grid_size=(13, 13), stride=32
        )

        total_obj_loss = (loss_p3_obj + loss_p4_obj + loss_p5_obj) / 3
        total_bbox_loss = (loss_p3_bbox + loss_p4_bbox + loss_p5_bbox) / 3
        total_cls_loss = (loss_p3_cls + loss_p4_cls + loss_p5_cls) / 3

        total_loss = total_obj_loss + 5.0 * total_bbox_loss + 2.0 * total_cls_loss

        return total_loss, {
            'obj': total_obj_loss.item(),
            'bbox': total_bbox_loss.item(),
            'class': total_cls_loss.item()
        }


#Metricas

class Metrics:
    def __init__(self):
        self.reset()

    def reset(self):
        self.losses = []
        self.obj_losses = []
        self.bbox_losses = []
        self.class_losses = []
        self.num_detections = []

    def update(self, loss, loss_dict):
        self.losses.append(loss)
        self.obj_losses.append(loss_dict.get('obj', 0))
        self.bbox_losses.append(loss_dict.get('bbox', 0))
        self.class_losses.append(loss_dict.get('class', 0))

    def get_metrics(self):
        if not self.losses:
            return {'loss': 0.0, 'obj_loss': 0.0, 'bbox_loss': 0.0,
                   'class_loss': 0.0}

        return {
            'loss': np.mean(self.losses),
            'obj_loss': np.mean(self.obj_losses), #detecta fresas
            'bbox_loss': np.mean(self.bbox_losses), #encierra las fresas
            'class_loss': np.mean(self.class_losses) #clasifica bien las clases
        }


# Early stopping

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001):
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

#Logica de checkpoints

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
        print(f"Mejor modelo guardado: {best_path}")
    else:
        checkpoint_path = Config.OUTPUT_PATH + f'/checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint guardado: {checkpoint_path}") #si es el mejor

# si no funciona y hay modelos guardados porfavor llamar al 967840901
def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint no encontrado: {checkpoint_path}")
        return 0, None, None

    print(f"Cargando checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Modelo restaurado desde época {checkpoint['epoch']}")

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


#Entrenamiento

def load_data():
    print("Cargando dataset")
    # primero cargas el dataset, defines la cantidad para entrenamiento, validación y test
    with open(Config.ANNOTATIONS_PATH, 'r') as f:
        coco_data = json.load(f)

    print(f"Imágenes: {len(coco_data['images'])}")
    print(f"Anotaciones: {len(coco_data['annotations'])}")

    image_ids = [img['id'] for img in coco_data['images']]
    #en esta parte es donde ocurre la divición 70/15/15 houlding validation
    train_ids, temp_ids = train_test_split(image_ids, test_size=0.3, random_state=42)
    val_ids, test_ids = train_test_split(temp_ids, test_size=0.5, random_state=42)

    print(f"Train: {len(train_ids)} | Val: {len(val_ids)} | Test: {len(test_ids)}")
    return coco_data, train_ids, val_ids, test_ids

# luego creas los
def create_dataloaders(coco_data, train_ids, val_ids, test_ids):
    #primero llama a get transforms que seran las transformaciones que se aplicaran a los datos de entrenamiento y validacion
    train_transform, val_transform = get_transforms()
    # crea tres instancias pues esa clase es la que sabe como cargar la imagen a partir del ID
    train_dataset = StrawberryDataset(coco_data, Config.IMAGES_PATH, train_ids, train_transform)
    val_dataset = StrawberryDataset(coco_data, Config.IMAGES_PATH, val_ids, val_transform)
    test_dataset = StrawberryDataset(coco_data, Config.IMAGES_PATH, test_ids, val_transform)

    """
    shuffle es como decirle que te barajee las imagenes por eso solo se aplica a entrenamiento
    bach size define cuantas imagenes cargar y pasar por lote
    dataloader es un iterador para facilitar la carga de imagenes
    """
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
                             collate_fn=collate_fn, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                           collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                            collate_fn=collate_fn, num_workers=2)

    return train_loader, val_loader, test_loader


def train_epoch(model, train_loader, optimizer, criterion):
    model.train()
    metrics = Metrics()

    progress_bar = tqdm(train_loader, desc="Training")
    for images, target_boxes, target_labels in progress_bar:
        images = images.to(Config.DEVICE)

        optimizer.zero_grad()
        predictions = model(images)  # dict con p3, p4, p5
        loss, loss_dict = criterion(predictions, target_boxes, target_labels)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()

        metrics.update(loss.item(), loss_dict)

        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Obj': f'{loss_dict["obj"]:.3f}',
            'Box': f'{loss_dict["bbox"]:.3f}',
            'Cls': f'{loss_dict["class"]:.3f}'
        })

    return metrics.get_metrics()


def validate_epoch(model, val_loader, criterion):
    model.eval()
    metrics = Metrics()

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validating")
        for images, target_boxes, target_labels in progress_bar:
            images = images.to(Config.DEVICE)

            predictions = model(images)
            loss, loss_dict = criterion(predictions, target_boxes, target_labels)

            metrics.update(loss.item(), loss_dict)
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})

    return metrics.get_metrics()


def train_sgsnet(resume_from_checkpoint=None):
    print(f"Classes: {Config.CLASS_NAMES}\n")

    # Cargar datos
    coco_data, train_ids, val_ids, test_ids = load_data()
    train_loader, val_loader, test_loader = create_dataloaders(coco_data, train_ids, val_ids, test_ids)

    # Modelo
    model = SGSNet(Config.NUM_CLASSES).to(Config.DEVICE)
    params = model.count_parameters()

    # Optimizador y pérdida
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE, weight_decay=0.0001)
    criterion = SGSNetLoss(Config.NUM_CLASSES)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=Config.EARLY_STOP_PATIENCE)

    # Tracking
    start_epoch = 1
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_obj': [],
        'val_obj': [],
        'train_bbox': [],
        'val_bbox': [],
        'train_class': [],
        'val_class': []
    }
    best_val_loss = float('inf')

    # Cargar checkpoint si existe
    if resume_from_checkpoint:
        if resume_from_checkpoint == 'latest':
            checkpoint_path = find_latest_checkpoint(Config.OUTPUT_PATH)
            if checkpoint_path:
                print(f"Checkpoint encontrado {checkpoint_path}")
            else:
                print("No hay checkpoints")
                checkpoint_path = None
        else:
            checkpoint_path = resume_from_checkpoint

        if checkpoint_path and os.path.exists(checkpoint_path):
            start_epoch, loaded_history, es_state = load_checkpoint(
                checkpoint_path, model, optimizer, scheduler
            )

            if loaded_history:
                history = loaded_history
                best_val_loss = min(history['val_loss'])

            if es_state:
                early_stopping.counter = es_state['counter']
                early_stopping.best_loss = es_state['best_loss']

            print(f"Reanudando desde {start_epoch} epoca \n")

    print(f"Entrenando desde hasta {Config.EPOCHS}...")
    start_time = time.time()

    for epoch in range(start_epoch, Config.EPOCHS + 1):
        print(f"Epoca {epoch}/{Config.EPOCHS}")

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion)
        history['train_loss'].append(train_metrics['loss'])
        history['train_obj'].append(train_metrics['obj_loss'])
        history['train_bbox'].append(train_metrics['bbox_loss'])
        history['train_class'].append(train_metrics['class_loss'])

        # Validate
        val_metrics = validate_epoch(model, val_loader, criterion)
        history['val_loss'].append(val_metrics['loss'])
        history['val_obj'].append(val_metrics['obj_loss'])
        history['val_bbox'].append(val_metrics['bbox_loss'])
        history['val_class'].append(val_metrics['class_loss'])

        scheduler.step()

        # Print summary
        print(f"\n{'-'*60}")
        print(f"Resumen Epoch {epoch}:")
        print(f"Train Loss: {train_metrics['loss']:.4f} | Val Loss: {val_metrics['loss']:.4f}")
        print(f"  Obj:  {train_metrics['obj_loss']:.4f} | {val_metrics['obj_loss']:.4f}")
        print(f"  Bbox: {train_metrics['bbox_loss']:.4f} | {val_metrics['bbox_loss']:.4f}")
        print(f"  Cls:  {train_metrics['class_loss']:.4f} | {val_metrics['class_loss']:.4f}")
        print(f"\n{'-'*60}")

        # Guardar mejor modelo
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
            print(f"\nNuevo mejor modelo (val_loss: {best_val_loss:.4f})")
            save_checkpoint(epoch, model, optimizer, scheduler, early_stopping,
                          train_metrics, val_metrics, history, is_best=True)
        elif epoch % Config.SAVE_EVERY == 0:
            save_checkpoint(epoch, model, optimizer, scheduler, early_stopping,
                          train_metrics, val_metrics, history, is_best=False)

        # Early stopping
        if early_stopping(val_metrics['loss']):
            print(f"\nEarly stopping activado en época {epoch}")
            save_checkpoint(epoch, model, optimizer, scheduler, early_stopping,
                          train_metrics, val_metrics, history, is_best=False)
            break

    # Test evaluation
    print("EVALUACIÓN FINAL EN TEST SET")
    test_metrics = validate_epoch(model, test_loader, criterion)
    print(f"Test Loss: {test_metrics['loss']:.4f}")
    print(f"  Obj Loss:  {test_metrics['obj_loss']:.4f}")
    print(f"  Bbox Loss: {test_metrics['bbox_loss']:.4f}")
    print(f"  Cls Loss:  {test_metrics['class_loss']:.4f}")

    # Guardar resultados finales
    total_time = time.time() - start_time
    print(f"\nEntrenamiento completado en {total_time/60:.1f} minutos")
    print(f"Mejor val_loss: {best_val_loss:.4f}")

    # Guardar historial
    with open(Config.OUTPUT_PATH + '/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Plotear resultados
    plot_results(history)

    return model


def plot_results(history):
    """Visualización de resultados"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Loss total general para todos
    axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Total')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Objectness
    axes[0, 1].plot(history['train_obj'], label='Train', linewidth=2)
    axes[0, 1].plot(history['val_obj'], label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Objectness Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Bbox
    axes[1, 0].plot(history['train_bbox'], label='Train', linewidth=2)
    axes[1, 0].plot(history['val_bbox'], label='Val', linewidth=2)
    axes[1, 0].set_xlabel('Época')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Bounding Box Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Classification
    axes[1, 1].plot(history['train_class'], label='Train', linewidth=2)
    axes[1, 1].plot(history['val_class'], label='Val', linewidth=2)
    axes[1, 1].set_xlabel('Época')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Classification Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Config.OUTPUT_PATH + '/training_curves.png', dpi=150, bbox_inches='tight')
    print(f"\nGráficas guardadas en: {Config.OUTPUT_PATH}/training_curves.png")
    plt.show()


def main(resume=True):
    print("Strawberry Detector")

    # Verificar rutas
    if not os.path.exists(Config.IMAGES_PATH):
        print(f"Imágenes no encontradas  {Config.IMAGES_PATH}")
        return

    if not os.path.exists(Config.ANNOTATIONS_PATH):
        print(f"Anotaciones no encontradas {Config.ANNOTATIONS_PATH}")
        return

    print(f"\nConfiguración verificada")
    print(f"  Imágenes: {Config.IMAGES_PATH}")
    print(f"  Anotaciones: {Config.ANNOTATIONS_PATH}")
    print(f"  Salida: {Config.OUTPUT_PATH}\n")

    checkpoint_to_resume = 'latest' if resume else None
    model = train_sgsnet(resume_from_checkpoint=checkpoint_to_resume)

    print("Entrenamiento completado PORFIN ")
    print(f"Modelos guardados en: {Config.OUTPUT_PATH}")


if __name__ == "__main__":
    main()