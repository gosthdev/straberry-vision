"""
Dataset y transformaciones para detección de fresas
"""
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
from sklearn.model_selection import train_test_split

from .config import Config


class StrawberryDataset(Dataset):
    """
    Dataset personalizado para detección de estados de madurez de fresas
    Compatible con formato COCO
    """
    def __init__(self, coco_data, image_dir, image_ids, transform=None):
        self.coco_data = coco_data
        self.image_dir = Path(image_dir)
        self.image_ids = image_ids
        self.transform = transform

        # Mapeo de categorías
        self.cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
        self.name_to_label = {name: i for i, name in enumerate(Config.CLASS_NAMES)}

        # Organizar anotaciones por imagen
        self.annotations = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.annotations:
                self.annotations[img_id] = []
            self.annotations[img_id].append(ann)

        # Filtrar IDs válidos (que tengan anotaciones)
        self.valid_ids = [img_id for img_id in image_ids
                         if img_id in self.annotations and len(self.annotations[img_id]) > 0]

    def __len__(self):
        return len(self.valid_ids)

    def __getitem__(self, idx):
        img_id = self.valid_ids[idx]
        img_info = next(img for img in self.coco_data['images'] if img['id'] == img_id)
        img_path = self.image_dir / img_info['file_name']

        # Buscar archivo con diferentes extensiones
        if not img_path.exists():
            for ext in ['.jpg', '.jpeg', '.png', '.webp']:
                test_path = img_path.with_suffix(ext)
                if test_path.exists():
                    img_path = test_path
                    break

        # Cargar imagen
        image = cv2.imread(str(img_path))
        if image is None:
            raise ValueError(f"No se pudo cargar: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Procesar anotaciones
        anns = self.annotations[img_id]
        bboxes = []
        class_labels = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            if w > 5 and h > 5 and x >= 0 and y >= 0:
                bboxes.append([x, y, w, h])
                cat_name = self.cat_id_to_name[ann['category_id']]
                class_labels.append(self.name_to_label.get(cat_name, 0))

        # Asegurar al menos una caja
        if not bboxes:
            bboxes = [[10, 10, 50, 50]]
            class_labels = [0]

        # Aplicar transformaciones
        transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
        image = transformed['image']
        boxes = transformed['bboxes']
        labels = transformed['class_labels']

        # Normalizar cajas a formato [cx, cy, w, h] en rango [0,1]
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

        # Asegurar al menos una caja válida
        if not normalized_boxes:
            normalized_boxes = [[0.5, 0.5, 0.1, 0.1]]
            valid_labels = [0]

        return image, torch.FloatTensor(normalized_boxes), torch.LongTensor(valid_labels)


def get_transforms():
    """
    Obtiene las transformaciones para entrenamiento y validación
    
    Returns:
        train_transform: Transformaciones con augmentación para entrenamiento
        val_transform: Transformaciones sin augmentación para validación
    """
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
    """
    Función personalizada para combinar muestras en un batch
    Necesaria porque cada imagen puede tener diferente número de cajas
    """
    images, boxes, labels = zip(*batch)
    images = torch.stack(images, 0)
    return images, boxes, labels


def load_data():
    """
    Carga el dataset COCO y divide en train/val/test
    
    Returns:
        coco_data: Diccionario con datos COCO
        train_ids: IDs de imágenes para entrenamiento
        val_ids: IDs de imágenes para validación
        test_ids: IDs de imágenes para test
    """
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
    """
    Crea los DataLoaders para train, val y test
    
    Returns:
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validación
        test_loader: DataLoader de test
    """
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
