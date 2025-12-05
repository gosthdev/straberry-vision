"""
Dataset y transformaciones para detección de fresas v2
Incluye: Mosaic, MixUp, y soporte para negative samples
"""
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import json
import random
from sklearn.model_selection import train_test_split

from .config import Config


class StrawberryDataset(Dataset):
    """
    Dataset personalizado para detección de estados de madurez de fresas
    Compatible con formato COCO
    """
    def __init__(self, coco_data, image_dir, image_ids, transform=None, is_training=False):
        self.coco_data = coco_data
        self.image_dir = Path(image_dir)
        self.image_ids = image_ids
        self.transform = transform
        self.is_training = is_training

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

        # Crear mapa de imagen id a info
        self.id_to_img_info = {img['id']: img for img in coco_data['images']}

        # Filtrar IDs válidos (que tengan anotaciones)
        self.valid_ids = [img_id for img_id in image_ids
                         if img_id in self.annotations and len(self.annotations[img_id]) > 0]
        
        # IDs para mosaic (necesitamos al menos 4)
        self.mosaic_ids = self.valid_ids.copy()

    def __len__(self):
        return len(self.valid_ids)
    
    def _load_image_and_boxes(self, img_id):
        """Carga una imagen y sus bounding boxes"""
        img_info = self.id_to_img_info[img_id]
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
        anns = self.annotations.get(img_id, [])
        bboxes = []
        class_labels = []

        for ann in anns:
            x, y, w, h = ann['bbox']
            if w > 5 and h > 5 and x >= 0 and y >= 0:
                bboxes.append([x, y, w, h])
                cat_name = self.cat_id_to_name[ann['category_id']]
                class_labels.append(self.name_to_label.get(cat_name, 0))

        return image, bboxes, class_labels

    def _apply_mosaic(self, idx):
        """
        Aplica Mosaic augmentation: combina 4 imágenes en una
        Esto es crucial para entrenar detección de múltiples objetos
        Versión simplificada y robusta
        """
        mosaic_size = Config.IMAGE_SIZE
        
        # Seleccionar 4 imágenes (la actual + 3 aleatorias)
        indices = [idx] + random.sample(range(len(self.valid_ids)), 3)
        
        # Crear canvas mosaic 2x2
        mosaic_img = np.zeros((mosaic_size, mosaic_size, 3), dtype=np.uint8)
        
        all_boxes = []
        all_labels = []
        
        # Dividir en 4 cuadrantes
        half = mosaic_size // 2
        positions = [
            (0, 0, half, half),           # top-left
            (half, 0, mosaic_size, half), # top-right  
            (0, half, half, mosaic_size), # bottom-left
            (half, half, mosaic_size, mosaic_size)  # bottom-right
        ]
        
        for i, index in enumerate(indices):
            img_id = self.valid_ids[index]
            image, bboxes, labels = self._load_image_and_boxes(img_id)
            
            x1, y1, x2, y2 = positions[i]
            quadrant_w = x2 - x1
            quadrant_h = y2 - y1
            
            # Redimensionar imagen al tamaño del cuadrante
            img_resized = cv2.resize(image, (quadrant_w, quadrant_h))
            mosaic_img[y1:y2, x1:x2] = img_resized
            
            # Escalar y trasladar bboxes
            orig_h, orig_w = image.shape[:2]
            scale_x = quadrant_w / orig_w
            scale_y = quadrant_h / orig_h
            
            for bbox, label in zip(bboxes, labels):
                bx, by, bw, bh = bbox
                
                # Escalar al tamaño del cuadrante
                new_x = bx * scale_x + x1
                new_y = by * scale_y + y1
                new_w = bw * scale_x
                new_h = bh * scale_y
                
                # Asegurar que la caja está dentro del canvas
                new_x = max(0, min(new_x, mosaic_size - 1))
                new_y = max(0, min(new_y, mosaic_size - 1))
                new_w = min(new_w, mosaic_size - new_x)
                new_h = min(new_h, mosaic_size - new_y)
                
                # Solo añadir si la caja es significativa
                if new_w > 5 and new_h > 5:
                    all_boxes.append([new_x, new_y, new_w, new_h])
                    all_labels.append(label)
        
        return mosaic_img, all_boxes, all_labels

    def _apply_mixup(self, image, boxes, labels, alpha=0.5):
        """
        Aplica MixUp augmentation: mezcla dos imágenes
        """
        # Copiar listas para no modificar originales
        boxes = list(boxes)
        labels = list(labels)
        
        # Seleccionar otra imagen aleatoria
        other_idx = random.randint(0, len(self.valid_ids) - 1)
        other_id = self.valid_ids[other_idx]
        other_image, other_boxes, other_labels = self._load_image_and_boxes(other_id)
        
        # Redimensionar la otra imagen al mismo tamaño
        h, w = image.shape[:2]
        other_image = cv2.resize(other_image, (w, h))
        
        # Ratio de mezcla aleatorio
        lam = np.random.beta(alpha, alpha)
        lam = max(lam, 1 - lam)  # Asegurar que la imagen principal domine
        
        # Mezclar imágenes
        mixed_image = (image * lam + other_image * (1 - lam)).astype(np.uint8)
        
        # Escalar boxes de la otra imagen
        orig_h, orig_w = other_image.shape[:2]
        for bbox, label in zip(other_boxes, other_labels):
            bx, by, bw, bh = bbox
            # Las boxes ya están en coordenadas de la imagen redimensionada
            # Verificar que están dentro
            if bx >= 0 and by >= 0 and bx + bw <= w and by + bh <= h:
                boxes.append([bx, by, bw, bh])
                labels.append(label)
        
        return mixed_image, boxes, labels

    def __getitem__(self, idx):
        img_id = self.valid_ids[idx]
        
        # Decidir si aplicar Mosaic (solo en entrenamiento)
        use_mosaic = (self.is_training and Config.USE_MOSAIC and 
                      random.random() < Config.MOSAIC_PROB and len(self.valid_ids) >= 4)
        
        if use_mosaic:
            image, bboxes, class_labels = self._apply_mosaic(idx)
        else:
            image, bboxes, class_labels = self._load_image_and_boxes(img_id)
        
        # Decidir si aplicar MixUp (solo en entrenamiento, y menos probable si ya se usó mosaic)
        use_mixup = (self.is_training and Config.USE_MIXUP and 
                     random.random() < Config.MIXUP_PROB and not use_mosaic)
        
        if use_mixup:
            image, bboxes, class_labels = self._apply_mixup(image, bboxes, class_labels)

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


class NegativeSampleDataset(Dataset):
    """
    Dataset de imágenes negativas (sin fresas) para reducir falsos positivos
    Usa imágenes de fondos/texturas similares al entorno de fresas
    """
    def __init__(self, image_dir, transform=None, max_samples=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        
        # Buscar imágenes
        self.image_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
            self.image_paths.extend(list(self.image_dir.glob(ext)))
        
        if max_samples and len(self.image_paths) > max_samples:
            self.image_paths = random.sample(self.image_paths, max_samples)
        
        print(f"NegativeSampleDataset: {len(self.image_paths)} imágenes de fondo cargadas")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        image = cv2.imread(str(img_path))
        if image is None:
            # Imagen por defecto si falla
            image = np.zeros((Config.IMAGE_SIZE, Config.IMAGE_SIZE, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Sin bounding boxes (imagen negativa)
        bboxes = []
        class_labels = []
        
        if self.transform:
            transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
            image = transformed['image']
        
        # Retornar sin boxes
        return image, torch.FloatTensor([]), torch.LongTensor([])


class CombinedDataset(Dataset):
    """
    Dataset combinado que incluye ejemplos positivos y negativos
    """
    def __init__(self, positive_dataset, negative_dataset=None, negative_ratio=0.1):
        self.positive_dataset = positive_dataset
        self.negative_dataset = negative_dataset
        self.negative_ratio = negative_ratio
        
        # Calcular número de negativos a incluir
        if negative_dataset and len(negative_dataset) > 0:
            max_negatives = int(len(positive_dataset) * negative_ratio)
            self.num_negatives = min(max_negatives, len(negative_dataset))
        else:
            self.num_negatives = 0
        
        self.total_length = len(positive_dataset) + self.num_negatives
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        if idx < len(self.positive_dataset):
            return self.positive_dataset[idx]
        else:
            neg_idx = idx - len(self.positive_dataset)
            return self.negative_dataset[neg_idx]


def get_transforms():
    """
    Obtiene las transformaciones para entrenamiento y validación
    Incluye augmentaciones más agresivas para multi-objeto
    """
    train_transform = A.Compose([
        # Primero redimensionar para asegurar tamaño mínimo
        A.LongestMaxSize(max_size=Config.IMAGE_SIZE + 64),
        A.PadIfNeeded(min_height=Config.IMAGE_SIZE, min_width=Config.IMAGE_SIZE, 
                      border_mode=cv2.BORDER_CONSTANT, value=0),
        # Transformaciones geométricas
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        # Color augmentation
        A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
        A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        # Noise y blur
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.MotionBlur(blur_limit=(3, 7), p=0.3),
        ], p=0.3),
        # Scale augmentation (más seguro que crop)
        A.RandomScale(scale_limit=0.2, p=0.3),
        # Simula oclusiones
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, 
                        min_holes=1, min_height=8, min_width=8,
                        fill_value=0, p=0.2),
        # Resize final al tamaño objetivo
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
    Con soporte para negative samples
    """
    train_transform, val_transform = get_transforms()
    
    # Dataset principal de entrenamiento con augmentation
    train_dataset = StrawberryDataset(
        coco_data, Config.IMAGES_PATH, train_ids, 
        train_transform, is_training=True
    )
    val_dataset = StrawberryDataset(
        coco_data, Config.IMAGES_PATH, val_ids, 
        val_transform, is_training=False
    )
    test_dataset = StrawberryDataset(
        coco_data, Config.IMAGES_PATH, test_ids, 
        val_transform, is_training=False
    )
    
    # Intentar cargar negative samples si está habilitado
    if Config.USE_NEGATIVE_SAMPLES:
        negative_path = Path(Config.IMAGES_PATH).parent / 'negatives'
        if negative_path.exists():
            max_negatives = int(len(train_ids) * Config.NEGATIVE_SAMPLE_RATIO)
            negative_dataset = NegativeSampleDataset(
                negative_path, train_transform, max_samples=max_negatives
            )
            train_dataset = CombinedDataset(
                train_dataset, negative_dataset, 
                negative_ratio=Config.NEGATIVE_SAMPLE_RATIO
            )
            print(f"✓ Negative samples habilitados: {len(negative_dataset)} imágenes")
        else:
            print(f"⚠ Directorio de negatives no encontrado: {negative_path}")
            print("  Para usar negative samples, crea el directorio y añade imágenes de fondo")

    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True
    )

    return train_loader, val_loader, test_loader
