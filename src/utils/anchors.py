import numpy as np
from sklearn.cluster import KMeans
import torch
from tqdm import tqdm
import sys
from pathlib import Path

# Add project root to path to allow absolute imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.core.config import Config

def calculate_optimal_anchors(coco_data, image_ids, num_anchors=3, image_size=416):
    """
    Calcula los anchors óptimos usando K-Means sobre las cajas del dataset de entrenamiento.
    
    Args:
        coco_data: Datos COCO cargados
        image_ids: Lista de IDs de imágenes de entrenamiento
        num_anchors: Número de anchors a generar
        image_size: Tamaño de imagen para normalización
        
    Returns:
        torch.Tensor: Tensor con los nuevos anchors ordenados por área
    """
    print("\nCalculando anchors dinámicos basados en el dataset...")
    
    # Recolectar dimensiones de todas las cajas (normalizadas)
    wh = []
    
    # Crear mapa de anotaciones para acceso rápido
    img_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
        
    count = 0
    for img_id in tqdm(image_ids, desc="Extrayendo cajas"):
        if img_id not in img_to_anns:
            continue
            
        for ann in img_to_anns[img_id]:
            w, h = ann['bbox'][2], ann['bbox'][3]
            
            # Normalizar w y h relativos a la imagen original
            img_info = next((img for img in coco_data['images'] if img['id'] == img_id), None)
            if img_info:
                img_w = img_info['width']
                img_h = img_info['height']
                
                norm_w = w / img_w
                norm_h = h / img_h
                
                wh.append([norm_w, norm_h])
                count += 1

    if len(wh) == 0:
        print("Advertencia: No se encontraron cajas para calcular anchors. Usando por defecto.")
        return Config.ANCHORS

    wh = np.array(wh)
    
    # K-Means
    kmeans = KMeans(n_clusters=num_anchors, random_state=42, n_init=10)
    kmeans.fit(wh)
    
    # Obtener centros y ordenar por área
    anchors = kmeans.cluster_centers_
    areas = anchors[:, 0] * anchors[:, 1]
    indices = np.argsort(areas)
    anchors = anchors[indices]
    
    print(f"Anchors calculados (w, h): \n{anchors}")
    
    return torch.tensor(anchors, dtype=torch.float32)
