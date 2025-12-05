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

from src.core.model.config import Config


def calculate_optimal_anchors_multiscale(coco_data, image_ids, anchors_per_scale=3, num_scales=3):
    """
    Calcula los anchors óptimos para múltiples escalas usando K-Means.
    Distribuye los anchors por tamaño a cada escala.
    
    Args:
        coco_data: Datos COCO cargados
        image_ids: Lista de IDs de imágenes de entrenamiento
        anchors_per_scale: Número de anchors por escala
        num_scales: Número de escalas de detección
        
    Returns:
        list: Lista de tensores de anchors por escala [anchors_p3, anchors_p4, anchors_p5]
    """
    print("\nCalculando anchors multi-escala basados en el dataset...")
    
    total_anchors = anchors_per_scale * num_scales
    
    # Recolectar dimensiones de todas las cajas (normalizadas)
    wh = []
    
    # Crear mapa de anotaciones para acceso rápido
    img_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    id_to_img_info = {img['id']: img for img in coco_data['images']}
    
    count = 0
    for img_id in tqdm(image_ids, desc="Extrayendo cajas"):
        if img_id not in img_to_anns:
            continue
            
        for ann in img_to_anns[img_id]:
            w, h = ann['bbox'][2], ann['bbox'][3]
            
            # Normalizar w y h relativos a la imagen original
            img_info = id_to_img_info.get(img_id)
            if img_info:
                img_w = img_info['width']
                img_h = img_info['height']
                
                norm_w = w / img_w
                norm_h = h / img_h
                
                # Filtrar cajas muy pequeñas o inválidas
                if norm_w > 0.01 and norm_h > 0.01:
                    wh.append([norm_w, norm_h])
                    count += 1

    if len(wh) == 0:
        print("Advertencia: No se encontraron cajas para calcular anchors. Usando por defecto.")
        return Config.ANCHORS

    wh = np.array(wh)
    print(f"Total de cajas procesadas: {count}")
    
    # K-Means para encontrar todos los anchors
    kmeans = KMeans(n_clusters=total_anchors, random_state=42, n_init=10)
    kmeans.fit(wh)
    
    # Obtener centros y ordenar por área
    anchors = kmeans.cluster_centers_
    areas = anchors[:, 0] * anchors[:, 1]
    indices = np.argsort(areas)
    anchors = anchors[indices]
    
    # Dividir en escalas (pequeño, mediano, grande)
    anchors_per_scale_list = []
    for i in range(num_scales):
        start_idx = i * anchors_per_scale
        end_idx = start_idx + anchors_per_scale
        scale_anchors = torch.tensor(anchors[start_idx:end_idx], dtype=torch.float32)
        anchors_per_scale_list.append(scale_anchors)
    
    # Imprimir resultados
    scale_names = ['P3 (pequeño)', 'P4 (mediano)', 'P5 (grande)']
    print(f"\nAnchors calculados por escala:")
    for i, (scale_anchors, name) in enumerate(zip(anchors_per_scale_list, scale_names)):
        print(f"\n{name}:")
        for j, anchor in enumerate(scale_anchors):
            print(f"  Anchor {j+1}: w={anchor[0]:.4f}, h={anchor[1]:.4f}")
    
    return anchors_per_scale_list


def calculate_optimal_anchors(coco_data, image_ids, num_anchors=3, image_size=416):
    """
    Versión legacy: Calcula los anchors óptimos usando K-Means para una sola escala.
    Mantenida para compatibilidad.
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
    
    id_to_img_info = {img['id']: img for img in coco_data['images']}
        
    count = 0
    for img_id in tqdm(image_ids, desc="Extrayendo cajas"):
        if img_id not in img_to_anns:
            continue
            
        for ann in img_to_anns[img_id]:
            w, h = ann['bbox'][2], ann['bbox'][3]
            
            # Normalizar w y h relativos a la imagen original
            img_info = id_to_img_info.get(img_id)
            if img_info:
                img_w = img_info['width']
                img_h = img_info['height']
                
                norm_w = w / img_w
                norm_h = h / img_h
                
                wh.append([norm_w, norm_h])
                count += 1

    if len(wh) == 0:
        print("Advertencia: No se encontraron cajas para calcular anchors. Usando por defecto.")
        return Config.ANCHORS[1]  # Retornar anchors de escala media por defecto

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


def analyze_box_distribution(coco_data, image_ids):
    """
    Analiza la distribución de tamaños de cajas en el dataset.
    Útil para entender qué escalas necesitan más anchors.
    """
    wh = []
    
    img_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    id_to_img_info = {img['id']: img for img in coco_data['images']}
    
    for img_id in image_ids:
        if img_id not in img_to_anns:
            continue
            
        for ann in img_to_anns[img_id]:
            w, h = ann['bbox'][2], ann['bbox'][3]
            img_info = id_to_img_info.get(img_id)
            if img_info:
                norm_w = w / img_info['width']
                norm_h = h / img_info['height']
                wh.append([norm_w, norm_h])
    
    wh = np.array(wh)
    areas = wh[:, 0] * wh[:, 1]
    
    print("\n" + "="*50)
    print("ANÁLISIS DE DISTRIBUCIÓN DE CAJAS")
    print("="*50)
    print(f"Total de cajas: {len(wh)}")
    print(f"\nAncho (normalizado):")
    print(f"  Min: {wh[:, 0].min():.4f}")
    print(f"  Max: {wh[:, 0].max():.4f}")
    print(f"  Media: {wh[:, 0].mean():.4f}")
    print(f"  Std: {wh[:, 0].std():.4f}")
    
    print(f"\nAlto (normalizado):")
    print(f"  Min: {wh[:, 1].min():.4f}")
    print(f"  Max: {wh[:, 1].max():.4f}")
    print(f"  Media: {wh[:, 1].mean():.4f}")
    print(f"  Std: {wh[:, 1].std():.4f}")
    
    print(f"\nÁrea (normalizada):")
    print(f"  Min: {areas.min():.6f}")
    print(f"  Max: {areas.max():.4f}")
    print(f"  Media: {areas.mean():.4f}")
    
    # Clasificar por tamaño
    small = (areas < 0.01).sum()
    medium = ((areas >= 0.01) & (areas < 0.05)).sum()
    large = (areas >= 0.05).sum()
    
    print(f"\nDistribución por tamaño:")
    print(f"  Pequeño (área < 0.01): {small} ({100*small/len(areas):.1f}%)")
    print(f"  Mediano (0.01 <= área < 0.05): {medium} ({100*medium/len(areas):.1f}%)")
    print(f"  Grande (área >= 0.05): {large} ({100*large/len(areas):.1f}%)")
    
    return wh, areas
