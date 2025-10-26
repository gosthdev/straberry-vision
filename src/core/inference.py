"""
Funciones de inferencia y testing del modelo
"""
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

from .config import Config
from .dataset import get_transforms


def test_model_on_image(model_path, image_path, conf_threshold=0.2, save_output=True):
    """
    Prueba el modelo en una sola imagen
    
    Args:
        model_path: Ruta al checkpoint del modelo
        image_path: Ruta a la imagen a testear
        conf_threshold: Umbral de confianza para detecciones
        save_output: Si guardar la imagen con detecciones
        
    Returns:
        dict: Resultados de las detecciones
    """
    from .architecture import SGSNet
    
    print("="*60)
    print("TESTING MODELO EN IMAGEN")
    print("="*60)

    # 1. Cargar modelo
    print(f"\n[1/5] Cargando modelo desde: {model_path}")
    model = SGSNet(Config.NUM_CLASSES).to(Config.DEVICE)
    checkpoint = torch.load(model_path, map_location=Config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("✓ Modelo cargado exitosamente")

    # 2. Cargar imagen original
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

    # Convertir BGR a RGB
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # 3. Preprocesamiento
    print(f"\n[3/5] Preprocesando imagen...")
    print(f"  → Redimensionar: {orig_w}x{orig_h} → {Config.IMAGE_SIZE}x{Config.IMAGE_SIZE}")
    print(f"  → Normalizar: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]")

    _, val_transform = get_transforms()
    transformed = val_transform(image=image_rgb, bboxes=[], class_labels=[])
    image_tensor = transformed['image'].unsqueeze(0).to(Config.DEVICE)
    print(f"✓ Tensor shape: {image_tensor.shape}")

    # 4. Inferencia
    print(f"\n[4/5] Ejecutando inferencia...")
    with torch.no_grad():
        predictions = model(image_tensor)

    # Procesar predicciones
    detections = extract_detections(predictions, conf_threshold)
    print(f"✓ Detecciones encontradas: {len(detections)}")

    # Mostrar estadísticas
    B, C, H, W = predictions.shape
    pred_reshaped = predictions.view(B, 3, 5 + Config.NUM_CLASSES, H, W)
    pred_reshaped = pred_reshaped.permute(0, 1, 3, 4, 2).contiguous()
    obj_scores = torch.sigmoid(pred_reshaped[0, :, :, :, 0])
    
    print(f"\nMÉTRICAS:")
    print(f"  Confianza máxima: {obj_scores.max().item():.4f}")
    print(f"  Confianza promedio: {obj_scores.mean().item():.4f}")
    print(f"  Predicciones > {conf_threshold}: {(obj_scores > conf_threshold).sum().item()}")

    # 5. Visualización
    print(f"\n[5/5] Generando visualización...")
    vis_image = visualize_detections(image_rgb, detections)
    
    # Mostrar resultado
    show_results(original_image, vis_image, detections, conf_threshold)

    # Resumen por clase
    print_detection_summary(detections)

    return {
        'detections': detections,
        'num_detections': len(detections),
        'max_confidence': obj_scores.max().item(),
        'mean_confidence': obj_scores.mean().item(),
        'image_shape': (orig_h, orig_w),
        'visualization': vis_image
    }


def extract_detections(predictions, conf_threshold):
    """
    Extrae detecciones de las predicciones del modelo
    
    Args:
        predictions: Tensor de predicciones
        conf_threshold: Umbral de confianza
        
    Returns:
        list: Lista de detecciones
    """
    B, C, H, W = predictions.shape
    num_anchors = 3
    pred_reshaped = predictions.view(B, num_anchors, 5 + Config.NUM_CLASSES, H, W)
    pred_reshaped = pred_reshaped.permute(0, 1, 3, 4, 2).contiguous()

    detections = []
    
    for anchor_idx in range(num_anchors):
        for gy in range(H):
            for gx in range(W):
                obj_conf = torch.sigmoid(pred_reshaped[0, anchor_idx, gy, gx, 0]).item()
                
                if obj_conf > conf_threshold:
                    # Decodificar bbox
                    dx = pred_reshaped[0, anchor_idx, gy, gx, 1].item()
                    dy = pred_reshaped[0, anchor_idx, gy, gx, 2].item()
                    dw = pred_reshaped[0, anchor_idx, gy, gx, 3].item()
                    dh = pred_reshaped[0, anchor_idx, gy, gx, 4].item()

                    cx = (gx + dx) / W
                    cy = (gy + dy) / H
                    w = dw
                    h = dh

                    # Obtener clase
                    class_scores = torch.sigmoid(pred_reshaped[0, anchor_idx, gy, gx, 5:])
                    class_conf, class_idx = torch.max(class_scores, dim=0)

                    detections.append({
                        'bbox': [cx, cy, w, h],
                        'obj_conf': obj_conf,
                        'class_idx': class_idx.item(),
                        'class_conf': class_conf.item(),
                        'class_name': Config.CLASS_NAMES[class_idx.item()]
                    })
    
    return detections


def visualize_detections(image, detections):
    """
    Dibuja las detecciones sobre la imagen
    
    Args:
        image: Imagen RGB
        detections: Lista de detecciones
        
    Returns:
        np.array: Imagen con detecciones dibujadas
    """
    vis_image = image.copy()
    colors = [
        (255, 0, 0),      # flowering - Rojo
        (0, 255, 0),      # growing_g - Verde
        (0, 255, 255),    # growing_w - Cyan
        (255, 255, 0),    # nearly_m - Amarillo
        (255, 0, 255)     # mature - Magenta
    ]

    for det in detections:
        cx, cy, w, h = det['bbox']
        
        # Convertir a pixeles
        x1 = int((cx - w/2) * Config.IMAGE_SIZE)
        y1 = int((cy - h/2) * Config.IMAGE_SIZE)
        x2 = int((cx + w/2) * Config.IMAGE_SIZE)
        y2 = int((cy + h/2) * Config.IMAGE_SIZE)

        color = colors[det['class_idx']]
        cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

        label = f"{det['class_name']}: {det['obj_conf']:.2f}"
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(vis_image, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
        cv2.putText(vis_image, label, (x1, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return vis_image


def show_results(original_image, vis_image, detections, conf_threshold):
    """
    Muestra los resultados en una figura
    
    Args:
        original_image: Imagen original (BGR)
        vis_image: Imagen con visualizaciones (RGB)
        detections: Lista de detecciones
        conf_threshold: Umbral usado
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Imagen original
    axes[0].imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Imagen Original\n{original_image.shape[1]}x{original_image.shape[0]} px', fontsize=12)
    axes[0].axis('off')

    # Imagen con detecciones
    axes[1].imshow(vis_image)
    axes[1].set_title(f'Detecciones: {len(detections)}\nConf > {conf_threshold}', fontsize=12)
    axes[1].axis('off')

    # Leyenda
    colors = [(255, 0, 0), (0, 255, 0), (0, 255, 255), (255, 255, 0), (255, 0, 255)]
    legend_text = "CLASES:\n"
    for name in Config.CLASS_NAMES:
        legend_text += f"  {name}\n"

    plt.figtext(0.02, 0.02, legend_text, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.show()


def print_detection_summary(detections):
    """
    Imprime resumen de detecciones por clase
    
    Args:
        detections: Lista de detecciones
    """
    print("\n" + "="*60)
    print("DETECCIONES POR CLASE:")
    print("="*60)
    for class_name in Config.CLASS_NAMES:
        count = sum(1 for d in detections if d['class_name'] == class_name)
        if count > 0:
            avg_conf = np.mean([d['obj_conf'] for d in detections if d['class_name'] == class_name])
            print(f"  {class_name:<15}: {count:>3} detecciones (conf promedio: {avg_conf:.3f})")


def test_model_on_folder(model_path, folder_path, conf_threshold=0.5, max_images=10):
    """
    Testea el modelo en múltiples imágenes de una carpeta
    
    Args:
        model_path: Ruta al modelo
        folder_path: Ruta a la carpeta con imágenes
        conf_threshold: Umbral de confianza
        max_images: Número máximo de imágenes a procesar
        
    Returns:
        list: Lista de resultados por imagen
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
        return []

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
