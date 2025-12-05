"""
Funciones de inferencia y testing del modelo SGSNet v2 - Multi-escala
Incluye Soft-NMS para mejor manejo de objetos superpuestos
"""
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from urllib.parse import urlparse, unquote
from .config import Config
from .dataset import get_transforms
from .architecture import SGSNet


def soft_nms(detections, sigma=0.5, score_threshold=0.01):
    """
    Soft-NMS: reduce scores de detecciones superpuestas en lugar de eliminarlas
    Mejor para objetos cercanos/superpuestos como múltiples fresas
    
    Args:
        detections: Lista de detecciones con 'bbox', 'obj_conf', etc.
        sigma: Parámetro de decay gaussiano
        score_threshold: Umbral mínimo de score
        
    Returns:
        Lista de detecciones filtradas
    """
    if len(detections) == 0:
        return []
    
    # Convertir a arrays para procesamiento eficiente
    boxes = np.array([d['bbox'] for d in detections])  # [cx, cy, w, h]
    scores = np.array([d['obj_conf'] for d in detections])
    
    # Convertir a formato [x1, y1, x2, y2]
    x1 = boxes[:, 0] - boxes[:, 2] / 2
    y1 = boxes[:, 1] - boxes[:, 3] / 2
    x2 = boxes[:, 0] + boxes[:, 2] / 2
    y2 = boxes[:, 1] + boxes[:, 3] / 2
    
    areas = (x2 - x1) * (y2 - y1)
    
    # Ordenar por score descendente
    order = scores.argsort()[::-1]
    
    keep_indices = []
    
    while len(order) > 0:
        i = order[0]
        keep_indices.append(i)
        
        if len(order) == 1:
            break
        
        # Calcular IoU con resto de cajas
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-16)
        
        # Soft-NMS: decay gaussiano
        weights = np.exp(-(iou ** 2) / sigma)
        scores[order[1:]] *= weights
        
        # Filtrar por score threshold
        remaining = np.where(scores[order[1:]] >= score_threshold)[0]
        order = order[remaining + 1]
    
    # Filtrar detecciones
    return [detections[i] for i in keep_indices if scores[i] >= score_threshold]


def extract_detections_multiscale(predictions_list, conf_threshold, anchors_list=None):
    """
    Extrae detecciones de predicciones multi-escala
    
    Args:
        predictions_list: Lista de predicciones [pred_p3, pred_p4, pred_p5]
        conf_threshold: Umbral de confianza
        anchors_list: Lista de anchors por escala (opcional)
        
    Returns:
        list: Lista de detecciones combinadas de todas las escalas
    """
    all_detections = []
    
    if anchors_list is None:
        anchors_list = Config.ANCHORS
    
    for scale_idx, predictions in enumerate(predictions_list):
        B, C, H, W = predictions.shape
        num_anchors = Config.ANCHORS_PER_SCALE
        anchors = anchors_list[scale_idx]
        
        # Reformatear: [B, A*(5+C), H, W] -> [B, A, H, W, 5+C]
        pred_reshaped = predictions.view(B, num_anchors, 5 + Config.NUM_CLASSES, H, W)
        pred_reshaped = pred_reshaped.permute(0, 1, 3, 4, 2).contiguous()
        
        for anchor_idx in range(num_anchors):
            for gy in range(H):
                for gx in range(W):
                    obj_conf = torch.sigmoid(pred_reshaped[0, anchor_idx, gy, gx, 0]).item()
                    
                    if obj_conf > conf_threshold:
                        # Decodificar bbox
                        dx = torch.sigmoid(pred_reshaped[0, anchor_idx, gy, gx, 1]).item()
                        dy = torch.sigmoid(pred_reshaped[0, anchor_idx, gy, gx, 2]).item()
                        dw = torch.sigmoid(pred_reshaped[0, anchor_idx, gy, gx, 3]).item()
                        dh = torch.sigmoid(pred_reshaped[0, anchor_idx, gy, gx, 4]).item()
                        
                        # Convertir a coordenadas normalizadas [0, 1]
                        cx = (gx + dx) / W
                        cy = (gy + dy) / H
                        w = dw  # Ya está en rango [0, 1] después de sigmoid
                        h = dh
                        
                        # Obtener clase
                        class_scores = torch.sigmoid(pred_reshaped[0, anchor_idx, gy, gx, 5:])
                        class_conf, class_idx = torch.max(class_scores, dim=0)
                        
                        all_detections.append({
                            'bbox': [cx, cy, w, h],
                            'obj_conf': obj_conf,
                            'class_idx': class_idx.item(),
                            'class_conf': class_conf.item(),
                            'class_name': Config.CLASS_NAMES[class_idx.item()],
                            'scale': scale_idx
                        })
    
    # Aplicar Soft-NMS
    all_detections = soft_nms(all_detections, sigma=0.5, score_threshold=conf_threshold * 0.5)
    
    return all_detections


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
    print("="*60)
    print("TESTING MODELO SGSNet v2 EN IMAGEN")
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
    print(f"\n[4/5] Ejecutando inferencia multi-escala...")
    with torch.no_grad():
        predictions_list = model(image_tensor)

    # Procesar predicciones multi-escala
    detections = extract_detections_multiscale(predictions_list, conf_threshold)
    print(f"✓ Detecciones encontradas: {len(detections)}")

    # Mostrar estadísticas por escala
    print(f"\nDETECCIONES POR ESCALA:")
    for scale_idx, scale_name in enumerate(['P3 (64x64)', 'P4 (32x32)', 'P5 (16x16)']):
        scale_dets = [d for d in detections if d.get('scale') == scale_idx]
        print(f"  {scale_name}: {len(scale_dets)} detecciones")

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
        'image_shape': (orig_h, orig_w),
        'visualization': vis_image
    }


def extract_detections(predictions, conf_threshold):
    """
    Extrae detecciones de las predicciones del modelo
    Versión compatible con modelo v1 (single scale) y v2 (multi-scale)
    
    Args:
        predictions: Tensor o lista de predicciones
        conf_threshold: Umbral de confianza
        
    Returns:
        list: Lista de detecciones
    """
    # Si es una lista, usar extracción multi-escala
    if isinstance(predictions, list):
        return extract_detections_multiscale(predictions, conf_threshold)
    
    # Compatibilidad con modelo v1 (single scale)
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

def load_model(model_path, device=None):
    device = device or Config.DEVICE
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = SGSNet(Config.NUM_CLASSES).to(device)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    model.eval()
    return model

def _resolve_image_path(image_path):
    """Convierte rutas tipo file:///... al path del sistema."""
    if isinstance(image_path, str) and image_path.startswith("file:"):
        parsed = urlparse(image_path)
        return unquote(parsed.path)
    return image_path


def preprocess_image(image_path):
    image_path = _resolve_image_path(image_path)
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image_rgb.shape[:2]
    _, val_transform = get_transforms()
    transformed = val_transform(image=image_rgb, bboxes=[], class_labels=[])
    tensor = transformed['image'].unsqueeze(0).to(Config.DEVICE)
    return image_rgb, tensor, (orig_w, orig_h)

def detections_to_pixels(detections, orig_size):
    orig_w, orig_h = orig_size
    scale_x = orig_w / Config.IMAGE_SIZE
    scale_y = orig_h / Config.IMAGE_SIZE
    pixel_dets = []
    for det in detections:
        cx, cy, w, h = det['bbox']
        x1 = max(0.0, (cx - w / 2) * Config.IMAGE_SIZE) * scale_x
        y1 = max(0.0, (cy - h / 2) * Config.IMAGE_SIZE) * scale_y
        x2 = min(1.0, (cx + w / 2) * Config.IMAGE_SIZE / Config.IMAGE_SIZE) * orig_w if Config.IMAGE_SIZE else (cx + w / 2) * Config.IMAGE_SIZE * scale_x
        y2 = min(1.0, (cy + h / 2) * Config.IMAGE_SIZE / Config.IMAGE_SIZE) * orig_h if Config.IMAGE_SIZE else (cy + h / 2) * Config.IMAGE_SIZE * scale_y
        pixel_dets.append({
            "label": det['class_name'],
            "score": float(det['obj_conf']),
            "xmin": float(x1),
            "ymin": float(y1),
            "xmax": float(x2),
            "ymax": float(y2),
        })
    return pixel_dets

def run_inference_on_image(model, image_path, conf_threshold=None):
    conf_threshold = conf_threshold or Config.CONF_THRESHOLD
    image_rgb, tensor, orig_size = preprocess_image(image_path)
    with torch.no_grad():
        predictions = model(tensor)
    detections = extract_detections(predictions, conf_threshold)
    torch.cuda.empty_cache()  # si usas GPU
    del tensor
    del predictions
    return detections_to_pixels(detections, orig_size)

def run_batch_inference(image_paths, model_path, conf_threshold=None):
    conf_threshold = conf_threshold or Config.CONF_THRESHOLD
    model = load_model(model_path)
    results = []
    for path in image_paths:
        preds = run_inference_on_image(model, path, conf_threshold)
        results.append(preds)
    return results