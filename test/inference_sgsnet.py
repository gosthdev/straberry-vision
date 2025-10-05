# inference_sgsnet.py (Actualizado para el formato YOLO/SSD de SGSNet)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import glob
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
# --- IMPORTAR AQUÍ LA FUNCIÓN DE NMS SI ES NECESARIA ---
# from torchvision.ops import nms 

# ==============================================================================
# 1. ARQUITECTURA (Incluye Configuración y Clases)
# (Mantenido igual que en la respuesta anterior para autocontención)
# ==============================================================================

# ... (Clases Config, ConvBNAct, UniversalInvertedBottleneck, 
#      GrowthNetBackbone, SimpleUpsampling, FeatureFusionBlock, 
#      SGSNeck, DetectionHead, SGSNet de la respuesta anterior) ...

class Config:
    # Rutas (ADJUSTAR SEGÚN DÓNDE ESTÉ EL MODELO .pth)
    MODEL_PATH = '/content/drive/MyDrive/percepcion/models/strawberry_detector_best.pth' 
    
    # Clases (DEBE COINCIDIR CON EL ENTRENAMIENTO)
    CLASS_NAMES = ['flowering', 'growing_g', 'growing_w', 'nearly_m', 'mature'] 
    NUM_CLASSES = len(CLASS_NAMES)

    # Hiperparámetros (DEBE COINCIDIR CON EL ENTRENAMIENTO)
    IMAGE_SIZE = 416 
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Parámetros de Inferencia
    CONF_THRESHOLD = 0.5  # Umbral de confianza mínimo para mostrar una detección
    IOU_THRESHOLD = 0.45  # Umbral IoU para Non-Max Suppression (NMS)

# ... (Definiciones de las clases de la arquitectura van aquí) ...


# ==============================================================================
# 2. FUNCIONES DE INFERENCIA Y POST-PROCESAMIENTO
# ==============================================================================

def preprocess_image(image_path: str, size: int):
    """Carga, redimensiona y normaliza una imagen para la inferencia."""
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"No se pudo cargar la imagen: {image_path}")
        
    # Importante: Usar las mismas transformaciones de normalización y redimensionamiento
    transform = A.Compose([
        A.Resize(size, size, interpolation=cv2.INTER_LINEAR),
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1), max_pixel_value=255.0),
        ToTensorV2(),
    ])
    
    transformed = transform(image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) # Convertir a RGB antes de la normalización
    image_tensor = transformed['image'].unsqueeze(0) # Añadir dimensión de batch
    
    return image_tensor, image # Devolvemos la imagen original (en BGR)


def post_process_output(raw_output_list, conf_thresh, iou_thresh, num_classes, image_width, image_height):
    """
    *** PUNTO CLAVE: IMPLEMENTAR LA LÓGICA DE DECODIFICACIÓN ESPECÍFICA DE SGSNET ***
    Decodifica las salidas codificadas del modelo (raw_output_list) a cajas y clases finales.
    """
    
    all_detections = []
    
    # --- IMPLEMENTACIÓN NECESARIA ---
    # 1. Iterar sobre las 3 salidas (P3, P4, P5)
    # 2. Por cada salida, decodificar las coordenadas y la confianza usando la lógica 
    #    inversa a tu función de pérdida (uso de anclas y celdas de cuadrícula).
    # 3. Aplicar el umbral de confianza (conf > conf_thresh).
    # 4. Almacenar todas las detecciones válidas en una lista unificada.
    # 5. Aplicar NMS (Supresión No Máxima) sobre la lista unificada.
    
    print("\n--- ¡ATENCIÓN! Se necesita implementación de la Lógica de Decodificación y NMS ---")
    print("La función a continuación es un SIMULACRO para demostrar el flujo de inferencia.")
    
    # --- SIMULACIÓN (Reemplazar con su lógica real) ---
    # La salida final debe ser: [x1_norm, y1_norm, x2_norm, y2_norm, confidence, class_id]
    if image_width > 0:
         simulated_detections = [
            # Detección de Fresa Maduro (simulada)
            [150/image_width, 150/image_height, 300/image_width, 300/image_height, 0.95, 4], 
            # Detección de Fresa en Floración (simulada)
            [400/image_width, 100/image_height, 550/image_width, 250/image_height, 0.80, 0] 
         ]
         
         for det in simulated_detections:
             if det[4] >= conf_thresh:
                all_detections.append(det)

    return all_detections


def draw_detections(image_original, detections, class_names):
    """Dibuja las cajas delimitadoras y etiquetas en la imagen original (en BGR)."""
    img_h, img_w = image_original.shape[:2]
    
    for det in detections:
        # Decodificación de coordenadas (asumiendo que están normalizadas 0-1)
        x1 = int(det[0] * img_w)
        y1 = int(det[1] * img_h)
        x2 = int(det[2] * img_w)
        y2 = int(det[3] * img_h)
        conf = det[4]
        class_id = int(det[5])
        class_name = class_names[class_id]
        
        label = f"{class_name}: {conf:.2f}"
        
        # Dibujar la caja delimitadora (Bounding Box) en BGR
        color = (0, 255, 0) # Verde
        cv2.rectangle(image_original, (x1, y1), (x2, y2), color, 2)
        
        # Dibujar el fondo del texto (mismo color)
        # Se calcula el tamaño del texto para el rectángulo de fondo
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(image_original, (x1, y1 - h - 5), (x1 + w, y1), color, -1)
        
        # Escribir la etiqueta (texto negro)
        cv2.putText(image_original, label, (x1, y1 - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
    return image_original


def run_inference(image_path: str, model: SGSNet, config: Config):
    """Función principal para realizar la inferencia y visualizar el resultado."""
    
    print(f"\n--- Probando imagen: {Path(image_path).name} ---")
    
    try:
        # 1. Preprocesar la imagen
        input_tensor, original_image = preprocess_image(image_path, config.IMAGE_SIZE)
        input_tensor = input_tensor.to(config.DEVICE)
        
        img_h_orig, img_w_orig = original_image.shape[:2]

        # 2. Inferir
        start_time = time.time()
        with torch.no_grad(): # Desactivar el cálculo de gradientes
            raw_outputs = model(input_tensor)
        inference_time = time.time() - start_time
        
        # El tiempo de inferencia es una métrica clave para sistemas ligeros
        print(f"✅ Tiempo de Inferencia (Forward Pass): {inference_time*1000:.2f} ms")

        # 3. Post-procesamiento (Decodificación de coordenadas y NMS)
        detections = post_process_output(
            raw_outputs, 
            config.CONF_THRESHOLD, 
            config.IOU_THRESHOLD, 
            config.NUM_CLASSES,
            img_w_orig, 
            img_h_orig
        )

        # 4. Dibujar el resultado
        image_with_detections = draw_detections(original_image, detections, config.CLASS_NAMES)

        # 5. Mostrar/Guardar resultado
        output_dir = 'test_outputs'
        os.makedirs(output_dir, exist_ok=True)
        output_name = os.path.join(output_dir, f"output_{Path(image_path).name}")
        cv2.imwrite(output_name, image_with_detections)
        print(f"✅ Imagen de salida guardada como: {output_name}")
        print(f"Total Detecciones (post-NMS): {len(detections)}")
        
        return image_with_detections, detections

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return None, None
    except Exception as e:
        print(f"❌ Ocurrió un error durante la inferencia: {e}")
        return None, None

# ==============================================================================
# 3. INICIALIZACIÓN Y PRUEBA
# ==============================================================================

if __name__ == '__main__':
    
    # 1. Inicializar Modelo
    # ... (Lógica de carga del modelo: inicialización, .load_state_dict, .eval()) ...
    
    # 2. Definir imágenes para la prueba (Asegúrese de tener estas imágenes)
    TEST_IMAGES = ["fresa_test_1.jpg", "fresa_test_2.png"]
    
    # Lógica de generación de imágenes dummy para la prueba de concepto
    # ... (Mantenida igual para crear archivos si no existen) ...
    
    # 3. Ejecutar inferencia
    for img_path in TEST_IMAGES:
        # Aquí se llama a la función principal de inferencia
        run_inference(img_path, model, Config)