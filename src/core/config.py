"""
Configuración del modelo SGSNet
"""
import torch
import os


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


# Crear directorio de salida
os.makedirs(Config.OUTPUT_PATH, exist_ok=True)
