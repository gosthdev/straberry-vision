"""
Configuración del modelo SGSNet v2 - Multi-escala
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
    IMAGE_SIZE = 512
    BATCH_SIZE = 16  # Reducido para GPU de 4GB (GTX 1650)
    EPOCHS = 100  # Más épocas para mejor convergencia
    LEARNING_RATE = 0.002
    WEIGHT_DECAY = 0.0005
    EARLY_STOP_PATIENCE = 25
    SAVE_EVERY = 5

    # Multi-escala: 3 escalas de detección
    # Strides para cada nivel de la pirámide
    STRIDES = [8, 16, 32]  # P3, P4, P5
    
    # Grid sizes para input 512x512
    GRID_SIZES = [64, 32, 16]  # 512/8, 512/16, 512/32
    
    # Anchors por escala (w, h en proporción de imagen)
    # P3 (64x64): objetos pequeños - anchors pequeños
    # P4 (32x32): objetos medianos - anchors medianos  
    # P5 (16x16): objetos grandes - anchors grandes
    # Aspect ratios: ~0.8 (alto), 1.0 (cuadrado), ~1.2 (ancho)
    ANCHORS = [
        # Escala P3 (pequeño) - stride 8
        torch.tensor([
            [0.02, 0.025],   # Pequeño alto
            [0.03, 0.03],    # Pequeño cuadrado
            [0.04, 0.032],   # Pequeño ancho
        ]),
        # Escala P4 (mediano) - stride 16
        torch.tensor([
            [0.06, 0.075],   # Mediano alto
            [0.08, 0.08],    # Mediano cuadrado
            [0.10, 0.08],    # Mediano ancho
        ]),
        # Escala P5 (grande) - stride 32
        torch.tensor([
            [0.12, 0.15],    # Grande alto
            [0.15, 0.15],    # Grande cuadrado
            [0.20, 0.16],    # Grande ancho
        ]),
    ]
    
    # Número de anchors por escala
    ANCHORS_PER_SCALE = 3
    NUM_SCALES = 3

    # Thresholds
    CONF_THRESHOLD = 0.3  # Reducido para capturar más detecciones
    IOU_THRESHOLD = 0.45
    NMS_IOU_THRESHOLD = 0.5  # Para Soft-NMS

    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # DEBUG: Visualización durante entrenamiento
    VISUALIZE_TRAINING = True  # CAMBIAR A False PARA DESACTIVAR
    VIS_INTERVAL = 5  # Visualizar cada N épocas
    
    # Data Augmentation avanzada
    USE_MOSAIC = True  # Habilitar Mosaic augmentation
    MOSAIC_PROB = 0.5  # Probabilidad de aplicar mosaic
    USE_MIXUP = True   # Habilitar MixUp
    MIXUP_PROB = 0.3   # Probabilidad de aplicar mixup
    
    # Negative samples (fondos sin fresas)
    USE_NEGATIVE_SAMPLES = True
    NEGATIVE_SAMPLE_RATIO = 0.1  # 10% del dataset como backgrounds


# Crear directorio de salida
os.makedirs(Config.OUTPUT_PATH, exist_ok=True)
