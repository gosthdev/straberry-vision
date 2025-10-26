"""
SGSNet - Strawberry Grading System Network
Módulo para detección de estados de madurez de fresas

Exporta las clases y funciones principales para facilitar su uso
"""

# Configuración
from .config import Config

# Arquitectura
from .architecture import SGSNet

# Dataset
from .dataset import (
    StrawberryDataset,
    get_transforms,
    load_data,
    create_dataloaders
)

# Pérdida
from .loss import SGSNetLoss, calculate_iou

# Métricas
from .metrics import Metrics, non_max_suppression

# Entrenamiento
from .trainer import (
    train_epoch,
    validate_epoch,
    train_model
)

# Utilidades
from .extras import (
    EarlyStopping,
    save_checkpoint,
    load_checkpoint,
    find_latest_checkpoint,
    visualize_predictions,
    plot_training_curves,
    save_training_history
)

# Inferencia
from .inference import (
    test_model_on_image,
    test_model_on_folder
)

# Función principal
from .model import train_sgsnet, main

__all__ = [
    # Configuración
    'Config',
    
    # Arquitectura
    'SGSNet',
    
    # Dataset
    'StrawberryDataset',
    'get_transforms',
    'load_data',
    'create_dataloaders',
    
    # Pérdida
    'SGSNetLoss',
    'calculate_iou',
    
    # Métricas
    'Metrics',
    'non_max_suppression',
    
    # Entrenamiento
    'train_epoch',
    'validate_epoch',
    'train_model',
    
    # Utilidades
    'EarlyStopping',
    'save_checkpoint',
    'load_checkpoint',
    'find_latest_checkpoint',
    'visualize_predictions',
    'plot_training_curves',
    'save_training_history',
    
    # Inferencia
    'test_model_on_image',
    'test_model_on_folder',
    
    # Función principal
    'train_sgsnet',
    'main'
]
