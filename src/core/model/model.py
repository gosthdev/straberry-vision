"""
SGSNet v2 para Detección de Estados de Madurez de Fresas
Versión Multi-escala con FPN

Este archivo sirve como punto de entrada principal para entrenar e inferir el modelo.
La implementación está dividida en módulos especializados:
    - config.py: Configuración del modelo
    - architecture.py: Arquitectura de la red neuronal (MobileNetV3-Large + FPN)
    - dataset.py: Manejo de datasets y transformaciones (Mosaic, MixUp)
    - loss.py: Funciones de pérdida (CIoU)
    - metrics.py: Métricas y evaluación
    - trainer.py: Lógica de entrenamiento
    - extras.py: Utilidades (checkpoints, visualización)
    - inference.py: Inferencia en imágenes (Soft-NMS)
"""
import torch
import torch.optim as optim
import os
import warnings
warnings.filterwarnings('ignore')

# Importar módulos propios
from .config import Config
from .metrics import non_max_suppression
from .architecture import SGSNet
from .dataset import load_data, create_dataloaders
from .loss import SGSNetLoss
from .trainer import train_model
from .extras import (
    EarlyStopping, 
    load_checkpoint, 
    find_latest_checkpoint,
    plot_training_curves,
    save_training_history
)
from .inference import test_model_on_image, test_model_on_folder


# ============================================================================
# FUNCIÓN PRINCIPAL DE ENTRENAMIENTO
# ============================================================================
def train_sgsnet(resume_from_checkpoint=None, calculate_anchors=True):
    """
    Función principal para entrenar el modelo SGSNet v2 Multi-escala
    
    Args:
        resume_from_checkpoint: 'latest' para continuar desde último checkpoint,
                                ruta específica para cargar checkpoint específico,
                                None para empezar desde cero
        calculate_anchors: Si calcular anchors óptimos del dataset (recomendado primera vez)
                                
    Returns:
        model: Modelo entrenado
    """
    # Import dentro de la función para evitar circular import
    from src.utils.anchors import calculate_optimal_anchors_multiscale, analyze_box_distribution
    print("="*60)
    print("STRAWBERRY DETECTION - SGSNET v2 MULTI-ESCALA")
    print("="*60)
    print(f"Clases: {Config.CLASS_NAMES}")
    print(f"Device: {Config.DEVICE}")
    print(f"Escalas de detección: {Config.NUM_SCALES}")
    print(f"Anchors por escala: {Config.ANCHORS_PER_SCALE}")
    print(f"Total anchors: {Config.NUM_SCALES * Config.ANCHORS_PER_SCALE}\n")

    # Cargar datos
    coco_data, train_ids, val_ids, test_ids = load_data()
    
    # Analizar distribución de cajas
    print("\nAnalizando distribución de tamaños de cajas...")
    analyze_box_distribution(coco_data, train_ids)
    
    # Calcular anchors dinámicos multi-escala basados en el dataset de entrenamiento
    if calculate_anchors:
        Config.ANCHORS = calculate_optimal_anchors_multiscale(
            coco_data, train_ids, 
            anchors_per_scale=Config.ANCHORS_PER_SCALE,
            num_scales=Config.NUM_SCALES
        )
    else:
        print("\nUsando anchors predefinidos en Config")
    
    train_loader, val_loader, test_loader = create_dataloaders(
        coco_data, train_ids, val_ids, test_ids
    )

    # Crear modelo
    model = SGSNet(Config.NUM_CLASSES, anchors_per_scale=Config.ANCHORS_PER_SCALE).to(Config.DEVICE)
    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
    print(f"\n✓ Modelo SGSNet v2 creado:")
    print(f"  - Parámetros totales: {total_params:.2f}M")
    print(f"  - Parámetros entrenables: {trainable_params:.2f}M")
    print(f"  - Backbone: MobileNetV3-Large")
    print(f"  - FPN: 3 escalas (P3, P4, P5)")
    print(f"  - Attention: CBAM\n")

    # Optimizador, pérdida y scheduler
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=Config.LEARNING_RATE, 
        weight_decay=Config.WEIGHT_DECAY
    )
    
    # Pérdida multi-escala con anchors por escala
    criterion = SGSNetLoss(Config.NUM_CLASSES, Config.ANCHORS)
    
    # Scheduler con warmup
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    scaler = torch.cuda.amp.GradScaler()
    early_stopping = EarlyStopping(patience=Config.EARLY_STOP_PATIENCE)

    # Variables de tracking
    start_epoch = 1
    history = None
    best_val_loss = float('inf')

    # Cargar checkpoint si existe
    if resume_from_checkpoint:
        if resume_from_checkpoint == 'latest':
            checkpoint_path = find_latest_checkpoint(Config.OUTPUT_PATH)
        else:
            checkpoint_path = resume_from_checkpoint

        if checkpoint_path and os.path.exists(checkpoint_path):
            start_epoch, loaded_history, es_state = load_checkpoint(
                checkpoint_path, model, optimizer, scheduler
            )
            if loaded_history:
                history = loaded_history
                best_val_loss = min(history['val_loss']) if history['val_loss'] else float('inf')
            if es_state:
                early_stopping.counter = es_state['counter']
                early_stopping.best_loss = es_state['best_loss']
            print(f"✓ Reanudando desde época {start_epoch}\n")

    # Entrenar modelo
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        early_stopping=early_stopping,
        start_epoch=start_epoch,
        history=history,
        best_val_loss=best_val_loss
    )

    # Guardar historial y gráficas
    save_training_history(history)
    plot_training_curves(history)

    print("\n" + "="*60)
    print("✓ ENTRENAMIENTO COMPLETADO")
    print(f"✓ Modelos guardados en: {Config.OUTPUT_PATH}")
    print("="*60)
    
    return model


# ============================================================================
# MAIN
# ============================================================================
def main(resume=True, calculate_anchors=True):
    """
    Función principal del script
    
    Args:
        resume: Si True, intenta resumir desde el último checkpoint
        calculate_anchors: Si calcular anchors óptimos del dataset
    """
    print("\nSTRAWBERRY MATURITY DETECTION - SGSNET v2")
    print("="*60)

    # Verificar rutas
    if not os.path.exists(Config.IMAGES_PATH):
        print(f"Imágenes no encontradas: {Config.IMAGES_PATH}")
        return

    if not os.path.exists(Config.ANNOTATIONS_PATH):
        print(f"Anotaciones no encontradas: {Config.ANNOTATIONS_PATH}")
        return

    print(f"✓ Configuración verificada")
    print(f"  Imágenes: {Config.IMAGES_PATH}")
    print(f"  Anotaciones: {Config.ANNOTATIONS_PATH}")
    print(f"  Salida: {Config.OUTPUT_PATH}")
    print(f"  Visualización: {'ACTIVADA' if Config.VISUALIZE_TRAINING else 'DESACTIVADA'}")
    print(f"  Mosaic Aug: {'ACTIVADA' if Config.USE_MOSAIC else 'DESACTIVADA'}")
    print(f"  MixUp Aug: {'ACTIVADA' if Config.USE_MIXUP else 'DESACTIVADA'}\n")

    # Entrenar
    checkpoint_to_resume = 'latest' if resume else None
    model = train_sgsnet(
        resume_from_checkpoint=checkpoint_to_resume,
        calculate_anchors=calculate_anchors
    )

    return model


if __name__ == "__main__":
    # Entrenar modelo (resumiendo desde último checkpoint si existe)
    # Para entrenamiento desde cero sin calcular anchors: main(resume=False, calculate_anchors=False)
    main(resume=False, calculate_anchors=True)
    
    # EJEMPLO: Test en imágenes después del entrenamiento
    # Descomentar las siguientes líneas para probar el modelo entrenado
    """
    result = test_model_on_folder(
        model_path='src/data/processed/models/best_model.pth',
        folder_path='test/files/',
        conf_threshold=0.3,
        max_images=10
    )
    """
