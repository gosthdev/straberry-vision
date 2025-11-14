"""
Funciones de entrenamiento y validación
"""
import torch
import torch.optim as optim
from tqdm import tqdm

from .config import Config
from .metrics import Metrics


def train_epoch(model, train_loader, optimizer, criterion, scaler):
    """
    Entrena el modelo por una época
    
    Args:
        model: Modelo a entrenar
        train_loader: DataLoader de entrenamiento
        optimizer: Optimizador
        criterion: Función de pérdida
        scaler: GradScaler para mixed precision
        
    Returns:
        dict: Métricas de entrenamiento
    """
    model.train()
    metrics = Metrics()

    progress_bar = tqdm(train_loader, desc="Training", 
                       mininterval=1.0,
                       maxinterval=2.0)
    
    for images, target_boxes, target_labels in progress_bar:
        images = images.to(Config.DEVICE)

        # Forward pass con mixed precision
        with torch.cuda.amp.autocast():
            predictions = model(images)
            loss, loss_dict = criterion(predictions, target_boxes, target_labels)

        # Backward pass
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        scaler.step(optimizer)
        scaler.update()

        # Actualizar métricas
        metrics.update(loss.item(), loss_dict)
        
        # Actualizar barra de progreso
        progress_bar.set_postfix({
            'L': f'{loss.item():.3f}',
            'O': f'{loss_dict["obj"]:.3f}',
            'B': f'{loss_dict["bbox"]:.3f}',
            'C': f'{loss_dict["class"]:.3f}'
        }, refresh=True)

    return metrics.get_metrics()


def validate_epoch(model, val_loader, criterion):
    """
    Valida el modelo en el conjunto de validación
    
    Args:
        model: Modelo a validar
        val_loader: DataLoader de validación
        criterion: Función de pérdida
        
    Returns:
        dict: Métricas de validación
    """
    model.eval()
    metrics = Metrics()

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validating",
                          mininterval=1.0,
                          maxinterval=2.0)
        
        for images, target_boxes, target_labels in progress_bar:
            images = images.to(Config.DEVICE)
            predictions = model(images)
            loss, loss_dict = criterion(predictions, target_boxes, target_labels)

            metrics.update(loss.item(), loss_dict, predictions, target_boxes, target_labels)
            
            progress_bar.set_postfix({
                'L': f'{loss.item():.3f}',
                'O': f'{loss_dict["obj"]:.3f}',
                'B': f'{loss_dict["bbox"]:.3f}',
                'C': f'{loss_dict["class"]:.3f}'
            }, refresh=True)

    return metrics.get_metrics()


def train_model(model, train_loader, val_loader, test_loader, 
                criterion, optimizer, scheduler, scaler, 
                early_stopping, start_epoch=1, history=None, best_val_loss=float('inf')):
    """
    Loop principal de entrenamiento
    
    Args:
        model: Modelo a entrenar
        train_loader: DataLoader de entrenamiento
        val_loader: DataLoader de validación
        test_loader: DataLoader de test
        criterion: Función de pérdida
        optimizer: Optimizador
        scheduler: Learning rate scheduler
        scaler: GradScaler para mixed precision
        early_stopping: Objeto EarlyStopping
        start_epoch: Época inicial (para resumir entrenamiento)
        history: Diccionario con historial previo (opcional)
        best_val_loss: Mejor pérdida de validación hasta ahora
        
    Returns:
        model: Modelo entrenado
        history: Historial de métricas
    """
    import time
    from .extras import save_checkpoint, visualize_predictions
    
    if history is None:
        history = {
            'train_loss': [], 'val_loss': [],
            'train_obj': [], 'val_obj': [],
            'train_bbox': [], 'val_bbox': [],
            'train_class': [], 'val_class': [],
            'train_precision': [], 'val_precision': [],
            'train_recall': [], 'val_recall': [],
            'train_f1': [], 'val_f1': [],
        }
    
    print(f"Entrenando hasta época {Config.EPOCHS}...\n")
    start_time = time.time()

    for epoch in range(start_epoch, Config.EPOCHS + 1):
        print(f"\n{'='*60}")
        print(f"ÉPOCA {epoch}/{Config.EPOCHS}")
        print(f"{'='*60}")

        # Entrenar
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, scaler)
        history['train_loss'].append(train_metrics['loss'])
        history['train_obj'].append(train_metrics['obj_loss'])
        history['train_bbox'].append(train_metrics['bbox_loss'])
        history['train_class'].append(train_metrics['class_loss'])
        history['train_precision'].append(train_metrics['precision'])
        history['train_recall'].append(train_metrics['recall'])
        history['train_f1'].append(train_metrics['f1_score'])

        # Validar
        val_metrics = validate_epoch(model, val_loader, criterion)
        history['val_loss'].append(val_metrics['loss'])
        history['val_obj'].append(val_metrics['obj_loss'])
        history['val_bbox'].append(val_metrics['bbox_loss'])
        history['val_class'].append(val_metrics['class_loss'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1_score'])

        scheduler.step()

        # Visualización
        if epoch % Config.VIS_INTERVAL == 0:
            visualize_predictions(model, val_loader, epoch, Config.OUTPUT_PATH)

        # Mostrar resumen
        print_epoch_summary(train_metrics, val_metrics)

        # Guardar mejor modelo
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
            print(f"\n🎯 NUEVO MEJOR MODELO (val_loss: {best_val_loss:.4f})")
            save_checkpoint(epoch, model, optimizer, scheduler, early_stopping,
                          train_metrics, val_metrics, history, is_best=True)
        
        if epoch % Config.SAVE_EVERY == 0:
            save_checkpoint(epoch, model, optimizer, scheduler, early_stopping,
                          train_metrics, val_metrics, history, is_best=False)

        # Early stopping
        if early_stopping(val_metrics['loss']):
            print(f"\n⚠️  Early stopping activado en época {epoch}")
            save_checkpoint(epoch, model, optimizer, scheduler, early_stopping,
                          train_metrics, val_metrics, history, is_best=False)
            break

    # Evaluación final en test set
    print("\n" + "="*60)
    print("EVALUACIÓN FINAL EN TEST SET")
    print("="*60)
    test_metrics = validate_epoch(model, test_loader, criterion)
    print_test_summary(test_metrics)

    total_time = time.time() - start_time
    print(f"\n✓ Entrenamiento completado en {total_time/60:.1f} minutos")
    print(f"✓ Mejor val_loss: {best_val_loss:.4f}")
    
    return model, history


def print_epoch_summary(train_metrics, val_metrics):
    """Imprime resumen de métricas de la época"""
    print(f"\n{'─'*60}")
    print(f"{'Métrica':<20} {'Train':>12} {'Val':>12} {'Δ':>8}")
    print(f"{'─'*60}")
    print(f"{'Loss Total':<20} {train_metrics['loss']:>12.4f} {val_metrics['loss']:>12.4f} {(val_metrics['loss']-train_metrics['loss']):>8.4f}")
    print(f"{'  - Objectness':<20} {train_metrics['obj_loss']:>12.4f} {val_metrics['obj_loss']:>12.4f}")
    print(f"{'  - BBox':<20} {train_metrics['bbox_loss']:>12.4f} {val_metrics['bbox_loss']:>12.4f}")
    print(f"{'  - Class':<20} {train_metrics['class_loss']:>12.4f} {val_metrics['class_loss']:>12.4f}")
    print(f"{'─'*60}")
    print(f"{'Precision':<20} {train_metrics['precision']:>12.4f} {val_metrics['precision']:>12.4f}")
    print(f"{'Recall':<20} {train_metrics['recall']:>12.4f} {val_metrics['recall']:>12.4f}")
    print(f"{'F1 Score':<20} {train_metrics['f1_score']:>12.4f} {val_metrics['f1_score']:>12.4f}")
    print(f"{'─'*60}")


def print_test_summary(test_metrics):
    """Imprime resumen de métricas de test"""
    print(f"\n{'Métrica':<20} {'Valor':>12}")
    print(f"{'─'*40}")
    print(f"{'Loss Total':<20} {test_metrics['loss']:>12.4f}")
    print(f"{'  - Objectness':<20} {test_metrics['obj_loss']:>12.4f}")
    print(f"{'  - BBox':<20} {test_metrics['bbox_loss']:>12.4f}")
    print(f"{'  - Class':<20} {test_metrics['class_loss']:>12.4f}")
    print(f"{'─'*40}")
    print(f"{'Precision':<20} {test_metrics['precision']:>12.4f}")
    print(f"{'Recall':<20} {test_metrics['recall']:>12.4f}")
    print(f"{'F1 Score':<20} {test_metrics['f1_score']:>12.4f}")
    print(f"{'─'*40}")
