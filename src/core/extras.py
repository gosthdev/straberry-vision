"""
Utilidades: checkpoints, early stopping, visualización
"""
import torch
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
import random
import json

from .config import Config
from .metrics import non_max_suppression


class EarlyStopping:
    """
    Clase para implementar early stopping durante el entrenamiento
    """
    def __init__(self, patience=10, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss):
        """
        Verifica si se debe detener el entrenamiento
        
        Args:
            val_loss: Pérdida de validación actual
            
        Returns:
            bool: True si se debe detener, False en caso contrario
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
            return self.should_stop


def save_checkpoint(epoch, model, optimizer, scheduler, early_stopping,
                   train_metrics, val_metrics, history, is_best=False):
    """
    Guarda un checkpoint del modelo
    
    Args:
        epoch: Época actual
        model: Modelo
        optimizer: Optimizador
        scheduler: Scheduler
        early_stopping: Objeto EarlyStopping
        train_metrics: Métricas de entrenamiento
        val_metrics: Métricas de validación
        history: Historial completo
        is_best: Si es el mejor modelo hasta ahora
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'early_stopping_counter': early_stopping.counter,
        'early_stopping_best_loss': early_stopping.best_loss,
        'train_loss': train_metrics['loss'],
        'val_loss': val_metrics['loss'],
        'history': history,
        'config': {
            'learning_rate': Config.LEARNING_RATE,
            'batch_size': Config.BATCH_SIZE,
            'num_classes': Config.NUM_CLASSES
        }
    }

    if is_best:
        best_path = Config.OUTPUT_PATH + '/best_model.pth'
        torch.save(checkpoint, best_path)
        print(f" Mejor modelo guardado: {best_path}")
    else:
        checkpoint_path = Config.OUTPUT_PATH + f'/checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        print(f" Checkpoint guardado: {checkpoint_path}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """
    Carga un checkpoint del modelo
    
    Args:
        checkpoint_path: Ruta al checkpoint
        model: Modelo donde cargar los pesos
        optimizer: Optimizador (opcional)
        scheduler: Scheduler (opcional)
        
    Returns:
        start_epoch: Época desde donde continuar
        history: Historial de entrenamiento
        early_stopping_state: Estado del early stopping
    """
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint no encontrado: {checkpoint_path}")
        return 0, None, None

    print(f"Cargando checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=Config.DEVICE, weights_only=False)

    model.load_state_dict(checkpoint['model_state_dict'])
    print(f" Modelo restaurado desde época {checkpoint['epoch']}")

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    early_stopping_state = {
        'counter': checkpoint.get('early_stopping_counter', 0),
        'best_loss': checkpoint.get('early_stopping_best_loss', float('inf'))
    }

    history = checkpoint.get('history', None)
    start_epoch = checkpoint['epoch'] + 1

    return start_epoch, history, early_stopping_state


def find_latest_checkpoint(checkpoint_dir):
    """
    Encuentra el checkpoint más reciente en un directorio
    
    Args:
        checkpoint_dir: Directorio donde buscar
        
    Returns:
        str: Ruta al checkpoint más reciente o None
    """
    pattern = os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pth')
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    
    def get_epoch_num(path):
        basename = os.path.basename(path)
        try:
            return int(basename.split('_')[-1].replace('.pth', ''))
        except:
            return 0
    
    return max(checkpoints, key=get_epoch_num)


def visualize_predictions(model, val_loader, epoch, save_path):
    """
    Visualiza predicciones del modelo durante el entrenamiento
    
    Args:
        model: Modelo
        val_loader: DataLoader de validación
        epoch: Época actual
        save_path: Directorio donde guardar
    """
    if not Config.VISUALIZE_TRAINING:
        return
    
    model.eval()
    with torch.no_grad():
        try:
            total_batches = len(val_loader)
        except TypeError:
            total_batches = None

        if total_batches:
            random_batch_idx = random.randrange(total_batches)
            selected_batch = None
            for idx, batch in enumerate(val_loader):
                if idx == random_batch_idx:
                    selected_batch = batch
                    break
            if selected_batch is None:
                selected_batch = next(iter(val_loader))
        else:
            selected_batch = next(iter(val_loader))

        images, target_boxes, target_labels = selected_batch
        images = images.to(Config.DEVICE)
        predictions = model(images)
        
        # Aplicar NMS
        batch_detections = non_max_suppression(
            predictions,
            conf_threshold=0.3,
            iou_threshold=0.4
        )
        
        num_display = min(4, images.shape[0])
        fig, axes = plt.subplots(2, num_display, figsize=(20, 10))
        if num_display == 1:
            axes = np.array(axes).reshape(2, 1)
        
        colors = ['red', 'blue', 'cyan', 'yellow', 'magenta']
        
        for idx in range(num_display):
            img = images[idx].cpu().permute(1, 2, 0).numpy()
            img = (img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406]))
            img = np.clip(img, 0, 1)
            
            # Ground truth
            ax_gt = axes[0, idx] if num_display > 1 else axes[0]
            ax_gt.imshow(img)
            ax_gt.set_title(f'Epoch {epoch} - GT')
            ax_gt.axis('off')
            
            for gt_box, gt_label in zip(target_boxes[idx], target_labels[idx]):
                cx, cy, w, h = gt_box.cpu().numpy()
                x1 = int((cx - w/2) * Config.IMAGE_SIZE)
                y1 = int((cy - h/2) * Config.IMAGE_SIZE)
                x2 = int((cx + w/2) * Config.IMAGE_SIZE)
                y2 = int((cy + h/2) * Config.IMAGE_SIZE)
                
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='green', linewidth=2)
                ax_gt.add_patch(rect)
                ax_gt.text(x1, y1-5, Config.CLASS_NAMES[gt_label], color='green', fontsize=8, weight='bold')
            
            # Predictions con NMS
            ax_pred = axes[1, idx] if num_display > 1 else axes[1]
            ax_pred.imshow(img)
            
            pred_boxes, pred_labels, pred_scores = batch_detections[idx]
            ax_pred.set_title(f'Epoch {epoch} - Pred (NMS: {len(pred_boxes)} boxes)')
            ax_pred.axis('off')
            
            for pred_box, pred_label, pred_score in zip(pred_boxes, pred_labels, pred_scores):
                cx, cy, w, h = pred_box.cpu().numpy()
                x1 = int((cx - w/2) * Config.IMAGE_SIZE)
                y1 = int((cy - h/2) * Config.IMAGE_SIZE)
                x2 = int((cx + w/2) * Config.IMAGE_SIZE)
                y2 = int((cy + h/2) * Config.IMAGE_SIZE)
                
                color = colors[int(pred_label) % len(colors)]
                rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color=color, linewidth=2)
                ax_pred.add_patch(rect)
                ax_pred.text(x1, y1-5, f'{Config.CLASS_NAMES[int(pred_label)]}:{pred_score:.2f}', 
                           color=color, fontsize=8, weight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{save_path}/predictions_epoch_{epoch}.png', dpi=100)
        plt.close()
        print(f"  → Visualización guardada: predictions_epoch_{epoch}.png")


def plot_training_curves(history):
    """
    Grafica las curvas de entrenamiento
    
    Args:
        history: Diccionario con historial de métricas
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Loss total
    axes[0, 0].plot(history['train_loss'], label='Train', linewidth=2)
    axes[0, 0].plot(history['val_loss'], label='Val', linewidth=2)
    axes[0, 0].set_xlabel('Época')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Total')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Precision
    axes[0, 1].plot(history['train_precision'], label='Train', linewidth=2)
    axes[0, 1].plot(history['val_precision'], label='Val', linewidth=2)
    axes[0, 1].set_xlabel('Época')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Recall
    axes[0, 2].plot(history['train_recall'], label='Train', linewidth=2)
    axes[0, 2].plot(history['val_recall'], label='Val', linewidth=2)
    axes[0, 2].set_xlabel('Época')
    axes[0, 2].set_ylabel('Recall')
    axes[0, 2].set_title('Recall')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # F1 Score
    axes[1, 0].plot(history['train_f1'], label='Train', linewidth=2)
    axes[1, 0].plot(history['val_f1'], label='Val', linewidth=2)
    axes[1, 0].set_xlabel('Época')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # BBox Loss
    axes[1, 1].plot(history['train_bbox'], label='Train', linewidth=2)
    axes[1, 1].plot(history['val_bbox'], label='Val', linewidth=2)
    axes[1, 1].set_xlabel('Época')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Bounding Box Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Class Loss
    axes[1, 2].plot(history['train_class'], label='Train', linewidth=2)
    axes[1, 2].plot(history['val_class'], label='Val', linewidth=2)
    axes[1, 2].set_xlabel('Época')
    axes[1, 2].set_ylabel('Loss')
    axes[1, 2].set_title('Classification Loss')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(Config.OUTPUT_PATH + '/training_curves.png', dpi=150, bbox_inches='tight')
    print(f"\nGráficas guardadas: {Config.OUTPUT_PATH}/training_curves.png")
    plt.show()


def save_training_history(history):
    """
    Guarda el historial de entrenamiento en JSON
    
    Args:
        history: Diccionario con historial
    """
    with open(Config.OUTPUT_PATH + '/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    print(f" Historial guardado: {Config.OUTPUT_PATH}/training_history.json")
