#!/usr/bin/env python3
"""
Script de entrenamiento para SGSNet v2
Ejecutar desde la raíz del proyecto: python train.py
"""
import gc
import torch

# Limpiar memoria CUDA antes de empezar
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()

from src.core.model.model import train_sgsnet

if __name__ == "__main__":
    print("=" * 60)
    print("Iniciando entrenamiento SGSNet")
    print("=" * 60)
    
    # Verificar GPU disponible
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memoria total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"Memoria libre: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB reservada")
    else:
        print("⚠️ CUDA no disponible, entrenando en CPU (muy lento)")
    
    print("=" * 60)
    
    # Entrenar - resume='latest' busca el último checkpoint automáticamente
    train_sgsnet(resume_from_checkpoint='latest', calculate_anchors=True)
