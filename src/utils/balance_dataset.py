# src/utils/balance_dataset.py
"""
Script para balancear el dataset eliminando imágenes con alta concentración 
de una clase específica (por defecto: growing_g).
"""

import json
import random
from pathlib import Path
from collections import defaultdict

# Rutas
ANNOTATIONS_IN = Path("src/data/processed/coco_out/annotations.json")
ANNOTATIONS_OUT = Path("src/data/processed/coco_out/annotations_balanced.json")

# Configuración
TARGET_CLASS_ID = 2  # growing_g
TARGET_COUNT = 3000  # Objetivo de anotaciones para esa clase
SEED = 42


def balance_by_image_sampling(
    input_json: Path,
    output_json: Path,
    target_class_id: int = TARGET_CLASS_ID,
    target_count: int = TARGET_COUNT,
    seed: int = SEED
):
    """
    Reduce una clase eliminando imágenes con alta concentración de esa clase.
    Preserva la coherencia: si se elimina una imagen, se eliminan TODAS sus anotaciones.
    """
    random.seed(seed)
    
    print(f"Cargando {input_json}...")
    data = json.loads(input_json.read_text(encoding="utf-8"))
    
    # Mapeo de categorías
    cat_map = {c["id"]: c["name"] for c in data["categories"]}
    target_class_name = cat_map.get(target_class_id, f"class_{target_class_id}")
    
    # Contar anotaciones de la clase objetivo por imagen
    img_target_count = defaultdict(int)
    for ann in data["annotations"]:
        if ann["category_id"] == target_class_id:
            img_target_count[ann["image_id"]] += 1
    
    # Contar total actual
    current_count = sum(img_target_count.values())
    print(f"\nClase objetivo: {target_class_name} (id={target_class_id})")
    print(f"Anotaciones actuales: {current_count}")
    print(f"Objetivo: {target_count}")
    
    if current_count <= target_count:
        print("Ya está en o por debajo del objetivo. No se necesitan cambios.")
        return
    
    to_remove = current_count - target_count
    print(f"Anotaciones a eliminar: ~{to_remove}")
    
    # Ordenar imágenes por cantidad de anotaciones de la clase objetivo (descendente)
    sorted_imgs = sorted(img_target_count.items(), key=lambda x: -x[1])
    
    # Eliminar imágenes con más anotaciones de la clase objetivo hasta alcanzar el objetivo
    removed_img_ids = set()
    removed_count = 0
    
    for img_id, count in sorted_imgs:
        if removed_count >= to_remove:
            break
        removed_img_ids.add(img_id)
        removed_count += count
    
    print(f"\nImágenes a eliminar: {len(removed_img_ids)}")
    
    # Filtrar imágenes y anotaciones
    original_img_count = len(data["images"])
    original_ann_count = len(data["annotations"])
    
    data["images"] = [i for i in data["images"] if i["id"] not in removed_img_ids]
    data["annotations"] = [a for a in data["annotations"] if a["image_id"] not in removed_img_ids]
    
    # Re-indexar anotaciones
    for i, ann in enumerate(data["annotations"], start=1):
        ann["id"] = i
    
    # Guardar
    print(f"\nGuardando en {output_json}...")
    output_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    
    # Estadísticas finales
    print("\n" + "=" * 50)
    print("RESUMEN")
    print("=" * 50)
    print(f"Imágenes: {original_img_count} → {len(data['images'])} (-{original_img_count - len(data['images'])})")
    print(f"Anotaciones totales: {original_ann_count} → {len(data['annotations'])} (-{original_ann_count - len(data['annotations'])})")
    
    # Verificar distribución final
    print("\nDistribución final de clases:")
    print("-" * 30)
    conteo = defaultdict(int)
    for ann in data["annotations"]:
        nombre = cat_map.get(ann["category_id"], "desconocido")
        conteo[nombre] += 1
    
    for nombre, cantidad in sorted(conteo.items()):
        print(f"  {nombre:<12}: {cantidad}")
    
    final_target_count = sum(1 for a in data["annotations"] if a["category_id"] == target_class_id)
    print(f"\n✓ {target_class_name} reducido de {current_count} a {final_target_count}")


if __name__ == "__main__":
    if not ANNOTATIONS_IN.exists():
        print(f"Error: No se encontró {ANNOTATIONS_IN}")
    else:
        balance_by_image_sampling(ANNOTATIONS_IN, ANNOTATIONS_OUT)
        print(f"\n¡Listo! Usa '{ANNOTATIONS_OUT}' para entrenar.")
