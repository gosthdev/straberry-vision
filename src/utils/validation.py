import json
from pathlib import Path

def validate_coco(json_path, images_root=None, num_categories=None, save_clean=True):
    p = Path(json_path)
    data = json.loads(p.read_text(encoding="utf-8"))
    errors = []

    # Estructura básica
    for k in ("images", "annotations", "categories"):
        if k not in data or not isinstance(data[k], list):
            errors.append(f"Falta sección '{k}' o no es lista")
            return False, errors, data

    # Categories
    cat_ids = {c.get("id") for c in data["categories"]}
    if None in cat_ids:
        errors.append("Alguna categoría no tiene 'id'")
    if num_categories and len(cat_ids) != num_categories:
        errors.append(f"Se esperaban {num_categories} categorías, hay {len(cat_ids)}")

    # Images
    img_ids = set()
    for im in data["images"]:
        if "id" not in im or "file_name" not in im or "width" not in im or "height" not in im:
            errors.append(f"Imagen inválida: {im}")
        else:
            img_ids.add(im["id"])
        if images_root:
            if not Path(images_root, im.get("file_name", "")).exists():
                errors.append(f"No existe archivo de imagen: {im.get('file_name')}")

    # Annotations
    clean_ann = []
    for ann in data["annotations"]:
        error_ann = []
        if "image_id" not in ann or ann["image_id"] not in img_ids:
            error_ann.append(f"annotation {ann.get('id')} con image_id inválido")
        if "category_id" not in ann or ann["category_id"] not in cat_ids:
            error_ann.append(f"annotation {ann.get('id')} con category_id inválido")
        bbox = ann.get("bbox")
        if (not isinstance(bbox, list)) or len(bbox) != 4 or any((not isinstance(v,(int,float))) for v in bbox):
            error_ann.append(f"annotation {ann.get('id')} con bbox inválido: {bbox}")
        elif bbox[2] <= 0 or bbox[3] <= 0:
            error_ann.append(f"annotation {ann.get('id')} con bbox no positiva: {bbox}")

        if error_ann:
            errors.extend(error_ann)  # ahora sí se registran
        else:
            clean_ann.append(ann)

    data["annotations"] = clean_ann

    # Guardar JSON limpio solo si hubo errores
    if save_clean and errors:
        print("Errores encontrados:", len(errors))
        for e in errors[:10]:
            print("-", e)
        out_path = p.parent / f"{p.stem}_clean.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"JSON limpio guardado en: {out_path}")
    
    ok = len(errors) == 0
    return ok, errors, data

def main():
    json_path = "src/data/processed/coco_out/annotations.json" #cambiar
    images_root = "src/data/raw/images" #cambiar
    num_categories = 5 #cambiar

    ok, errs,data_limpio = validate_coco(json_path, images_root, num_categories)
    #if ok:
    #    print("El JSON COCO es válido")
    #else:
    #    print("El JSON COCO tiene errores") 

if __name__ == "__main__":
    main()
    