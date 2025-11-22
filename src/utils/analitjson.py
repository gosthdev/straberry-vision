# src/utils/analisis.py
from pathlib import Path
import json
from PIL import Image, ImageStat

# Rutas
ANNOTATIONS = Path("src/data/processed/coco_out/annotations.json")
IMAGES_ROOT = Path("src/data/raw/images")


def guardar_json(data):
    ANNOTATIONS.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n Archivo anotations actualizado: {ANNOTATIONS}")


def analizar_json():
    print("\nPaso 1.1: Verificando integridad del JSON...")
    data = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))

    errores = []
    if "images" not in data or "annotations" not in data or "categories" not in data:
        errores.append("Faltan secciones obligatorias en el JSON")

    img_ids = {im["id"] for im in data.get("images", [])}
    cat_ids = {c["id"] for c in data.get("categories", [])}

    cleaned_annotations = []
    removed = 0

    for ann in data.get("annotations", []):
        motivos = []
        if ann["image_id"] not in img_ids:
            motivos.append("image_id inválido")
        if ann["category_id"] not in cat_ids:
            motivos.append("category_id inválido")
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            motivos.append(f"bbox no positiva: {ann['bbox']}")

        if motivos:
            errores.append(f"annotation {ann['id']} removida ({', '.join(motivos)})")
            removed += 1
        else:
            cleaned_annotations.append(ann)

    if removed:
        data["annotations"] = cleaned_annotations
        print(f" Se eliminaron {removed} anotaciones inválidas")

    if errores:
        print(" Errores corregidos:")
        for e in errores[:10]:
            print("-", e)
    else:
        print(" JSON válido, sin errores graves")

    return data


def verificar_imagenes(data):
    print("\nPaso 1.2: Verificando consistencia entre JSON e imágenes...")
    errores = []

    nuevas_imagenes = []
    ids_eliminados = set()

    for im in data["images"]:
        ruta = IMAGES_ROOT / im["file_name"]
        if ruta.exists():
            nuevas_imagenes.append(im)
        else:
            errores.append(f"No existe la imagen: {ruta}")
            ids_eliminados.add(im["id"])

    if ids_eliminados:
        data["images"] = nuevas_imagenes
        antes = len(data["annotations"])
        data["annotations"] = [ann for ann in data["annotations"] if ann["image_id"] not in ids_eliminados]
        print(
            f" Se eliminaron {len(ids_eliminados)} imágenes faltantes y {antes - len(data['annotations'])} anotaciones asociadas"
        )

    if errores:
        print(" Errores de consistencia corregidos")
        for e in errores[:10]:
            print("-", e)
    else:
        print(" Todas las imágenes del JSON existen en disco")


def validar_calidad(data):
    print("\nPaso 1.3: Validando calidad de imágenes...")

    problemas = []
    for im in data["images"]:
        img_file = IMAGES_ROOT / im["file_name"]
        if not img_file.exists():
            continue

        try:
            with Image.open(img_file) as img:
                gray = img.convert("L")
                stat = ImageStat.Stat(gray)
                varianza = stat.var[0]

                if varianza < 5:  # umbral
                    problemas.append(
                        f"- Imagen sospechosa (varianza {varianza:.2f}): {img_file}"
                    )
        except Exception as e:
            problemas.append(f"- Error al abrir {img_file}: {e}")

    if problemas:
        print(" Problemas detectados:")
        for p in problemas:
            print(p)
    else:
        print(" Todas las imágenes parecen tener suficiente variación de color")


def distribucion_clases(data):
    print("\nPaso 1.4: Analizando distribución de clases...")
    conteo = {}
    cat_map = {c["id"]: c["name"] for c in data["categories"]}

    for ann in data["annotations"]:
        nombre = cat_map.get(ann["category_id"], "desconocido")
        conteo[nombre] = conteo.get(nombre, 0) + 1

    print("\nClase        | # N°fresas detectadas")
    print("-------------|--------------")
    for k, v in conteo.items():
        print(f"{k:<12} | {v}")


if __name__ == "__main__":
    if not ANNOTATIONS.exists():
        print(f" No se encontró el archivo {ANNOTATIONS}")
    else:
        data = analizar_json()
        verificar_imagenes(data)
        validar_calidad(data)
        distribucion_clases(data)
        guardar_json(data)