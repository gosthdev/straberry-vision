# src/utils/analisis.py
from pathlib import Path
import json
from PIL import Image, ImageStat

# Rutas
ANNOTATIONS = Path("src/data/processed/coco_out/annotations.json")
IMAGES_ROOT = Path("src/data/raw/images")


def analizar_json():
    print("\nPaso 1.1: Verificando integridad del JSON...")
    data = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))

    errores = []
    if "images" not in data or "annotations" not in data or "categories" not in data:
        errores.append("Faltan secciones obligatorias en el JSON")

    img_ids = {im["id"] for im in data.get("images", [])}
    cat_ids = {c["id"] for c in data.get("categories", [])}

    for ann in data.get("annotations", []):
        if ann["image_id"] not in img_ids:
            errores.append(f"annotation {ann['id']} con image_id inválido")
        if ann["category_id"] not in cat_ids:
            errores.append(f"annotation {ann['id']} con category_id inválido")
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            errores.append(f"annotation {ann['id']} con bbox no positiva: {ann['bbox']}")

    if errores:
        print(" Errores encontrados:")
        for e in errores[:10]:
            print("-", e)
    else:
        print(" JSON válido, sin errores graves")

    return data


def verificar_imagenes(data):
    print("\nPaso 1.2: Verificando consistencia entre JSON e imágenes...")
    errores = []

    for im in data["images"]:
        ruta = IMAGES_ROOT / im["file_name"]
        if not ruta.exists():
            errores.append(f"No existe la imagen: {ruta}")

    if errores:
        print(" Errores de consistencia:")
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