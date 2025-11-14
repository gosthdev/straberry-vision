"""Convierte imágenes PNG a WEBP con compresión controlada."""
import argparse
from pathlib import Path

from PIL import Image


def convert_image(
    root: Path,
    source_path: Path,
    output_path: Path,
    quality: int,
    overwrite: bool,
    delete_original: bool,
) -> None:
    if not overwrite and output_path.exists():
        print(f"[omitido] {output_path} ya existe")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(source_path) as img:
        mode = "RGBA" if img.mode in {"RGBA", "LA", "P"} else "RGB"
        image = img.convert(mode)
        image.save(output_path, "WEBP", quality=quality, method=6)

    if delete_original:
        source_path.unlink()

    try:
        relative = output_path.relative_to(root)
    except ValueError:
        relative = output_path

    print(f"[ok] {source_path.name} → {relative}")


def iter_png_files(root: Path):
    for file_path in root.rglob("*"):
        if file_path.is_file() and file_path.suffix.lower() == ".png":
            yield file_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convertir PNGs a WEBP.")
    parser.add_argument(
        "root",
        type=Path,
        help="Directorio raíz donde buscar archivos .png",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=90,
        help="Calidad WEBP (0-100).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reemplaza archivos .webp existentes.",
    )
    parser.add_argument(
        "--keep-original",
        action="store_true",
        help="Conserva los PNG después de convertirlos.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()

    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Directorio inválido: {root}")

    png_files = list(iter_png_files(root))
    if not png_files:
        print("No se encontraron PNGs para convertir.")
        return

    print(f"Convertiendo {len(png_files)} archivos desde {root} con calidad {args.quality}")

    delete_original = not args.keep_original

    for png_path in png_files:
        webp_path = png_path.with_suffix(".webp")
        convert_image(
            root=root,
            source_path=png_path,
            output_path=webp_path,
            quality=args.quality,
            overwrite=args.overwrite,
            delete_original=delete_original,
        )


if __name__ == "__main__":
    main()
