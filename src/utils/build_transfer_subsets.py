"""Utility to build reduced object and face datasets for transfer learning experiments."""
import argparse
import random
import shutil
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from PIL import Image
from torchvision import datasets


def _select_subset(items: Sequence[int], k: int, rng: random.Random) -> List[int]:
    if len(items) <= k:
        return list(items)
    return rng.sample(items, k)


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _save_image(image: Image.Image, destination: Path, size: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image = image.convert("RGB")
    if size > 0:
        image = image.resize((size, size), Image.BICUBIC)
    image.save(destination.with_suffix(".webp"), "WEBP", quality=90)


def build_objects_subset(
    source_dir: Path,
    output_dir: Path,
    classes: Iterable[str] | None,
    max_per_class: int,
    image_size: int,
    rng: random.Random,
) -> None:
    dataset = datasets.ImageFolder(str(source_dir))
    selected_classes = set(classes) if classes else set(dataset.classes)

    print(f"[objects] clases disponibles: {dataset.classes}")
    print(f"[objects] clases seleccionadas: {sorted(selected_classes)}")

    _clear_dir(output_dir)
    all_counts: dict[str, int] = {}

    for class_name, class_idx in dataset.class_to_idx.items():
        if class_name not in selected_classes:
            continue

        items = [idx for idx, (_, label) in enumerate(dataset.imgs) if label == class_idx]
        chosen = _select_subset(items, max_per_class, rng)
        rng.shuffle(chosen)

        split_idx = int(len(chosen) * 0.8)
        train_idx = chosen[:split_idx]
        val_idx = chosen[split_idx:]

        for subset_name, subset_indices in ("train", train_idx), ("val", val_idx):
            for i, sample_idx in enumerate(subset_indices):
                img_path, _ = dataset.imgs[sample_idx]
                image = Image.open(img_path)
                destination = output_dir / subset_name / class_name / f"{class_name}_{i:05d}"
                _save_image(image, destination, image_size)

        all_counts[class_name] = len(chosen)

    print(f"[objects] total guardado por clase: {all_counts}")


def build_faces_subset(
    root_dir: Path,
    output_dir: Path,
    subset_size: int,
    image_size: int,
    rng: random.Random,
    download: bool,
) -> None:
    dataset = datasets.CelebA(root=str(root_dir), split="train", download=download)

    indices = _select_subset(list(range(len(dataset))), subset_size, rng)
    rng.shuffle(indices)

    split_idx = int(len(indices) * 0.8)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    _clear_dir(output_dir)

    def _dump(indices_subset: Sequence[int], subset_name: str) -> None:
        for i, idx in enumerate(indices_subset):
            image, _ = dataset[idx]
            destination = output_dir / subset_name / "face" / f"face_{i:05d}"
            _save_image(image, destination, image_size)

    _dump(train_idx, "train")
    _dump(val_idx, "val")

    print(f"[faces] imágenes guardadas → train: {len(train_idx)}, val: {len(val_idx)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build reduced object/face datasets.")
    parser.add_argument("--objects-src", type=Path, help="Ruta con dataset estilo ImageFolder.")
    parser.add_argument(
        "--objects-classes",
        type=str,
        nargs="*",
        help="Clases a conservar (por defecto todas).",
    )
    parser.add_argument(
        "--objects-max-per-class",
        type=int,
        default=500,
        help="Máximo de imágenes por clase en el subset.",
    )
    parser.add_argument(
        "--faces-root",
        type=Path,
        help="Directorio raíz para descargar/leer CelebA.",
    )
    parser.add_argument(
        "--faces-max",
        type=int,
        default=5000,
        help="Número de rostros a conservar.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/transfer_subsets"),
        help="Directorio donde se escribirá el resultado.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="Resolución cuadrada de salida (px).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para muestreo aleatorio.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Si existe CelebA localmente, evita forzar descarga.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    if not args.objects_src and not args.faces_root:
        raise SystemExit("Debe proveerse al menos --objects-src o --faces-root.")

    output_dir = args.output_dir.resolve()
    print(f"Salida → {output_dir}")

    if args.objects_src:
        build_objects_subset(
            source_dir=args.objects_src.resolve(),
            output_dir=output_dir / "objects",
            classes=args.objects_classes,
            max_per_class=args.objects_max_per_class,
            image_size=args.image_size,
            rng=rng,
        )

    if args.faces_root:
        build_faces_subset(
            root_dir=args.faces_root.resolve(),
            output_dir=output_dir / "faces",
            subset_size=args.faces_max,
            image_size=args.image_size,
            rng=rng,
            download=not args.skip_download,
        )


if __name__ == "__main__":
    main()
