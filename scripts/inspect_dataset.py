"""Validate the structure, images and YOLO labels of a dataset export."""

from __future__ import annotations

import argparse
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_CLASSES = {"capacete", "colete", "pessoa"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True, help="caminho para data.yaml")
    parser.add_argument("--min-per-class", type=int, default=30)
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"data.yaml não encontrado: {path}")
    with path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("data.yaml deve conter um objeto YAML")
    return config


def class_names(config: dict[str, Any]) -> list[str]:
    names = config.get("names")
    if isinstance(names, dict):
        try:
            names = [names[index] if index in names else names[str(index)] for index in range(len(names))]
        except (KeyError, TypeError) as exc:
            raise ValueError("IDs do mapa 'names' devem ser sequenciais a partir de zero") from exc
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError("'names' deve ser uma lista de nomes de classes")
    nc = config.get("nc", len(names))
    if nc != len(names):
        raise ValueError(f"nc={nc}, mas names contém {len(names)} classes")
    if {name.strip().casefold() for name in names} != EXPECTED_CLASSES:
        raise ValueError(f"classes esperadas: {sorted(EXPECTED_CLASSES)}; encontradas: {names}")
    return names


def dataset_base(yaml_path: Path, config: dict[str, Any]) -> Path:
    configured = Path(str(config.get("path", yaml_path.parent)))
    if not configured.is_absolute():
        configured = yaml_path.parent / configured
    return configured.resolve()


def resolve_images(base: Path, config: dict[str, Any], key: str) -> Path:
    value = config.get(key)
    if not isinstance(value, str):
        raise ValueError(f"split '{key}' não possui um caminho textual")
    path = Path(value)
    return (path if path.is_absolute() else base / path).resolve()


def inspect_split(images_dir: Path, class_count: int) -> tuple[Counter[int], int, list[str]]:
    issues: list[str] = []
    counts: Counter[int] = Counter()
    if not images_dir.is_dir():
        return counts, 0, [f"diretório de imagens ausente: {images_dir}"]
    labels_dir = images_dir.parent / "labels"
    if not labels_dir.is_dir():
        return counts, 0, [f"diretório de labels ausente: {labels_dir}"]

    images = sorted(path for path in images_dir.iterdir() if path.suffix.casefold() in IMAGE_EXTENSIONS)
    stems = {path.stem for path in images}
    for image in images:
        decoded = cv2.imread(str(image))
        if decoded is None or decoded.size == 0:
            issues.append(f"imagem ausente/corrompida: {image}")
        label = labels_dir / f"{image.stem}.txt"
        if not label.is_file():
            issues.append(f"label ausente: {label}")
            continue
        for line_number, raw_line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
            if not raw_line.strip():
                continue
            fields = raw_line.split()
            location = f"{label}:{line_number}"
            if len(fields) != 5:
                issues.append(f"label inválido ({len(fields)} campos): {location}")
                continue
            try:
                class_value = float(fields[0])
                coordinates = [float(value) for value in fields[1:]]
            except ValueError:
                issues.append(f"valor não numérico: {location}")
                continue
            if not class_value.is_integer() or not 0 <= class_value < class_count:
                issues.append(f"ID de classe inexistente ({fields[0]}): {location}")
                continue
            if not all(math.isfinite(value) for value in coordinates):
                issues.append(f"coordenada não finita: {location}")
                continue
            x, y, width, height = coordinates
            if not (0 <= x <= 1 and 0 <= y <= 1 and 0 < width <= 1 and 0 < height <= 1):
                issues.append(f"coordenadas fora do intervalo YOLO: {location}")
                continue
            counts[int(class_value)] += 1

    for label in labels_dir.glob("*.txt"):
        if label.stem not in stems:
            issues.append(f"label sem imagem correspondente: {label}")
    return counts, len(images), issues


def main() -> int:
    args = parse_args()
    if args.min_per_class < 0:
        print("ERRO: --min-per-class não pode ser negativo", file=sys.stderr)
        return 2
    try:
        yaml_path = args.dataset.resolve()
        config = load_config(yaml_path)
        names = class_names(config)
        base = dataset_base(yaml_path, config)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"ERRO: {exc}", file=sys.stderr)
        return 1

    print("=" * 64)
    print(f"Inspeção do Dataset: {base.name}")
    print("=" * 64)
    print(f"data.yaml: {yaml_path}")
    print(f"path resolvido: {base}")
    print(f"Classes ({len(names)}): {names}")
    all_issues: list[str] = []
    train_counts: Counter[int] = Counter()

    for display_name, key in (("TRAIN", "train"), ("VALID", "val"), ("TEST", "test")):
        try:
            images_dir = resolve_images(base, config, key)
        except ValueError as exc:
            all_issues.append(str(exc))
            continue
        counts, image_count, issues = inspect_split(images_dir, len(names))
        all_issues.extend(issues)
        if key == "train":
            train_counts = counts
        print(f"\n[{display_name}] {image_count} imagens | {sum(counts.values())} anotações")
        for class_id, name in enumerate(names):
            print(f"  {name:15s} {counts[class_id]:6d}")

    for class_id, name in enumerate(names):
        if train_counts[class_id] < args.min_per_class:
            all_issues.append(
                f"classe '{name}' abaixo do mínimo no treino: "
                f"{train_counts[class_id]} < {args.min_per_class}"
            )
    if train_counts:
        nonzero = [train_counts[index] for index in range(len(names)) if train_counts[index] > 0]
        if nonzero and max(nonzero) / min(nonzero) >= 5:
            all_issues.append("desbalanceamento severo no treino (razão entre classes >= 5)")

    print("\n" + "=" * 64)
    if all_issues:
        print(f"Dataset reprovado: {len(all_issues)} problema(s) encontrado(s).")
        for issue in all_issues[:50]:
            print(f"- {issue}")
        if len(all_issues) > 50:
            print(f"- ... e mais {len(all_issues) - 50} problema(s)")
        return 1
    print("Dataset aprovado para treinamento")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
