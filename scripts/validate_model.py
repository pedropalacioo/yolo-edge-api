"""Bloqueia o deploy se o modelo EPI for incompatível ou ficar abaixo do mAP."""

import argparse
import sys
from pathlib import Path

import yaml


DEFAULT_THRESHOLD = 0.60
EXPECTED_CLASSES = {"capacete", "colete", "pessoa"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/yolo-epi.pt")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--dataset", default="dataset/exports/epi-v1/data.yaml")
    return parser.parse_args()


def normalized_names(names):
    if isinstance(names, dict):
        names = names.values()
    return {str(name).strip().casefold() for name in names}


def main():
    args = parse_args()
    model_path = Path(args.model)
    dataset_path = Path(args.dataset)
    if not model_path.is_file():
        print(f"[ERRO] Modelo não encontrado: {model_path}")
        return 1
    if not dataset_path.is_file():
        print(f"[ERRO] Dataset não encontrado: {dataset_path}")
        return 1
    if not 0 <= args.threshold <= 1:
        print("[ERRO] O limiar deve estar entre 0 e 1.")
        return 2

    from ultralytics import YOLO

    with dataset_path.open(encoding="utf-8") as stream:
        dataset_config = yaml.safe_load(stream)
    dataset_names = normalized_names(dataset_config.get("names", []))
    if dataset_names != EXPECTED_CLASSES:
        print(f"[ERRO] Classes inesperadas no dataset: {sorted(dataset_names)}")
        return 1

    model = YOLO(str(model_path))
    model_names = normalized_names(model.names)
    if model_names != EXPECTED_CLASSES:
        print(
            "[ERRO] Classes incompatíveis. "
            f"Modelo={sorted(model_names)} Dataset={sorted(dataset_names)}"
        )
        return 1

    print(f"[INFO] Validando modelo EPI: {model_path}")
    print(f"[INFO] Validando com dataset real: {dataset_path}")
    metrics = model.val(data=str(dataset_path), split="val", verbose=False)
    map50 = float(metrics.box.map50)
    print(f"[INFO] mAP@0.5 = {map50:.4f} | Limiar: {args.threshold:.4f}")
    if map50 < args.threshold:
        print("[FALHA] mAP abaixo do limiar. Deploy bloqueado.")
        return 1

    print("[OK] Quality gate aprovado. Deploy autorizado.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
