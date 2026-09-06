"""Shared, reproducible evaluator for lesson 5 image experiments."""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import yaml

Transform = Callable[[np.ndarray], np.ndarray]
EXPECTED_CLASSES = {"capacete", "colete", "pessoa"}


def _names(value) -> set[str]:
    if isinstance(value, dict):
        value = value.values()
    return {str(item).strip().casefold() for item in value}


def evaluate_pipeline(
    label: str,
    transform: Transform | None = None,
    *,
    model_path: Path = Path("models/yolo-epi.pt"),
    dataset_yaml: Path = Path("dataset/exports/epi-v1/data.yaml"),
    split: str = "val",
    imgsz: int = 640,
) -> dict[str, float | int | str]:
    """Evaluate a transform, refusing incompatible EPI models."""
    from ultralytics import YOLO

    if not model_path.is_file():
        raise FileNotFoundError(f"compatible EPI model not found: {model_path}")
    if not dataset_yaml.is_file():
        raise FileNotFoundError(f"dataset config not found: {dataset_yaml}")
    with dataset_yaml.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    model = YOLO(str(model_path))
    if _names(config["names"]) != EXPECTED_CLASSES or _names(model.names) != EXPECTED_CLASSES:
        raise ValueError("model and dataset must both contain Capacete, Colete and Pessoa")

    evaluation_yaml = dataset_yaml
    preproc_ms = 0.0
    image_count = 0
    if transform is not None:
        source_name = {"val": "valid", "test": "test", "train": "train"}[split]
        source = dataset_yaml.parent / source_name
        destination = Path("preprocessing/outputs/evaluation") / label
        if destination.exists():
            shutil.rmtree(destination)
        images_out = destination / "images"
        labels_out = destination / "labels"
        images_out.mkdir(parents=True)
        labels_out.mkdir()
        timings = []
        for image_path in sorted((source / "images").iterdir()):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            started = time.perf_counter()
            output = transform(image)
            timings.append((time.perf_counter() - started) * 1000)
            cv2.imwrite(str(images_out / image_path.name), output)
            source_label = source / "labels" / f"{image_path.stem}.txt"
            if source_label.exists():
                shutil.copy2(source_label, labels_out / source_label.name)
            image_count += 1
        preproc_ms = float(np.mean(timings)) if timings else 0.0
        temp_config = {
            "path": str(destination.resolve()),
            "train": "images",
            "val": "images",
            "names": config["names"],
        }
        evaluation_yaml = destination / "data.yaml"
        evaluation_yaml.write_text(yaml.safe_dump(temp_config), encoding="utf-8")

    started = time.perf_counter()
    metrics = model.val(data=str(evaluation_yaml), split="val", imgsz=imgsz, verbose=False)
    result = {
        "label": label,
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
        "preproc_ms": preproc_ms,
        "images": image_count,
        "elapsed_s": time.perf_counter() - started,
    }
    print(json.dumps(result, ensure_ascii=False))
    return result
