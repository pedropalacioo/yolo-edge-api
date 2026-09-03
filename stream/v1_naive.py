"""Naive, sequential Raspberry Pi camera and YOLO inference pipeline.

Each loop waits for one complete MJPEG frame and only then runs inference.  This
is intentionally simple and provides the baseline used by the threaded and
optimized pipelines.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from ultralytics import YOLO

from stream.camera import MjpegCamera


@dataclass
class Metrics:
    capture_ms: list[float]
    inference_ms: list[float]
    cycle_ms: list[float]
    detections: int = 0


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("o valor deve ser maior que zero")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("models/yolov8n.pt"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=positive_int, default=1280)
    parser.add_argument("--height", type=positive_int, default=720)
    parser.add_argument("--fps", type=positive_int, default=30)
    parser.add_argument("--frames", type=positive_int, default=50)
    parser.add_argument("--imgsz", type=positive_int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    return parser.parse_args()


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def run(args: argparse.Namespace) -> int:
    if not args.model.is_file():
        raise FileNotFoundError(f"modelo não encontrado: {args.model}")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf deve estar entre 0 e 1")

    print("=== Pipeline V1: diagnóstico ingênuo ===", flush=True)
    print(
        f"Comando: {' '.join(sys.argv)}\n"
        f"Configuração: camera={args.camera} resolução={args.width}x{args.height} "
        f"fps_solicitado={args.fps} frames={args.frames} imgsz={args.imgsz} "
        f"conf={args.conf:.2f} modelo={args.model}",
        flush=True,
    )

    model = YOLO(str(args.model))
    metrics = Metrics([], [], [])
    run_started = time.perf_counter()

    with MjpegCamera(args.camera, args.width, args.height, args.fps) as camera:
        for index in range(1, args.frames + 1):
            cycle_started = time.perf_counter()
            capture_started = cycle_started
            frame = camera.read()
            capture_done = time.perf_counter()
            results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
            inference_done = time.perf_counter()

            capture_ms = (capture_done - capture_started) * 1000
            inference_ms = (inference_done - capture_done) * 1000
            cycle_ms = (inference_done - cycle_started) * 1000
            detected = len(results[0].boxes)
            metrics.capture_ms.append(capture_ms)
            metrics.inference_ms.append(inference_ms)
            metrics.cycle_ms.append(cycle_ms)
            metrics.detections += detected
            print(
                f"frame={index:03d}/{args.frames} capture_ms={capture_ms:.2f} "
                f"inference_ms={inference_ms:.2f} cycle_ms={cycle_ms:.2f} "
                f"detections={detected}",
                flush=True,
            )

    elapsed = time.perf_counter() - run_started
    sustained_fps = args.frames / elapsed
    print("=== Relatório final V1 ===")
    print(f"frames_processados={args.frames}")
    print(f"tempo_total_s={elapsed:.2f}")
    print(f"fps_sustentado={sustained_fps:.2f}")
    print(f"captura_media_ms={mean(metrics.capture_ms):.2f}")
    print(f"inferencia_media_ms={mean(metrics.inference_ms):.2f}")
    print(f"ciclo_medio_ms={mean(metrics.cycle_ms):.2f}")
    print(f"deteccoes_totais={metrics.detections}")
    print("camera_encerrada=sim")
    return 0


def main() -> int:
    try:
        return run(parse_args())
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário; encerrando a câmera.", file=sys.stderr)
        return 130
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
