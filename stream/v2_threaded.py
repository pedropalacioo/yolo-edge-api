"""Threaded YOLO pipeline with a latest-frame-only camera buffer."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ultralytics import YOLO

from stream.camera import LatestFrameCamera
from stream.v1_naive import mean, positive_int


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


def run(args: argparse.Namespace) -> int:
    if not args.model.is_file():
        raise FileNotFoundError(f"modelo não encontrado: {args.model}")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf deve estar entre 0 e 1")

    print("=== Pipeline V2: captura concorrente ===", flush=True)
    print(
        f"Comando: {' '.join(sys.argv)}\nConfiguração: camera={args.camera} "
        f"resolução={args.width}x{args.height} fps_solicitado={args.fps} "
        f"frames={args.frames} imgsz={args.imgsz} conf={args.conf:.2f} modelo={args.model}",
        flush=True,
    )
    model = YOLO(str(args.model))
    ages: list[float] = []
    inferences: list[float] = []
    cycles: list[float] = []
    detections = 0
    last_sequence = -1
    run_started = time.perf_counter()

    with LatestFrameCamera(args.camera, args.width, args.height, args.fps) as camera:
        for index in range(1, args.frames + 1):
            cycle_started = time.perf_counter()
            frame, sequence, captured_at = camera.read_latest(last_sequence)
            last_sequence = sequence
            inference_started = time.perf_counter()
            results = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)
            inference_done = time.perf_counter()
            age_ms = (inference_started - captured_at) * 1000
            inference_ms = (inference_done - inference_started) * 1000
            cycle_ms = (inference_done - cycle_started) * 1000
            skipped = max(0, sequence - (1 if index == 1 else previous_sequence) - 1)
            previous_sequence = sequence
            detected = len(results[0].boxes)
            ages.append(age_ms)
            inferences.append(inference_ms)
            cycles.append(cycle_ms)
            detections += detected
            print(
                f"frame={index:03d}/{args.frames} camera_seq={sequence} descartados={skipped} "
                f"idade_ms={age_ms:.2f} inference_ms={inference_ms:.2f} "
                f"cycle_ms={cycle_ms:.2f} detections={detected}",
                flush=True,
            )
        captured = camera.captured

    elapsed = time.perf_counter() - run_started
    print("=== Relatório final V2 ===")
    print(f"frames_processados={args.frames}")
    print(f"frames_capturados={captured}")
    print(f"frames_descartados={max(0, captured - args.frames)}")
    print(f"tempo_total_s={elapsed:.2f}")
    print(f"fps_sustentado={args.frames / elapsed:.2f}")
    print(f"idade_media_frame_ms={mean(ages):.2f}")
    print(f"inferencia_media_ms={mean(inferences):.2f}")
    print(f"ciclo_medio_ms={mean(cycles):.2f}")
    print(f"deteccoes_totais={detections}")
    print("buffer=ultimo_frame")
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
