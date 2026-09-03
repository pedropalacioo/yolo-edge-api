"""Optimized YOLO pipeline with frame skipping, OSD, display and AVI output."""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
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
    parser.add_argument("--frames", type=positive_int, default=150)
    parser.add_argument("--imgsz", type=positive_int, default=416)
    parser.add_argument("--infer-every", type=positive_int, default=2)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--output-fps", type=float, default=10.0)
    return parser.parse_args()


def add_osd(frame, fps: float, latency_ms: float, detections: int, inferred: bool) -> None:
    lines = [
        f"FPS: {fps:.2f}",
        f"Latencia: {latency_ms:.1f} ms",
        f"Deteccoes: {detections}",
        f"Inferencia: {'sim' if inferred else 'reutilizada'}",
    ]
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (310, 125), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.62, frame, 0.38, 0, frame)
    for row, line in enumerate(lines):
        cv2.putText(frame, line, (22, 38 + row * 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)


def run(args: argparse.Namespace) -> int:
    if not args.model.is_file():
        raise FileNotFoundError(f"modelo não encontrado: {args.model}")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf deve estar entre 0 e 1")
    if args.output_fps <= 0:
        raise ValueError("--output-fps deve ser maior que zero")
    if not args.headless and not sys.stdout.isatty():
        print("Aviso: terminal não interativo; use --headless se não houver display.", file=sys.stderr)

    print("=== Pipeline V3: otimizado ===", flush=True)
    print(
        f"Comando: {' '.join(sys.argv)}\nConfiguração: camera={args.camera} "
        f"resolução={args.width}x{args.height} fps_solicitado={args.fps} "
        f"frames={args.frames} imgsz={args.imgsz} infer_every={args.infer_every} "
        f"headless={args.headless} output={args.output or 'nenhum'}",
        flush=True,
    )

    model = YOLO(str(args.model))
    writer = None
    inference_times: list[float] = []
    latencies: list[float] = []
    fps_window: deque[float] = deque(maxlen=30)
    last_sequence = -1
    last_result = None
    last_detections = 0
    inference_count = 0
    started = time.perf_counter()

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(args.output), cv2.VideoWriter_fourcc(*"MJPG"), args.output_fps,
            (args.width, args.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"não foi possível abrir o AVI para gravação: {args.output}")

    try:
        with LatestFrameCamera(args.camera, args.width, args.height, args.fps) as camera:
            for index in range(args.frames):
                frame, last_sequence, captured_at = camera.read_latest(last_sequence)
                now = time.perf_counter()
                inferred = index % args.infer_every == 0 or last_result is None
                if inferred:
                    infer_started = time.perf_counter()
                    last_result = model.predict(frame, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
                    infer_done = time.perf_counter()
                    inference_times.append((infer_done - infer_started) * 1000)
                    inference_count += 1
                    last_detections = len(last_result.boxes)
                annotated = last_result.plot(img=frame, labels=True, conf=True)
                latency_ms = (time.perf_counter() - captured_at) * 1000
                latencies.append(latency_ms)
                fps_window.append(time.perf_counter())
                display_fps = 0.0
                if len(fps_window) > 1:
                    display_fps = (len(fps_window) - 1) / (fps_window[-1] - fps_window[0])
                add_osd(annotated, display_fps, latency_ms, last_detections, inferred)
                if writer is not None:
                    writer.write(annotated)
                if not args.headless:
                    cv2.imshow("YOLO V3", annotated)
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        print("Encerramento solicitado pela interface.")
                        break
                if (index + 1) % 10 == 0 or index == 0:
                    print(
                        f"frame={index + 1:03d}/{args.frames} camera_seq={last_sequence} "
                        f"fps={display_fps:.2f} latency_ms={latency_ms:.2f} "
                        f"inferred={inferred} detections={last_detections}",
                        flush=True,
                    )
            captured = camera.captured
    finally:
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    elapsed = time.perf_counter() - started
    print("=== Relatório final V3 ===")
    print(f"frames_processados={len(latencies)}")
    print(f"frames_capturados={captured}")
    print(f"inferencias_executadas={inference_count}")
    print(f"infer_every={args.infer_every}")
    print(f"tempo_total_s={elapsed:.2f}")
    print(f"fps_saida={len(latencies) / elapsed:.2f}")
    print(f"latencia_media_ms={mean(latencies):.2f}")
    print(f"inferencia_media_ms={mean(inference_times):.2f}")
    print(f"arquivo_saida={args.output or 'nenhum'}")
    print("camera_encerrada=sim")
    return 0


def main() -> int:
    try:
        return run(parse_args())
    except KeyboardInterrupt:
        print("\nInterrompido pelo usuário; encerrando recursos.", file=sys.stderr)
        return 130
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
