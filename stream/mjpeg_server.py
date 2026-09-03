"""Flask MJPEG server for the optimized Raspberry Pi YOLO pipeline."""

from __future__ import annotations

import argparse
import atexit
import os
import signal
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import cv2
from flask import Flask, Response, jsonify
from ultralytics import YOLO

from stream.camera import LatestFrameCamera
from stream.v1_naive import positive_int
from stream.v3_optimized import add_osd


PAGE = """<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>YOLO Edge Stream</title>
  <style>
    body { margin: 0; background: #111; color: #eee; font: 16px sans-serif; text-align: center; }
    h1 { margin: 1rem 0 .5rem; }
    img { width: min(96vw, 1280px); height: auto; border: 1px solid #555; }
    a { color: #79d4ff; }
  </style>
</head>
<body>
  <h1>YOLO Edge — câmera CSI</h1>
  <p><a href="/health">Estado do serviço</a></p>
  <img src="/stream" alt="Stream YOLO anotado">
</body>
</html>"""


class StreamPipeline:
    def __init__(
        self, model_path: Path, camera_index: int, width: int, height: int,
        camera_fps: int, imgsz: int, infer_every: int, confidence: float,
        jpeg_quality: int,
    ) -> None:
        self.model_path = model_path
        self.camera = LatestFrameCamera(camera_index, width, height, camera_fps)
        self.imgsz = imgsz
        self.infer_every = infer_every
        self.confidence = confidence
        self.jpeg_quality = jpeg_quality
        self.condition = threading.Condition()
        self.latest_jpeg: bytes | None = None
        self.sequence = 0
        self.detections = 0
        self.output_fps = 0.0
        self.latency_ms = 0.0
        self.updated_at: float | None = None
        self.error: str | None = None
        self.started_at: float | None = None
        self.stopping = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        if self.thread is not None:
            return
        self.started_at = time.time()
        self.camera.start()
        self.thread = threading.Thread(target=self._run, name="yolo-stream", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        timestamps: deque[float] = deque(maxlen=30)
        camera_sequence = -1
        result = None
        try:
            model = YOLO(str(self.model_path))
            frame_index = 0
            while not self.stopping.is_set():
                frame, camera_sequence, captured_at = self.camera.read_latest(camera_sequence)
                inferred = frame_index % self.infer_every == 0 or result is None
                if inferred:
                    result = model.predict(
                        frame, imgsz=self.imgsz, conf=self.confidence, verbose=False,
                    )[0]
                    self.detections = len(result.boxes)
                annotated = result.plot(img=frame, labels=True, conf=True)
                now = time.perf_counter()
                self.latency_ms = (now - captured_at) * 1000
                timestamps.append(now)
                if len(timestamps) > 1:
                    self.output_fps = (len(timestamps) - 1) / (timestamps[-1] - timestamps[0])
                add_osd(annotated, self.output_fps, self.latency_ms, self.detections, inferred)
                ok, encoded = cv2.imencode(
                    ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
                )
                if not ok:
                    raise RuntimeError("OpenCV não conseguiu codificar o frame como JPEG")
                with self.condition:
                    self.latest_jpeg = encoded.tobytes()
                    self.sequence += 1
                    self.updated_at = time.time()
                    self.condition.notify_all()
                frame_index += 1
        except BaseException as exc:
            if not self.stopping.is_set():
                self.error = f"{type(exc).__name__}: {exc}"
                with self.condition:
                    self.condition.notify_all()
        finally:
            self.camera.close()

    def frames(self):
        seen = -1
        while not self.stopping.is_set():
            with self.condition:
                ready = self.condition.wait_for(
                    lambda: self.sequence > seen or self.error is not None or self.stopping.is_set(),
                    timeout=10,
                )
                if not ready:
                    continue
                if self.error is not None:
                    return
                if self.latest_jpeg is None:
                    continue
                seen = self.sequence
                jpeg = self.latest_jpeg
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

    def health(self) -> tuple[dict[str, Any], int]:
        age = None if self.updated_at is None else time.time() - self.updated_at
        healthy = self.error is None and age is not None and age < 5 and not self.stopping.is_set()
        payload = {
            "status": "healthy" if healthy else "starting" if self.error is None and age is None else "unhealthy",
            "camera": "active" if healthy else "inactive",
            "model": self.model_path.name,
            "frames_processed": self.sequence,
            "fps": round(self.output_fps, 2),
            "latency_ms": round(self.latency_ms, 2),
            "detections": self.detections,
            "last_frame_age_s": None if age is None else round(age, 3),
            "error": self.error,
        }
        return payload, 200 if healthy else 503

    def close(self) -> None:
        if self.stopping.is_set():
            return
        self.stopping.set()
        self.camera.close()
        with self.condition:
            self.condition.notify_all()
        if self.thread is not None:
            self.thread.join(timeout=5)


def create_app(pipeline: StreamPipeline) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index() -> Response:
        return Response(PAGE, mimetype="text/html")

    @app.get("/stream")
    def stream() -> Response:
        return Response(
            pipeline.frames(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )

    @app.get("/health")
    def health():
        payload, status = pipeline.health()
        return jsonify(payload), status

    return app


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.getenv("STREAM_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=positive_int, default=env_int("STREAM_PORT", 5000))
    parser.add_argument("--model", type=Path, default=Path(os.getenv("MODEL_PATH", "models/yolov8n.pt")))
    parser.add_argument("--camera", type=int, default=env_int("CAMERA_INDEX", 0))
    parser.add_argument("--width", type=positive_int, default=env_int("CAMERA_WIDTH", 1280))
    parser.add_argument("--height", type=positive_int, default=env_int("CAMERA_HEIGHT", 720))
    parser.add_argument("--fps", type=positive_int, default=env_int("CAMERA_FPS", 30))
    parser.add_argument("--imgsz", type=positive_int, default=env_int("INFERENCE_SIZE", 416))
    parser.add_argument("--infer-every", type=positive_int, default=env_int("INFER_EVERY", 2))
    parser.add_argument("--conf", type=float, default=float(os.getenv("CONFIDENCE", "0.25")))
    parser.add_argument("--jpeg-quality", type=int, default=env_int("JPEG_QUALITY", 80))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.model.is_file():
        raise SystemExit(f"Modelo não encontrado: {args.model}")
    if not 1 <= args.jpeg_quality <= 100:
        raise SystemExit("--jpeg-quality deve estar entre 1 e 100")
    pipeline = StreamPipeline(
        args.model, args.camera, args.width, args.height, args.fps,
        args.imgsz, args.infer_every, args.conf, args.jpeg_quality,
    )
    pipeline.start()
    atexit.register(pipeline.close)

    def stop(_signum: int, _frame: Any) -> None:
        pipeline.close()
        raise SystemExit(0)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    create_app(pipeline).run(host=args.host, port=args.port, threaded=True, use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
