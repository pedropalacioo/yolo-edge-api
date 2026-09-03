"""Raw camera preview with MJPEG, snapshot and health endpoints."""

from __future__ import annotations

import argparse
import atexit
import os
import signal
import threading
import time
from pathlib import Path
from typing import Any

import cv2
from flask import Flask, Response, jsonify

from stream.camera import LatestFrameCamera
from stream.v1_naive import positive_int


PAGE = """<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Preview bruto</title><style>body{background:#111;color:#eee;text-align:center;font-family:sans-serif}
img{width:min(96vw,1280px);height:auto}</style></head><body><h1>Preview bruto da câmera CSI</h1>
<p>Sem inferência ou anotações YOLO — <a href="/snapshot">baixar snapshot</a></p>
<img src="/stream" alt="Preview bruto"></body></html>"""


class RawPipeline:
    def __init__(self, camera: int, width: int, height: int, fps: int, quality: int) -> None:
        self.camera = LatestFrameCamera(camera, width, height, fps)
        self.quality = quality
        self.condition = threading.Condition()
        self.jpeg: bytes | None = None
        self.sequence = 0
        self.updated_at: float | None = None
        self.error: str | None = None
        self.stopping = threading.Event()
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.camera.start()
        self.thread = threading.Thread(target=self._run, name="raw-stream", daemon=True)
        self.thread.start()

    def _run(self) -> None:
        camera_sequence = -1
        try:
            while not self.stopping.is_set():
                frame, camera_sequence, _ = self.camera.read_latest(camera_sequence)
                ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
                if not ok:
                    raise RuntimeError("falha ao codificar JPEG")
                with self.condition:
                    self.jpeg = encoded.tobytes()
                    self.sequence += 1
                    self.updated_at = time.time()
                    self.condition.notify_all()
        except BaseException as exc:
            if not self.stopping.is_set():
                self.error = f"{type(exc).__name__}: {exc}"
                with self.condition:
                    self.condition.notify_all()
        finally:
            self.camera.close()

    def wait_for_jpeg(self, after: int = -1, timeout: float = 10) -> tuple[bytes, int] | None:
        with self.condition:
            ready = self.condition.wait_for(
                lambda: self.sequence > after or self.error is not None or self.stopping.is_set(), timeout,
            )
            if not ready or self.jpeg is None or self.error is not None:
                return None
            return self.jpeg, self.sequence

    def frames(self):
        seen = -1
        while not self.stopping.is_set():
            item = self.wait_for_jpeg(seen)
            if item is None:
                return
            jpeg, seen = item
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"

    def close(self) -> None:
        if self.stopping.is_set():
            return
        self.stopping.set()
        self.camera.close()
        with self.condition:
            self.condition.notify_all()
        if self.thread is not None:
            self.thread.join(timeout=5)


def create_app(pipeline: RawPipeline) -> Flask:
    app = Flask(__name__)

    @app.get("/")
    def index() -> Response:
        return Response(PAGE, mimetype="text/html")

    @app.get("/stream")
    def stream() -> Response:
        return Response(pipeline.frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.get("/snapshot")
    def snapshot() -> Response:
        item = pipeline.wait_for_jpeg()
        if item is None:
            return Response("Frame indisponível\n", status=503, mimetype="text/plain")
        jpeg, _ = item
        return Response(jpeg, mimetype="image/jpeg", headers={"Content-Disposition": "inline; filename=snapshot.jpg"})

    @app.get("/health")
    def health():
        age = None if pipeline.updated_at is None else time.time() - pipeline.updated_at
        healthy = pipeline.error is None and age is not None and age < 5
        return jsonify({
            "status": "healthy" if healthy else "unhealthy",
            "camera": "active" if healthy else "inactive",
            "frames": pipeline.sequence,
            "last_frame_age_s": None if age is None else round(age, 3),
            "annotations": False,
            "error": pipeline.error,
        }), 200 if healthy else 503

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=os.getenv("RAW_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=positive_int, default=int(os.getenv("RAW_PORT", "5001")))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=positive_int, default=1280)
    parser.add_argument("--height", type=positive_int, default=720)
    parser.add_argument("--fps", type=positive_int, default=30)
    parser.add_argument("--jpeg-quality", type=positive_int, default=85)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.jpeg_quality > 100:
        raise SystemExit("--jpeg-quality deve estar entre 1 e 100")
    pipeline = RawPipeline(args.camera, args.width, args.height, args.fps, args.jpeg_quality)
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
