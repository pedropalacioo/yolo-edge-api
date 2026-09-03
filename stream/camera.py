"""Shared Raspberry Pi MJPEG camera primitives."""

from __future__ import annotations

import shutil
import signal
import subprocess
import threading
import time

import cv2
import numpy as np


JPEG_START = b"\xff\xd8"
JPEG_END = b"\xff\xd9"


class MjpegCamera:
    """Own an rpicam-vid subprocess and decode its MJPEG byte stream."""

    def __init__(self, camera: int, width: int, height: int, fps: int) -> None:
        executable = shutil.which("rpicam-vid")
        if executable is None:
            raise RuntimeError("rpicam-vid não foi encontrado no PATH")
        self.command = [
            executable,
            "--camera", str(camera),
            "--width", str(width),
            "--height", str(height),
            "--framerate", str(fps),
            "--codec", "mjpeg",
            "--quality", "85",
            "--timeout", "0",
            "--nopreview",
            "--output", "-",
        ]
        self.process: subprocess.Popen[bytes] | None = None
        self.buffer = bytearray()

    def start(self) -> None:
        if self.process is not None:
            raise RuntimeError("processo da câmera já foi iniciado")
        self.process = subprocess.Popen(
            self.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=0,
        )

    def read(self) -> np.ndarray:
        if self.process is None or self.process.stdout is None:
            raise RuntimeError("processo da câmera não foi iniciado")
        while True:
            start = self.buffer.find(JPEG_START)
            end = self.buffer.find(JPEG_END, max(start + 2, 0))
            if start >= 0 and end >= 0:
                encoded = bytes(self.buffer[start : end + 2])
                del self.buffer[: end + 2]
                frame = cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame
            chunk = self.process.stdout.read(65536)
            if not chunk:
                code = self.process.poll()
                raise RuntimeError(f"rpicam-vid encerrou sem entregar um frame (código {code})")
            self.buffer.extend(chunk)

    def close(self) -> None:
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        if self.process.stdout is not None:
            self.process.stdout.close()

    def __enter__(self) -> "MjpegCamera":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class LatestFrameCamera:
    """Capture continuously, keeping only the newest decoded frame."""

    def __init__(self, camera: int, width: int, height: int, fps: int) -> None:
        self.camera = MjpegCamera(camera, width, height, fps)
        self.condition = threading.Condition()
        self.frame: np.ndarray | None = None
        self.sequence = 0
        self.captured = 0
        self.error: BaseException | None = None
        self.stopping = False
        self.thread: threading.Thread | None = None

    def start(self) -> None:
        self.camera.start()
        self.thread = threading.Thread(target=self._capture_loop, name="camera-capture", daemon=True)
        self.thread.start()

    def _capture_loop(self) -> None:
        try:
            while not self.stopping:
                frame = self.camera.read()
                captured_at = time.perf_counter()
                with self.condition:
                    self.frame = frame
                    self.sequence += 1
                    self.captured += 1
                    self.captured_at = captured_at
                    self.condition.notify_all()
        except BaseException as exc:
            if not self.stopping:
                with self.condition:
                    self.error = exc
                    self.condition.notify_all()

    def read_latest(self, after_sequence: int = -1, timeout: float = 5.0) -> tuple[np.ndarray, int, float]:
        deadline = time.monotonic() + timeout
        with self.condition:
            while self.frame is None or self.sequence <= after_sequence:
                if self.error is not None:
                    raise RuntimeError(f"falha na thread de captura: {self.error}") from self.error
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("tempo esgotado aguardando frame da câmera")
                self.condition.wait(remaining)
            return self.frame.copy(), self.sequence, self.captured_at

    def close(self) -> None:
        self.stopping = True
        self.camera.close()
        if self.thread is not None:
            self.thread.join(timeout=3)

    def __enter__(self) -> "LatestFrameCamera":
        self.start()
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
