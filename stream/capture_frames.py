"""Capture raw frames from the CSI camera or an MJPEG URL."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np

from stream.camera import JPEG_END, JPEG_START, LatestFrameCamera
from stream.v1_naive import positive_int


class HttpMjpegSource:
    def __init__(self, url: str) -> None:
        self.url = url
        self.response = None
        self.buffer = bytearray()

    def __enter__(self) -> "HttpMjpegSource":
        self.response = urlopen(self.url, timeout=10)
        return self

    def read(self) -> np.ndarray:
        if self.response is None:
            raise RuntimeError("fonte MJPEG não foi aberta")
        while True:
            start = self.buffer.find(JPEG_START)
            end = self.buffer.find(JPEG_END, max(start + 2, 0))
            if start >= 0 and end >= 0:
                encoded = bytes(self.buffer[start : end + 2])
                del self.buffer[: end + 2]
                frame = cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    return frame
            chunk = self.response.read(65536)
            if not chunk:
                raise RuntimeError("fonte MJPEG encerrou o fluxo")
            self.buffer.extend(chunk)

    def __exit__(self, *_: object) -> None:
        if self.response is not None:
            self.response.close()


class CsiSource:
    def __init__(self, camera: int, width: int, height: int, fps: int) -> None:
        self.camera = LatestFrameCamera(camera, width, height, fps)
        self.sequence = -1

    def __enter__(self) -> "CsiSource":
        self.camera.start()
        return self

    def read(self) -> np.ndarray:
        frame, self.sequence, _ = self.camera.read_latest(self.sequence)
        return frame

    def __exit__(self, *_: object) -> None:
        self.camera.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--mjpeg-url", help="fonte HTTP, por exemplo http://host:5001/stream")
    source.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=positive_int, default=1280)
    parser.add_argument("--height", type=positive_int, default=720)
    parser.add_argument("--fps", type=positive_int, default=30)
    parser.add_argument("--count", type=positive_int, default=10)
    parser.add_argument("--interval", type=float, default=1.0, help="segundos entre capturas")
    parser.add_argument("--output", type=Path, default=Path("dataset/raw"))
    parser.add_argument("--prefix", default="frame")
    parser.add_argument("--manual", action="store_true", help="aguarda ENTER antes de cada captura")
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    if args.interval < 0:
        raise ValueError("--interval não pode ser negativo")
    args.output.mkdir(parents=True, exist_ok=True)
    source = HttpMjpegSource(args.mjpeg_url) if args.mjpeg_url else CsiSource(
        args.camera, args.width, args.height, args.fps,
    )
    print(f"fonte={args.mjpeg_url or f'camera:{args.camera}'} destino={args.output} quantidade={args.count}")
    with source:
        for index in range(1, args.count + 1):
            if args.manual:
                input(f"[{index}/{args.count}] Pressione ENTER para capturar...")
            frame = source.read()
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            path = args.output / f"{args.prefix}_{timestamp}_{index:04d}.jpg"
            if not cv2.imwrite(str(path), frame):
                raise RuntimeError(f"não foi possível gravar {path}")
            print(f"capturado={path} resolução={frame.shape[1]}x{frame.shape[0]}", flush=True)
            if index < args.count and not args.manual:
                time.sleep(args.interval)
    print(f"capturas_concluidas={args.count}")
    return 0


def main() -> int:
    try:
        return run(parse_args())
    except KeyboardInterrupt:
        print("\nCaptura interrompida; liberando a fonte.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
