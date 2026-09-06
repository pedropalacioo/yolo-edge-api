"""Configurable BGR-to-model preprocessing with geometry metadata."""

from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from preprocessing.utils.letterbox import adjust_bboxes, letterbox


@dataclass(frozen=True)
class PreprocessConfig:
    infer_size: int = 320
    convert_rgb: bool = True
    use_letterbox: bool = True
    gaussian_blur: bool = False
    gaussian_ksize: int = 3
    gaussian_sigma: float = 0.8
    median_blur: bool = False
    median_ksize: int = 3
    clahe: bool = False
    clahe_clip: float = 2.0
    clahe_tile: int = 8
    clahe_space: str = "lab"
    normalize: bool = False


@dataclass(frozen=True)
class PreprocessResult:
    frame: np.ndarray
    scale: float
    scale_x: float
    scale_y: float
    pad_w: int
    pad_h: int
    orig_size: tuple[int, int]


class Preprocessor:
    """Stateless preprocessing pipeline. Input is uint8 BGR by contract."""

    def __init__(self, config: PreprocessConfig | None = None):
        self.cfg = config or CONFIG_DEFAULT
        self._validate_config()

    def _validate_config(self) -> None:
        cfg = self.cfg
        if cfg.infer_size <= 0:
            raise ValueError("infer_size must be greater than zero")
        for enabled, kernel, name in (
            (cfg.gaussian_blur, cfg.gaussian_ksize, "gaussian_ksize"),
            (cfg.median_blur, cfg.median_ksize, "median_ksize"),
        ):
            if enabled and (kernel <= 1 or kernel % 2 == 0):
                raise ValueError(f"{name} must be an odd integer greater than one")
        if cfg.gaussian_blur and cfg.median_blur:
            raise ValueError("gaussian_blur and median_blur are mutually exclusive")
        if cfg.clahe_space not in {"lab", "hsv"}:
            raise ValueError("clahe_space must be 'lab' or 'hsv'")
        if cfg.clahe_clip <= 0 or cfg.clahe_tile <= 0:
            raise ValueError("CLAHE values must be greater than zero")

    def process(self, frame: np.ndarray) -> PreprocessResult:
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("frame must be an HxWx3 NumPy array")
        if frame.dtype != np.uint8:
            raise ValueError("frame must use uint8 BGR pixels")
        orig_h, orig_w = frame.shape[:2]
        if orig_h == 0 or orig_w == 0:
            raise ValueError("frame must have non-zero dimensions")
        out = frame.copy()

        if self.cfg.clahe:
            out = self._apply_clahe(out)
        if self.cfg.convert_rgb:
            out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        if self.cfg.gaussian_blur:
            out = cv2.GaussianBlur(
                out,
                (self.cfg.gaussian_ksize, self.cfg.gaussian_ksize),
                self.cfg.gaussian_sigma,
            )
        elif self.cfg.median_blur:
            out = cv2.medianBlur(out, self.cfg.median_ksize)

        if self.cfg.use_letterbox:
            out, scale, (pad_w, pad_h) = letterbox(out, self.cfg.infer_size)
            scale_x = scale_y = scale
        else:
            out = cv2.resize(out, (self.cfg.infer_size, self.cfg.infer_size))
            scale_x = self.cfg.infer_size / orig_w
            scale_y = self.cfg.infer_size / orig_h
            scale = min(scale_x, scale_y)
            pad_w = pad_h = 0

        if self.cfg.normalize:
            out = out.astype(np.float32) / 255.0
        return PreprocessResult(
            out, scale, scale_x, scale_y, pad_w, pad_h, (orig_h, orig_w)
        )

    def adjust_boxes(self, boxes_xyxy: np.ndarray, result: PreprocessResult) -> np.ndarray:
        return adjust_bboxes(
            boxes_xyxy,
            scale_x=result.scale_x,
            scale_y=result.scale_y,
            pad_w=result.pad_w,
            pad_h=result.pad_h,
            original_size=result.orig_size,
        )

    def _apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        clahe = cv2.createCLAHE(
            clipLimit=self.cfg.clahe_clip,
            tileGridSize=(self.cfg.clahe_tile, self.cfg.clahe_tile),
        )
        if self.cfg.clahe_space == "lab":
            converted = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            luminance, first, second = cv2.split(converted)
            merged = cv2.merge((clahe.apply(luminance), first, second))
            return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        converted = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        first, second, luminance = cv2.split(converted)
        merged = cv2.merge((first, second, clahe.apply(luminance)))
        return cv2.cvtColor(merged, cv2.COLOR_HSV2BGR)


CONFIG_DEFAULT = PreprocessConfig()
CONFIG_LOW_LIGHT = replace(CONFIG_DEFAULT, clahe=True)
CONFIG_HIGH_QUALITY = replace(CONFIG_DEFAULT, infer_size=640)
