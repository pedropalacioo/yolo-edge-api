"""Eight deterministic tests for the reusable preprocessor."""

import numpy as np

from preprocessing.preprocessor import (
    CONFIG_DEFAULT,
    CONFIG_HIGH_QUALITY,
    CONFIG_LOW_LIGHT,
    PreprocessConfig,
    Preprocessor,
)


def frame(height=480, width=640):
    values = np.arange(height * width * 3, dtype=np.uint32) % 256
    return values.astype(np.uint8).reshape(height, width, 3)


def test_output_shape_with_letterbox():
    assert Preprocessor(PreprocessConfig(infer_size=416)).process(frame()).frame.shape == (
        416,
        416,
        3,
    )


def test_uint8_is_preserved_without_normalization():
    assert Preprocessor().process(frame()).frame.dtype == np.uint8


def test_normalization_returns_float32_in_unit_range():
    output = Preprocessor(PreprocessConfig(normalize=True)).process(frame()).frame
    assert output.dtype == np.float32
    assert 0.0 <= output.min() <= output.max() <= 1.0


def test_bgr_is_converted_to_rgb_without_mutating_input():
    source = np.array([[[10, 20, 30]]], dtype=np.uint8)
    original = source.copy()
    result = Preprocessor(PreprocessConfig(infer_size=1, use_letterbox=False)).process(source)
    assert result.frame.tolist() == [[[30, 20, 10]]]
    assert np.array_equal(source, original)


def test_letterbox_records_scale_and_symmetric_padding():
    result = Preprocessor(PreprocessConfig(infer_size=416)).process(frame(480, 640))
    assert result.scale == result.scale_x == result.scale_y == 0.65
    assert result.pad_w == 0
    assert result.pad_h == 52
    assert result.orig_size == (480, 640)


def test_adjust_boxes_reverses_letterbox_and_clips():
    processor = Preprocessor(PreprocessConfig(infer_size=416))
    result = processor.process(frame(480, 640))
    boxes = np.array([[0, 52, 416, 364]], dtype=float)
    assert np.allclose(processor.adjust_boxes(boxes, result), [[0, 0, 640, 480]])


def test_non_letterbox_adjustment_uses_independent_scales():
    processor = Preprocessor(PreprocessConfig(infer_size=416, use_letterbox=False))
    result = processor.process(frame(480, 640))
    boxes = np.array([[0, 0, 416, 416]], dtype=float)
    assert result.scale_x != result.scale_y
    assert np.allclose(processor.adjust_boxes(boxes, result), [[0, 0, 640, 480]])


def test_required_presets_are_distinct_and_usable():
    assert CONFIG_DEFAULT.infer_size == 320 and not CONFIG_DEFAULT.clahe
    assert CONFIG_LOW_LIGHT.clahe and CONFIG_LOW_LIGHT.clahe_space == "lab"
    assert CONFIG_HIGH_QUALITY.infer_size == 640
    for config in (CONFIG_DEFAULT, CONFIG_LOW_LIGHT, CONFIG_HIGH_QUALITY):
        assert Preprocessor(config).process(frame(8, 12)).frame.shape[:2] == (
            config.infer_size,
            config.infer_size,
        )
