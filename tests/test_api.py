"""Smoke tests, testes unitários e testes de integração da API."""

import base64
import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import _decode_image, app

client = TestClient(app)
ASSETS = Path(__file__).parent / "assets"


class TestSmoke:
    def test_health_status_200(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_health_payload_structure(self):
        data = client.get("/health").json()
        assert {"status", "model_loaded", "model_name"} <= data.keys()
        assert data["model_loaded"] is True

    def test_metrics_endpoint_accessible(self):
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "total_requests" in response.json()


class TestDecodeImage:
    @staticmethod
    def _image_b64(fmt="JPEG"):
        image = Image.new("RGB", (10, 10), color=(255, 255, 255))
        buffer = io.BytesIO()
        image.save(buffer, format=fmt)
        return base64.b64encode(buffer.getvalue()).decode()

    def test_returns_numpy_array(self):
        assert isinstance(_decode_image(self._image_b64()), np.ndarray)

    def test_correct_shape(self):
        assert _decode_image(self._image_b64()).shape == (10, 10, 3)

    def test_png_format(self):
        assert _decode_image(self._image_b64("PNG")).shape == (10, 10, 3)

    def test_invalid_base64_raises(self):
        with pytest.raises(Exception):
            _decode_image("dado_invalido_nao_e_base64")


class TestPredictEndpoint:
    @pytest.fixture
    def zidane_b64(self):
        return base64.b64encode((ASSETS / "zidane.jpg").read_bytes()).decode()

    def test_predict_returns_200(self, zidane_b64):
        response = client.post("/predict", json={"image_base64": zidane_b64})
        assert response.status_code == 200

    def test_predict_detects_objects(self, zidane_b64):
        data = client.post("/predict", json={"image_base64": zidane_b64}).json()
        assert len(data["detections"]) >= 1

    def test_predict_response_schema(self, zidane_b64):
        data = client.post("/predict", json={"image_base64": zidane_b64}).json()
        expected = {"detections", "inference_ms", "model_used", "image_width", "image_height"}
        assert expected <= data.keys()

    def test_predict_detection_fields(self, zidane_b64):
        response = client.post("/predict", json={"image_base64": zidane_b64})
        detection = response.json()["detections"][0]
        assert {"label", "confidence", "bbox"} <= detection.keys()
        assert len(detection["bbox"]) == 4

    def test_predict_missing_input_returns_422(self):
        assert client.post("/predict", json={}).status_code == 422


class TestBatchEndpoint:
    @pytest.fixture
    def two_images_b64(self):
        encoded = base64.b64encode((ASSETS / "zidane.jpg").read_bytes()).decode()
        return [encoded, encoded]

    def test_batch_returns_correct_count(self, two_images_b64):
        response = client.post("/predict/batch", json={"images_base64": two_images_b64})
        assert response.status_code == 200
        assert len(response.json()["results"]) == 2

    def test_batch_total_ms_is_positive(self, two_images_b64):
        data = client.post("/predict/batch", json={"images_base64": two_images_b64}).json()
        assert data["total_inference_ms"] > 0
