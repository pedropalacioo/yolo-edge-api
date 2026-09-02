import base64
import io
import json
import time
import uuid

import httpx
import numpy as np
from fastapi import FastAPI, HTTPException, Response
from PIL import Image

from app.model import get_default_model_name, load_model
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    Detection,
    HealthResponse,
    MetricsResponse,
    PredictRequest,
    PredictResponse,
)

app = FastAPI(
    title="YOLO Inference API",
    description="API REST para inferência com YOLOv8 no Raspberry Pi 5",
    version="1.0.0",
)


def log_event(event: str, level: str = "INFO", **kwargs):
    """Emite um evento estruturado em JSON para stdout."""
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "level": level,
        "event": event,
        **kwargs,
    }
    print(json.dumps(record, ensure_ascii=False), flush=True)


# ── Métricas simples em memória ─────────────────────────────

_metrics = {
    "total": 0,
    "success": 0,
    "total_ms": 0.0,
}


# ── Funções auxiliares ──────────────────────────────────────

def _decode_image(image_base64: str) -> np.ndarray:
    """Converte base64 → numpy array RGB."""

    raw = base64.b64decode(image_base64)
    img = Image.open(io.BytesIO(raw)).convert("RGB")

    return np.array(img)


def _load_image_from_request(request: PredictRequest) -> np.ndarray:
    """Lê a imagem a partir de Base64 ou URL pública sempre em RGB."""

    if not request.image_base64 and not request.image_url:
        raise HTTPException(
            status_code=422,
            detail="Forneça image_base64 ou image_url.",
        )

    try:
        if request.image_base64:
            return _decode_image(request.image_base64)

        response = httpx.get(
            request.image_url,
            timeout=10.0,
            follow_redirects=True,
        )
        response.raise_for_status()

        img = Image.open(
            io.BytesIO(response.content)
        ).convert("RGB")

        return np.array(img)

    except HTTPException:
        raise

    except Exception as e:  # noqa: BLE001 - converte falhas de leitura em HTTP 400
        raise HTTPException(
            status_code=400,
            detail=f"Erro ao carregar imagem: {e!s}",
        )


def _run_inference(
    image: np.ndarray,
    model_name: str,
    confidence: float,
) -> PredictResponse:
    """Executa inferência YOLO e converte o resultado para o schema da API."""

    model = load_model(model_name)

    start = time.perf_counter()

    results = model.predict(
        source=image,
        conf=confidence,
        verbose=False,
    )

    inference_ms = (
        time.perf_counter() - start
    ) * 1000

    detections = []

    for result in results:
        if result.boxes is None:
            continue

        for box in result.boxes:
            class_id = int(box.cls[0].item())

            label = model.names[class_id]

            conf = float(box.conf[0].item())

            bbox = [
                float(value)
                for value in box.xyxy[0].tolist()
            ]

            detections.append(
                Detection(
                    label=label,
                    confidence=round(conf, 4),
                    bbox=bbox,
                )
            )

    return PredictResponse(
        detections=detections,
        inference_ms=round(inference_ms, 2),
        model_used=model_name,
        image_width=int(image.shape[1]),
        image_height=int(image.shape[0]),
    )


# ── Health Check ────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
)
def health():
    model_name = get_default_model_name()

    try:
        load_model(model_name)

        return HealthResponse(
            status="ok",
            model_loaded=True,
            model_name=model_name,
        )

    except Exception:  # noqa: BLE001 - health check deve reportar qualquer falha do modelo
        return HealthResponse(
            status="error",
            model_loaded=False,
            model_name=model_name,
        )


# ── Inferência individual ───────────────────────────────────

@app.post(
    "/predict",
    response_model=PredictResponse,
)
def predict(request: PredictRequest):
    request_id = str(uuid.uuid4())[:8]
    _metrics["total"] += 1

    log_event(
        "predict_start",
        request_id=request_id,
        model=request.model_name,
        confidence=request.confidence,
    )

    try:
        image = _load_image_from_request(request)

        result = _run_inference(
            image,
            request.model_name,
            request.confidence,
        )

        _metrics["success"] += 1
        _metrics["total_ms"] += result.inference_ms

        log_event(
            "predict_complete",
            request_id=request_id,
            model=result.model_used,
            detections=len(result.detections),
            inference_ms=result.inference_ms,
            image_size=f"{result.image_width}x{result.image_height}",
        )

        return result

    except HTTPException as e:
        log_event(
            "predict_error",
            level="WARN" if e.status_code < 500 else "ERROR",
            request_id=request_id,
            reason=str(e.detail),
        )
        raise

    except FileNotFoundError as e:
        log_event("predict_error", level="ERROR", request_id=request_id, reason=str(e))
        raise HTTPException(
            status_code=404,
            detail=str(e),
        )

    except Exception as e:  # noqa: BLE001 - fronteira da API registra falhas inesperadas
        log_event("predict_error", level="ERROR", request_id=request_id, reason=str(e))
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ── Inferência retornando imagem anotada ────────────────────

@app.post("/predict/image")
def predict_image(request: PredictRequest):

    try:
        image = _load_image_from_request(request)

        model = load_model(request.model_name)

        results = model.predict(
            source=image,
            conf=request.confidence,
            verbose=False,
        )

        annotated = results[0].plot()

        annotated_rgb = annotated[:, :, ::-1]

        output = Image.fromarray(annotated_rgb)

        buffer = io.BytesIO()

        output.save(
            buffer,
            format="JPEG",
        )

        return Response(
            content=buffer.getvalue(),
            media_type="image/jpeg",
        )

    except HTTPException:
        raise

    except Exception as e:  # noqa: BLE001 - converte falhas do YOLO/Pillow em HTTP 500
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )


# ── Inferência Batch ────────────────────────────────────────

@app.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
)
def predict_batch(request: BatchPredictRequest):

    t_total = time.perf_counter()

    results = []

    for img_b64 in request.images_base64:

        img = _decode_image(img_b64)

        results.append(
            _run_inference(
                img,
                request.model_name,
                request.confidence,
            )
        )

    total_ms = (
        time.perf_counter() - t_total
    ) * 1000

    return BatchPredictResponse(
        results=results,
        total_inference_ms=round(total_ms, 2),
    )


# ── Métricas ────────────────────────────────────────────────

@app.get(
    "/metrics",
    response_model=MetricsResponse,
)
async def get_metrics():

    avg = (
        _metrics["total_ms"] / _metrics["success"]
        if _metrics["success"] > 0
        else 0.0
    )

    return MetricsResponse(
        total_requests=_metrics["total"],
        successful_requests=_metrics["success"],
        avg_inference_ms=round(avg, 2),
    )
