from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    image_base64: str | None = Field(
        None,
        description="Imagem codificada em base64"
    )

    image_url: str | None = Field(
        None,
        description="URL pública da imagem"
    )

    confidence: float = Field(
        0.25,
        ge=0.0,
        le=1.0
    )

    model_name: str = Field(
        "yolov8n.pt"
    )


class Detection(BaseModel):
    label: str
    confidence: float
    bbox: list[float]


class PredictResponse(BaseModel):
    detections: list[Detection]
    inference_ms: float
    model_used: str
    image_width: int
    image_height: int


class BatchPredictRequest(BaseModel):
    images_base64: list[str]

    confidence: float = Field(
        0.25,
        ge=0.0,
        le=1.0
    )

    model_name: str = Field(
        "yolov8n.pt"
    )


class BatchPredictResponse(BaseModel):
    results: list[PredictResponse]
    total_inference_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str


class MetricsResponse(BaseModel):
    total_requests: int
    successful_requests: int
    avg_inference_ms: float
