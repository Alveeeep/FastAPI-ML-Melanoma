from __future__ import annotations

import hashlib
from io import BytesIO
from time import perf_counter
from typing import Literal

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from PIL import Image, UnidentifiedImageError

from app.database import (
    close_database,
    configure_database,
    database_enabled,
    init_database,
    log_prediction_request,
)
from app.predictor import MelanomaPredictor
from app.schemas import (
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    PredictionWithHeatmapResponse,
)
from app.security import auth_dependency
from app.settings import get_settings

settings = get_settings()
configure_database(settings)
predictor = MelanomaPredictor(settings)

app = FastAPI(title=settings.app_name, version=settings.app_version)


@app.on_event("startup")
async def _startup() -> None:
    predictor.load()
    await init_database()


@app.on_event("shutdown")
async def _shutdown() -> None:
    await close_database()


async def _read_upload_to_image(upload: UploadFile) -> tuple[Image.Image, bytes]:
    if upload.content_type and not upload.content_type.startswith("image/"):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file must be an image.")
    content = await upload.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Empty file.")
    try:
        return Image.open(BytesIO(content)).convert("RGB"), content
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Could not parse image.") from exc


def _ready_or_503() -> None:
    if not predictor.loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model weights are not loaded. Expected file: {predictor.weights_path}",
        )


def _resolve_threshold(mode: str) -> float:
    if mode == "checkpoint":
        return predictor.decision_threshold
    if mode == "screening":
        if settings.screening_threshold is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="SCREENING_THRESHOLD is not configured.",
            )
        return float(settings.screening_threshold)
    if mode == "high_specificity":
        if settings.high_specificity_threshold is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="HIGH_SPECIFICITY_THRESHOLD is not configured.",
            )
        return float(settings.high_specificity_threshold)
    raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"Unknown mode '{mode}'.")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        model_loaded=predictor.loaded,
        model_name=settings.model_name,
        database_enabled=database_enabled(),
    )


@app.get("/v1/model-info", response_model=ModelInfoResponse)
def model_info(_: None = Depends(auth_dependency(settings))) -> ModelInfoResponse:
    return ModelInfoResponse(
        model_name=settings.model_name,
        classes=predictor.class_names,
        image_size=settings.image_size,
        tta_runs=settings.tta_runs,
        decision_threshold=predictor.decision_threshold,
        screening_threshold=settings.screening_threshold,
        high_specificity_threshold=settings.high_specificity_threshold,
        weights_path=str(settings.model_weights_path),
        loaded=predictor.loaded,
        database_enabled=database_enabled(),
    )


@app.post("/v1/predict", response_model=PredictionResponse)
async def predict(
    image: UploadFile = File(...),
    mode: Literal["checkpoint", "screening", "high_specificity"] = "checkpoint",
    _: None = Depends(auth_dependency(settings)),
) -> PredictionResponse:
    _ready_or_503()
    started = perf_counter()
    pil_image, content = await _read_upload_to_image(image)
    threshold = _resolve_threshold(mode)
    result = predictor.predict(pil_image, threshold_override=threshold)
    response = PredictionResponse(**result.__dict__)
    processing_ms = (perf_counter() - started) * 1000.0
    await log_prediction_request(
        endpoint="/v1/predict",
        mode=mode,
        filename=image.filename,
        content_type=image.content_type,
        image_size_bytes=len(content),
        image_sha256=hashlib.sha256(content).hexdigest(),
        request_payload={
            "mode": mode,
            "filename": image.filename,
            "content_type": image.content_type,
            "image_size_bytes": len(content),
        },
        response_payload=response.model_dump(),
        model_name=settings.model_name,
        class_label=response.class_label,
        melanoma_probability=response.melanoma_probability,
        threshold_used=response.threshold_used,
        risk_band=response.risk_band,
        needs_review=response.needs_review,
        processing_ms=processing_ms,
    )
    return response


@app.post("/v1/predict-with-heatmap", response_model=PredictionWithHeatmapResponse)
async def predict_with_heatmap(
    image: UploadFile = File(...),
    mode: Literal["checkpoint", "screening", "high_specificity"] = "checkpoint",
    _: None = Depends(auth_dependency(settings)),
) -> PredictionWithHeatmapResponse:
    _ready_or_503()
    started = perf_counter()
    pil_image, content = await _read_upload_to_image(image)
    threshold = _resolve_threshold(mode)
    result = predictor.predict(pil_image, threshold_override=threshold)
    heatmap = predictor.gradcam_overlay(pil_image)
    response = PredictionWithHeatmapResponse(**result.__dict__, heatmap_base64_png=heatmap)
    processing_ms = (perf_counter() - started) * 1000.0
    await log_prediction_request(
        endpoint="/v1/predict-with-heatmap",
        mode=mode,
        filename=image.filename,
        content_type=image.content_type,
        image_size_bytes=len(content),
        image_sha256=hashlib.sha256(content).hexdigest(),
        request_payload={
            "mode": mode,
            "filename": image.filename,
            "content_type": image.content_type,
            "image_size_bytes": len(content),
            "heatmap_requested": True,
        },
        response_payload=response.model_dump(exclude={"heatmap_base64_png"}) | {"heatmap_base64_png": "<base64 omitted>"},
        model_name=settings.model_name,
        class_label=response.class_label,
        melanoma_probability=response.melanoma_probability,
        threshold_used=response.threshold_used,
        risk_band=response.risk_band,
        needs_review=response.needs_review,
        processing_ms=processing_ms,
    )
    return response
