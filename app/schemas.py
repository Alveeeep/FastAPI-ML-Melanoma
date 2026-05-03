from __future__ import annotations

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    model_name: str


class ModelInfoResponse(BaseModel):
    model_name: str
    classes: list[str]
    image_size: int
    tta_runs: int
    decision_threshold: float = Field(ge=0.0, le=1.0)
    screening_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    high_specificity_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    weights_path: str
    loaded: bool


class PredictionResponse(BaseModel):
    class_label: str
    melanoma_probability: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    uncertainty: float = Field(ge=0.0, le=1.0)
    threshold_used: float = Field(ge=0.0, le=1.0)


class PredictionWithHeatmapResponse(PredictionResponse):
    heatmap_base64_png: str
