from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _to_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    app_name: str
    app_version: str
    model_name: str
    model_weights_path: Path
    image_size: int
    device: str
    tta_runs: int
    screening_threshold: float | None
    high_specificity_threshold: float | None
    enable_auth: bool
    api_token: str


def _to_optional_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


def get_settings() -> Settings:
    return Settings(
        app_name=os.getenv("APP_NAME", "melanoma-cds-service"),
        app_version=os.getenv("APP_VERSION", "0.1.0"),
        model_name=os.getenv("MODEL_NAME", "resnet50"),
        model_weights_path=Path(os.getenv("MODEL_WEIGHTS_PATH", "models/resnet50_best.pt")),
        image_size=int(os.getenv("IMAGE_SIZE", "224")),
        device=os.getenv("DEVICE", "cpu"),
        tta_runs=int(os.getenv("TTA_RUNS", "4")),
        screening_threshold=_to_optional_float(os.getenv("SCREENING_THRESHOLD")),
        high_specificity_threshold=_to_optional_float(os.getenv("HIGH_SPECIFICITY_THRESHOLD")),
        enable_auth=_to_bool(os.getenv("ENABLE_AUTH"), default=False),
        api_token=os.getenv("API_TOKEN", "change-me"),
    )
