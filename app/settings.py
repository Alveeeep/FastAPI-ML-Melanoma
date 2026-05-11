from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (value.startswith('"') and value.endswith('"')) or (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        os.environ.setdefault(key, value)


_load_dotenv_file(Path(".env"))


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
    review_margin: float
    enable_auth: bool
    api_token: str
    database_url: str | None
    database_echo: bool


def _to_optional_float(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    return float(value)


def get_settings() -> Settings:
    return Settings(
        app_name=os.getenv("APP_NAME", "melanoma-cds-service"),
        app_version=os.getenv("APP_VERSION", "0.1.0"),
        model_name=os.getenv("MODEL_NAME", "resnet50v2"),
        model_weights_path=Path(os.getenv("MODEL_WEIGHTS_PATH", "models/resnet50v2_c5_baseline_best.pt")),
        image_size=int(os.getenv("IMAGE_SIZE", "224")),
        device=os.getenv("DEVICE", "cpu"),
        tta_runs=int(os.getenv("TTA_RUNS", "4")),
        screening_threshold=_to_optional_float(os.getenv("SCREENING_THRESHOLD")),
        high_specificity_threshold=_to_optional_float(os.getenv("HIGH_SPECIFICITY_THRESHOLD")),
        review_margin=float(os.getenv("REVIEW_MARGIN", "0.05")),
        enable_auth=_to_bool(os.getenv("ENABLE_AUTH"), default=False),
        api_token=os.getenv("API_TOKEN", "change-me"),
        database_url=os.getenv("DATABASE_URL") or None,
        database_echo=_to_bool(os.getenv("DATABASE_ECHO"), default=False),
    )
