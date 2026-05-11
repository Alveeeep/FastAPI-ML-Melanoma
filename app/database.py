from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import JSON, Boolean, DateTime, Float, Integer, String
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.settings import Settings


class Base(DeclarativeBase):
    pass


class PredictionRequestLog(Base):
    __tablename__ = "prediction_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(UTC))
    endpoint: Mapped[str] = mapped_column(String(64), nullable=False)
    mode: Mapped[str] = mapped_column(String(32), nullable=False)
    filename: Mapped[str | None] = mapped_column(String(512), nullable=True)
    content_type: Mapped[str | None] = mapped_column(String(128), nullable=True)
    image_size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    request_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    response_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    model_name: Mapped[str] = mapped_column(String(128), nullable=False)
    class_label: Mapped[str] = mapped_column(String(32), nullable=False)
    melanoma_probability: Mapped[float] = mapped_column(Float, nullable=False)
    threshold_used: Mapped[float] = mapped_column(Float, nullable=False)
    risk_band: Mapped[str] = mapped_column(String(32), nullable=False)
    needs_review: Mapped[bool] = mapped_column(Boolean, nullable=False)
    processing_ms: Mapped[float] = mapped_column(Float, nullable=False)


_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker | None = None


def configure_database(settings: Settings) -> None:
    global _engine, _session_factory
    if not settings.database_url:
        return
    _engine = create_async_engine(str(settings.database_url), echo=settings.database_echo)
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)


async def init_database() -> None:
    if _engine is None:
        return
    async with _engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def close_database() -> None:
    global _engine, _session_factory
    if _engine is None:
        return
    await _engine.dispose()
    _engine = None
    _session_factory = None


def database_enabled() -> bool:
    return _session_factory is not None


async def log_prediction_request(
    *,
    endpoint: str,
    mode: str,
    filename: str | None,
    content_type: str | None,
    image_size_bytes: int | None,
    image_sha256: str | None,
    request_payload: dict[str, Any],
    response_payload: dict[str, Any],
    model_name: str,
    class_label: str,
    melanoma_probability: float,
    threshold_used: float,
    risk_band: str,
    needs_review: bool,
    processing_ms: float,
) -> None:
    if _session_factory is None:
        return

    row = PredictionRequestLog(
        endpoint=endpoint,
        mode=mode,
        filename=filename,
        content_type=content_type,
        image_size_bytes=image_size_bytes,
        image_sha256=image_sha256,
        request_payload=request_payload,
        response_payload=response_payload,
        model_name=model_name,
        class_label=class_label,
        melanoma_probability=melanoma_probability,
        threshold_used=threshold_used,
        risk_band=risk_band,
        needs_review=needs_review,
        processing_ms=processing_ms,
    )
    try:
        async with _session_factory() as session:
            session.add(row)
            await session.commit()
    except SQLAlchemyError:
        # Request logging must not break medical-image inference.
        return
