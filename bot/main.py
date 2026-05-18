from __future__ import annotations

import asyncio
import os
from io import BytesIO
from typing import Literal

import httpx
from aiogram import Bot, Dispatcher, F
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiogram.client.default import DefaultBotProperties


Mode = Literal["checkpoint", "screening", "high_specificity"]


def _env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None or value == "":
        raise RuntimeError(f"Environment variable {name} is required.")
    return value


BOT_TOKEN = _env("TELEGRAM_BOT_TOKEN")
API_BASE_URL = _env("API_BASE_URL", "http://127.0.0.1:8000")
API_MODE: Mode = os.getenv("API_MODE", "checkpoint").strip().lower()  # type: ignore[assignment]
API_TIMEOUT = float(os.getenv("API_TIMEOUT_SECONDS", "60"))
ENABLE_API_AUTH = os.getenv("ENABLE_API_AUTH", "false").strip().lower() in {"1", "true", "yes", "on"}
API_TOKEN = os.getenv("API_TOKEN", "")

if API_MODE not in {"checkpoint", "screening", "high_specificity"}:
    raise RuntimeError("API_MODE must be one of: checkpoint, screening, high_specificity")


def _auth_headers() -> dict[str, str]:
    if not ENABLE_API_AUTH:
        return {}
    if not API_TOKEN:
        raise RuntimeError("ENABLE_API_AUTH=true but API_TOKEN is empty.")
    return {"Authorization": f"Bearer {API_TOKEN}"}


def _format_prediction(payload: dict) -> str:
    return (
        "Результат анализа:\n"
        f"- Класс: <b>{payload.get('class_label', 'n/a')}</b>\n"
        f"- Вероятность melanoma: <b>{payload.get('melanoma_probability', 0):.4f}</b>\n"
        f"- Confidence: <b>{payload.get('confidence', 0):.4f}</b>\n"
        f"- Uncertainty: <b>{payload.get('uncertainty', 0):.4f}</b>\n"
        f"- Использованный порог: <b>{payload.get('threshold_used', 0):.4f}</b>\n"
        f"- Режим: <b>{API_MODE}</b>"
    )


async def _predict_image(image_bytes: bytes, filename: str) -> dict:
    endpoint = f"{API_BASE_URL.rstrip('/')}/v1/predict"
    params = {"mode": API_MODE}
    files = {"image": (filename, image_bytes, "image/jpeg")}
    headers = _auth_headers()
    async with httpx.AsyncClient(timeout=API_TIMEOUT) as client:
        response = await client.post(endpoint, params=params, files=files, headers=headers)
    response.raise_for_status()
    return response.json()


async def _extract_image_bytes(bot: Bot, message: Message) -> tuple[bytes, str]:
    if message.photo:
        file_id = message.photo[-1].file_id
        file = await bot.get_file(file_id)
        buff = BytesIO()
        await bot.download(file, destination=buff)
        return buff.getvalue(), f"{file_id}.jpg"

    if message.document and message.document.mime_type and message.document.mime_type.startswith("image/"):
        file_id = message.document.file_id
        file = await bot.get_file(file_id)
        buff = BytesIO()
        await bot.download(file, destination=buff)
        filename = message.document.file_name or f"{file_id}.jpg"
        return buff.getvalue(), filename

    raise ValueError("Пришлите изображение как photo или image/* документ.")


dp = Dispatcher()


@dp.message(CommandStart())
async def on_start(message: Message) -> None:
    await message.answer(
        "Пришлите фото родинки.\n"
        "Бот отправит изображение в FastAPI и вернет вероятность melanoma."
    )


@dp.message(F.photo | F.document)
async def on_image(message: Message, bot: Bot) -> None:
    status_msg = await message.answer("Обрабатываю изображение...")
    try:
        image_bytes, filename = await _extract_image_bytes(bot, message)
        payload = await _predict_image(image_bytes, filename)
        await status_msg.edit_text(_format_prediction(payload), parse_mode=ParseMode.HTML)
    except httpx.HTTPStatusError as exc:
        body = exc.response.text[:500] if exc.response is not None else str(exc)
        await status_msg.edit_text(f"Ошибка API: {body}")
    except Exception as exc:
        await status_msg.edit_text(f"Ошибка: {exc}")


@dp.message()
async def on_other(message: Message) -> None:
    await message.answer("Отправьте изображение (photo или image-файл).")


async def main() -> None:
    bot = Bot(token=BOT_TOKEN, default=DefaultBotProperties(parse_mode=ParseMode.HTML))
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

