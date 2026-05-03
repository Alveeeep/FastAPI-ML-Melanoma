from __future__ import annotations

import base64
from io import BytesIO

import numpy as np
from PIL import Image


def uncertainty_from_probabilities(probabilities: np.ndarray) -> float:
    eps = 1e-12
    entropy = -np.sum(probabilities * np.log(probabilities + eps))
    max_entropy = np.log(probabilities.shape[-1])
    if max_entropy <= 0:
        return 0.0
    return float(np.clip(entropy / max_entropy, 0.0, 1.0))


def heatmap_overlay_base64(image_rgb: Image.Image, heatmap: np.ndarray) -> str:
    img = np.asarray(image_rgb, dtype=np.float32)
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Expected RGB image for heatmap overlay.")

    heat = np.clip(heatmap, 0.0, 1.0).astype(np.float32)
    heat = np.stack(
        [
            heat,  # red
            0.3 * (1.0 - heat),  # green
            1.0 - heat,  # blue
        ],
        axis=-1,
    )
    heat = heat * 255.0

    overlay = 0.65 * img + 0.35 * heat
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    out = Image.fromarray(overlay, mode="RGB")
    buf = BytesIO()
    out.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return encoded

