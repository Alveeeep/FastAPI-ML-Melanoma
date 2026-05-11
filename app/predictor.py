from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from app.explainability import heatmap_overlay_base64, uncertainty_from_probabilities
from app.modeling import create_model
from app.settings import Settings


CLASS_NAMES = ["benign", "melanoma"]


@dataclass
class PredictionResult:
    class_label: str
    melanoma_probability: float
    confidence: float
    uncertainty: float
    threshold_used: float
    risk_band: str
    needs_review: bool


class MelanomaPredictor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.device = torch.device(settings.device)
        self.model: nn.Module | None = None
        self.loaded = False
        self.weights_path = Path(settings.model_weights_path)
        self.transform = T.Compose(
            [
                T.Resize((settings.image_size, settings.image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.class_names = CLASS_NAMES.copy()
        self.melanoma_index = 1
        self.decision_threshold = 0.5
        self.review_margin = max(0.0, float(settings.review_margin))

    def load(self) -> None:
        model = create_model(self.settings.model_name, num_classes=2)
        if not self.weights_path.exists():
            self.model = model.to(self.device).eval()
            self.loaded = False
            return

        checkpoint: dict[str, Any] = torch.load(self.weights_path, map_location=self.device)
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint

        class_to_idx = checkpoint.get("class_to_idx")
        if isinstance(class_to_idx, dict) and class_to_idx:
            idx_to_class = sorted(class_to_idx.items(), key=lambda kv: kv[1])
            self.class_names = [cls for cls, _ in idx_to_class]
            if "melanoma" in class_to_idx:
                self.melanoma_index = int(class_to_idx["melanoma"])
            else:
                self.melanoma_index = min(1, len(self.class_names) - 1)
        if "selected_threshold" in checkpoint:
            try:
                threshold = float(checkpoint["selected_threshold"])
                if 0.0 <= threshold <= 1.0:
                    self.decision_threshold = threshold
            except (TypeError, ValueError):
                pass

        model.load_state_dict(state_dict, strict=True)
        self.model = model.to(self.device).eval()
        self.loaded = True

    def _ensure_ready(self) -> nn.Module:
        if self.model is None:
            raise RuntimeError("Model is not initialized.")
        return self.model

    def _prepare(self, image: Image.Image) -> torch.Tensor:
        rgb = image.convert("RGB")
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        return tensor

    def _tta_batch(self, x: torch.Tensor) -> torch.Tensor:
        variants = [x]
        if self.settings.tta_runs > 1:
            variants.append(torch.flip(x, dims=[3]))
        if self.settings.tta_runs > 2:
            variants.append(torch.flip(x, dims=[2]))
        if self.settings.tta_runs > 3:
            variants.append(torch.rot90(x, k=1, dims=[2, 3]))
        if self.settings.tta_runs > 4:
            variants.append(torch.rot90(x, k=3, dims=[2, 3]))
        variants = variants[: self.settings.tta_runs]
        return torch.cat(variants, dim=0)

    @torch.inference_mode()
    def predict(self, image: Image.Image, threshold_override: float | None = None) -> PredictionResult:
        model = self._ensure_ready()
        x = self._prepare(image)
        tta_batch = self._tta_batch(x)
        logits = model(tta_batch)
        probs = torch.softmax(logits, dim=1).mean(dim=0).cpu().numpy()

        uncertainty = uncertainty_from_probabilities(probs)
        melanoma_probability = float(probs[self.melanoma_index])
        threshold = self.decision_threshold if threshold_override is None else float(threshold_override)
        class_idx = self.melanoma_index if melanoma_probability >= threshold else 1 - self.melanoma_index
        confidence = melanoma_probability if class_idx == self.melanoma_index else (1.0 - melanoma_probability)
        delta = melanoma_probability - threshold
        if delta >= self.review_margin:
            risk_band = "high"
        elif delta <= -self.review_margin:
            risk_band = "low"
        else:
            risk_band = "borderline"
        needs_review = risk_band == "borderline"

        return PredictionResult(
            class_label=self.class_names[class_idx],
            melanoma_probability=melanoma_probability,
            confidence=confidence,
            uncertainty=uncertainty,
            threshold_used=threshold,
            risk_band=risk_band,
            needs_review=needs_review,
        )

    def _find_last_conv(self, model: nn.Module) -> nn.Module:
        conv = None
        for layer in model.modules():
            if isinstance(layer, nn.Conv2d):
                conv = layer
        if conv is None:
            raise RuntimeError("No Conv2d layer found for Grad-CAM.")
        return conv

    def gradcam_overlay(self, image: Image.Image, target_class: int | None = None) -> str:
        model = self._ensure_ready()
        model.eval()

        x = self._prepare(image)
        x = x.requires_grad_(True)

        activations = {}
        gradients = {}
        target_layer = self._find_last_conv(model)

        def _forward_hook(_: nn.Module, __: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            activations["value"] = output

        def _backward_hook(
            _: nn.Module,
            __: tuple[torch.Tensor | None, ...],
            grad_output: tuple[torch.Tensor, ...],
        ) -> None:
            gradients["value"] = grad_output[0]

        h1 = target_layer.register_forward_hook(_forward_hook)
        h2 = target_layer.register_full_backward_hook(_backward_hook)
        try:
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            cls = int(torch.argmax(probs, dim=1).item()) if target_class is None else int(target_class)
            score = logits[:, cls].sum()
            model.zero_grad(set_to_none=True)
            score.backward(retain_graph=False)

            acts = activations["value"]
            grads = gradients["value"]
            weights = grads.mean(dim=(2, 3), keepdim=True)
            cam = (weights * acts).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=(self.settings.image_size, self.settings.image_size), mode="bilinear")
            cam = cam.squeeze().detach().cpu().numpy()
            cam = cam - cam.min()
            denom = cam.max() - cam.min() + 1e-8
            cam = cam / denom
        finally:
            h1.remove()
            h2.remove()

        rendered_image = image.convert("RGB").resize((self.settings.image_size, self.settings.image_size))
        return heatmap_overlay_base64(rendered_image, cam)
