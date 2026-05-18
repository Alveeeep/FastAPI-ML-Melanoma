from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from app.modeling import SUPPORTED_MODELS, create_model
from training.dataset import create_dataloaders
from training.metrics import classification_metrics, optimize_threshold
from training.plots import (
    plot_confusion_matrix_heatmap,
    plot_probability_histogram,
    plot_roc,
)


@torch.inference_mode()
def run_inference(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    positive_class_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_targets: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)[:, positive_class_index]
        all_targets.append(targets.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    y_true_raw = np.concatenate(all_targets)
    y_true = (y_true_raw == int(positive_class_index)).astype(np.int64)
    y_prob = np.concatenate(all_probs)
    return y_true, y_prob


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final validation of a trained checkpoint on val split.")
    parser.add_argument("--data-dir", required=True, help="Prepared dataset root with train/val folders.")
    parser.add_argument("--weights-path", required=True, help="Path to *.pt checkpoint.")
    parser.add_argument("--model-name", default=None, choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target-specificity", type=float, default=0.85)
    parser.add_argument("--output-dir", default="outputs/final_validation")
    parser.add_argument("--output-stem", default=None, help="Output file stem. Defaults to '<model>_external_val'.")
    parser.add_argument("--plot-suffix", default="", help="Suffix for generated plot and metrics files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use --device cpu or install CUDA-enabled torch.")

    ckpt_path = Path(args.weights_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    checkpoint_model_name = checkpoint.get("model_name")
    model_name = args.model_name or checkpoint_model_name
    if not model_name:
        raise ValueError("model_name is missing. Pass --model-name explicitly.")
    model_name = str(model_name).lower()
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{model_name}'. Allowed: {sorted(SUPPORTED_MODELS)}")

    image_size = int(args.image_size or checkpoint.get("image_size", 224))
    class_to_idx = checkpoint.get("class_to_idx") or {"benign": 0, "melanoma": 1}
    positive_class_index = int(class_to_idx.get("melanoma", 1))
    decision_threshold = float(checkpoint.get("selected_threshold", 0.5))
    if not (0.0 <= decision_threshold <= 1.0):
        decision_threshold = 0.5

    device = torch.device(args.device)
    _, val_loader, val_class_to_idx = create_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=image_size,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )
    if set(val_class_to_idx.keys()) != {"benign", "melanoma"}:
        raise ValueError(
            f"Validation dataset must have benign/melanoma class folders. Found: {sorted(val_class_to_idx.keys())}"
        )

    model = create_model(model_name, num_classes=2).to(device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    y_true, y_prob = run_inference(
        model=model,
        loader=val_loader,
        device=device,
        positive_class_index=positive_class_index,
    )
    fixed_metrics = classification_metrics(y_true, y_prob, threshold=decision_threshold)
    tuned_metrics = optimize_threshold(y_true, y_prob, target_specificity=float(args.target_specificity))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = str(args.output_stem or f"{model_name}_external_val")
    file_stem = f"{stem}{args.plot_suffix}"

    cm_path = output_dir / f"{file_stem}_confusion_matrix_heatmap.png"
    roc_path = output_dir / f"{file_stem}_roc_curve.png"
    probs_path = output_dir / f"{file_stem}_probability_histogram.png"
    plot_confusion_matrix_heatmap(y_true, y_prob, cm_path, threshold=decision_threshold)
    plot_roc(y_true, y_prob, roc_path)
    plot_probability_histogram(y_true, y_prob, probs_path)

    payload = {
        "weights_path": str(ckpt_path),
        "data_dir": str(Path(args.data_dir)),
        "model_name": model_name,
        "image_size": image_size,
        "class_to_idx": class_to_idx,
        "decision_threshold_from_checkpoint": decision_threshold,
        "target_specificity_for_tuning": float(args.target_specificity),
        "val_samples": int(y_true.shape[0]),
        "metrics_at_checkpoint_threshold": fixed_metrics,
        "metrics_with_threshold_optimized_on_this_validation": tuned_metrics,
        "artifacts": {
            "confusion_matrix_heatmap_png": str(cm_path),
            "roc_curve_png": str(roc_path),
            "probability_histogram_png": str(probs_path),
        },
    }

    metrics_path = output_dir / f"{file_stem}.metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved final-validation metrics: {metrics_path}")
    print(f"Saved confusion matrix heatmap: {cm_path}")
    print(f"Saved ROC curve: {roc_path}")
    print(f"Saved probability histogram: {probs_path}")


if __name__ == "__main__":
    main()
