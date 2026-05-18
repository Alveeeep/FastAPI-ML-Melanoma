from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from app.modeling import SUPPORTED_MODELS, create_model
from training.dataset import create_dataloaders
from training.evaluate_checkpoint import run_inference
from training.metrics import (
    classification_metrics,
    optimize_threshold,
    optimize_threshold_for_sensitivity,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune API thresholds for screening and high_specificity modes on a validation dataset."
    )
    parser.add_argument("--data-dir", required=True, help="Prepared dataset root with train/val folders.")
    parser.add_argument("--weights-path", required=True, help="Path to *.pt checkpoint.")
    parser.add_argument("--model-name", default=None, choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--target-sensitivity", type=float, default=0.80)
    parser.add_argument("--target-specificity", type=float, default=0.85)
    parser.add_argument("--output-path", default="outputs/api_thresholds.json")
    return parser.parse_args()


def _validate_target(name: str, value: float) -> None:
    if not (0.0 < value <= 1.0):
        raise ValueError(f"{name} must be in range (0, 1].")


def main() -> None:
    args = parse_args()
    _validate_target("target-sensitivity", float(args.target_sensitivity))
    _validate_target("target-specificity", float(args.target_specificity))
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Use --device cpu or install CUDA-enabled torch.")

    ckpt_path = Path(args.weights_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model_name = str(args.model_name or checkpoint.get("model_name") or "").lower()
    if not model_name:
        raise ValueError("model_name is missing. Pass --model-name explicitly.")
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{model_name}'. Allowed: {sorted(SUPPORTED_MODELS)}")

    image_size = int(args.image_size or checkpoint.get("image_size", 224))
    class_to_idx = checkpoint.get("class_to_idx") or {"benign": 0, "melanoma": 1}
    positive_class_index = int(class_to_idx.get("melanoma", 1))
    checkpoint_threshold = float(checkpoint.get("selected_threshold", 0.5))
    if not (0.0 <= checkpoint_threshold <= 1.0):
        checkpoint_threshold = 0.5

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

    y_true, y_prob = run_inference(
        model=model,
        loader=val_loader,
        device=device,
        positive_class_index=positive_class_index,
    )

    checkpoint_metrics = classification_metrics(y_true, y_prob, threshold=checkpoint_threshold)
    screening_metrics = optimize_threshold_for_sensitivity(
        y_true,
        y_prob,
        target_sensitivity=float(args.target_sensitivity),
    )
    high_specificity_metrics = optimize_threshold(
        y_true,
        y_prob,
        target_specificity=float(args.target_specificity),
    )

    payload = {
        "weights_path": str(ckpt_path),
        "data_dir": str(Path(args.data_dir)),
        "model_name": model_name,
        "image_size": image_size,
        "class_to_idx": class_to_idx,
        "val_samples": int(y_true.shape[0]),
        "checkpoint": {
            "threshold_source": "checkpoint selected_threshold",
            "threshold": checkpoint_threshold,
            "metrics": checkpoint_metrics,
        },
        "screening": {
            "threshold_source": f"optimized for sensitivity >= {float(args.target_sensitivity):.4f}",
            "env_name": "SCREENING_THRESHOLD",
            "metrics": screening_metrics,
        },
        "high_specificity": {
            "threshold_source": f"optimized for specificity >= {float(args.target_specificity):.4f}",
            "env_name": "HIGH_SPECIFICITY_THRESHOLD",
            "metrics": high_specificity_metrics,
        },
        "recommended_env": {
            "SCREENING_THRESHOLD": screening_metrics.get("threshold"),
            "HIGH_SPECIFICITY_THRESHOLD": high_specificity_metrics.get("threshold"),
        },
    }

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Saved threshold tuning report: {output_path}")
    print("Recommended .env values:")
    print(f"SCREENING_THRESHOLD={screening_metrics.get('threshold')}")
    print(f"HIGH_SPECIFICITY_THRESHOLD={high_specificity_metrics.get('threshold')}")
    print()
    print("Screening metrics:")
    print(
        "threshold={threshold:.6f} sensitivity={sensitivity:.4f} "
        "specificity={specificity:.4f} precision={precision:.4f} f1={f1:.4f}".format(**screening_metrics)
    )
    print("High-specificity metrics:")
    print(
        "threshold={threshold:.6f} sensitivity={sensitivity:.4f} "
        "specificity={specificity:.4f} precision={precision:.4f} f1={f1:.4f}".format(**high_specificity_metrics)
    )


if __name__ == "__main__":
    main()
