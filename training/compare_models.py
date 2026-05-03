from __future__ import annotations

import argparse
import csv
from dataclasses import replace
from pathlib import Path

from app.modeling import SUPPORTED_MODELS
from training.plots import plot_model_comparison
from training.train import TrainConfig, run_training


DEFAULT_MODEL_ORDER = ["resnet50", "resnet50v2", "alexnet", "vgg19"]


def parse_models(raw: str) -> list[str]:
    models = [m.strip().lower() for m in raw.split(",") if m.strip()]
    if not models:
        raise ValueError("--models must not be empty.")
    unknown = [m for m in models if m not in SUPPORTED_MODELS]
    if unknown:
        raise ValueError(f"Unknown models in --models: {unknown}. Allowed: {sorted(SUPPORTED_MODELS)}")
    # Keep declared order but remove duplicates.
    seen: set[str] = set()
    unique: list[str] = []
    for m in models:
        if m not in seen:
            unique.append(m)
            seen.add(m)
    return unique


def is_windows_shared_mapping_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "couldn't open shared file mapping" in text or "error code: <1455>" in text


def parse_args():
    parser = argparse.ArgumentParser(description="Compare ResNet50 / ResNet50v2 / AlexNet / VGG19.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--models",
        default=",".join(DEFAULT_MODEL_ORDER),
        help=f"Comma-separated models to run. Allowed: {', '.join(sorted(SUPPORTED_MODELS))}",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--uniform-hparams", action="store_true", help="Use same optimizer/lr settings for all models.")
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--class-weight-mode", default="inverse_frequency", choices=["none", "inverse_frequency"])
    parser.add_argument("--target-specificity", type=float, default=0.85)
    parser.add_argument("--loss-mode", default="cross_entropy", choices=["cross_entropy", "focal"])
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--scheduler", default="none", choices=["none", "cosine", "plateau", "onecycle"])
    parser.add_argument("--scheduler-t-max", type=int, default=0)
    parser.add_argument("--scheduler-eta-min", type=float, default=1e-6)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=2)
    parser.add_argument(
        "--selection-metric",
        default="auc",
        choices=[
            "accuracy",
            "auc",
            "pr_auc",
            "precision",
            "recall",
            "f1",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "selected_sensitivity",
            "selected_specificity",
            "selected_precision",
            "selected_f1",
            "selected_balanced_accuracy",
            "loss",
            "error",
        ],
    )
    parser.add_argument(
        "--early-stopping-metric",
        default="pr_auc",
        choices=[
            "accuracy",
            "auc",
            "pr_auc",
            "precision",
            "recall",
            "f1",
            "balanced_accuracy",
            "sensitivity",
            "specificity",
            "selected_sensitivity",
            "selected_specificity",
            "selected_precision",
            "selected_f1",
            "selected_balanced_accuracy",
            "loss",
            "error",
        ],
    )
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_models = parse_models(args.models)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initial settings aligned with your article/presentation logic.
    model_defaults: dict[str, dict[str, float | str]] = {
        "resnet50": {"optimizer": "sgd", "lr": 0.01, "momentum": 0.9},
        "resnet50v2": {"optimizer": "sgd", "lr": 0.01, "momentum": 0.9},
        "alexnet": {"optimizer": "sgd", "lr": 0.003, "momentum": 0.9},
        "vgg19": {"optimizer": "adamw", "lr": 0.0001, "momentum": 0.9},
    }
    configs: list[TrainConfig] = []
    for model_name in selected_models:
        defaults = model_defaults[model_name]
        optimizer = args.optimizer if args.uniform_hparams else str(defaults["optimizer"])
        lr = args.lr if args.uniform_hparams else float(defaults["lr"])
        momentum = args.momentum if args.uniform_hparams else float(defaults["momentum"])
        configs.append(
            TrainConfig(
                data_dir=args.data_dir,
                model_name=model_name,
                epochs=args.epochs,
                batch_size=args.batch_size,
                image_size=args.image_size,
                lr=lr,
                momentum=momentum,
                weight_decay=args.weight_decay,
                optimizer=optimizer,
                device=args.device,
                num_workers=args.num_workers,
                output_path=str(output_dir / f"{model_name}_best.pt"),
                log_every=args.log_every,
                max_train_batches=args.max_train_batches,
                max_val_batches=args.max_val_batches,
                class_weight_mode=args.class_weight_mode,
                target_specificity=args.target_specificity,
                loss_mode=args.loss_mode,
                focal_gamma=args.focal_gamma,
                scheduler=args.scheduler,
                scheduler_t_max=args.scheduler_t_max,
                scheduler_eta_min=args.scheduler_eta_min,
                scheduler_factor=args.scheduler_factor,
                scheduler_patience=args.scheduler_patience,
                selection_metric=args.selection_metric,
                early_stopping_metric=args.early_stopping_metric,
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_min_delta=args.early_stopping_min_delta,
            )
        )

    rows: list[dict[str, float | str]] = []
    for config in configs:
        print(
            f"Starting model comparison run: model={config.model_name}, "
            f"device={config.device}, epochs={config.epochs}"
        )
        try:
            _, metrics = run_training(config)
        except RuntimeError as exc:
            if is_windows_shared_mapping_error(exc) and config.num_workers > 0:
                print(
                    f"[WARN] Shared mapping error for model={config.model_name} with num_workers={config.num_workers}. "
                    "Retrying automatically with num_workers=0."
                )
                fallback_config = replace(config, num_workers=0)
                _, metrics = run_training(fallback_config)
            else:
                raise
        rows.append(
            {
                "model": config.model_name,
                "optimizer": config.optimizer,
                "lr": config.lr,
                "epochs": config.epochs,
                "accuracy": metrics.get("accuracy", float("nan")),
                "auc": metrics.get("auc", float("nan")),
                "pr_auc": metrics.get("pr_auc", float("nan")),
                "balanced_accuracy": metrics.get("balanced_accuracy", float("nan")),
                "precision": metrics.get("precision", float("nan")),
                "recall": metrics.get("recall", float("nan")),
                "f1": metrics.get("f1", float("nan")),
                "sensitivity": metrics.get("sensitivity", float("nan")),
                "specificity": metrics.get("specificity", float("nan")),
                "selected_threshold": metrics.get("selected_threshold", float("nan")),
                "selected_sensitivity": metrics.get("selected_sensitivity", float("nan")),
                "selected_specificity": metrics.get("selected_specificity", float("nan")),
                "selected_f1": metrics.get("selected_f1", float("nan")),
                "loss": metrics.get("loss", float("nan")),
                "train_seconds": metrics.get("train_seconds", float("nan")),
            }
        )

    summary_path = output_dir / "comparison_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved comparison table: {summary_path}")

    comparison_plot_path = output_dir / "model_comparison_metrics.png"
    plot_model_comparison(rows, comparison_plot_path)
    print(f"Saved model comparison plot: {comparison_plot_path}")


if __name__ == "__main__":
    main()
