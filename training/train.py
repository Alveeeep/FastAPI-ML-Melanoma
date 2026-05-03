from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from app.modeling import SUPPORTED_MODELS, create_model
from training.dataset import create_dataloaders
from training.metrics import classification_metrics, optimize_threshold
from training.plots import (
    plot_confusion_matrix_heatmap,
    plot_probability_histogram,
    plot_roc,
    plot_training_curves,
)


@dataclass
class TrainConfig:
    data_dir: str
    model_name: str = "resnet50"
    epochs: int = 20
    batch_size: int = 32
    image_size: int = 224
    lr: float = 0.003
    momentum: float = 0.9
    weight_decay: float = 1e-4
    optimizer: str = "sgd"
    device: str = "cpu"
    num_workers: int = 0
    output_path: str = "models/resnet50_best.pt"
    seed: int = 42
    log_every: int = 100
    max_train_batches: int = 0
    max_val_batches: int = 0
    class_weight_mode: str = "inverse_frequency"
    target_specificity: float = 0.85
    selection_metric: str = "auc"
    early_stopping_metric: str = "pr_auc"
    early_stopping_patience: int = 0
    early_stopping_min_delta: float = 0.0
    loss_mode: str = "cross_entropy"
    focal_gamma: float = 2.0
    scheduler: str = "none"
    scheduler_t_max: int = 0
    scheduler_eta_min: float = 1e-6
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_optimizer(config: TrainConfig, model: nn.Module):
    params = model.parameters()
    if config.optimizer.lower() == "adamw":
        return AdamW(params, lr=config.lr, weight_decay=config.weight_decay)
    return SGD(params, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = float(gamma)
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()


def build_criterion(config: TrainConfig, class_weights: torch.Tensor | None) -> nn.Module:
    mode = config.loss_mode.strip().lower()
    if mode == "cross_entropy":
        return nn.CrossEntropyLoss(weight=class_weights)
    if mode == "focal":
        return FocalLoss(gamma=config.focal_gamma, weight=class_weights)
    raise ValueError("Unsupported loss_mode. Allowed: cross_entropy, focal")


def build_scheduler(
    config: TrainConfig,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
):
    scheduler_name = config.scheduler.strip().lower()
    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        t_max = config.scheduler_t_max if config.scheduler_t_max > 0 else config.epochs
        return CosineAnnealingLR(optimizer, T_max=t_max, eta_min=config.scheduler_eta_min)
    if scheduler_name == "plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
        )
    if scheduler_name == "onecycle":
        total_steps = max(1, config.epochs * max(1, len(train_loader)))
        return OneCycleLR(optimizer, max_lr=config.lr, total_steps=total_steps)
    raise ValueError("Unsupported scheduler. Allowed: none, cosine, plateau, onecycle")


MAXIMIZE_METRICS = {
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
}
MINIMIZE_METRICS = {"loss", "error"}


def normalize_metric_name(metric: str) -> str:
    return metric.strip().lower()


def metric_direction(metric: str) -> str:
    name = normalize_metric_name(metric)
    if name in MAXIMIZE_METRICS:
        return "max"
    if name in MINIMIZE_METRICS:
        return "min"
    raise ValueError(
        f"Unsupported metric '{metric}'. Allowed max metrics: {sorted(MAXIMIZE_METRICS)}; "
        f"min metrics: {sorted(MINIMIZE_METRICS)}"
    )


def metric_value(metrics: dict[str, float], metric: str) -> float:
    value = metrics.get(normalize_metric_name(metric), float("nan"))
    return float(value)


def is_improved(current: float, best: float, mode: str, min_delta: float) -> bool:
    if np.isnan(current):
        return False
    if mode == "max":
        return current > (best + min_delta)
    return current < (best - min_delta)


def build_class_weights(
    class_to_idx: dict[str, int],
    train_loader: DataLoader,
    mode: str,
    device: torch.device,
) -> torch.Tensor | None:
    mode = mode.strip().lower()
    if mode == "none":
        return None
    if mode != "inverse_frequency":
        raise ValueError("Unsupported class_weight_mode. Allowed: none, inverse_frequency")

    dataset = train_loader.dataset
    targets = getattr(dataset, "targets", None)
    if targets is None:
        return None

    targets_np = np.asarray(targets, dtype=np.int64)
    if targets_np.size == 0:
        return None

    num_classes = len(class_to_idx)
    counts = np.bincount(targets_np, minlength=num_classes).astype(np.float64)
    if np.any(counts == 0):
        return None

    # normalized inverse-frequency weights
    weights = targets_np.size / (num_classes * counts)
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32, device=device)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    model_name: str,
    epoch: int,
    log_every: int,
    max_batches: int,
    scheduler: Any | None = None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_count = 0
    total_correct = 0
    started = perf_counter()
    for step, (images, targets) in enumerate(loader, start=1):
        if max_batches > 0 and step > max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        if scheduler is not None and scheduler.__class__.__name__ == "OneCycleLR":
            scheduler.step()

        batch = images.shape[0]
        preds = torch.argmax(logits, dim=1)
        total_loss += float(loss.item()) * batch
        total_count += batch
        total_correct += int((preds == targets).sum().item())

        if log_every > 0 and (step % log_every == 0):
            elapsed = perf_counter() - started
            avg_loss = total_loss / max(total_count, 1)
            avg_acc = total_correct / max(total_count, 1)
            speed = step / max(elapsed, 1e-9)
            print(
                f"[{model_name}] epoch={epoch:02d} train step={step}/{len(loader)} "
                f"loss={avg_loss:.4f} acc={avg_acc:.4f} speed={speed:.2f} batch/s"
            )

    loss_value = total_loss / max(total_count, 1)
    accuracy = float(total_correct / max(total_count, 1))
    return {
        "loss": loss_value,
        "accuracy": accuracy,
        "error": 1.0 - accuracy,
    }


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    positive_class_index: int,
    target_specificity: float,
    model_name: str,
    epoch: int,
    log_every: int,
    max_batches: int,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_count = 0
    total_correct = 0
    all_targets: list[np.ndarray] = []
    all_probs: list[np.ndarray] = []

    started = perf_counter()
    for step, (images, targets) in enumerate(loader, start=1):
        if max_batches > 0 and step > max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        preds = torch.argmax(logits, dim=1)
        probs = torch.softmax(logits, dim=1)[:, positive_class_index]

        batch = images.shape[0]
        total_loss += float(loss.item()) * batch
        total_count += batch
        total_correct += int((preds == targets).sum().item())
        all_targets.append(targets.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

        if log_every > 0 and (step % log_every == 0):
            elapsed = perf_counter() - started
            avg_loss = total_loss / max(total_count, 1)
            avg_acc = total_correct / max(total_count, 1)
            speed = step / max(elapsed, 1e-9)
            print(
                f"[{model_name}] epoch={epoch:02d} val step={step}/{len(loader)} "
                f"loss={avg_loss:.4f} acc={avg_acc:.4f} speed={speed:.2f} batch/s"
            )

    y_true_raw = np.concatenate(all_targets)
    y_true = (y_true_raw == int(positive_class_index)).astype(np.int64)
    y_prob = np.concatenate(all_probs)
    metrics = classification_metrics(y_true, y_prob, threshold=0.5)
    tuned = optimize_threshold(y_true, y_prob, target_specificity=target_specificity)

    metrics["loss"] = total_loss / max(total_count, 1)
    metrics["error"] = 1.0 - metrics["accuracy"]
    metrics["selected_threshold"] = float(tuned.get("threshold", 0.5))
    metrics["selected_target_specificity"] = float(tuned.get("target_specificity", target_specificity))
    metrics["selected_sensitivity"] = float(tuned.get("sensitivity", 0.0))
    metrics["selected_specificity"] = float(tuned.get("specificity", 0.0))
    metrics["selected_precision"] = float(tuned.get("precision", 0.0))
    metrics["selected_f1"] = float(tuned.get("f1", 0.0))
    metrics["selected_balanced_accuracy"] = float(tuned.get("balanced_accuracy", 0.0))
    metrics["selected_tp"] = float(tuned.get("tp", 0.0))
    metrics["selected_fp"] = float(tuned.get("fp", 0.0))
    metrics["selected_tn"] = float(tuned.get("tn", 0.0))
    metrics["selected_fn"] = float(tuned.get("fn", 0.0))
    return metrics, y_true, y_prob


def write_history_csv(history: list[dict[str, float]], path: Path) -> None:
    if not history:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(history[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def run_training(config: TrainConfig) -> tuple[Path, dict[str, float]]:
    set_seed(config.seed)
    if not (0.0 < config.target_specificity <= 1.0):
        raise ValueError("target_specificity must be in range (0, 1].")
    if config.early_stopping_patience < 0:
        raise ValueError("early_stopping_patience must be >= 0.")
    if config.early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must be >= 0.")
    if config.focal_gamma < 0:
        raise ValueError("focal_gamma must be >= 0.")
    if not (0.0 < config.scheduler_factor < 1.0):
        raise ValueError("scheduler_factor must be in range (0, 1).")
    if config.scheduler_patience < 0:
        raise ValueError("scheduler_patience must be >= 0.")

    best_key = normalize_metric_name(config.selection_metric)
    early_key = normalize_metric_name(config.early_stopping_metric)
    best_mode = metric_direction(best_key)
    early_mode = metric_direction(early_key)

    if config.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch does not see CUDA. "
            "Install CUDA-enabled torch build or use --device cpu."
        )

    device = torch.device(config.device)
    train_loader, val_loader, class_to_idx = create_dataloaders(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        image_size=config.image_size,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"),
    )
    if set(class_to_idx.keys()) != {"benign", "melanoma"}:
        print(f"[WARN] Expected classes benign/melanoma. Found: {sorted(class_to_idx.keys())}")

    model = create_model(config.model_name, num_classes=2).to(device)
    class_weights = build_class_weights(
        class_to_idx=class_to_idx,
        train_loader=train_loader,
        mode=config.class_weight_mode,
        device=device,
    )
    class_weights_list: list[float] | None = None
    if class_weights is not None:
        class_weights_list = class_weights.detach().cpu().numpy().tolist()
        print(f"Using class weights: {class_weights_list}")
    else:
        print("Class weights: disabled")
    criterion = build_criterion(config, class_weights)
    optimizer = build_optimizer(config, model)
    scheduler = build_scheduler(config, optimizer, train_loader)

    best_score = float("-inf") if best_mode == "max" else float("inf")
    early_best_score = float("-inf") if early_mode == "max" else float("inf")
    epochs_without_improvement = 0
    best_metrics: dict[str, float] = {}
    best_epoch = 0
    best_val_true: np.ndarray | None = None
    best_val_prob: np.ndarray | None = None
    history: list[dict[str, float]] = []
    output_path = Path(config.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_stem = output_path.stem
    positive_class_index = class_to_idx.get("melanoma", 1)

    print(
        f"Training started: model={config.model_name}, device={config.device}, "
        f"epochs={config.epochs}, train_batches={len(train_loader)}, val_batches={len(val_loader)}, "
        f"class_weight_mode={config.class_weight_mode}, target_specificity={config.target_specificity}, "
        f"loss_mode={config.loss_mode}, scheduler={config.scheduler}, "
        f"selection_metric={best_key}, early_stopping_metric={early_key}, "
        f"early_stopping_patience={config.early_stopping_patience}, "
        f"early_stopping_min_delta={config.early_stopping_min_delta}"
    )

    started = perf_counter()
    for epoch in range(1, config.epochs + 1):
        print(f"[{config.model_name}] epoch={epoch:02d} started")
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            model_name=config.model_name,
            epoch=epoch,
            log_every=config.log_every,
            max_batches=config.max_train_batches,
            scheduler=scheduler,
        )
        val_metrics, y_true, y_prob = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            positive_class_index=positive_class_index,
            target_specificity=config.target_specificity,
            model_name=config.model_name,
            epoch=epoch,
            log_every=config.log_every,
            max_batches=config.max_val_batches,
        )

        epoch_row = {
            "epoch": float(epoch),
            "train_loss": float(train_metrics["loss"]),
            "val_loss": float(val_metrics["loss"]),
            "train_accuracy": float(train_metrics["accuracy"]),
            "val_accuracy": float(val_metrics["accuracy"]),
            "train_error": float(train_metrics["error"]),
            "val_error": float(val_metrics["error"]),
            "val_auc": float(val_metrics["auc"]),
            "val_pr_auc": float(val_metrics["pr_auc"]),
            "val_precision": float(val_metrics["precision"]),
            "val_recall": float(val_metrics["recall"]),
            "val_f1": float(val_metrics["f1"]),
            "val_balanced_accuracy": float(val_metrics["balanced_accuracy"]),
            "val_sensitivity": float(val_metrics["sensitivity"]),
            "val_specificity": float(val_metrics["specificity"]),
            "val_selected_threshold": float(val_metrics["selected_threshold"]),
            "val_selected_sensitivity": float(val_metrics["selected_sensitivity"]),
            "val_selected_specificity": float(val_metrics["selected_specificity"]),
            "val_selected_f1": float(val_metrics["selected_f1"]),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_row)

        score = metric_value(val_metrics, best_key)
        if np.isnan(score):
            fallback = metric_value(val_metrics, "accuracy")
            print(f"[WARN] Metric '{best_key}' is NaN. Falling back to accuracy={fallback:.4f}.")
            score = fallback

        print(
            f"[{config.model_name}] epoch={epoch:02d} "
            f"train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"auc={val_metrics['auc']:.4f} pr_auc={val_metrics['pr_auc']:.4f} "
            f"sens@0.5={val_metrics['sensitivity']:.4f} spec@0.5={val_metrics['specificity']:.4f} "
            f"best_thr={val_metrics['selected_threshold']:.4f} "
            f"sens@best={val_metrics['selected_sensitivity']:.4f} "
            f"spec@best={val_metrics['selected_specificity']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.6f}"
        )

        if is_improved(score, best_score, best_mode, min_delta=0.0):
            best_score = score
            best_metrics = val_metrics
            best_epoch = epoch
            best_val_true = y_true.copy()
            best_val_prob = y_prob.copy()
            checkpoint = {
                "model_name": config.model_name,
                "class_to_idx": class_to_idx,
                "image_size": config.image_size,
                "best_epoch": best_epoch,
                "selected_threshold": float(val_metrics.get("selected_threshold", 0.5)),
                "selected_target_specificity": float(val_metrics.get("selected_target_specificity", config.target_specificity)),
                "model_state_dict": model.state_dict(),
            }
            torch.save(checkpoint, output_path)

        early_score = metric_value(val_metrics, early_key)
        if np.isnan(early_score):
            epochs_without_improvement += 1
            print(
                f"[{config.model_name}] epoch={epoch:02d} early-stop metric '{early_key}' is NaN "
                f"(no-improve count={epochs_without_improvement}/{config.early_stopping_patience})"
            )
        elif is_improved(
            early_score,
            early_best_score,
            early_mode,
            min_delta=config.early_stopping_min_delta,
        ):
            early_best_score = early_score
            epochs_without_improvement = 0
            print(
                f"[{config.model_name}] epoch={epoch:02d} early-stop metric improved: "
                f"{early_key}={early_score:.4f}"
            )
        else:
            epochs_without_improvement += 1
            print(
                f"[{config.model_name}] epoch={epoch:02d} no improvement for early-stop metric "
                f"{early_key}={early_score:.4f} "
                f"(count={epochs_without_improvement}/{config.early_stopping_patience})"
            )

        if (
            config.early_stopping_patience > 0
            and epochs_without_improvement >= config.early_stopping_patience
        ):
            print(
                f"[{config.model_name}] Early stopping triggered at epoch={epoch:02d} "
                f"(metric={early_key}, patience={config.early_stopping_patience}, "
                f"min_delta={config.early_stopping_min_delta})."
            )
            break

        if scheduler is not None:
            if scheduler.__class__.__name__ == "ReduceLROnPlateau":
                scheduler.step(val_metrics.get("pr_auc", float("nan")))
            elif scheduler.__class__.__name__ != "OneCycleLR":
                scheduler.step()

    elapsed = perf_counter() - started
    best_metrics["best_epoch"] = float(best_epoch)
    best_metrics["train_seconds"] = elapsed

    metrics_path = output_path.with_suffix(".metrics.json")
    history_csv_path = output_path.with_name(f"{output_stem}_history.csv")
    history_json_path = output_path.with_name(f"{output_stem}_history.json")
    curves_path = output_path.with_name(f"{output_stem}_training_curves.png")
    cm_path = output_path.with_name(f"{output_stem}_confusion_matrix_heatmap.png")
    roc_path = output_path.with_name(f"{output_stem}_roc_curve.png")
    probs_path = output_path.with_name(f"{output_stem}_probability_histogram.png")

    write_history_csv(history, history_csv_path)
    history_json_path.write_text(json.dumps(history, indent=2), encoding="utf-8")

    if history:
        plot_training_curves(history, curves_path)
    if best_val_true is not None and best_val_prob is not None:
        best_threshold = float(best_metrics.get("selected_threshold", 0.5))
        plot_confusion_matrix_heatmap(best_val_true, best_val_prob, cm_path, threshold=best_threshold)
        plot_roc(best_val_true, best_val_prob, roc_path)
        plot_probability_histogram(best_val_true, best_val_prob, probs_path)

    payload = {
        "config": asdict(config),
        "class_weights": class_weights_list,
        "best_metrics": best_metrics,
        "artifacts": {
            "history_csv": str(history_csv_path),
            "history_json": str(history_json_path),
            "training_curves_png": str(curves_path),
            "confusion_matrix_heatmap_png": str(cm_path),
            "roc_curve_png": str(roc_path),
            "probability_histogram_png": str(probs_path),
        },
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved best checkpoint: {output_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved history CSV: {history_csv_path}")
    print(f"Saved curves: {curves_path}")
    print(f"Saved confusion matrix heatmap: {cm_path}")
    print(f"Saved ROC curve: {roc_path}")
    print(f"Saved probability histogram: {probs_path}")

    return output_path, best_metrics


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Train a melanoma classifier.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--model-name", default="resnet50", choices=sorted(SUPPORTED_MODELS))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adamw"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-path", default="models/resnet50_best.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--class-weight-mode", default="inverse_frequency", choices=["none", "inverse_frequency"])
    parser.add_argument("--target-specificity", type=float, default=0.85)
    parser.add_argument("--loss-mode", default="cross_entropy", choices=["cross_entropy", "focal"])
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--scheduler", default="none", choices=["none", "cosine", "plateau", "onecycle"])
    parser.add_argument("--scheduler-t-max", type=int, default=0, help="Used by cosine scheduler. 0 means epochs.")
    parser.add_argument("--scheduler-eta-min", type=float, default=1e-6, help="Used by cosine scheduler.")
    parser.add_argument("--scheduler-factor", type=float, default=0.5, help="Used by plateau scheduler.")
    parser.add_argument("--scheduler-patience", type=int, default=2, help="Used by plateau scheduler.")
    parser.add_argument(
        "--selection-metric",
        default="auc",
        choices=sorted(MAXIMIZE_METRICS | MINIMIZE_METRICS),
        help="Metric used to choose and save best checkpoint.",
    )
    parser.add_argument(
        "--early-stopping-metric",
        default="pr_auc",
        choices=sorted(MAXIMIZE_METRICS | MINIMIZE_METRICS),
        help="Validation metric monitored for early stopping.",
    )
    parser.add_argument("--early-stopping-patience", type=int, default=0, help="0 disables early stopping.")
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    run_training(parse_args())
