from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc as sklearn_auc
from sklearn.metrics import confusion_matrix, roc_curve

RUSSIAN_CLASS_NAMES = ("доброкачественное", "меланома")


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _to_array(values: Sequence[float]) -> np.ndarray:
    return np.asarray(list(values), dtype=np.float32)


def _plot_confusion_matrix(
    cm: np.ndarray,
    output_path: str | Path,
    class_names: tuple[str, str] = RUSSIAN_CLASS_NAMES,
) -> None:
    path = Path(output_path)
    _ensure_parent(path)

    cm = np.asarray(cm, dtype=np.int64)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=np.float32), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=plt.cm.Blues, vmin=0.0, vmax=1.0)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=np.arange(2),
        yticks=np.arange(2),
        xticklabels=[f"Предсказано: {class_names[0]}", f"Предсказано: {class_names[1]}"],
        yticklabels=[f"Истинно: {class_names[0]}", f"Истинно: {class_names[1]}"],
        ylabel="Истинный класс",
        xlabel="Предсказанный класс",
        title="Матрица ошибок, нормализованная по истинному классу",
    )
    plt.setp(ax.get_xticklabels(), rotation=18, ha="right", rotation_mode="anchor")

    for i in range(2):
        for j in range(2):
            ax.text(
                j,
                i,
                f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                ha="center",
                va="center",
                color="white" if cm_norm[i, j] > 0.5 else "black",
                fontsize=10,
            )

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_training_curves(history: list[dict[str, float]], output_path: str | Path) -> None:
    if not history:
        return
    path = Path(output_path)
    _ensure_parent(path)

    epochs = _to_array([row["epoch"] for row in history])
    train_loss = _to_array([row["train_loss"] for row in history])
    val_loss = _to_array([row["val_loss"] for row in history])
    train_acc = _to_array([row["train_accuracy"] for row in history])
    val_acc = _to_array([row["val_accuracy"] for row in history])
    train_err = _to_array([row["train_error"] for row in history])
    val_err = _to_array([row["val_error"] for row in history])
    val_sens = _to_array([row["val_sensitivity"] for row in history])
    val_spec = _to_array([row["val_specificity"] for row in history])
    val_auc = _to_array([row["val_auc"] for row in history])
    val_pr_auc = _to_array([row.get("val_pr_auc", float("nan")) for row in history])
    val_sel_sens = _to_array([row.get("val_selected_sensitivity", float("nan")) for row in history])
    val_sel_spec = _to_array([row.get("val_selected_specificity", float("nan")) for row in history])

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, marker="o", label="Обучение")
    ax.plot(epochs, val_loss, marker="o", label="Валидация")
    ax.set_title("Функция потерь по эпохам")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Значение loss")
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(epochs, train_acc, marker="o", label="Обучение")
    ax.plot(epochs, val_acc, marker="o", label="Валидация")
    ax.set_title("Точность по эпохам")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 0]
    ax.plot(epochs, train_err, marker="o", label="Обучение")
    ax.plot(epochs, val_err, marker="o", label="Валидация")
    ax.set_title("Доля ошибок по эпохам")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Доля ошибок")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(epochs, val_sens, marker="o", label="Чувствительность")
    ax.plot(epochs, val_spec, marker="o", label="Специфичность")
    if np.any(~np.isnan(val_auc)):
        ax.plot(epochs, val_auc, marker="o", label="ROC-AUC")
    if np.any(~np.isnan(val_pr_auc)):
        ax.plot(epochs, val_pr_auc, marker="o", label="PR-AUC")
    if np.any(~np.isnan(val_sel_sens)):
        ax.plot(epochs, val_sel_sens, marker="o", label="Чувствительность при выбранном пороге")
    if np.any(~np.isnan(val_sel_spec)):
        ax.plot(epochs, val_sel_spec, marker="o", label="Специфичность при выбранном пороге")
    ax.set_title("Метрики качества на валидации")
    ax.set_xlabel("Эпоха")
    ax.set_ylabel("Значение метрики")
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_confusion_matrix_heatmap(
    y_true: np.ndarray,
    y_prob_pos: np.ndarray,
    output_path: str | Path,
    threshold: float = 0.5,
    class_names: tuple[str, str] = RUSSIAN_CLASS_NAMES,
) -> None:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob_pos = np.asarray(y_prob_pos, dtype=np.float32)
    y_pred = (y_prob_pos >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    _plot_confusion_matrix(cm, output_path, class_names=class_names)


def plot_confusion_matrix_from_counts(
    tn: float,
    fp: float,
    fn: float,
    tp: float,
    output_path: str | Path,
    class_names: tuple[str, str] = RUSSIAN_CLASS_NAMES,
) -> None:
    cm = np.asarray([[int(tn), int(fp)], [int(fn), int(tp)]], dtype=np.int64)
    _plot_confusion_matrix(cm, output_path, class_names=class_names)


def plot_roc(
    y_true: np.ndarray,
    y_prob_pos: np.ndarray,
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    _ensure_parent(path)

    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob_pos = np.asarray(y_prob_pos, dtype=np.float32)
    if len(np.unique(y_true)) < 2:
        return

    fpr, tpr, _ = roc_curve(y_true, y_prob_pos)
    roc_auc = sklearn_auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#1f77b4", lw=2, label=f"ROC-AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle="--")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("Доля ложноположительных срабатываний")
    ax.set_ylabel("Доля истинноположительных срабатываний")
    ax.set_title("ROC-кривая")
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_probability_histogram(
    y_true: np.ndarray,
    y_prob_pos: np.ndarray,
    output_path: str | Path,
) -> None:
    path = Path(output_path)
    _ensure_parent(path)

    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob_pos = np.asarray(y_prob_pos, dtype=np.float32)

    benign_probs = y_prob_pos[y_true == 0]
    melanoma_probs = y_prob_pos[y_true == 1]
    if benign_probs.size == 0 and melanoma_probs.size == 0:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0.0, 1.0, 30)
    if benign_probs.size > 0:
        ax.hist(benign_probs, bins=bins, alpha=0.65, label="Доброкачественные", color="#2ca02c", density=True)
    if melanoma_probs.size > 0:
        ax.hist(melanoma_probs, bins=bins, alpha=0.65, label="Меланома", color="#d62728", density=True)
    ax.axvline(0.5, color="black", linestyle="--", linewidth=1.3, label="Порог 0.5")
    ax.set_title("Распределение вероятности класса melanoma")
    ax.set_xlabel("Предсказанная вероятность melanoma")
    ax.set_ylabel("Плотность")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_model_comparison(rows: list[dict[str, float | str]], output_path: str | Path) -> None:
    if not rows:
        return
    path = Path(output_path)
    _ensure_parent(path)

    model_names = [str(row["model"]) for row in rows]
    metrics = ("auc", "pr_auc", "selected_sensitivity", "selected_specificity")
    metric_titles = {
        "auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
        "selected_sensitivity": "Чувствительность",
        "selected_specificity": "Специфичность",
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for idx, metric in enumerate(metrics):
        values = []
        for row in rows:
            raw = row.get(metric, float("nan"))
            values.append(float(raw) if raw is not None else float("nan"))
        ax = axes[idx]
        x = np.arange(len(model_names))
        bars = ax.bar(x, values, color="#1f77b4", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=15, ha="right")
        ax.set_ylim(0.0, 1.02)
        ax.set_title(metric_titles.get(metric, metric))
        ax.grid(axis="y", alpha=0.25)
        for bar, value in zip(bars, values):
            if np.isnan(value):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2.0, value + 0.01, f"{value:.3f}", ha="center", va="bottom")

    fig.suptitle("Сравнение моделей по метрикам качества", fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
