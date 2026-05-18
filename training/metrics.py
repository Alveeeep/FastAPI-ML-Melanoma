from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    f1_score,
)


def classification_metrics(y_true: np.ndarray, y_prob_pos: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob_pos = np.asarray(y_prob_pos, dtype=np.float32)
    y_pred = (y_prob_pos >= threshold).astype(np.int64)
    acc = float(accuracy_score(y_true, y_pred))

    auc = float("nan")
    if len(np.unique(y_true)) > 1:
        auc = float(roc_auc_score(y_true, y_prob_pos))
    pr_auc = float("nan")
    if len(np.unique(y_true)) > 1:
        pr_auc = float(average_precision_score(y_true, y_prob_pos))

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = float(tp / (tp + fn)) if (tp + fn) else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) else 0.0
    precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    balanced_acc = float(balanced_accuracy_score(y_true, y_pred))

    return {
        "threshold": float(threshold),
        "accuracy": acc,
        "auc": auc,
        "pr_auc": pr_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced_acc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "tp": float(tp),
        "fp": float(fp),
        "tn": float(tn),
        "fn": float(fn),
    }


def optimize_threshold(
    y_true: np.ndarray,
    y_prob_pos: np.ndarray,
    target_specificity: float = 0.85,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob_pos = np.asarray(y_prob_pos, dtype=np.float32)
    if y_true.size == 0:
        return {"threshold": 0.5}

    thresholds = np.unique(y_prob_pos)
    if thresholds.size == 0:
        return {"threshold": 0.5}

    candidates: list[dict[str, float]] = []
    for threshold in thresholds:
        metrics = classification_metrics(y_true, y_prob_pos, threshold=float(threshold))
        candidates.append(metrics)

    valid = [m for m in candidates if m.get("specificity", 0.0) >= target_specificity]
    if valid:
        valid.sort(
            key=lambda m: (
                m.get("sensitivity", 0.0),
                m.get("specificity", 0.0),
                -abs(m.get("threshold", 0.5) - 0.5),
            ),
            reverse=True,
        )
        best = valid[0]
    else:
        # fallback to Youden's J statistic
        candidates.sort(
            key=lambda m: (m.get("sensitivity", 0.0) + m.get("specificity", 0.0) - 1.0),
            reverse=True,
        )
        best = candidates[0]

    result = dict(best)
    result["target_specificity"] = float(target_specificity)
    return result


def optimize_threshold_for_sensitivity(
    y_true: np.ndarray,
    y_prob_pos: np.ndarray,
    target_sensitivity: float = 0.80,
) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob_pos = np.asarray(y_prob_pos, dtype=np.float32)
    if y_true.size == 0:
        return {"threshold": 0.5}

    thresholds = np.unique(y_prob_pos)
    if thresholds.size == 0:
        return {"threshold": 0.5}

    candidates: list[dict[str, float]] = []
    for threshold in thresholds:
        metrics = classification_metrics(y_true, y_prob_pos, threshold=float(threshold))
        candidates.append(metrics)

    valid = [m for m in candidates if m.get("sensitivity", 0.0) >= target_sensitivity]
    if valid:
        valid.sort(
            key=lambda m: (
                m.get("specificity", 0.0),
                m.get("sensitivity", 0.0),
                -abs(m.get("threshold", 0.5) - 0.5),
            ),
            reverse=True,
        )
        best = valid[0]
    else:
        # If the target sensitivity is unreachable, keep the most sensitive threshold.
        candidates.sort(
            key=lambda m: (
                m.get("sensitivity", 0.0),
                m.get("specificity", 0.0),
                -abs(m.get("threshold", 0.5) - 0.5),
            ),
            reverse=True,
        )
        best = candidates[0]

    result = dict(best)
    result["target_sensitivity"] = float(target_sensitivity)
    result["target_sensitivity_reached"] = float(result.get("sensitivity", 0.0) >= target_sensitivity)
    return result
