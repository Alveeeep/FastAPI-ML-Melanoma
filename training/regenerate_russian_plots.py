from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from training.plots import (
    plot_confusion_matrix_from_counts,
    plot_model_comparison,
    plot_training_curves,
)


DEFAULT_METRICS_JSON = [
    Path("outputs/phaseC/resnet50v2_c5_baseline_best.metrics.json"),
    Path("outputs/final_validation/ham10000/resnet50v2_external_val.metrics.json"),
    Path("outputs/final_validation/isic2019/resnet50v2_external_val.metrics.json"),
]

DEFAULT_SUMMARY_CSV = [
    Path("outputs/phaseA_uniform/comparison_summary.csv"),
    Path("outputs/phaseB1/best_per_model.csv"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Regenerate diploma plots with Russian labels from saved history/metrics JSON and CSV summaries. "
            "ROC curves require y_true/y_prob arrays, so use training.evaluate_checkpoint for them."
        )
    )
    parser.add_argument(
        "--metrics-json",
        nargs="*",
        type=Path,
        default=None,
        help="Metrics JSON files to process. Defaults to the key diploma metrics files.",
    )
    parser.add_argument(
        "--summary-csv",
        nargs="*",
        type=Path,
        default=None,
        help="CSV summary files for model-comparison plots. Defaults to phase A and phase B summaries.",
    )
    parser.add_argument("--output-suffix", default="_ru", help="Suffix for regenerated PNG files.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite original plot paths instead of writing *_ru.png files.",
    )
    return parser.parse_args()


def as_repo_path(raw: str | None) -> Path | None:
    if not raw:
        return None
    return Path(raw.replace("\\", "/"))


def output_path(original: Path, suffix: str, overwrite: bool) -> Path:
    if overwrite:
        return original
    return original.with_name(f"{original.stem}{suffix}{original.suffix}")


def parse_float(raw: Any) -> float:
    if raw is None:
        return float("nan")
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip().replace(",", ".")
    if not text:
        return float("nan")
    return float(text)


def load_csv_rows(path: Path) -> list[dict[str, float | str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        rows: list[dict[str, float | str]] = []
        for row in reader:
            parsed: dict[str, float | str] = {}
            for key, value in row.items():
                if key in {"model", "config_id", "optimizer", "class_weight_mode", "metrics_file"}:
                    parsed[key] = value or ""
                else:
                    try:
                        parsed[key] = parse_float(value)
                    except ValueError:
                        parsed[key] = value or ""
            rows.append(parsed)
    return rows


def metrics_for_confusion_matrix(payload: dict[str, Any]) -> dict[str, Any] | None:
    if "best_metrics" in payload:
        metrics = payload["best_metrics"]
        selected_keys = {"selected_tn", "selected_fp", "selected_fn", "selected_tp"}
        if selected_keys <= set(metrics):
            return {
                "tn": metrics["selected_tn"],
                "fp": metrics["selected_fp"],
                "fn": metrics["selected_fn"],
                "tp": metrics["selected_tp"],
            }
        return metrics
    if "metrics_at_checkpoint_threshold" in payload:
        return payload["metrics_at_checkpoint_threshold"]
    return None


def regenerate_from_metrics_json(path: Path, suffix: str, overwrite: bool) -> list[Path]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    artifacts = payload.get("artifacts", {})
    written: list[Path] = []

    history_path = as_repo_path(artifacts.get("history_json"))
    curves_path = as_repo_path(artifacts.get("training_curves_png"))
    if history_path and curves_path and history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        target = output_path(curves_path, suffix, overwrite)
        plot_training_curves(history, target)
        written.append(target)

    cm_path = as_repo_path(artifacts.get("confusion_matrix_heatmap_png"))
    cm_metrics = metrics_for_confusion_matrix(payload)
    if cm_path and cm_metrics:
        keys = {"tn", "fp", "fn", "tp"}
        if keys <= set(cm_metrics):
            target = output_path(cm_path, suffix, overwrite)
            plot_confusion_matrix_from_counts(
                tn=parse_float(cm_metrics["tn"]),
                fp=parse_float(cm_metrics["fp"]),
                fn=parse_float(cm_metrics["fn"]),
                tp=parse_float(cm_metrics["tp"]),
                output_path=target,
            )
            written.append(target)

    roc_path = as_repo_path(artifacts.get("roc_curve_png"))
    if roc_path:
        print(
            "[WARN] ROC-кривая не восстановлена из "
            f"{path}: в metrics.json нет массивов y_true/y_prob. "
            "Для ROC выполните training.evaluate_checkpoint."
        )

    return written


def regenerate_from_summary_csv(path: Path, suffix: str, overwrite: bool) -> Path | None:
    rows = load_csv_rows(path)
    if not rows:
        return None
    target_name = "model_comparison_metrics.png"
    if path.name == "best_per_model.csv":
        target_name = "best_per_model_metrics.png"
    target = output_path(path.with_name(target_name), suffix, overwrite)
    plot_model_comparison(rows, target)
    return target


def main() -> None:
    args = parse_args()
    metrics_files = args.metrics_json if args.metrics_json is not None else DEFAULT_METRICS_JSON
    summary_files = args.summary_csv if args.summary_csv is not None else DEFAULT_SUMMARY_CSV

    written: list[Path] = []
    for path in metrics_files:
        if not path.exists():
            print(f"[WARN] Metrics JSON not found: {path}")
            continue
        written.extend(regenerate_from_metrics_json(path, args.output_suffix, args.overwrite))

    for path in summary_files:
        if not path.exists():
            print(f"[WARN] Summary CSV not found: {path}")
            continue
        result = regenerate_from_summary_csv(path, args.output_suffix, args.overwrite)
        if result:
            written.append(result)

    print("Generated Russian-labeled plots:")
    for path in written:
        print(f"- {path}")


if __name__ == "__main__":
    main()
