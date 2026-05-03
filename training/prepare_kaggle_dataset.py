from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_POSITIVE_KEYWORDS = ("melanoma", "malignant", "cancer", "positive")
DEFAULT_NEGATIVE_KEYWORDS = ("benign", "nevus", "naevus", "normal", "negative")
DEFAULT_METADATA_ID_COLUMNS = ("image_name", "isic_id", "image_id", "id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Kaggle dataset and split it to train/val in benign/melanoma format."
    )
    parser.add_argument("--dataset", required=True, help="Kaggle dataset slug, for example user/dataset-name")
    parser.add_argument("--download-dir", default="data/raw", help="Where raw downloaded files will be stored")
    parser.add_argument("--prepared-dir", default="data/prepared", help="Output dir with train/val folders")
    parser.add_argument(
        "--source-subdir",
        default=None,
        help="Optional relative path inside downloaded dataset where class folders are located",
    )
    parser.add_argument(
        "--metadata-csv",
        default=None,
        help=(
            "Optional metadata CSV path for flat datasets (for example train-metadata.csv). "
            "If set, class folders are not required."
        ),
    )
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Optional image folder path (relative to source dir) used with --metadata-csv",
    )
    parser.add_argument(
        "--id-column",
        default=None,
        help="Metadata column containing image id/name (auto-detected if omitted)",
    )
    parser.add_argument(
        "--label-column",
        default="target",
        help="Metadata column containing class label (default: target)",
    )
    parser.add_argument(
        "--metadata-label-map",
        default=None,
        help='Optional explicit metadata label mapping: "1=melanoma,0=benign,malignant=melanoma"',
    )
    parser.add_argument(
        "--strict-metadata",
        action="store_true",
        help="Fail if metadata contains unknown labels or missing image files",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio, e.g. 0.2")
    parser.add_argument(
        "--balance-train",
        default="none",
        choices=["none", "oversample", "undersample"],
        help="Optional balancing strategy applied to TRAIN split only.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    parser.add_argument(
        "--class-map",
        default=None,
        help='Optional explicit mapping: "malignant=melanoma,benign=benign"',
    )
    parser.add_argument(
        "--positive-keywords",
        default=",".join(DEFAULT_POSITIVE_KEYWORDS),
        help="Comma-separated keywords for melanoma class auto-detection",
    )
    parser.add_argument(
        "--negative-keywords",
        default=",".join(DEFAULT_NEGATIVE_KEYWORDS),
        help="Comma-separated keywords for benign class auto-detection",
    )
    parser.add_argument("--skip-download", action="store_true", help="Skip Kaggle download step")
    parser.add_argument("--force", action="store_true", help="Overwrite prepared-dir if it exists")
    return parser.parse_args()


def parse_csv_list(value: str) -> tuple[str, ...]:
    return tuple(x.strip().lower() for x in value.split(",") if x.strip())


def parse_class_map(value: str | None) -> dict[str, str]:
    if not value:
        return {}
    result: dict[str, str] = {}
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Invalid class mapping item '{chunk}'. Expected source=target.")
        source, target = chunk.split("=", 1)
        source = normalize_source_name(source)
        target = target.strip().lower()
        if target not in {"benign", "melanoma"}:
            raise ValueError(f"Invalid target class '{target}'. Allowed: benign, melanoma.")
        result[source] = target
    return result


def parse_metadata_label_map(value: str | None) -> dict[str, str]:
    if not value:
        return {}
    result: dict[str, str] = {}
    for chunk in value.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Invalid metadata label mapping '{chunk}'. Expected source=target.")
        source, target = chunk.split("=", 1)
        source = source.strip().lower()
        target = target.strip().lower()
        if target not in {"benign", "melanoma"}:
            raise ValueError(f"Invalid target class '{target}'. Allowed: benign, melanoma.")
        result[source] = target
    return result


def get_kaggle_api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise RuntimeError("Package 'kaggle' is not installed. Install it and retry.") from exc

    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:
        raise RuntimeError(
            "Kaggle authentication failed. Configure KAGGLE_API_TOKEN env var "
            "or ~/.kaggle/access_token. Legacy fallback: ~/.kaggle/kaggle.json."
        ) from exc
    return api


def download_dataset(dataset: str, download_dir: Path, skip_download: bool) -> Path:
    dataset_dir = download_dir / dataset.replace("/", "__")
    dataset_dir.mkdir(parents=True, exist_ok=True)
    if skip_download:
        if not any(dataset_dir.rglob("*")):
            raise FileNotFoundError(f"--skip-download set, but no files found in {dataset_dir}")
        return dataset_dir

    api = get_kaggle_api()
    api.dataset_download_files(dataset=dataset, path=str(dataset_dir), unzip=True, quiet=False)
    return dataset_dir


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS


def image_count_recursive(folder: Path) -> int:
    return sum(1 for p in folder.rglob("*") if is_image(p))


def detect_class_root(base_dir: Path) -> Path:
    image_count_cache: dict[Path, int] = {}

    def _count(folder: Path) -> int:
        if folder not in image_count_cache:
            image_count_cache[folder] = image_count_recursive(folder)
        return image_count_cache[folder]

    best_dir: Path | None = None
    best_count = -1

    for candidate in [base_dir, *base_dir.rglob("*")]:
        if not candidate.is_dir():
            continue
        children = [x for x in candidate.iterdir() if x.is_dir()]
        if len(children) < 2:
            continue
        valid_children = [x for x in children if _count(x) > 0]
        if len(valid_children) < 2:
            continue
        total = sum(_count(x) for x in valid_children)
        if total > best_count:
            best_count = total
            best_dir = candidate

    if best_dir is None:
        raise FileNotFoundError(
            f"Could not detect class folders automatically in {base_dir}. "
            "Use --source-subdir to point to the class root folder."
        )
    return best_dir


def normalize_source_name(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", name.lower()).strip()


def map_class_name(
    source_name: str,
    explicit_map: dict[str, str],
    positive_keywords: tuple[str, ...],
    negative_keywords: tuple[str, ...],
) -> str | None:
    normalized = normalize_source_name(source_name)
    if normalized in explicit_map:
        return explicit_map[normalized]

    positive_match = any(key in normalized for key in positive_keywords)
    negative_match = any(key in normalized for key in negative_keywords)

    if positive_match and not negative_match:
        return "melanoma"
    if negative_match and not positive_match:
        return "benign"
    return None


def collect_class_files(
    class_root: Path,
    explicit_map: dict[str, str],
    positive_keywords: tuple[str, ...],
    negative_keywords: tuple[str, ...],
) -> tuple[dict[str, list[Path]], dict[str, str]]:
    target_to_files: dict[str, list[Path]] = defaultdict(list)
    source_to_target: dict[str, str] = {}
    unknown_classes: list[str] = []

    for source_dir in sorted([x for x in class_root.iterdir() if x.is_dir()], key=lambda p: p.name.lower()):
        files = [p for p in source_dir.rglob("*") if is_image(p)]
        if not files:
            continue

        mapped = map_class_name(source_dir.name, explicit_map, positive_keywords, negative_keywords)
        if mapped is None:
            unknown_classes.append(source_dir.name)
            continue
        source_to_target[source_dir.name] = mapped
        target_to_files[mapped].extend(files)

    if unknown_classes:
        unknown_str = ", ".join(unknown_classes)
        raise ValueError(
            "Could not map one or more class folders to benign/melanoma: "
            f"{unknown_str}. Provide explicit mapping via --class-map."
        )

    if not target_to_files.get("benign") or not target_to_files.get("melanoma"):
        raise ValueError(
            "Expected both classes after mapping. Ensure dataset has both benign and melanoma samples."
        )
    return target_to_files, source_to_target


def resolve_path(base: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return base / path


def detect_metadata_csv(source_dir: Path, metadata_csv: str | None) -> Path:
    if metadata_csv:
        path = resolve_path(source_dir, metadata_csv)
        if not path.exists():
            raise FileNotFoundError(f"Metadata CSV not found: {path}")
        return path

    candidates: list[Path] = []
    for name in ("train-metadata.csv", "train_metadata.csv", "train.csv", "metadata.csv"):
        candidates.extend(source_dir.rglob(name))
    if not candidates:
        raise FileNotFoundError(
            "Could not detect metadata CSV automatically. Use --metadata-csv to set it explicitly."
        )
    candidates.sort(key=lambda p: (len(p.parts), str(p).lower()))
    return candidates[0]


def detect_image_dir(source_dir: Path, image_dir: str | None) -> Path:
    if image_dir:
        path = resolve_path(source_dir, image_dir)
        if not path.exists():
            raise FileNotFoundError(f"Image dir not found: {path}")
        return path

    train_dir = source_dir / "train"
    if train_dir.exists() and any(is_image(p) for p in train_dir.rglob("*")):
        return train_dir
    return source_dir


def _normalize_label(raw_label: Any) -> str:
    return str(raw_label).strip().lower()


def map_metadata_label(raw_label: Any, label_map: dict[str, str]) -> str | None:
    normalized = _normalize_label(raw_label)
    if normalized in label_map:
        return label_map[normalized]

    if normalized in {"1", "true", "t", "yes", "y", "melanoma", "malignant", "positive", "cancer"}:
        return "melanoma"
    if normalized in {"0", "false", "f", "no", "n", "benign", "negative", "nevus", "naevus", "normal"}:
        return "benign"
    return None


def build_image_index(image_dir: Path) -> tuple[dict[str, Path], dict[str, list[Path]]]:
    by_name: dict[str, Path] = {}
    by_stem: dict[str, list[Path]] = defaultdict(list)
    for file in image_dir.rglob("*"):
        if not is_image(file):
            continue
        by_name[file.name.lower()] = file
        by_stem[file.stem.lower()].append(file)
    return by_name, by_stem


def resolve_image_file(image_id: Any, by_name: dict[str, Path], by_stem: dict[str, list[Path]]) -> Path | None:
    value = str(image_id).strip()
    if not value:
        return None

    key = value.lower()
    if key in by_name:
        return by_name[key]

    candidate = Path(value)
    if candidate.suffix:
        stem_key = candidate.stem.lower()
    else:
        stem_key = key
    options = by_stem.get(stem_key, [])
    if options:
        return options[0]
    return None


def detect_column(fieldnames: list[str], column_name: str, required_label: str) -> str:
    normalized_map = {name.strip().lower(): name for name in fieldnames}
    key = column_name.strip().lower()
    if key not in normalized_map:
        raise ValueError(f"{required_label} column '{column_name}' not found in CSV columns: {fieldnames}")
    return normalized_map[key]


def detect_id_column(fieldnames: list[str], id_column: str | None) -> str:
    normalized_map = {name.strip().lower(): name for name in fieldnames}
    if id_column:
        return detect_column(fieldnames, id_column, "id")

    for default_name in DEFAULT_METADATA_ID_COLUMNS:
        if default_name in normalized_map:
            return normalized_map[default_name]
    raise ValueError(
        "Could not auto-detect id column. Use --id-column explicitly. "
        f"Available columns: {fieldnames}"
    )


def collect_metadata_files(
    source_dir: Path,
    metadata_csv: str | None,
    image_dir: str | None,
    id_column: str | None,
    label_column: str,
    label_map: dict[str, str],
    strict_metadata: bool,
) -> tuple[dict[str, list[Path]], dict[str, Any]]:
    metadata_path = detect_metadata_csv(source_dir, metadata_csv)
    images_root = detect_image_dir(source_dir, image_dir)

    by_name, by_stem = build_image_index(images_root)
    if not by_name:
        raise FileNotFoundError(f"No image files found in {images_root}")

    target_to_files: dict[str, list[Path]] = defaultdict(list)
    unknown_labels = 0
    missing_images = 0
    total_rows = 0

    with metadata_path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames:
            raise ValueError(f"Metadata CSV has no header: {metadata_path}")

        id_col = detect_id_column(reader.fieldnames, id_column)
        label_col = detect_column(reader.fieldnames, label_column, "label")

        for row in reader:
            total_rows += 1
            target = map_metadata_label(row.get(label_col), label_map)
            if target is None:
                unknown_labels += 1
                continue

            image_file = resolve_image_file(row.get(id_col), by_name, by_stem)
            if image_file is None:
                missing_images += 1
                continue
            target_to_files[target].append(image_file)

    if strict_metadata and (unknown_labels > 0 or missing_images > 0):
        raise ValueError(
            "Metadata strict mode failed: "
            f"unknown_labels={unknown_labels}, missing_images={missing_images}"
        )

    if not target_to_files.get("benign") or not target_to_files.get("melanoma"):
        raise ValueError(
            "Expected both classes after metadata mapping. "
            "Check --label-column and --metadata-label-map."
        )

    info = {
        "mode": "metadata_csv",
        "metadata_csv": str(metadata_path),
        "images_root": str(images_root),
        "id_column": id_col,
        "label_column": label_col,
        "total_rows": total_rows,
        "unknown_labels": unknown_labels,
        "missing_images": missing_images,
        "matched_images": len(target_to_files.get("benign", [])) + len(target_to_files.get("melanoma", [])),
    }
    return target_to_files, info


def split_paths(paths: list[Path], val_ratio: float, rng: random.Random) -> tuple[list[Path], list[Path]]:
    items = paths.copy()
    rng.shuffle(items)

    total = len(items)
    if total == 0:
        return [], []
    if total == 1:
        return items, []

    val_count = int(total * val_ratio)
    if val_count <= 0:
        val_count = 1
    if val_count >= total:
        val_count = total - 1

    val_paths = items[:val_count]
    train_paths = items[val_count:]
    return train_paths, val_paths


def apply_train_balancing(
    train_files_by_class: dict[str, list[Path]],
    strategy: str,
    rng: random.Random,
) -> tuple[dict[str, list[Path]], dict[str, int]]:
    strategy = strategy.strip().lower()
    if strategy == "none":
        return train_files_by_class, {
            "before_benign": len(train_files_by_class["benign"]),
            "before_melanoma": len(train_files_by_class["melanoma"]),
            "after_benign": len(train_files_by_class["benign"]),
            "after_melanoma": len(train_files_by_class["melanoma"]),
        }

    benign = train_files_by_class["benign"]
    melanoma = train_files_by_class["melanoma"]
    if not benign or not melanoma:
        return train_files_by_class, {
            "before_benign": len(benign),
            "before_melanoma": len(melanoma),
            "after_benign": len(benign),
            "after_melanoma": len(melanoma),
        }

    if strategy == "undersample":
        target = min(len(benign), len(melanoma))
        benign_new = rng.sample(benign, target) if len(benign) > target else benign.copy()
        melanoma_new = rng.sample(melanoma, target) if len(melanoma) > target else melanoma.copy()
    elif strategy == "oversample":
        target = max(len(benign), len(melanoma))
        benign_new = benign.copy()
        melanoma_new = melanoma.copy()
        while len(benign_new) < target:
            benign_new.append(rng.choice(benign))
        while len(melanoma_new) < target:
            melanoma_new.append(rng.choice(melanoma))
    else:
        raise ValueError("Unsupported --balance-train strategy.")

    balanced = {"benign": benign_new, "melanoma": melanoma_new}
    stats = {
        "before_benign": len(benign),
        "before_melanoma": len(melanoma),
        "after_benign": len(benign_new),
        "after_melanoma": len(melanoma_new),
    }
    return balanced, stats


def safe_name(path: Path) -> str:
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", path.stem).strip("_")
    if not stem:
        stem = "img"
    return stem[:80]


def copy_split(
    files: list[Path],
    split_name: str,
    class_name: str,
    output_root: Path,
) -> int:
    split_dir = output_root / split_name / class_name
    split_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    for idx, src in enumerate(files, start=1):
        filename = f"{idx:06d}_{safe_name(src)}{src.suffix.lower()}"
        destination = split_dir / filename
        attempt = 1
        while destination.exists():
            filename = f"{idx:06d}_{safe_name(src)}_{attempt:04d}{src.suffix.lower()}"
            destination = split_dir / filename
            attempt += 1
        shutil.copy2(src, destination)
        copied += 1
    return copied


def prepare_dataset(
    source_dir: Path,
    prepared_dir: Path,
    val_ratio: float,
    balance_train: str,
    seed: int,
    explicit_map: dict[str, str],
    positive_keywords: tuple[str, ...],
    negative_keywords: tuple[str, ...],
    metadata_csv: str | None,
    image_dir: str | None,
    id_column: str | None,
    label_column: str,
    metadata_label_map: dict[str, str],
    strict_metadata: bool,
    force: bool,
) -> dict[str, object]:
    if prepared_dir.exists():
        if force:
            shutil.rmtree(prepared_dir)
        else:
            raise FileExistsError(f"{prepared_dir} already exists. Use --force to overwrite.")
    prepared_dir.mkdir(parents=True, exist_ok=True)

    if metadata_csv:
        target_to_files, metadata_info = collect_metadata_files(
            source_dir=source_dir,
            metadata_csv=metadata_csv,
            image_dir=image_dir,
            id_column=id_column,
            label_column=label_column,
            label_map=metadata_label_map,
            strict_metadata=strict_metadata,
        )
        summary: dict[str, object] = {
            **metadata_info,
            "val_ratio": val_ratio,
            "balance_train": balance_train,
            "seed": seed,
            "counts": {},
        }
    else:
        class_root = detect_class_root(source_dir)
        target_to_files, source_to_target = collect_class_files(
            class_root=class_root,
            explicit_map=explicit_map,
            positive_keywords=positive_keywords,
            negative_keywords=negative_keywords,
        )
        summary = {
            "mode": "class_folders",
            "class_root": str(class_root),
            "source_to_target_map": source_to_target,
            "val_ratio": val_ratio,
            "balance_train": balance_train,
            "seed": seed,
            "counts": {},
        }

    rng = random.Random(seed)
    split_files: dict[str, dict[str, list[Path]]] = {}

    for target_class in ("benign", "melanoma"):
        train_files, val_files = split_paths(target_to_files[target_class], val_ratio, rng)
        split_files[target_class] = {"train": train_files, "val": val_files}

    balanced_train_files, balance_stats = apply_train_balancing(
        {"benign": split_files["benign"]["train"], "melanoma": split_files["melanoma"]["train"]},
        strategy=balance_train,
        rng=rng,
    )
    summary["train_balance"] = balance_stats

    for target_class in ("benign", "melanoma"):
        train_files = balanced_train_files[target_class]
        val_files = split_files[target_class]["val"]
        copied_train = copy_split(train_files, "train", target_class, prepared_dir)
        copied_val = copy_split(val_files, "val", target_class, prepared_dir)
        summary["counts"][target_class] = {
            "source_total": len(target_to_files[target_class]),
            "train": copied_train,
            "val": copied_val,
        }

    benign_total = len(target_to_files["benign"])
    melanoma_total = len(target_to_files["melanoma"])
    imbalance_ratio = float(benign_total / melanoma_total) if melanoma_total > 0 else float("inf")
    summary["source_distribution"] = {
        "benign_total": benign_total,
        "melanoma_total": melanoma_total,
        "benign_to_melanoma_ratio": imbalance_ratio,
    }
    if imbalance_ratio > 3.0:
        summary["warning"] = (
            "Source dataset is imbalanced by design. "
            "This is not a parsing error. Use --balance-train if you need balanced TRAIN split."
        )

    manifest_path = prepared_dir / "dataset_manifest.json"
    manifest_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    summary["manifest_path"] = str(manifest_path)
    return summary


def main() -> None:
    args = parse_args()
    if not (0.0 < args.val_ratio < 1.0):
        raise ValueError("--val-ratio must be in range (0, 1)")

    explicit_map = parse_class_map(args.class_map)
    metadata_label_map = parse_metadata_label_map(args.metadata_label_map)
    positive_keywords = parse_csv_list(args.positive_keywords)
    negative_keywords = parse_csv_list(args.negative_keywords)

    raw_dir = download_dataset(
        dataset=args.dataset,
        download_dir=Path(args.download_dir),
        skip_download=args.skip_download,
    )

    if args.source_subdir:
        source_dir = raw_dir / args.source_subdir
        if not source_dir.exists():
            raise FileNotFoundError(f"source-subdir path not found: {source_dir}")
    else:
        source_dir = raw_dir

    summary = prepare_dataset(
        source_dir=source_dir,
        prepared_dir=Path(args.prepared_dir),
        val_ratio=args.val_ratio,
        balance_train=args.balance_train,
        seed=args.seed,
        explicit_map=explicit_map,
        positive_keywords=positive_keywords,
        negative_keywords=negative_keywords,
        metadata_csv=args.metadata_csv,
        image_dir=args.image_dir,
        id_column=args.id_column,
        label_column=args.label_column,
        metadata_label_map=metadata_label_map,
        strict_metadata=args.strict_metadata,
        force=args.force,
    )

    print("Dataset prepared successfully.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
