# Melanoma CDS Microservice (Diploma Starter)

This repository provides a practical start for your diploma project:

- CNN model training and model comparison (`ResNet50`, `ResNet50v2`, `AlexNet`, `VGG19`)
- FastAPI microservice for inference
- Output fields aligned with your presentation:
  - melanoma probability
  - predicted class
  - confidence
  - uncertainty
  - Grad-CAM heatmap (base64 PNG)

## 1. Setup

```bash
uv venv
# Windows PowerShell:
. .venv/Scripts/Activate.ps1
uv sync
```

Optional env config:

```bash
copy .env.example .env
```

## 2. Kaggle Download + Auto-split

Before running download, configure Kaggle credentials:

```text
KAGGLE_API_TOKEN
or
%USERPROFILE%/.kaggle/access_token
legacy fallback:
%USERPROFILE%/.kaggle/kaggle.json
```

Prepare dataset in the expected structure with one command:

```bash
python -m training.prepare_kaggle_dataset \
  --dataset kmader/skin-cancer-mnist-ham10000 \
  --download-dir data/raw \
  --prepared-dir data/prepared \
  --val-ratio 0.2 \
  --force
```

For ISIC 2020 style datasets (flat images + metadata CSV with `target`):

```bash
python -m training.prepare_kaggle_dataset \
  --dataset nischaydnk/isic-2020-jpg-224x224-resized \
  --metadata-csv train-metadata.csv \
  --image-dir train-image/image \
  --id-column isic_id \
  --label-column target \
  --prepared-dir data/prepared \
  --val-ratio 0.2 \
  --force
```

If your CSV name differs, replace `--metadata-csv` with the real filename (for example `train.csv`).
This ISIC 2020 dataset is naturally imbalanced (~32542 benign vs ~584 melanoma), so this is expected.

To balance TRAIN split (recommended for training), use:

```bash
python -m training.prepare_kaggle_dataset \
  --dataset nischaydnk/isic-2020-jpg-224x224-resized \
  --metadata-csv train-metadata.csv \
  --image-dir train-image/image \
  --id-column isic_id \
  --label-column target \
  --balance-train oversample \
  --prepared-dir data/prepared \
  --val-ratio 0.2 \
  --force
```

If class folders are nested inside a subdirectory, specify it:

```bash
python -m training.prepare_kaggle_dataset \
  --dataset <owner>/<slug> \
  --source-subdir train \
  --prepared-dir data/prepared \
  --force
```

If folder names are non-standard, map explicitly:

```bash
python -m training.prepare_kaggle_dataset \
  --dataset <owner>/<slug> \
  --class-map "malignant=melanoma,non_melanoma=benign" \
  --prepared-dir data/prepared \
  --force
```

Output layout:

```text
data/prepared/
  train/
    benign/
    melanoma/
  val/
    benign/
    melanoma/
```

## 3. Train One Model

```bash
python -m training.train \
  --data-dir data/prepared \
  --model-name resnet50 \
  --class-weight-mode inverse_frequency \
  --target-specificity 0.85 \
  --epochs 20 \
  --device cuda \
  --output-path models/resnet50_best.pt
```

The script also writes `models/resnet50_best.metrics.json`.
Available values for `--model-name`: `resnet50`, `resnet50v2`, `alexnet`, `vgg19`.
It also generates:
- `<model>_history.csv` and `<model>_history.json` (epoch-by-epoch train/val metrics)
- `<model>_training_curves.png` (loss/accuracy/error/validation metrics)
- `<model>_confusion_matrix_heatmap.png`
- `<model>_roc_curve.png`
- `<model>_probability_histogram.png`
Best-metrics include: ROC-AUC, PR-AUC, precision/recall/F1, balanced accuracy,
and selected threshold metrics for target specificity.

## 4. Compare All 4 Models

```bash
python -m training.compare_models \
  --data-dir data/prepared \
  --models resnet50,resnet50v2,alexnet,vgg19 \
  --uniform-hparams \
  --optimizer sgd \
  --lr 0.003 \
  --momentum 0.9 \
  --weight-decay 1e-4 \
  --class-weight-mode inverse_frequency \
  --target-specificity 0.85 \
  --epochs 20 \
  --device cuda \
  --output-dir outputs
```

Result:
- `outputs/comparison_summary.csv`
- `outputs/model_comparison_metrics.png`
- best checkpoint + metrics for each model

For CPU debug / quick health-check:

```bash
python -m training.compare_models \
  --data-dir data/prepared \
  --models resnet50 \
  --device cpu \
  --epochs 1 \
  --batch-size 8 \
  --num-workers 0 \
  --log-every 20 \
  --max-train-batches 100 \
  --max-val-batches 30 \
  --output-dir outputs/debug
```

Run comparison for a specific model only:

```bash
python -m training.compare_models \
  --data-dir data/prepared \
  --models vgg19 \
  --uniform-hparams \
  --epochs 20 \
  --device cuda \
  --output-dir outputs/vgg19_only
```

## 5. Run API

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Endpoints

- `GET /health`
- `GET /v1/model-info`
- `POST /v1/predict` (multipart form-data: `image`, query `mode=checkpoint|screening|high_specificity`)
- `POST /v1/predict-with-heatmap` (multipart form-data: `image`, query `mode=checkpoint|screening|high_specificity`)

Threshold modes:
- `checkpoint`: threshold saved in model checkpoint.
- `screening`: uses the configurable `SCREENING_THRESHOLD` env var.
- `high_specificity`: uses `HIGH_SPECIFICITY_THRESHOLD` env var.

## 6. Quick cURL Examples

```bash
curl -X POST "http://127.0.0.1:8000/v1/predict" \
  -F "image=@C:/path/to/image.jpg"
```

```bash
curl -X POST "http://127.0.0.1:8000/v1/predict-with-heatmap?mode=high_specificity" \
  -F "image=@C:/path/to/image.jpg"
```

## 8. Docker / Docker Compose (uv)

Create `.env` from `.env.example`, set `MODEL_WEIGHTS_PATH` and thresholds.

Build and run:

```bash
docker compose up --build -d
```

Check:

```bash
curl http://127.0.0.1:8000/health
```

Check recent prediction logs in PostgreSQL:

```bash
docker compose exec postgres psql -U melanoma -d melanoma -c "SELECT id, created_at, mode, class_label, melanoma_probability, processing_ms FROM prediction_requests ORDER BY id DESC LIMIT 5;"
```

Stop:

```bash
docker compose down
```

## 9. Telegram Bot (aiogram)

The bot accepts image messages and forwards them to your API `/v1/predict`.

Required env vars:
- `TELEGRAM_BOT_TOKEN`
- `API_BASE_URL` (example: `http://127.0.0.1:8000`)
- `API_MODE` (`checkpoint` | `screening` | `high_specificity`)
- if API auth is enabled: `ENABLE_API_AUTH=true` and `API_TOKEN=<token>`

Run bot:

```bash
python -m bot.main
```

## 10. What to Add Next (for full diploma scope)

- probability calibration (temperature scaling / isotonic)
- threshold tuning for target sensitivity/specificity
- optional medical-case history and image object storage
- API gateway + JWT/OAuth2
- MLflow experiment tracking
- Docker Compose with monitoring/logging stack

## 11. Final External Validation (No Retraining)

1) Download and prepare an external dataset (example: ISIC 2019):

```bash
python -m training.prepare_kaggle_dataset \
  --dataset andrewmvd/isic-2019 \
  --download-dir data/raw \
  --prepared-dir data/prepared_external_isic2019 \
  --source-subdir train \
  --class-map "melanoma=melanoma,melanocytic nevus=benign,basal cell carcinoma=benign,actinic keratosis=benign,benign keratosis=benign,dermatofibroma=benign,vascular lesion=benign,squamous cell carcinoma=benign,none of the above=benign" \
  --val-ratio 0.2 \
  --seed 42 \
  --force
```

2) Run final validation of your selected checkpoint (example with ResNet50v2):

```bash
python -m training.evaluate_checkpoint \
  --data-dir data/prepared_external_isic2019 \
  --weights-path outputs/phaseB3/resnet50v2_c5_seed42_best.pt \
  --device cuda \
  --batch-size 16 \
  --target-specificity 0.85 \
  --output-dir outputs/final_validation/isic2019
```

Result files:
- `outputs/final_validation/isic2019/*_external_val.metrics.json`
- ROC, confusion matrix heatmap, probability histogram PNGs.
