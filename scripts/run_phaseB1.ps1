$models = @("resnet50","resnet50v2","alexnet","vgg19")
$configs = @(
  @{id="c1"; opt="sgd";   lr="0.003";  wd="0.0001"; cwm="inverse_frequency"; ts="0.85"},
  @{id="c2"; opt="sgd";   lr="0.001";  wd="0.0001"; cwm="inverse_frequency"; ts="0.85"},
  @{id="c3"; opt="sgd";   lr="0.01";   wd="0.0001"; cwm="inverse_frequency"; ts="0.85"},
  @{id="c4"; opt="adamw"; lr="0.0003"; wd="0.0001"; cwm="inverse_frequency"; ts="0.85"},
  @{id="c5"; opt="adamw"; lr="0.0001"; wd="0.0001"; cwm="inverse_frequency"; ts="0.85"},
  @{id="c6"; opt="adamw"; lr="0.0003"; wd="0.001";  cwm="inverse_frequency"; ts="0.85"}
)

$runDir = "outputs/phaseB1"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

foreach ($m in $models) {
  foreach ($c in $configs) {
    $out = "$runDir/${m}_$($c.id)_best.pt"
    $metrics = $out -replace "\.pt$", ".metrics.json"

    if (Test-Path $metrics) {
      Write-Host "Skip existing: $metrics"
      continue
    }

    Write-Host "Run: model=$m config=$($c.id)"
    uv run python -m training.train `
      --data-dir data/prepared `
      --model-name $m `
      --epochs 30 `
      --batch-size 16 `
      --image-size 224 `
      --optimizer $($c.opt) `
      --lr $($c.lr) `
      --weight-decay $($c.wd) `
      --device cuda `
      --num-workers 0 `
      --class-weight-mode $($c.cwm) `
      --target-specificity $($c.ts) `
      --selection-metric pr_auc `
      --early-stopping-metric pr_auc `
      --early-stopping-patience 5 `
      --early-stopping-min-delta 0.001 `
      --seed 42 `
      --output-path $out
  }
}
