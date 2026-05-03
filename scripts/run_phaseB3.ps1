# Phase B3: stability check (3 seeds) for top-2 configs
# Models/configs from B1 winners:
# - resnet50 c5:   adamw lr=1e-4 wd=1e-4
# - resnet50v2 c5: adamw lr=1e-4 wd=1e-4

$runDir = "outputs/phaseB3"
New-Item -ItemType Directory -Force -Path $runDir | Out-Null

$seeds = @(42, 52, 62)

$runs = @(
  @{ model="resnet50";   cfg="c5"; opt="adamw"; lr="0.0001"; wd="0.0001"; cwm="inverse_frequency"; ts="0.85" },
  @{ model="resnet50v2"; cfg="c5"; opt="adamw"; lr="0.0001"; wd="0.0001"; cwm="inverse_frequency"; ts="0.85" }
)

foreach ($r in $runs) {
  foreach ($s in $seeds) {
    $out = "$runDir/$($r.model)_$($r.cfg)_seed$s" + "_best.pt"
    $metrics = $out -replace "\.pt$", ".metrics.json"

    if (Test-Path $metrics) {
      Write-Host "Skip existing: $metrics"
      continue
    }

    Write-Host "Run: model=$($r.model) cfg=$($r.cfg) seed=$s"
    uv run python -m training.train `
      --data-dir data/prepared `
      --model-name $($r.model) `
      --epochs 30 `
      --batch-size 16 `
      --image-size 224 `
      --optimizer $($r.opt) `
      --lr $($r.lr) `
      --weight-decay $($r.wd) `
      --device cuda `
      --num-workers 0 `
      --class-weight-mode $($r.cwm) `
      --target-specificity $($r.ts) `
      --selection-metric pr_auc `
      --early-stopping-metric pr_auc `
      --early-stopping-patience 5 `
      --early-stopping-min-delta 0.001 `
      --seed $s `
      --output-path $out
  }
}

# Aggregate B3 results (mean/std by model)
$rows = Get-ChildItem $runDir -Filter "*_best.metrics.json" | ForEach-Object {
  $j = Get-Content $_.FullName -Raw | ConvertFrom-Json
  $m = $j.best_metrics
  $cfg = $j.config
  $name = [System.IO.Path]::GetFileNameWithoutExtension($_.Name)
  $mx = [regex]::Match($name, '^(?<model>.+)_c5_seed(?<seed>\d+)_best\.metrics$')

  [PSCustomObject]@{
    model = $mx.Groups['model'].Value
    seed = [int]$mx.Groups['seed'].Value
    pr_auc = [double]$m.pr_auc
    auc = [double]$m.auc
    selected_sensitivity = [double]$m.selected_sensitivity
    selected_specificity = [double]$m.selected_specificity
    selected_f1 = [double]$m.selected_f1
    best_epoch = [int]$m.best_epoch
    train_seconds = [double]$m.train_seconds
    metrics_file = $_.FullName
  }
}

$rows | Sort-Object model, seed | Export-Csv "$runDir/runs_summary.csv" -NoTypeInformation -Encoding UTF8

$summary = $rows | Group-Object model | ForEach-Object {
  $g = $_.Group
  [PSCustomObject]@{
    model = $_.Name
    n = $g.Count
    pr_auc_mean = ($g | Measure-Object -Property pr_auc -Average).Average
    pr_auc_std  = [Math]::Sqrt((($g | ForEach-Object { [Math]::Pow($_.pr_auc - (($g | Measure-Object -Property pr_auc -Average).Average), 2) } | Measure-Object -Sum).Sum) / [Math]::Max(1, $g.Count - 1))
    auc_mean = ($g | Measure-Object -Property auc -Average).Average
    sel_sens_mean = ($g | Measure-Object -Property selected_sensitivity -Average).Average
    sel_spec_mean = ($g | Measure-Object -Property selected_specificity -Average).Average
    sel_f1_mean = ($g | Measure-Object -Property selected_f1 -Average).Average
    train_hours_mean = (($g | Measure-Object -Property train_seconds -Average).Average / 3600.0)
  }
}

$summary | Export-Csv "$runDir/model_stability_summary.csv" -NoTypeInformation -Encoding UTF8
$summary | Format-Table model,n,pr_auc_mean,pr_auc_std,auc_mean,sel_sens_mean,sel_spec_mean,sel_f1_mean,train_hours_mean -AutoSize
