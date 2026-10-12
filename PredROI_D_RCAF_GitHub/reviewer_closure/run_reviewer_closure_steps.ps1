param(
    [string]$Root = "C:\PATH\TO\brain_project",
    [string]$Manifest = "C:\PATH\TO\brain_project\cheng_patient_disjoint\cheng_manifest.csv",
    [string]$PredROIRoot = "C:\PATH\TO\brain_project\predroi_swin_seed5813_LOCKED",
    [string]$DRCAFRoot = "C:\PATH\TO\brain_project\drcaf_full5_seed5813",
    [string]$FixedRoot = "C:\PATH\TO\brain_project\fixed05_full5_seed5813",
    [string]$ExternalRefRoot = "C:\PATH\TO\brain_project\bdneuro_fixed05_external"
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "STEP 1/4 — Training-free paired statistics"
python "$Root\reviewer_missing_paired_stats.py" `
  --brain_root "$Root" `
  --fixed_patient_oof "$FixedRoot\patient_oof.csv" `
  --drcaf_patient_oof "$DRCAFRoot\patient_oof.csv" `
  --external_reference_csv "$ExternalRefRoot\external_predictions_three_methods.csv"

if ($LASTEXITCODE -ne 0) { throw "STEP 1 failed" }

Write-Host ""
Write-Host "STEP 2/4 — PredROI reconstruction replay audit on Cheng"
python "$Root\audit_predroi_reconstruction_on_cheng.py" `
  --locked_root "$PredROIRoot" `
  --manifest "$Manifest"

if ($LASTEXITCODE -ne 0) { throw "STEP 2 failed" }

Write-Host ""
Write-Host "CHEAP REVIEWER-CLOSURE STEPS FINISHED."
Write-Host "Inspect their outputs BEFORE running the two heavier controls."
Write-Host ""
Write-Host "Optional STEP 3 — D-RCAF component ablation (classifier-only; cached boxes)"
Write-Host "python `"$Root\cheng_drcaf_component_ablation_cached_boxes.py`" --manifest `"$Manifest`" --cache_root `"$DRCAFRoot`" --out_root `"$Root\drcaf_component_ablation_seed5813`""
Write-Host ""
Write-Host "Optional STEP 4 — 15-epoch full-image matched-budget control"
Write-Host "python `"$Root\cheng_swin_baseline_seed5813_15epoch_control.py`" --manifest `"$Manifest`" --out_root `"$Root\swin_baseline_seed5813_15epoch_control`" --train_folds 5 --amp_dtype bf16"
