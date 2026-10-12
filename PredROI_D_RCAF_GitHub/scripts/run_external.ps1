$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = "$Repo\src"

# EDIT THESE PATHS
$DatasetRoot   = "C:\PATH\TO\BDNeuro_MRI_v8"
$BaselineRoot  = "C:\PATH\TO\results\full_image_seed5813"
$PredROIRoot   = "C:\PATH\TO\results\predroi_seed5813"
$FixedRoot     = "C:\PATH\TO\results\fixed05_seed5813"
$DrcafRoot     = "C:\PATH\TO\results\drcaf_seed5813"
$FixedOut      = "C:\PATH\TO\results\bdneuro_fixed05"
$DrcafOut      = "C:\PATH\TO\results\bdneuro_drcaf"

# First create the frozen reference predictions (Full, reconstructed PredROI, Fixed-0.5).
python "$Repo\src\bdneuro_external_validation_fixed05_ensemble.py" `
  --dataset_root "$DatasetRoot" `
  --out_root "$FixedOut" `
  --baseline_root "$BaselineRoot" `
  --predroi_root "$PredROIRoot" `
  --fixed05_root "$FixedRoot" `
  --fixed_localizer_root "$DrcafRoot"

if ($LASTEXITCODE -ne 0) { throw "Fixed-0.5 external evaluation failed." }

# Then add D-RCAF and compare against the frozen reference predictions above.
python "$Repo\src\bdneuro_external_validation_drcaf_ensemble.py" `
  --dataset_root "$DatasetRoot" `
  --out_root "$DrcafOut" `
  --reference_root "$FixedOut" `
  --drcaf_root "$DrcafRoot"

if ($LASTEXITCODE -ne 0) { throw "D-RCAF external evaluation failed." }
