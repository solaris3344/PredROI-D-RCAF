$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = "$Repo\src"

# EDIT THESE PATHS
$Manifest = "C:\PATH\TO\cheng_manifest.csv"
$MatRoot  = "C:\PATH\TO\cheng_mat"
$OutRoot  = "C:\PATH\TO\results\drcaf_seed5813"

python "$Repo\src\cheng_drcaf_predroi_swin_full5.py" `
  --manifest "$Manifest" `
  --mat_root "$MatRoot" `
  --out_root "$OutRoot" `
  --outer_seed 5813 `
  --train_folds 5 `
  --amp_dtype bf16
