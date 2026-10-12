$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = "$Repo\src"

# EDIT THESE PATHS
$Manifest = "C:\PATH\TO\cheng_manifest.csv"
$MatRoot  = "C:\PATH\TO\cheng_mat"
$DrcafRoot = "C:\PATH\TO\results\drcaf_seed5813"
$OutRoot  = "C:\PATH\TO\results\fixed05_seed5813"

python "$Repo\src\cheng_fixed05_predroi_swin_full5.py" `
  --manifest "$Manifest" `
  --mat_root "$MatRoot" `
  --reuse_boxes_root "$DrcafRoot" `
  --out_root "$OutRoot" `
  --outer_seed 5813 `
  --train_folds 5 `
  --amp_dtype bf16
