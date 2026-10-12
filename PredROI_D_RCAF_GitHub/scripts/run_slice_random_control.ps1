$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = "$Repo\src"

# EDIT THESE PATHS
$Manifest = "C:\PATH\TO\cheng_manifest.csv"
$MatRoot  = "C:\PATH\TO\cheng_mat"
$OutRoot  = "C:\PATH\TO\results\slice_random_seed123"

python "$Repo\src\cheng_drcaf_slice_random_literature_control.py" `
  --manifest "$Manifest" `
  --mat_root "$MatRoot" `
  --out_root "$OutRoot" `
  --amp_dtype bf16
