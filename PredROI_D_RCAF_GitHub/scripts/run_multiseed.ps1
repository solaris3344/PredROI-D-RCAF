$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = "$Repo\src"

# EDIT THESE PATHS
$Manifest = "C:\PATH\TO\cheng_manifest.csv"
$MatRoot  = "C:\PATH\TO\cheng_mat"
$ResultsRoot = "C:\PATH\TO\results"

$SplitSeed = 5813
$RunSeeds = @(7253, 9341, 12011)

foreach ($RunSeed in $RunSeeds) {
  $Out = Join-Path $ResultsRoot ("drcaf_full5_split5813_run" + $RunSeed)
  python "$Repo\src\cheng_drcaf_predroi_swin_full5_fixedsplit.py" `
    --manifest "$Manifest" `
    --mat_root "$MatRoot" `
    --out_root "$Out" `
    --split_seed $SplitSeed `
    --run_seed $RunSeed `
    --train_folds 5 `
    --amp_dtype bf16
  if ($LASTEXITCODE -ne 0) { throw "Run seed $RunSeed failed." }
}

python "$Repo\analysis\summarize_drcaf_fixedsplit_multiseed.py" `
  --root "$ResultsRoot" `
  --original "drcaf_full5_seed5813" `
  --fresh "drcaf_full5_split5813_run7253" "drcaf_full5_split5813_run9341" "drcaf_full5_split5813_run12011" `
  --out_dir "drcaf_multiseed_fixedsplit_summary"
