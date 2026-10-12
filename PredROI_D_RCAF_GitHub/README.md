# Leakage-Safe PredROI + D-RCAF for Brain Tumor MRI

Code package for the manuscript **“Leakage-Safe Cross-Fitted Predicted-ROI Classification for Brain Tumor MRI: Patient-Disjoint Evaluation, Global-Local Fusion, and External Validation.”**

The repository implements a patient-disjoint, cross-fitted predicted-ROI (PredROI) pipeline for the Cheng CE-MRI cohort, controlled global-local fusion experiments (Fixed-0.5 and D-RCAF), a slice-random literature-matched control, fixed-split multi-seed sensitivity analysis, and frozen external evaluation on BDNeuro-MRI.

## What is the main methodological point?

The central contribution is the **leakage-safe crop-generation protocol**. Tumor masks supervise localizers only on training patients. Classifier-training ROIs are generated out of fold at the patient level, and validation/test ROIs are generated from the MRI image alone. The adaptive D-RCAF gate is a secondary fusion extension rather than the sole novelty claim.

## Repository layout

```text
PredROI_D_RCAF_GitHub/
├── src/                 # Core training and evaluation scripts
├── analysis/            # Paired statistics and multi-seed summaries
├── reviewer_closure/    # Post-lock audit / reviewer-closure scripts
├── scripts/             # PowerShell launchers with editable paths
├── examples/            # Example Cheng manifest
├── figures/             # Manuscript workflow figures
├── docs/                # Data, reproducibility, and manuscript mapping notes
├── requirements.txt
├── .gitignore
└── LICENSE_NOTICE.md
```

## Installation

Windows / PowerShell example:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Install a PyTorch/CUDA build appropriate for your GPU if the generic `pip install` command does not select the desired CUDA build.

## Data

Datasets and checkpoints are **not included**. See `docs/DATA_LAYOUT.md` and `examples/cheng_manifest_example.csv`.

The Cheng manifest must contain a valid image path, patient ID, and class label/name for every slice. The MAT directory must contain the mask/metadata file corresponding to each PNG stem.

## Quick start: locked D-RCAF patient-disjoint run

Edit the three paths in `scripts/run_locked_drcaf.ps1`, then run:

```powershell
.\scripts\run_locked_drcaf.ps1
```

Equivalent direct command:

```powershell
$env:PYTHONPATH = "$PWD\src"
python .\src\cheng_drcaf_predroi_swin_full5.py `
  --manifest "C:\PATH\TO\cheng_manifest.csv" `
  --mat_root "C:\PATH\TO\cheng_mat" `
  --out_root "C:\PATH\TO\results\drcaf_seed5813" `
  --outer_seed 5813 `
  --train_folds 5 `
  --amp_dtype bf16
```

## Fixed equal-weight control

```powershell
.\scripts\run_fixed05.ps1
```

This model uses the same shared global/PredROI encoder structure but fixes the fusion weight at 0.5.

## Slice-random control

```powershell
.\scripts\run_slice_random_control.ps1
```

This is a secondary literature-matched control. It is not evidence of patient-independent generalization.

## Fixed-split multi-seed sensitivity

```powershell
.\scripts\run_multiseed.ps1
```

Patient fold membership remains fixed at split seed 5813; only the stochastic training seed changes.

## External BDNeuro-MRI evaluation

See `scripts/run_external.ps1`. Before running it, provide the Cheng-trained Full-image, reconstructed PredROI, Fixed-0.5, and D-RCAF checkpoint roots in the PowerShell file. The script first creates the frozen Full/PredROI/Fixed external reference predictions and then evaluates D-RCAF against those predictions. The external analysis is image-level because patient IDs are unavailable. No BDNeuro fine-tuning is performed.

## Reviewer / audit analyses

The `reviewer_closure/` folder contains scripts for paired statistics, PredROI reconstruction audit, cached-box component ablation, and the 15-epoch full-image control. These analyses are separated from the primary pipeline so their post-lock status remains transparent.

## Reported headline results

- Locked Cheng full-image Swin: 90.1% patient accuracy.
- Locked Cheng PredROI-Swin: 94.4% patient accuracy.
- Locked Cheng Fixed-0.5: 96.1% patient accuracy.
- Locked Cheng D-RCAF: 96.1% patient accuracy.
- Slice-random D-RCAF control: 97.8% slice accuracy.
- External BDNeuro-MRI (image-level): Full 82.4%, protocol-reconstructed PredROI 90.5%, Fixed-0.5 88.6%, D-RCAF 91.3% accuracy.

These results use different evaluation units where stated and should not be interpreted as a single leaderboard.

## Figures

`figures/` contains the global/PredROI fusion graph and the patient-disjoint cross-fitting workflow used to explain the method.

## Citation

Please cite the associated manuscript when using this code. A final DOI/BibTeX entry can be added after publication.

## License

No open-source license is included in this archive. Add the intended license before public release.
