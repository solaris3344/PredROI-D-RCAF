REVIEWER CLOSURE CODES — Brain MRI PredROI / D-RCAF

Put these files in:
C:\PATH\TO\brain_project

Required existing project files in the same folder:
- cheng_patient_set_leakage_free_v2.py
- cheng_predroi_swin_screen_v1_audited.py
- cheng_predroi_swin_full5_locked_v1_1.py
- cheng_drcaf_predroi_swin_full5.py

Recommended order
=================

STEP 1 — TRAINING-FREE paired statistics
----------------------------------------
Computes:
A) Internal Fixed-0.5 vs D-RCAF:
   rescue/lost, exact McNemar, paired patient bootstrap.
B) External Fixed-0.5 vs PredROI:
   rescue/lost, descriptive paired image bootstrap.

Command:
python .\reviewer_missing_paired_stats.py `
  --brain_root "C:\PATH\TO\brain_project" `
  --fixed_patient_oof "C:\PATH\TO\brain_project\fixed05_full5_seed5813\patient_oof.csv" `
  --drcaf_patient_oof "C:\PATH\TO\brain_project\drcaf_full5_seed5813\patient_oof.csv" `
  --external_reference_csv "C:\PATH\TO\brain_project\bdneuro_fixed05_external\external_predictions_three_methods.csv"

STEP 2 — PredROI reconstruction audit on Cheng
-----------------------------------------------
No training. It reloads the reconstructed PredROI classifier/localizer checkpoints,
reruns Cheng outer-val inference, and compares against the original locked OOF CSVs.

Command:
python .\audit_predroi_reconstruction_on_cheng.py `
  --locked_root "C:\PATH\TO\brain_project\predroi_swin_seed5813_LOCKED" `
  --manifest "C:\PATH\TO\brain_project\cheng_patient_disjoint\cheng_manifest.csv"

If your locked PredROI folder has another name, change --locked_root only.

STEP 3 — D-RCAF mechanism ablation (HEAVIER)
--------------------------------------------
Trains classifiers only. It REUSES all locked predicted-box caches and does NOT
retrain localizers.

Variants:
- no_d
- no_q
- scalar

Reference full D-RCAF is reused from existing patient_oof.csv.

Command:
python .\cheng_drcaf_component_ablation_cached_boxes.py `
  --manifest "C:\PATH\TO\brain_project\cheng_patient_disjoint\cheng_manifest.csv" `
  --cache_root "C:\PATH\TO\brain_project\drcaf_full5_seed5813" `
  --out_root "C:\PATH\TO\brain_project\drcaf_component_ablation_seed5813" `
  --amp_dtype bf16

STEP 4 — 15-epoch full-image matched-budget control (HEAVIER)
-------------------------------------------------------------
Runs standard Swin under the same 15-epoch maximum selection budget used by
PredROI/fusion arms, on seed5813 patient folds.

Command:
python .\cheng_swin_baseline_seed5813_15epoch_control.py `
  --manifest "C:\PATH\TO\brain_project\cheng_patient_disjoint\cheng_manifest.csv" `
  --out_root "C:\PATH\TO\brain_project\swin_baseline_seed5813_15epoch_control" `
  --train_folds 5 `
  --amp_dtype bf16

FAST MASTER
===========
run_reviewer_closure_steps.ps1 runs only STEP 1 and STEP 2 automatically, then
prints commands for STEP 3 and STEP 4. This is intentional: inspect the cheap
results before spending GPU time.

PowerShell:
powershell -ExecutionPolicy Bypass -File .\run_reviewer_closure_steps.ps1

Scientific interpretation
=========================
- STEP 1 internal paired test is post-lock/descriptive.
- STEP 1 external bootstrap is IMAGE-level descriptive only because BDNeuro
  patient IDs are unavailable.
- STEP 2 can establish measured reconstruction concordance; if not exact, do
  not call the reconstructed model "identical".
- STEP 3 is a post-lock mechanism ablation and should be labelled as such.
- STEP 4 is a reviewer sensitivity control, not a replacement of the original
  locked historical baseline unless you explicitly decide to reframe the paper.
