# PowerShell launchers

All launcher files set `PYTHONPATH` to the repository `src` directory. Edit only the paths marked `EDIT THESE PATHS`.

- `run_locked_drcaf.ps1`: main locked patient-disjoint D-RCAF run.
- `run_fixed05.ps1`: deterministic equal-weight control using cached predicted boxes from the D-RCAF run.
- `run_slice_random_control.ps1`: secondary slice-random literature-matched control.
- `run_multiseed.ps1`: fixed patient folds, new stochastic seeds.
- `run_external.ps1`: BDNeuro external image-level evaluation.
