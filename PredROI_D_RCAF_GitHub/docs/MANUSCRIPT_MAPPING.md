# Manuscript-to-code mapping

| Manuscript component | Main code |
|---|---|
| Leakage-safe patient-disjoint PredROI training | `src/cheng_predroi_swin_full5_locked_v1_1.py` and shared utilities |
| D-RCAF patient-disjoint five-fold model | `src/cheng_drcaf_predroi_swin_full5.py` |
| Fixed equal-weight fusion | `src/cheng_fixed05_predroi_swin_full5.py` |
| Fixed-patient-split multi-seed D-RCAF | `src/cheng_drcaf_predroi_swin_full5_fixedsplit.py` |
| Slice-random literature-matched control | `src/cheng_drcaf_slice_random_literature_control.py` |
| External frozen ensemble baseline | `src/bdneuro_external_validation_frozen_ensemble.py` |
| External Fixed-0.5 evaluation | `src/bdneuro_external_validation_fixed05_ensemble.py` |
| External D-RCAF evaluation | `src/bdneuro_external_validation_drcaf_ensemble.py` |
| D-RCAF vs PredROI paired statistics | `analysis/compare_drcaf_predroi_paired.py` |
| Multi-seed summary | `analysis/summarize_drcaf_fixedsplit_multiseed.py` |
| Reviewer-closure analyses | `reviewer_closure/` |

## Important interpretation

The primary methodological contribution is the leakage-safe, patient-disjoint, cross-fitted PredROI protocol. D-RCAF is a secondary global-local fusion extension. On the locked Cheng patient-level benchmark, D-RCAF and Fixed-0.5 reached the same patient accuracy; therefore the repository should not be described as proving superiority of adaptive fusion over fixed equal weighting.
