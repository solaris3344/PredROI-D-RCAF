#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
audit_predroi_reconstruction_on_cheng.py

Audit the reconstructed PredROI-Swin fold checkpoints against the ORIGINAL
locked Cheng OOF predictions.

The original final PredROI checkpoints were not saved. Later, the fold models
were reconstructed from the locked protocol/selected epochs for external use.

This audit now:
  1) loads each reconstructed final_classifier.pt + final_localizer.pt;
  2) regenerates image-only OUTER-VAL boxes on Cheng;
  3) re-runs reconstructed PredROI inference on the same OUTER-VAL slices;
  4) compares replayed slice probabilities/predictions to the original locked
     fold_i/outer_slice_predictions.csv;
  5) compares patient predictions/metrics to original patient_oof.csv;
  6) reports foldwise probability error, slice agreement, patient agreement,
     and aggregate metrics.

Interpretation:
  - Exact replay is ideal but may fail if the original run was nondeterministic.
  - High patient/slice agreement still provides a quantitative reconstruction
    audit.
  - This script does NOT tune anything and does NOT use BDNeuro.
"""
from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

import cheng_patient_set_leakage_free_v2 as common
import cheng_predroi_swin_screen_v1_audited as pred
import cheng_predroi_swin_full5_locked_v1_1 as full5


CLASSES = common.CLASSES


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_state(path, device):
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"], obj
    return obj, {}


def norm_path(x):
    return str(Path(str(x)).resolve()).lower()


def load_runtime_args(locked_root: Path, cli):
    cfg_path = locked_root / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    a = Namespace(**cfg)

    defaults = {
        "outer_seed": 5813,
        "n_folds": 5,
        "seg_size": 256,
        "seg_eval_batch_size": 16,
        "box_threshold": 0.50,
        "box_fallback_threshold": 0.30,
        "crop_margin_factor": 1.10,
        "backbone": "swin_base_patch4_window7_224",
        "image_size": 224,
        "eval_batch_size": 32,
        "workers": 0,
        "dropout": 0.15,
        "amp_dtype": "bf16",
    }
    for k,v in defaults.items():
        if not hasattr(a, k):
            setattr(a, k, v)
    if cli.manifest:
        a.manifest = cli.manifest
    if cli.mat_root:
        a.mat_root = cli.mat_root
    a.outer_fold = 1
    return a


def validate_box_coverage(boxes, patients):
    df = pd.DataFrame(boxes)
    exp = {
        str(q): (str(p.patient_id), int(p.label))
        for p in patients for q in p.paths
    }
    if set(df["path"].astype(str)) != set(exp):
        raise RuntimeError("Replayed outer-val box path coverage mismatch.")
    for r in df.itertuples(index=False):
        pid, lab = exp[str(r.path)]
        if str(r.patient_id) != pid or int(r.label) != lab:
            raise RuntimeError("Replayed outer-val box patient/label mismatch.")


def compare_slice_frames(original, replay):
    a = original.copy()
    b = replay.copy()
    a["_key"] = a["path"].map(norm_path)
    b["_key"] = b["path"].map(norm_path)
    if a["_key"].duplicated().any() or b["_key"].duplicated().any():
        raise RuntimeError("Duplicate slice path during audit.")
    z = a.merge(b, on="_key", suffixes=("_orig","_replay"), validate="one_to_one")
    if len(z) != len(a) or len(z) != len(b):
        raise RuntimeError(
            f"Slice coverage mismatch: original={len(a)} replay={len(b)} merged={len(z)}"
        )

    truth_match = bool(np.array_equal(
        z["true_id_orig"].to_numpy(int), z["true_id_replay"].to_numpy(int)
    ))
    pred_match = float(np.mean(
        z["slice_pred_id_orig"].to_numpy(int) == z["slice_pred_id_replay"].to_numpy(int)
    ))

    diffs = []
    corrs = []
    for cname in CLASSES:
        x = z[f"prob_{cname}_orig"].to_numpy(float)
        y = z[f"prob_{cname}_replay"].to_numpy(float)
        diffs.append(np.abs(x-y))
        if np.std(x) > 0 and np.std(y) > 0:
            corrs.append(float(np.corrcoef(x,y)[0,1]))
    diffs = np.concatenate(diffs)
    return z, {
        "truth_match": truth_match,
        "slice_prediction_agreement_pct": 100.0 * pred_match,
        "max_abs_probability_diff": float(diffs.max()) if len(diffs) else 0.0,
        "mean_abs_probability_diff": float(diffs.mean()) if len(diffs) else 0.0,
        "mean_class_probability_correlation": float(np.mean(corrs)) if corrs else float("nan"),
    }


def compare_patient_frames(original, replay):
    a = original[["patient_id","true_id","pred_id"]].copy()
    b = replay[["patient_id","true_id","pred_id"]].copy()
    a["patient_id"] = a["patient_id"].astype(str)
    b["patient_id"] = b["patient_id"].astype(str)
    z = a.merge(b, on="patient_id", suffixes=("_orig","_replay"), validate="one_to_one")
    if len(z) != len(a) or len(z) != len(b):
        raise RuntimeError("Patient coverage mismatch.")
    if not np.array_equal(z["true_id_orig"].to_numpy(int), z["true_id_replay"].to_numpy(int)):
        raise RuntimeError("Patient true-label mismatch.")
    y = z["true_id_orig"].to_numpy(int)
    po = z["pred_id_orig"].to_numpy(int)
    pr = z["pred_id_replay"].to_numpy(int)
    return z, {
        "n": int(len(z)),
        "patient_prediction_agreement_pct": float(100.0 * np.mean(po == pr)),
        "original_accuracy": float(100.0 * accuracy_score(y,po)),
        "replay_accuracy": float(100.0 * accuracy_score(y,pr)),
        "original_macro_f1": float(100.0 * f1_score(y,po,average="macro",zero_division=0)),
        "replay_macro_f1": float(100.0 * f1_score(y,pr,average="macro",zero_division=0)),
        "original_errors": int(np.sum(po != y)),
        "replay_errors": int(np.sum(pr != y)),
        "prediction_changed_patients": int(np.sum(po != pr)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--locked_root",
        default=r"C:\PATH\TO\brain_project\predroi_swin_seed5813_LOCKED",
        help="Original locked PredROI output folder containing original OOF CSVs and reconstructed checkpoints.",
    )
    ap.add_argument("--manifest", default="")
    ap.add_argument("--mat_root", default="")
    ap.add_argument("--out_root", default="")
    ap.add_argument("--prob_exact_tolerance", type=float, default=5e-4)
    args_cli = ap.parse_args()

    root = Path(args_cli.locked_root)
    args = load_runtime_args(root, args_cli)
    out = Path(args_cli.out_root) if args_cli.out_root else root / "reconstruction_replay_audit"
    out.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    slice_rows, _ = common.load_manifest(Path(args.manifest))
    splits = common.outer_patient_splits(slice_rows, int(args.n_folds), int(args.outer_seed))

    original_patient_path = root / "patient_oof.csv"
    if not original_patient_path.exists():
        raise FileNotFoundError(original_patient_path)
    original_patient = pd.read_csv(original_patient_path, dtype={"patient_id": str})

    all_replay_slice = []
    all_replay_patient = []
    fold_rows = []

    print("=" * 150)
    print("PREDROI RECONSTRUCTION REPLAY AUDIT ON LOCKED CHENG OUTER-VAL")
    print("=" * 150)
    print(f"root       : {root}")
    print(f"device     : {device}")
    print(f"outer seed : {args.outer_seed}")
    print("BDNeuro    : NOT ACCESSED")
    print("tuning     : NONE")
    print("=" * 150)

    for fold, (_tr_idx, va_idx) in enumerate(splits, start=1):
        args.outer_fold = fold
        fdir = root / f"fold_{fold}"
        cls_path = fdir / "final_classifier.pt"
        loc_path = fdir / "final_localizer.pt"
        orig_slice_path = fdir / "outer_slice_predictions.csv"

        for p in [cls_path, loc_path, orig_slice_path]:
            if not p.exists():
                raise FileNotFoundError(p)

        outer_val = common.patient_subset_from_slice_indices(slice_rows, va_idx)

        # Reconstructed final localizer -> image-only boxes.
        loc = pred.ResNet18UNet(pretrained=False).to(device)
        loc_state, loc_meta = load_state(loc_path, device)
        loc.load_state_dict(loc_state, strict=True)
        loc.eval()

        boxes = full5.predict_boxes_image_only(
            loc,
            outer_val,
            args,
            device,
            seed=int(args.outer_seed) * 8100 + fold,
            tag=f"reconstruction_audit_outer_val_f{fold}",
        )
        validate_box_coverage(boxes, outer_val)

        # Reconstructed final PredROI classifier.
        clf = pred.SwinClassifier(
            args.backbone,
            pretrained=False,
            dropout=args.dropout,
        ).to(device)
        cls_state, cls_meta = load_state(cls_path, device)
        clf.load_state_dict(cls_state, strict=True)
        clf.eval()

        srows, prows, sm, pm = full5.evaluate_outer(
            clf, outer_val, boxes, args, device, fold
        )
        replay_slice = pd.DataFrame(srows)
        replay_patient = pd.DataFrame(prows)
        original_slice = pd.read_csv(orig_slice_path, dtype={"patient_id": str})

        aligned_slice, slice_cmp = compare_slice_frames(original_slice, replay_slice)

        orig_fold_pat = original_patient[
            original_patient["fold"].astype(int) == fold
        ].copy() if "fold" in original_patient.columns else None

        if orig_fold_pat is None or len(orig_fold_pat) == 0:
            # Aggregate original slice CSV if patient_oof lacks fold.
            orig_fold_pat = pd.DataFrame(
                pred.aggregate_patient_probs(original_slice.to_dict("records"))
            )
        _, pat_cmp = compare_patient_frames(orig_fold_pat, replay_patient)

        exactish = bool(
            slice_cmp["truth_match"]
            and slice_cmp["slice_prediction_agreement_pct"] == 100.0
            and slice_cmp["max_abs_probability_diff"] <= args_cli.prob_exact_tolerance
            and pat_cmp["patient_prediction_agreement_pct"] == 100.0
        )

        fold_rows.append({
            "fold": fold,
            "selected_epoch": cls_meta.get("selected_epoch"),
            "reconstructed_before_external": cls_meta.get("reconstructed_before_external"),
            "slice_prediction_agreement_pct": slice_cmp["slice_prediction_agreement_pct"],
            "max_abs_probability_diff": slice_cmp["max_abs_probability_diff"],
            "mean_abs_probability_diff": slice_cmp["mean_abs_probability_diff"],
            "mean_class_probability_correlation": slice_cmp["mean_class_probability_correlation"],
            "patient_prediction_agreement_pct": pat_cmp["patient_prediction_agreement_pct"],
            "original_patient_accuracy": pat_cmp["original_accuracy"],
            "replay_patient_accuracy": pat_cmp["replay_accuracy"],
            "original_patient_errors": pat_cmp["original_errors"],
            "replay_patient_errors": pat_cmp["replay_errors"],
            "prediction_changed_patients": pat_cmp["prediction_changed_patients"],
            "exact_within_tolerance": exactish,
        })

        replay_slice.to_csv(out / f"fold_{fold}_replayed_slice_predictions.csv", index=False)
        replay_patient.to_csv(out / f"fold_{fold}_replayed_patient_predictions.csv", index=False)
        all_replay_slice.extend(srows)
        all_replay_patient.extend(prows)

        print(
            f"Fold {fold} | slice-agree={slice_cmp['slice_prediction_agreement_pct']:.3f}% | "
            f"patient-agree={pat_cmp['patient_prediction_agreement_pct']:.3f}% | "
            f"max|dp|={slice_cmp['max_abs_probability_diff']:.3e} | "
            f"patient err {pat_cmp['original_errors']} -> {pat_cmp['replay_errors']} | "
            f"{'EXACT/PASS' if exactish else 'NON-IDENTICAL'}"
        )

        del clf, loc
        if device.type == "cuda":
            torch.cuda.empty_cache()

    replay_patient_all = pd.DataFrame(all_replay_patient)
    replay_slice_all = pd.DataFrame(all_replay_slice)
    _, overall_pat = compare_patient_frames(original_patient, replay_patient_all)

    fold_df = pd.DataFrame(fold_rows)
    fold_df.to_csv(out / "reconstruction_replay_fold_audit.csv", index=False)
    replay_patient_all.to_csv(out / "reconstructed_patient_oof.csv", index=False)
    replay_slice_all.to_csv(out / "reconstructed_slice_oof.csv", index=False)

    summary = {
        "locked_root": str(root),
        "outer_seed": int(args.outer_seed),
        "n_folds": int(args.n_folds),
        "bdneuro_accessed": False,
        "tuning_performed": False,
        "prob_exact_tolerance": float(args_cli.prob_exact_tolerance),
        "folds": fold_rows,
        "overall_patient_comparison": overall_pat,
        "all_folds_exact_within_tolerance": bool(fold_df["exact_within_tolerance"].all()),
        "mean_slice_prediction_agreement_pct": float(fold_df["slice_prediction_agreement_pct"].mean()),
        "mean_patient_prediction_agreement_pct": float(fold_df["patient_prediction_agreement_pct"].mean()),
        "max_probability_difference_over_folds": float(fold_df["max_abs_probability_diff"].max()),
        "interpretation": (
            "Exact replay is ideal. If not exact, report the measured patient/slice agreement and "
            "treat the external PredROI ensemble as a protocol reconstruction, not a bitwise replay."
        ),
    }
    save_json(summary, out / "reconstruction_replay_summary.json")

    print("=" * 150)
    print("FINAL PREDROI RECONSTRUCTION AUDIT")
    print("=" * 150)
    print(
        f"Overall patient agreement : {overall_pat['patient_prediction_agreement_pct']:.3f}% "
        f"({overall_pat['prediction_changed_patients']} changed patients)"
    )
    print(
        f"Original/replay Acc       : {overall_pat['original_accuracy']:.3f}% / "
        f"{overall_pat['replay_accuracy']:.3f}%"
    )
    print(
        f"Original/replay errors    : {overall_pat['original_errors']} / "
        f"{overall_pat['replay_errors']}"
    )
    print(
        "Exact within tolerance    : "
        + ("PASS" if summary["all_folds_exact_within_tolerance"] else "NO — use concordance wording, not equivalence")
    )
    print(f"Outputs                   : {out}")
    print("=" * 150)


if __name__ == "__main__":
    main()
