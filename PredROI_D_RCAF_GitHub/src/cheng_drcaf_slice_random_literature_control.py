#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
D-RCAF SLICE-RANDOM LITERATURE-MATCHED CONTROL
==============================================

Purpose
-------
This script evaluates the *same* D-RCAF architecture under a conventional
5-fold slice-random stratified split so that the paper can report a secondary
protocol-matched control alongside literature that evaluates individual MRI
slices/images rather than unseen patients.

IMPORTANT INTERPRETATION
------------------------
This is NOT the primary generalization estimate and must not replace the
locked patient-disjoint result. Because multiple slices from one Cheng patient
are deliberately allowed to appear in both training and validation folds,
patient-specific appearance can leak across the split. The script measures and
reports this source-patient overlap explicitly.

The primary paper result remains the locked patient-disjoint D-RCAF result.
This control is intended only for protocol-aware literature comparison and for
showing how much apparent performance changes when the split unit is changed.

Frozen control design
---------------------
- Dataset: Cheng 3-class CE-MRI, all 3,064 slices.
- Outer split: StratifiedKFold over *slices*, 5 folds, seed=123.
- Inner checkpoint selection: stratified split over outer-training slices only.
- D-RCAF architecture: unchanged.
- PredROI localizer: unchanged.
- Tight crop margin: 1.10x unchanged.
- Classifier loss/optimizer/augmentations: unchanged.
- Test/outer-validation masks: NEVER loaded for crop generation.
- Final endpoint: pooled OOF *slice-level* Accuracy, Macro-F1, class recalls.
- Source-patient overlap is intentionally audited and reported.

Implementation note
-------------------
The existing audited D-RCAF helpers operate on PatientRec containers. To reuse
that exact code path without modifying the architecture, each slice is wrapped
as a singleton pseudo-patient. Therefore, in imported localizer/helper logs,
"patient" counts may refer to singleton slice containers. Final outputs from
this script are explicitly labeled slice-level. The original source patient ID
is retained separately for overlap auditing and output CSVs.

Required neighboring files
--------------------------
Place this script next to:
    cheng_drcaf_predroi_swin_full5.py
    cheng_predroi_swin_screen_v1_audited.py
    cheng_patient_set_leakage_free_v2.py

Recommended final run (PowerShell)
----------------------------------
python .\cheng_drcaf_slice_random_literature_control.py `
  --manifest "C:\PATH\TO\brain_project\cheng_patient_disjoint\cheng_manifest.csv" `
  --mat_root "C:\PATH\TO\brain_project\cheng_brain_tumor" `
  --out_root "C:\PATH\TO\brain_project\drcaf_slice_random_seed123" `
  --amp_dtype bf16
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold

import cheng_patient_set_leakage_free_v2 as common
import cheng_predroi_swin_screen_v1_audited as pred
import cheng_drcaf_predroi_swin_full5 as drcaf


CLASSES = common.CLASSES
N_CLASSES = common.N_CLASSES
FROZEN_SLICE_RANDOM_SEED = 123
FROZEN_N_FOLDS = 5
LOCKED_PD_SLICE_ACC_REF = 94.158
LOCKED_PD_SLICE_F1_REF = 93.437


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def slice_singletons(slice_rows, indices):
    """Wrap each selected slice as a singleton PatientRec.

    The pseudo patient ID is unique per slice and is NOT the source Cheng PID.
    This makes all existing D-RCAF training/evaluation helpers operate on one
    slice per aggregation unit while preserving the original image path/label.
    """
    out = []
    seen_paths = set()
    seen_ids = set()

    for raw_idx in indices:
        idx = int(raw_idx)
        r = slice_rows[idx]
        p = str(r.path)
        if p in seen_paths:
            raise RuntimeError(f"Duplicate path in singleton conversion: {p}")
        pseudo_id = f"SLICE_{idx:05d}"
        if pseudo_id in seen_ids:
            raise RuntimeError(f"Duplicate pseudo ID: {pseudo_id}")
        seen_paths.add(p)
        seen_ids.add(pseudo_id)
        out.append(
            common.PatientRec(
                patient_id=pseudo_id,
                label=int(r.label),
                class_name=str(r.class_name),
                paths=[p],
            )
        )
    return out


def make_slice_random_splits(slice_rows, n_folds, seed):
    labels = np.asarray([int(r.label) for r in slice_rows], dtype=np.int64)
    skf = StratifiedKFold(
        n_splits=int(n_folds),
        shuffle=True,
        random_state=int(seed),
    )
    splits = list(skf.split(np.zeros(len(labels)), labels))

    # Strict index integrity: no slice can cross train/validation within a fold,
    # and every slice must appear exactly once in outer validation across folds.
    val_seen = []
    for fold, (tr, va) in enumerate(splits, start=1):
        if set(map(int, tr)) & set(map(int, va)):
            raise RuntimeError(f"Fold {fold}: exact slice index overlap detected.")
        val_seen.extend(map(int, va))

    if len(val_seen) != len(slice_rows):
        raise RuntimeError("Outer validation coverage length mismatch.")
    if len(set(val_seen)) != len(slice_rows):
        raise RuntimeError("A slice appears in outer validation more than once.")
    if set(val_seen) != set(range(len(slice_rows))):
        raise RuntimeError("Outer validation does not cover every slice exactly once.")

    return splits


def split_overlap_audit(slice_rows, tr_idx, va_idx):
    tr_idx = [int(i) for i in tr_idx]
    va_idx = [int(i) for i in va_idx]

    tr_paths = {str(slice_rows[i].path) for i in tr_idx}
    va_paths = {str(slice_rows[i].path) for i in va_idx}
    if tr_paths & va_paths:
        raise RuntimeError("Exact image path crossed the slice split.")

    tr_pids = {str(slice_rows[i].patient_id) for i in tr_idx}
    va_pids = {str(slice_rows[i].patient_id) for i in va_idx}
    overlap_pids = tr_pids & va_pids

    seen_count = sum(
        1 for i in va_idx
        if str(slice_rows[i].patient_id) in tr_pids
    )
    seen_pct = 100.0 * seen_count / max(1, len(va_idx))

    return {
        "train_slices": int(len(tr_idx)),
        "validation_slices": int(len(va_idx)),
        "train_source_patients": int(len(tr_pids)),
        "validation_source_patients": int(len(va_pids)),
        "overlap_source_patients": int(len(overlap_pids)),
        "validation_slices_from_source_patient_seen_in_train": int(seen_count),
        "validation_slices_from_seen_source_patient_pct": float(seen_pct),
        "exact_path_overlap": 0,
    }


def add_source_pid(rows, path_to_source_pid):
    for r in rows:
        p = str(r["path"])
        if p not in path_to_source_pid:
            raise RuntimeError(f"Missing original source PID for path: {p}")
        r["source_patient_id"] = str(path_to_source_pid[p])
    return rows


def metrics_with_recalls(slice_prediction_rows):
    y = np.asarray([int(r["true_id"]) for r in slice_prediction_rows], dtype=int)
    p = np.asarray([int(r["slice_pred_id"]) for r in slice_prediction_rows], dtype=int)

    recalls = recall_score(
        y,
        p,
        labels=list(range(N_CLASSES)),
        average=None,
        zero_division=0,
    )
    cm = confusion_matrix(y, p, labels=list(range(N_CLASSES)))

    out = {
        "n": int(len(y)),
        "accuracy": float(100.0 * accuracy_score(y, p)),
        "macro_f1": float(
            100.0 * f1_score(
                y,
                p,
                labels=list(range(N_CLASSES)),
                average="macro",
                zero_division=0,
            )
        ),
        "errors": int(np.sum(y != p)),
        "glioma_recall": float(100.0 * recalls[0]),
        "meningioma_recall": float(100.0 * recalls[1]),
        "pituitary_recall": float(100.0 * recalls[2]),
        "confusion_matrix": cm.astype(int).tolist(),
    }
    return out


def exact_box_cache(path: Path, expected_singletons):
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if "path" not in df.columns:
        return None

    expected = {
        str(q)
        for p in expected_singletons
        for q in p.paths
    }
    got = [str(x) for x in df["path"].tolist()]
    if len(got) != len(set(got)):
        return None
    if set(got) != expected:
        return None
    return df.to_dict("records")


def train_drcaf_classifier_slice(
    inner_train,
    inner_val,
    train_boxes,
    val_boxes,
    args,
    device,
    out_root: Path,
):
    """Same D-RCAF selection training, but explicitly labels metrics as slices."""
    common.seed_all(
        args.outer_seed * 50000 + args.outer_fold,
        args.deterministic,
    )

    tr_ds = drcaf._drcaf_make_dataset(
        inner_train,
        train_boxes,
        args,
        True,
        args.outer_seed * 51000 + args.outer_fold,
    )
    va_ds = drcaf._drcaf_make_dataset(
        inner_val,
        val_boxes,
        args,
        False,
        args.outer_seed * 52000 + args.outer_fold,
    )

    tr_loader = DataLoader(
        tr_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )
    va_loader = DataLoader(
        va_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    model = drcaf.DRCAFSharedSwin(
        args.backbone,
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
    ).to(device)

    # Because each container contains exactly one slice, this becomes a
    # slice-frequency class weighting scheme in this control.
    cw = common.patient_class_weights(inner_train, device)
    criterion = common.FocalLoss(
        cw,
        gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
    )

    drcaf.drcaf_set_encoder_trainable(model, False)
    opt = drcaf.drcaf_make_optimizer(
        model,
        args.backbone_lr,
        args.head_lr,
        args.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda" and args.amp_dtype == "fp16"),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=max(1, args.max_select_epochs),
        eta_min=args.minimum_lr,
    )

    history = []
    best = None
    best_slice_rows = None

    for epoch in range(1, args.max_select_epochs + 1):
        tr_ds.set_epoch(epoch)
        drcaf.drcaf_set_encoder_trainable(
            model,
            epoch > args.freeze_epochs,
        )
        model.train()
        opt.zero_grad(set_to_none=True)

        losses = []
        gates = []
        correct = 0
        n = 0

        for step, batch in enumerate(tr_loader, start=1):
            xg = batch["global_image"].to(device, non_blocking=True)
            xr = batch["roi_image"].to(device, non_blocking=True)
            q = batch["quality"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            with common.autocast_ctx(device, args.amp_dtype):
                logits, aux = model(xg, xr, q, return_aux=True)
                loss = criterion(logits, y)
                scaled = loss / args.accumulation_steps

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite D-RCAF slice-control loss epoch={epoch}"
                )

            scaler.scale(scaled).backward()

            if (
                step % args.accumulation_steps == 0
                or step == len(tr_loader)
            ):
                scaler.unscale_(opt)
                grad_params = [
                    p for p in model.parameters()
                    if p.requires_grad and p.grad is not None
                ]
                gnorm = torch.nn.utils.clip_grad_norm_(
                    grad_params,
                    args.grad_clip,
                )
                if not torch.isfinite(gnorm):
                    raise FloatingPointError(
                        "Non-finite D-RCAF slice-control gradient norm."
                    )
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            losses.append(float(loss.detach().cpu()))
            gates.extend(
                aux["roi_gate_mean"].detach().float().cpu().tolist()
            )
            correct += int((logits.argmax(1) == y).sum().item())
            n += int(y.numel())

        srows, _singleton_rows, sm, _singleton_metrics = (
            drcaf._drcaf_eval_loader(
                model,
                va_loader,
                args,
                device,
            )
        )

        tracc = 100.0 * correct / max(1, n)
        val_gate = float(np.mean([r["roi_gate_mean"] for r in srows]))
        val_dis = float(np.mean([r["disagreement_l1"] for r in srows]))

        history.append(
            {
                "epoch": int(epoch),
                "train_loss": float(np.mean(losses)),
                "train_slice_acc": float(tracc),
                "train_roi_gate_mean": float(np.mean(gates)),
                "inner_slice_acc": float(sm["accuracy"]),
                "inner_slice_f1": float(sm["macro_f1"]),
                "inner_slice_errors": int(sm["errors"]),
                "inner_roi_gate_mean": val_gate,
                "inner_disagreement_l1": val_dis,
            }
        )

        print(
            f"[D-RCAF/SR] epoch {epoch:02d}/{args.max_select_epochs} | "
            f"loss={np.mean(losses):.4f} train-slice={tracc:6.2f}% | "
            f"INNER slice Acc={sm['accuracy']:6.2f}% "
            f"F1={sm['macro_f1']:6.2f}% err={sm['errors']} | "
            f"ROI-gate={val_gate:.3f} dis={val_dis:.4f}"
        )

        if epoch > args.freeze_epochs:
            key = (sm["macro_f1"], sm["accuracy"], -epoch)
            if best is None or key > best["key"]:
                best = {
                    "key": key,
                    "epoch": int(epoch),
                    "slice": dict(sm),
                    "roi_gate_mean": val_gate,
                    "disagreement_l1": val_dis,
                }
                best_slice_rows = [dict(r) for r in srows]

        sched.step()

    if best is None:
        raise RuntimeError("No selectable slice-random D-RCAF epoch.")

    pd.DataFrame(history).to_csv(
        out_root / "drcaf_inner_slice_selection_history.csv",
        index=False,
    )
    pd.DataFrame(best_slice_rows).to_csv(
        out_root / "drcaf_inner_best_slice_predictions.csv",
        index=False,
    )
    return best


def synthetic_split_audit():
    # 60 slices, 12 source patients, 3 classes. Each source patient contributes
    # five adjacent slices. Slice-random splitting should preserve index
    # disjointness while deliberately creating source-patient overlap.
    rows = []
    for pid in range(12):
        lab = pid % 3
        for s in range(5):
            rows.append(
                common.SliceRec(
                    path=f"fake_{pid:02d}_{s:02d}.png",
                    patient_id=f"P{pid:02d}",
                    label=lab,
                    class_name=CLASSES[lab],
                )
            )

    splits = make_slice_random_splits(rows, 5, FROZEN_SLICE_RANDOM_SEED)
    overlaps = []
    for tr, va in splits:
        a = split_overlap_audit(rows, tr, va)
        overlaps.append(a["validation_slices_from_seen_source_patient_pct"])
        single = slice_singletons(rows, va)
        if len(single) != len(va):
            raise RuntimeError("Singleton conversion size audit failed.")
        if not all(len(p.paths) == 1 for p in single):
            raise RuntimeError("A pseudo-patient contains multiple slices.")

    if max(overlaps) <= 0.0:
        raise RuntimeError(
            "Synthetic slice-random audit unexpectedly produced no source-patient overlap."
        )
    return True


def dependency_api_audit():
    required_drcaf = [
        "_drcaf_make_dataset",
        "_drcaf_eval_loader",
        "DRCAFSharedSwin",
        "drcaf_set_encoder_trainable",
        "drcaf_make_optimizer",
        "crossfit_train_boxes_only",
        "refit_classifier_fixed_epoch",
        "predict_boxes_image_only",
        "evaluate_outer",
        "synthetic_drcaf_audit",
        "_load_cached_selection_boxes_if_valid",
    ]
    required_pred = [
        "crossfit_localizer_and_boxes",
        "train_segmenter",
        "preflight_manifest_mat",
        "synthetic_bbox_audit",
    ]

    missing = [x for x in required_drcaf if not hasattr(drcaf, x)]
    missing += [x for x in required_pred if not hasattr(pred, x)]
    if missing:
        raise RuntimeError(
            "Dependency API mismatch. Missing symbols: " + ", ".join(missing)
        )
    return True


def build_arg_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="")
    ap.add_argument("--mat_root", default="")
    ap.add_argument("--out_root", default="")

    ap.add_argument(
        "--outer_seed",
        type=int,
        default=FROZEN_SLICE_RANDOM_SEED,
        help="Frozen slice-random control seed. Must remain 123.",
    )
    ap.add_argument("--n_folds", type=int, default=FROZEN_N_FOLDS)
    ap.add_argument("--inner_n_splits", type=int, default=5)
    ap.add_argument("--inner_seed", type=int, default=31415)

    # Unchanged localizer.
    ap.add_argument("--seg_oof_folds", type=int, default=3)
    ap.add_argument("--seg_oof_seed", type=int, default=27182)
    ap.add_argument("--seg_epochs", type=int, default=8)
    ap.add_argument("--seg_size", type=int, default=256)
    ap.add_argument("--seg_batch_size", type=int, default=8)
    ap.add_argument("--seg_eval_batch_size", type=int, default=16)
    ap.add_argument("--seg_lr", type=float, default=2e-4)
    ap.add_argument("--seg_weight_decay", type=float, default=1e-4)

    # Unchanged predicted-box rule.
    ap.add_argument("--box_threshold", type=float, default=0.50)
    ap.add_argument("--box_fallback_threshold", type=float, default=0.30)
    ap.add_argument("--crop_margin_factor", type=float, default=1.10)

    # Unchanged D-RCAF classifier.
    ap.add_argument("--backbone", default="swin_base_patch4_window7_224")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--max_select_epochs", type=int, default=15)
    ap.add_argument("--freeze_epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--accumulation_steps", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--backbone_lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=2e-4)
    ap.add_argument("--minimum_lr", type=float, default=1e-7)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--label_smoothing", type=float, default=0.03)
    ap.add_argument("--grad_clip", type=float, default=3.0)
    ap.add_argument("--alignment_samples", type=int, default=48)
    ap.add_argument("--amp_dtype", choices=["fp16", "bf16"], default="bf16")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--no_save_models", action="store_true")
    ap.add_argument(
        "--self_test_only",
        action="store_true",
        help="Run code/split/API audits only; do not load the Cheng dataset.",
    )
    return ap


def main():
    ap = build_arg_parser()
    args = ap.parse_args()

    dependency_api_audit()
    synthetic_split_audit()
    drcaf.synthetic_drcaf_audit()
    pred.synthetic_bbox_audit()

    print("[AUDIT] dependency API: PASS")
    print("[AUDIT] slice-random index integrity + intentional PID overlap: PASS")
    print("[AUDIT] D-RCAF gate algebra: PASS")
    print("[AUDIT] PredROI bbox geometry: PASS")

    if args.self_test_only:
        print("[SELF TEST] PASS")
        return

    if int(args.outer_seed) != FROZEN_SLICE_RANDOM_SEED:
        raise ValueError(
            f"This protocol-matched control is frozen to --outer_seed "
            f"{FROZEN_SLICE_RANDOM_SEED}; do not seed-search."
        )
    if int(args.n_folds) != FROZEN_N_FOLDS:
        raise ValueError(
            f"This control is frozen to {FROZEN_N_FOLDS}-fold slice-random CV."
        )
    if args.max_select_epochs <= args.freeze_epochs:
        raise ValueError("--max_select_epochs must exceed --freeze_epochs.")
    if args.seg_oof_folds < 2:
        raise ValueError("--seg_oof_folds must be >= 2.")
    if not args.manifest or not args.mat_root or not args.out_root:
        raise ValueError("--manifest, --mat_root and --out_root are required for a real run.")

    common.seed_all(args.outer_seed, args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    slice_rows, _ = common.load_manifest(Path(args.manifest))
    if len(slice_rows) != 3064:
        print(
            f"[WARNING] Expected Cheng 3,064 slices, found {len(slice_rows)}. "
            "Continue only if this is intentional."
        )

    path_to_source_pid = {
        str(r.path): str(r.patient_id)
        for r in slice_rows
    }
    if len(path_to_source_pid) != len(slice_rows):
        raise RuntimeError("Manifest contains duplicate image paths.")

    preflight = pred.preflight_manifest_mat(
        slice_rows,
        Path(args.mat_root),
        args.alignment_samples,
    )

    splits = make_slice_random_splits(
        slice_rows,
        args.n_folds,
        args.outer_seed,
    )

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    print("=" * 168)
    print("D-RCAF — 5-FOLD SLICE-RANDOM LITERATURE-MATCHED CONTROL")
    print("=" * 168)
    print(f"device                    : {device}")
    print(f"slice-random seed         : {args.outer_seed} (FROZEN)")
    print(f"outer folds               : {args.n_folds}")
    print("split unit                : SLICE / IMAGE")
    print("source-patient separation : NO — overlap is intentional and audited")
    print("architecture              : unchanged D-RCAF")
    print("PredROI margin            : 1.10x (unchanged)")
    print("outer-val GT masks        : NEVER loaded for crop generation")
    print("selection endpoint        : inner slice Macro-F1")
    print("final endpoint            : pooled OOF slice-level metrics")
    print(
        f"MAT alignment median/min  : "
        f"{preflight['corr_median']:.6f}/{preflight['corr_min']:.6f}"
    )
    print("IMPORTANT                 : secondary protocol-matched control; NOT primary generalization evidence")
    print("=" * 168)

    save_json(
        {
            **vars(args),
            "method": "D-RCAF-slice-random-literature-control",
            "frozen_slice_random_seed": FROZEN_SLICE_RANDOM_SEED,
            "split_unit": "slice",
            "source_patient_disjoint": False,
            "interpretation": (
                "secondary protocol-matched literature control; not a patient-level generalization estimate"
            ),
            "outer_val_ground_truth_masks_loaded": False,
            "selection_metric": "inner slice Macro-F1",
            "final_metric_unit": "slice",
            "classifier_architecture": "unchanged D-RCAF",
            "predroi_margin_factor": args.crop_margin_factor,
            "locked_patient_disjoint_slice_reference": {
                "accuracy": LOCKED_PD_SLICE_ACC_REF,
                "macro_f1": LOCKED_PD_SLICE_F1_REF,
            },
            "preflight": preflight,
        },
        out_root / "run_config.json",
    )

    all_slice = []
    fold_summary = []
    weighted_seen_num = 0
    weighted_seen_den = 0

    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        args.outer_fold = int(fold)
        print("\n" + "=" * 168)
        print(f"SLICE-RANDOM OUTER FOLD {fold}/{args.n_folds}")
        print("=" * 168)

        audit = split_overlap_audit(slice_rows, tr_idx, va_idx)
        weighted_seen_num += audit[
            "validation_slices_from_source_patient_seen_in_train"
        ]
        weighted_seen_den += audit["validation_slices"]

        print(
            f"train/val slices          : "
            f"{audit['train_slices']}/{audit['validation_slices']}"
        )
        print(
            f"source patients tr/val    : "
            f"{audit['train_source_patients']}/{audit['validation_source_patients']}"
        )
        print(
            f"source PID overlap        : "
            f"{audit['overlap_source_patients']} patients | "
            f"{audit['validation_slices_from_seen_source_patient_pct']:.2f}% "
            "of validation slices come from a patient represented in training"
        )

        outer_train = slice_singletons(slice_rows, tr_idx)
        outer_val = slice_singletons(slice_rows, va_idx)

        inner_train, inner_val = common.make_inner_selection_split(
            outer_train,
            args.inner_n_splits,
            args.inner_seed + fold,
        )

        fdir = out_root / f"fold_{fold}"
        fdir.mkdir(parents=True, exist_ok=True)
        sdir = fdir / "selection"
        rdir = fdir / "refit"
        sdir.mkdir(parents=True, exist_ok=True)
        rdir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            [
                {
                    "fold": fold,
                    "partition": "train",
                    "path": str(slice_rows[int(i)].path),
                    "source_patient_id": str(slice_rows[int(i)].patient_id),
                    "label": int(slice_rows[int(i)].label),
                    "class_name": str(slice_rows[int(i)].class_name),
                }
                for i in tr_idx
            ]
            + [
                {
                    "fold": fold,
                    "partition": "validation",
                    "path": str(slice_rows[int(i)].path),
                    "source_patient_id": str(slice_rows[int(i)].patient_id),
                    "label": int(slice_rows[int(i)].label),
                    "class_name": str(slice_rows[int(i)].class_name),
                }
                for i in va_idx
            ]
        ).to_csv(fdir / "slice_split.csv", index=False)

        save_json(audit, fdir / "source_patient_overlap_audit.json")

        print("[STAGE A] INNER slice-only PredROI box generation + D-RCAF epoch selection")

        cached = drcaf._load_cached_selection_boxes_if_valid(
            sdir,
            inner_train,
            inner_val,
        )
        if cached is not None:
            inner_train_boxes, inner_val_boxes = cached
            print("[RESUME] Reusing exact Stage-A slice-control box caches.")
        else:
            inner_train_boxes, inner_val_boxes = pred.crossfit_localizer_and_boxes(
                inner_train,
                inner_val,
                args,
                device,
                sdir,
            )

        inner_best = train_drcaf_classifier_slice(
            inner_train,
            inner_val,
            inner_train_boxes,
            inner_val_boxes,
            args,
            device,
            sdir,
        )
        selected_epoch = int(inner_best["epoch"])

        print(
            f"[SELECTED] fold={fold} | epoch={selected_epoch} | "
            f"INNER slice Acc={inner_best['slice']['accuracy']:.3f}% | "
            f"F1={inner_best['slice']['macro_f1']:.3f}% | "
            f"err={inner_best['slice']['errors']}"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[STAGE B] OUTER-TRAIN slice-level OOF localizer boxes")
        refit_cache = exact_box_cache(
            rdir / "refit_oof_train_boxes.csv",
            outer_train,
        )
        if refit_cache is not None:
            refit_train_boxes = refit_cache
            print("[RESUME] Reusing exact Stage-B refit OOF box cache.")
        else:
            refit_train_boxes = drcaf.crossfit_train_boxes_only(
                outer_train,
                args,
                device,
                rdir,
                seed_offset=fold,
            )

        print("[STAGE C] D-RCAF reinit/refit on all outer-training slices")
        classifier = drcaf.refit_classifier_fixed_epoch(
            outer_train,
            refit_train_boxes,
            selected_epoch,
            args,
            device,
            seed=args.outer_seed * 7000 + fold,
            outdir=rdir,
        )

        print("[STAGE D] Final localizer on all outer-training slices")
        final_localizer = pred.train_segmenter(
            outer_train,
            args,
            device,
            seed=args.outer_seed * 8000 + fold,
            tag=f"SLICE_RANDOM_FINAL_F{fold}",
        )

        print("[STAGE E] Outer-validation image-only predicted boxes")
        outer_val_boxes = drcaf.predict_boxes_image_only(
            final_localizer,
            outer_val,
            args,
            device,
            seed=args.outer_seed * 8100 + fold,
            tag=f"slice_random_outer_val_fold_{fold}",
        )
        pd.DataFrame(outer_val_boxes).to_csv(
            rdir / "outer_val_predicted_boxes.csv",
            index=False,
        )

        found_rate = float(
            np.mean([int(r["found"]) for r in outer_val_boxes])
        )
        print(
            f"[OUTER LOCALIZER] bbox-found={100.0 * found_rate:.2f}% "
            "(image-only; no validation GT mask loaded)"
        )

        print("[STAGE F] Outer-validation D-RCAF slice classification")
        srows, _singleton_prows, sm, _singleton_pm = drcaf.evaluate_outer(
            classifier,
            outer_val,
            outer_val_boxes,
            args,
            device,
            fold,
        )
        srows = add_source_pid(srows, path_to_source_pid)

        pd.DataFrame(srows).to_csv(
            fdir / "outer_slice_predictions.csv",
            index=False,
        )

        if not args.no_save_models:
            torch.save(
                {
                    "state_dict": classifier.state_dict(),
                    "split_policy": "slice_random",
                    "slice_random_seed": args.outer_seed,
                    "fold": fold,
                    "selected_epoch": selected_epoch,
                    "method": "D-RCAF-slice-random-control",
                },
                fdir / "final_drcaf_classifier.pt",
            )
            torch.save(
                {
                    "state_dict": final_localizer.state_dict(),
                    "split_policy": "slice_random",
                    "slice_random_seed": args.outer_seed,
                    "fold": fold,
                    "seg_epochs": args.seg_epochs,
                    "method": "PredROI-localizer-slice-random-control",
                },
                fdir / "final_localizer.pt",
            )

        all_slice.extend(srows)

        fm = metrics_with_recalls(srows)
        fold_summary.append(
            {
                "fold": int(fold),
                "selected_epoch": selected_epoch,
                "n_validation_slices": int(fm["n"]),
                "slice_accuracy": fm["accuracy"],
                "slice_macro_f1": fm["macro_f1"],
                "slice_errors": fm["errors"],
                "glioma_recall": fm["glioma_recall"],
                "meningioma_recall": fm["meningioma_recall"],
                "pituitary_recall": fm["pituitary_recall"],
                "bbox_found_rate": found_rate,
                **audit,
            }
        )

        print(
            f"[OUTER SLICE RESULT] Acc={fm['accuracy']:.3f}% | "
            f"F1={fm['macro_f1']:.3f}% | "
            f"G-R={fm['glioma_recall']:.3f}% | "
            f"M-R={fm['meningioma_recall']:.3f}% | "
            f"P-R={fm['pituitary_recall']:.3f}% | "
            f"err={fm['errors']}"
        )

        del classifier
        del final_localizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(all_slice) != len(slice_rows):
        raise RuntimeError(
            f"Pooled OOF prediction count mismatch: {len(all_slice)} vs {len(slice_rows)}"
        )

    pred_paths = [str(r["path"]) for r in all_slice]
    if len(pred_paths) != len(set(pred_paths)):
        raise RuntimeError("Duplicate OOF slice prediction paths detected.")
    if set(pred_paths) != set(path_to_source_pid):
        raise RuntimeError("OOF predictions do not cover the manifest exactly once.")

    overall = metrics_with_recalls(all_slice)
    seen_pct = 100.0 * weighted_seen_num / max(1, weighted_seen_den)
    gate_mean = float(np.mean([float(r["roi_gate_mean"]) for r in all_slice]))
    disagreement_mean = float(
        np.mean([float(r["disagreement_l1"]) for r in all_slice])
    )

    delta_acc_vs_pd_slice = overall["accuracy"] - LOCKED_PD_SLICE_ACC_REF
    delta_f1_vs_pd_slice = overall["macro_f1"] - LOCKED_PD_SLICE_F1_REF

    pd.DataFrame(all_slice).to_csv(
        out_root / "slice_oof.csv",
        index=False,
    )
    pd.DataFrame(fold_summary).to_csv(
        out_root / "fold_summary.csv",
        index=False,
    )
    pd.DataFrame(
        overall["confusion_matrix"],
        index=[f"true_{c}" for c in CLASSES],
        columns=[f"pred_{c}" for c in CLASSES],
    ).to_csv(out_root / "confusion_matrix.csv")

    summary = {
        "method": "D-RCAF-slice-random-literature-control",
        "split_policy": "5-fold stratified slice-random",
        "slice_random_seed": args.outer_seed,
        "primary_endpoint_unit": "slice",
        "source_patient_disjoint": False,
        "source_patient_overlap_is_intentional": True,
        "validation_slices_from_source_patient_seen_in_train_pct": seen_pct,
        "slice_oof": overall,
        "roi_gate_mean": gate_mean,
        "disagreement_l1_mean": disagreement_mean,
        "locked_patient_disjoint_slice_reference": {
            "accuracy": LOCKED_PD_SLICE_ACC_REF,
            "macro_f1": LOCKED_PD_SLICE_F1_REF,
        },
        "delta_vs_locked_patient_disjoint_slice_reference_pp": {
            "accuracy": delta_acc_vs_pd_slice,
            "macro_f1": delta_f1_vs_pd_slice,
        },
        "folds": fold_summary,
        "interpretation": (
            "secondary protocol-matched slice-level control only; not a replacement "
            "for the locked patient-disjoint patient-level generalization estimate"
        ),
    }
    save_json(summary, out_root / "summary.json")

    print("\n" + "=" * 168)
    print("FINAL D-RCAF SLICE-RANDOM LITERATURE-MATCHED CONTROL")
    print("=" * 168)
    print(
        f"Slice-random OOF | Acc={overall['accuracy']:.3f}% | "
        f"F1={overall['macro_f1']:.3f}% | "
        f"G-R={overall['glioma_recall']:.3f}% | "
        f"M-R={overall['meningioma_recall']:.3f}% | "
        f"P-R={overall['pituitary_recall']:.3f}% | "
        f"err={overall['errors']}"
    )
    print(
        f"Seen-source PID   | {seen_pct:.2f}% of validation slices came from "
        "patients represented in the corresponding training fold"
    )
    print(
        f"vs locked PD slice| ΔAcc={delta_acc_vs_pd_slice:+.3f} pp | "
        f"ΔF1={delta_f1_vs_pd_slice:+.3f} pp "
        f"(reference Acc={LOCKED_PD_SLICE_ACC_REF:.3f}%, "
        f"F1={LOCKED_PD_SLICE_F1_REF:.3f}%)"
    )
    print(
        f"Gate diagnostics  | mean ROI gate={gate_mean:.4f} | "
        f"mean disagreement={disagreement_mean:.4f}"
    )
    print("INTERPRETATION     | secondary slice-level protocol-matched literature control")
    print("PRIMARY CLAIM      | remains locked patient-disjoint patient-level D-RCAF")
    print("=" * 168)
    print(f"Written: {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
