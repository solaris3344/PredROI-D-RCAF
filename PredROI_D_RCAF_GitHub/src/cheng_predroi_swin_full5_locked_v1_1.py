#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
cheng_predroi_swin_full5_locked_v1_1.py

LOCKED FULL-5 EVALUATION FOR PredROI-Swin v1
============================================

This script is the post-screen evaluation stage.

Development evidence already observed on seed42/fold1:
    PredROI-Swin inner-only:
        Acc=100.000%
        Macro-F1=100.000%
        errors=0
    Historical global Swin inner:
        Acc=94.737%
        Macro-F1=94.715%
        errors=2

The method is now FROZEN. This script does not tune:
    - segmenter architecture
    - segmenter epochs
    - box thresholds
    - crop margin
    - classifier architecture
    - loss
    - patient aggregation

Recommended primary run:
    use a fresh locked outer seed, default 5813.

Per outer fold:
    1) Build patient-disjoint INNER train/val from OUTER-TRAIN.
    2) Select classifier epoch using INNER only:
         - 3-fold patient-disjoint segmenter cross-fit on INNER-TRAIN
         - classifier train crops = OOF predicted boxes
         - INNER-VAL boxes = final segmenter trained only on INNER-TRAIN
         - select by INNER patient Macro-F1
    3) Refit on ALL OUTER-TRAIN using selected epoch:
         - 3-fold patient-disjoint segmenter cross-fit across OUTER-TRAIN
         - classifier train crops = OOF predicted boxes
         - classifier reinitialized and trained selected_epoch epochs
         - final segmenter trained on ALL OUTER-TRAIN
    4) OUTER-VAL:
         - image-only localizer inference
         - predicted box only
         - predicted ROI -> frozen classifier
         - patient probability = mean slice softmax probability
         - OUTER-VAL ground-truth masks are NEVER loaded

Optional:
    --baseline_patient_oof can point to the standard Swin patient_oof.csv
    generated on the EXACT SAME outer seed. Then the script reports paired
    bootstrap + exact McNemar.

Requirements:
    - cheng_predroi_swin_screen_v1_audited.py
    - cheng_patient_set_leakage_free_v2.py
must be in the same folder.

Recommended:
    First run the standard Swin baseline on the same fresh outer seed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score

import cheng_patient_set_leakage_free_v2 as common
import cheng_predroi_swin_screen_v1_audited as pred


CLASSES = common.CLASSES
N_CLASSES = common.N_CLASSES


def save_json(obj, path: Path):
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


# =============================================================================
# Image-only localizer inference for untouched OUTER validation
# =============================================================================

class SegImageOnlyDataset(Dataset):
    """
    Deliberately has no mat_root and cannot load a tumorMask.
    """

    def __init__(self, rows, seg_size, seed):
        self.rows = list(rows)
        self.seg_size = int(seg_size)
        self.seed = int(seed)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]

        img = cv2.imread(
            r["path"],
            cv2.IMREAD_GRAYSCALE,
        )

        if img is None:
            raise RuntimeError(
                f"Could not read {r['path']}"
            )

        clahe = cv2.createCLAHE(
            clipLimit=2.0,
            tileGridSize=(8, 8),
        )
        img = clahe.apply(img)

        img = cv2.resize(
            img,
            (self.seg_size, self.seg_size),
            interpolation=cv2.INTER_AREA,
        )

        x = torch.from_numpy(
            img.astype(np.float32) / 255.0
        ).unsqueeze(0)

        x = x.repeat(3, 1, 1)
        x = (x - pred.SEG_MEAN) / pred.SEG_STD

        return {
            "image": x,
            "path": str(r["path"]),
            "patient_id": str(r["patient_id"]),
            "label": torch.tensor(
                int(r["label"]),
                dtype=torch.long,
            ),
        }


@torch.no_grad()
def predict_boxes_image_only(
    model,
    patients,
    args,
    device,
    seed,
    tag,
):
    rows = pred.slice_rows_from_patients(
        patients
    )

    ds = SegImageOnlyDataset(
        rows,
        args.seg_size,
        seed,
    )

    loader = DataLoader(
        ds,
        batch_size=args.seg_eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    model.eval()
    out = []

    for batch in loader:
        x = batch["image"].to(
            device,
            non_blocking=True,
        )

        with common.autocast_ctx(
            device,
            args.amp_dtype,
        ):
            prob = torch.sigmoid(
                model(x).float()
            ).cpu().numpy()[:, 0]

        for i in range(len(prob)):
            path = str(
                batch["path"][i]
            )

            original = cv2.imread(
                path,
                cv2.IMREAD_GRAYSCALE,
            )

            if original is None:
                raise RuntimeError(
                    f"Could not read source image: {path}"
                )

            rec = pred.prob_to_bbox(
                prob[i],
                original.shape,
                args.box_threshold,
                args.box_fallback_threshold,
            )

            rec.update(
                {
                    "path": path,
                    "patient_id": str(
                        batch["patient_id"][i]
                    ),
                    "label": int(
                        batch["label"][i]
                    ),
                    "source": tag,
                }
            )

            out.append(rec)

    return out


# =============================================================================
# OOF boxes for OUTER-TRAIN refit
# =============================================================================

def crossfit_train_boxes_only(
    patients,
    args,
    device,
    outdir: Path,
    seed_offset,
):
    folds = pred.patient_seg_folds(
        patients,
        args.seg_oof_folds,
        args.seg_oof_seed + seed_offset,
    )

    all_ids = {
        str(p.patient_id)
        for p in patients
    }

    seen = set()
    boxes_all = []
    fold_rows = []

    for fold_idx, (tr_ids, ho_ids) in enumerate(
        folds,
        start=1,
    ):
        if tr_ids & ho_ids:
            raise RuntimeError(
                "Refit OOF segmenter patient leakage."
            )

        seen |= ho_ids

        tr_pat = pred.group_patients_by_ids(
            patients,
            tr_ids,
        )

        ho_pat = pred.group_patients_by_ids(
            patients,
            ho_ids,
        )

        model = pred.train_segmenter(
            tr_pat,
            args,
            device,
            seed=(
                args.outer_seed * 100000
                + seed_offset * 100
                + fold_idx
            ),
            tag=f"REFIT_OOF{fold_idx}",
        )

        boxes = pred.predict_boxes(
            model,
            ho_pat,
            args,
            device,
            seed=(
                args.outer_seed * 110000
                + seed_offset * 100
                + fold_idx
            ),
            tag=f"refit_oof_fold_{fold_idx}",
            evaluate_with_masks=True,
        )

        boxes_all.extend(boxes)

        fold_rows.append(
            {
                "fold": fold_idx,
                "train_patients": len(tr_pat),
                "holdout_patients": len(ho_pat),
                "holdout_slices": len(boxes),
                "mean_seg_iou_050": float(
                    np.mean(
                        [r["seg_iou_050"] for r in boxes]
                    )
                ),
                "bbox_found_rate": float(
                    np.mean(
                        [int(r["found"]) for r in boxes]
                    )
                ),
            }
        )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if seen != all_ids:
        raise RuntimeError(
            "Refit OOF patient coverage mismatch."
        )

    expected_paths = {
        str(q)
        for p in patients
        for q in p.paths
    }

    got_paths = [
        str(r["path"])
        for r in boxes_all
    ]

    if len(got_paths) != len(set(got_paths)):
        raise RuntimeError(
            "Duplicate refit OOF box paths."
        )

    if set(got_paths) != expected_paths:
        raise RuntimeError(
            "Refit OOF box path coverage mismatch."
        )

    pd.DataFrame(
        boxes_all
    ).to_csv(
        outdir / "refit_oof_train_boxes.csv",
        index=False,
    )

    pd.DataFrame(
        fold_rows
    ).to_csv(
        outdir / "refit_oof_seg_metrics.csv",
        index=False,
    )

    return boxes_all


# =============================================================================
# Classifier fixed-epoch refit
# =============================================================================

def refit_classifier_fixed_epoch(
    outer_train,
    train_boxes,
    selected_epoch,
    args,
    device,
    seed,
    outdir: Path,
):
    common.seed_all(
        seed,
        args.deterministic,
    )

    rows = pred.slice_rows_from_patients(
        outer_train
    )

    lookup = {
        str(r["path"]): r
        for r in train_boxes
    }

    expected = {
        str(r["path"])
        for r in rows
    }

    if set(lookup) != expected:
        raise RuntimeError(
            "Refit classifier predicted-box coverage mismatch."
        )

    ds = pred.PredROIDataset(
        rows,
        lookup,
        args.image_size,
        True,
        seed + 1000,
        args.crop_margin_factor,
    )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    model = pred.SwinClassifier(
        args.backbone,
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
    ).to(device)

    cw = common.patient_class_weights(
        outer_train,
        device,
    )

    criterion = common.FocalLoss(
        cw,
        gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
    )

    pred.set_encoder_trainable(
        model,
        False,
    )

    opt = pred.make_classifier_optimizer(
        model,
        args.backbone_lr,
        args.head_lr,
        args.weight_decay,
    )

    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(
            device.type == "cuda"
            and args.amp_dtype == "fp16"
        ),
    )

    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=max(1, int(selected_epoch)),
        eta_min=args.minimum_lr,
    )

    history = []

    for epoch in range(
        1,
        int(selected_epoch) + 1,
    ):
        ds.set_epoch(epoch)

        pred.set_encoder_trainable(
            model,
            epoch > args.freeze_epochs,
        )

        model.train()
        opt.zero_grad(set_to_none=True)

        losses = []
        correct = 0
        n = 0

        for step, batch in enumerate(
            loader,
            start=1,
        ):
            x = batch["image"].to(
                device,
                non_blocking=True,
            )
            y = batch["label"].to(
                device,
                non_blocking=True,
            )

            with common.autocast_ctx(
                device,
                args.amp_dtype,
            ):
                logits = model(x)
                loss = criterion(
                    logits,
                    y,
                )
                scaled = (
                    loss
                    / args.accumulation_steps
                )

            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite refit classifier loss epoch={epoch}"
                )

            scaler.scale(
                scaled
            ).backward()

            if (
                step
                % args.accumulation_steps
                == 0
                or step == len(loader)
            ):
                scaler.unscale_(opt)

                grad_params = [
                    p
                    for p in model.parameters()
                    if p.requires_grad
                    and p.grad is not None
                ]

                gnorm = torch.nn.utils.clip_grad_norm_(
                    grad_params,
                    args.grad_clip,
                )

                if not torch.isfinite(gnorm):
                    raise FloatingPointError(
                        "Non-finite refit classifier gradient."
                    )

                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            losses.append(
                float(loss.detach().cpu())
            )

            correct += int(
                (
                    logits.argmax(1)
                    == y
                ).sum().item()
            )

            n += int(
                y.numel()
            )

        sched.step()

        tracc = (
            100.0
            * correct
            / max(1, n)
        )

        history.append(
            {
                "epoch": epoch,
                "loss": float(
                    np.mean(losses)
                ),
                "train_slice_acc": tracc,
            }
        )

        print(
            f"      CLS REFIT epoch "
            f"{epoch:02d}/{selected_epoch} | "
            f"loss={np.mean(losses):.4f} | "
            f"slice-acc={tracc:.2f}%"
        )

    pd.DataFrame(
        history
    ).to_csv(
        outdir / "classifier_refit_history.csv",
        index=False,
    )

    return model


# =============================================================================
# OUTER evaluation
# =============================================================================

@torch.no_grad()
def evaluate_outer(
    classifier,
    outer_val,
    val_boxes,
    args,
    device,
    fold,
):
    rows = pred.slice_rows_from_patients(
        outer_val
    )

    lookup = {
        str(r["path"]): r
        for r in val_boxes
    }

    expected = {
        str(r["path"])
        for r in rows
    }

    if set(lookup) != expected:
        raise RuntimeError(
            "OUTER predicted-box coverage mismatch."
        )

    ds = pred.PredROIDataset(
        rows,
        lookup,
        args.image_size,
        False,
        args.outer_seed * 900000 + fold,
        args.crop_margin_factor,
    )

    loader = DataLoader(
        ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    classifier.eval()

    srows = []

    for batch in loader:
        x = batch["image"].to(
            device,
            non_blocking=True,
        )

        y = batch["label"].cpu().numpy()

        with common.autocast_ctx(
            device,
            args.amp_dtype,
        ):
            prob = F.softmax(
                classifier(x),
                dim=-1,
            ).float().cpu().numpy()

        pred_id = prob.argmax(
            axis=1
        )

        for i in range(len(y)):
            rec = {
                "patient_id": str(
                    batch["patient_id"][i]
                ),
                "path": str(
                    batch["path"][i]
                ),
                "true_id": int(y[i]),
                "slice_pred_id": int(
                    pred_id[i]
                ),
                "fold": int(fold),
            }

            for c, cname in enumerate(
                CLASSES
            ):
                rec[
                    f"prob_{cname}"
                ] = float(
                    prob[i, c]
                )

            srows.append(rec)

    prows = pred.aggregate_patient_probs(
        srows
    )

    for r in prows:
        r["fold"] = int(fold)

    sm = pred.slice_metrics(
        srows
    )

    pm = pred.patient_metrics(
        prows
    )

    return srows, prows, sm, pm


# =============================================================================
# Paired comparison
# =============================================================================

def paired_compare(
    predroi_patient_csv: Path,
    baseline_patient_csv: Path,
    n_boot,
    seed,
):
    a = pd.read_csv(
        predroi_patient_csv,
        dtype={"patient_id": str},
    )

    b = pd.read_csv(
        baseline_patient_csv,
        dtype={"patient_id": str},
    )

    aa = a[
        ["patient_id", "true_id", "pred_id"]
    ].rename(
        columns={
            "pred_id": "predroi_pred"
        }
    )

    bb = b[
        ["patient_id", "true_id", "pred_id"]
    ].rename(
        columns={
            "true_id": "true2",
            "pred_id": "baseline_pred",
        }
    )

    z = aa.merge(
        bb,
        on="patient_id",
        how="inner",
        validate="one_to_one",
    )

    if len(z) != len(aa) or len(z) != len(bb):
        raise RuntimeError(
            "Baseline/PredROI patient mismatch."
        )

    if not np.array_equal(
        z["true_id"].to_numpy(int),
        z["true2"].to_numpy(int),
    ):
        raise RuntimeError(
            "True labels differ in paired comparison."
        )

    y = z["true_id"].to_numpy(int)
    pp = z["predroi_pred"].to_numpy(int)
    pb = z["baseline_pred"].to_numpy(int)

    rescue, lost, p = common.exact_mcnemar(
        pb == y,
        pp == y,
    )

    boot = common.paired_bootstrap(
        y,
        pb,
        pp,
        n_boot=n_boot,
        seed=seed,
    )

    return {
        "n": int(len(y)),
        "baseline_acc": float(
            100.0 * accuracy_score(y, pb)
        ),
        "baseline_f1": float(
            100.0
            * f1_score(
                y,
                pb,
                average="macro",
                zero_division=0,
            )
        ),
        "predroi_acc": float(
            100.0 * accuracy_score(y, pp)
        ),
        "predroi_f1": float(
            100.0
            * f1_score(
                y,
                pp,
                average="macro",
                zero_division=0,
            )
        ),
        "rescue": int(rescue),
        "lost": int(lost),
        "mcnemar_p": float(p),
        "bootstrap": boot,
    }


# =============================================================================
# Safe Stage-A cache reuse
# =============================================================================

def _load_cached_selection_boxes_if_valid(
    sdir: Path,
    inner_train,
    inner_val,
):
    """Reuse completed Stage-A predicted-box CSVs only after exact coverage checks."""
    train_csv = sdir / "oof_train_predicted_boxes.csv"
    val_csv = sdir / "inner_val_predicted_boxes.csv"

    if not (train_csv.exists() and val_csv.exists()):
        return None

    try:
        tr_df = pd.read_csv(train_csv, dtype={"patient_id": str})
        va_df = pd.read_csv(val_csv, dtype={"patient_id": str})
    except Exception:
        return None

    required = {
        "path", "patient_id", "label", "found",
        "x0", "y0", "x1", "y1",
    }

    if not required.issubset(tr_df.columns):
        return None
    if not required.issubset(va_df.columns):
        return None

    expected_train = {
        str(q): (str(p.patient_id), int(p.label))
        for p in inner_train
        for q in p.paths
    }
    expected_val = {
        str(q): (str(p.patient_id), int(p.label))
        for p in inner_val
        for q in p.paths
    }

    if tr_df["path"].duplicated().any() or va_df["path"].duplicated().any():
        return None

    if set(tr_df["path"].astype(str)) != set(expected_train):
        return None
    if set(va_df["path"].astype(str)) != set(expected_val):
        return None

    for df, expected in ((tr_df, expected_train), (va_df, expected_val)):
        for r in df.itertuples(index=False):
            exp_pid, exp_label = expected[str(r.path)]
            if str(r.patient_id) != exp_pid:
                return None
            if int(r.label) != exp_label:
                return None

    return tr_df.to_dict("records"), va_df.to_dict("records")


# =============================================================================
# Main
# =============================================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--manifest",
        required=True,
    )
    ap.add_argument(
        "--mat_root",
        required=True,
    )
    ap.add_argument(
        "--out_root",
        required=True,
    )

    # Fresh locked seed by default.
    ap.add_argument(
        "--outer_seed",
        type=int,
        default=5813,
    )
    ap.add_argument(
        "--n_folds",
        type=int,
        default=5,
    )
    ap.add_argument(
        "--train_folds",
        type=int,
        default=5,
    )

    ap.add_argument(
        "--inner_n_splits",
        type=int,
        default=5,
    )
    ap.add_argument(
        "--inner_seed",
        type=int,
        default=31415,
    )

    # Frozen segmenter design.
    ap.add_argument(
        "--seg_oof_folds",
        type=int,
        default=3,
    )
    ap.add_argument(
        "--seg_oof_seed",
        type=int,
        default=27182,
    )
    ap.add_argument(
        "--seg_epochs",
        type=int,
        default=8,
    )
    ap.add_argument(
        "--seg_size",
        type=int,
        default=256,
    )
    ap.add_argument(
        "--seg_batch_size",
        type=int,
        default=8,
    )
    ap.add_argument(
        "--seg_eval_batch_size",
        type=int,
        default=16,
    )
    ap.add_argument(
        "--seg_lr",
        type=float,
        default=2e-4,
    )
    ap.add_argument(
        "--seg_weight_decay",
        type=float,
        default=1e-4,
    )

    # Frozen predicted-box rule.
    ap.add_argument(
        "--box_threshold",
        type=float,
        default=0.50,
    )
    ap.add_argument(
        "--box_fallback_threshold",
        type=float,
        default=0.30,
    )
    ap.add_argument(
        "--crop_margin_factor",
        type=float,
        default=1.10,
    )

    # Classifier.
    ap.add_argument(
        "--backbone",
        default=(
            "swin_base_patch4_window7_224"
        ),
    )
    ap.add_argument(
        "--image_size",
        type=int,
        default=224,
    )
    ap.add_argument(
        "--max_select_epochs",
        type=int,
        default=15,
    )
    ap.add_argument(
        "--freeze_epochs",
        type=int,
        default=3,
    )

    ap.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )
    ap.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
    )
    ap.add_argument(
        "--accumulation_steps",
        type=int,
        default=2,
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    ap.add_argument(
        "--dropout",
        type=float,
        default=0.15,
    )
    ap.add_argument(
        "--backbone_lr",
        type=float,
        default=2e-5,
    )
    ap.add_argument(
        "--head_lr",
        type=float,
        default=2e-4,
    )
    ap.add_argument(
        "--minimum_lr",
        type=float,
        default=1e-7,
    )
    ap.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )
    ap.add_argument(
        "--focal_gamma",
        type=float,
        default=2.0,
    )
    ap.add_argument(
        "--label_smoothing",
        type=float,
        default=0.03,
    )
    ap.add_argument(
        "--grad_clip",
        type=float,
        default=3.0,
    )

    ap.add_argument(
        "--alignment_samples",
        type=int,
        default=48,
    )

    ap.add_argument(
        "--amp_dtype",
        choices=["fp16", "bf16"],
        default="bf16",
    )

    ap.add_argument(
        "--deterministic",
        action="store_true",
    )
    ap.add_argument(
        "--no_pretrained",
        action="store_true",
    )

    ap.add_argument(
        "--save_models",
        action="store_true",
    )

    ap.add_argument(
        "--baseline_patient_oof",
        default="",
    )
    ap.add_argument(
        "--bootstrap",
        type=int,
        default=20000,
    )

    args = ap.parse_args()

    if args.train_folds > args.n_folds:
        raise ValueError(
            "--train_folds cannot exceed --n_folds."
        )

    if (
        args.max_select_epochs
        <= args.freeze_epochs
    ):
        raise ValueError(
            "--max_select_epochs must exceed --freeze_epochs."
        )

    if args.seg_oof_folds < 2:
        raise ValueError(
            "--seg_oof_folds must be >=2."
        )

    # Reuse the already audited geometry sanity check.
    pred.synthetic_bbox_audit()

    common.seed_all(
        args.outer_seed,
        args.deterministic,
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    slice_rows, _ = common.load_manifest(
        Path(args.manifest)
    )

    preflight = pred.preflight_manifest_mat(
        slice_rows,
        Path(args.mat_root),
        args.alignment_samples,
    )

    print("=" * 156)
    print(
        "PredROI-Swin v1.1 — LOCKED FULL-5 PATIENT-DISJOINT EVALUATION"
    )
    print("=" * 156)
    print(
        f"device                    : {device}"
    )
    print(
        f"LOCKED outer seed         : {args.outer_seed}"
    )
    print(
        f"outer folds               : {args.train_folds}/{args.n_folds}"
    )
    print(
        "method hyperparameters    : FROZEN from pre-evaluation design"
    )
    print(
        "outer-val ground-truth mask: NEVER LOADED"
    )
    print(
        "patient inference          : mean slice softmax probability"
    )
    print(
        f"MAT alignment median/min  : "
        f"{preflight['corr_median']:.6f}/"
        f"{preflight['corr_min']:.6f}"
    )
    print("=" * 156)

    out_root = Path(
        args.out_root
    )
    out_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_json(
        {
            **vars(args),
            "method": "PredROI-Swin-v1.1-locked-full5",
            "outer_val_ground_truth_mask_use": False,
            "classifier_outer_val_crop_source": (
                "final localizer trained on outer-train only"
            ),
            "classifier_outer_train_crop_source": (
                "patient-disjoint OOF segmenter predictions"
            ),
            "selection_metric": (
                "inner patient Macro-F1 only"
            ),
            "patient_inference": (
                "mean slice softmax probability"
            ),
            "preflight": preflight,
        },
        out_root / "run_config.json",
    )

    splits = common.outer_patient_splits(
        slice_rows,
        args.n_folds,
        args.outer_seed,
    )

    all_pat = []
    all_slice = []
    fold_summary = []

    for fold, (tr_idx, va_idx) in enumerate(
        splits,
        start=1,
    ):
        if fold > args.train_folds:
            continue

        # Compatibility with the audited screen helper. That helper uses
        # args.outer_fold only for deterministic RNG seeding.
        args.outer_fold = int(fold)

        print(
            "\n"
            + "=" * 156
        )
        print(
            f"OUTER FOLD {fold}"
        )
        print(
            "=" * 156
        )

        outer_train = (
            common.patient_subset_from_slice_indices(
                slice_rows,
                tr_idx,
            )
        )

        outer_val = (
            common.patient_subset_from_slice_indices(
                slice_rows,
                va_idx,
            )
        )

        tr_ids = {
            str(p.patient_id)
            for p in outer_train
        }

        va_ids = {
            str(p.patient_id)
            for p in outer_val
        }

        if tr_ids & va_ids:
            raise RuntimeError(
                "Outer patient leakage."
            )

        inner_train, inner_val = (
            common.make_inner_selection_split(
                outer_train,
                args.inner_n_splits,
                args.inner_seed + fold,
            )
        )

        print(
            f"outer train/val = "
            f"{len(outer_train)}/{len(outer_val)} patients | "
            f"inner train/val = "
            f"{len(inner_train)}/{len(inner_val)}"
        )

        fdir = (
            out_root
            / f"fold_{fold}"
        )
        fdir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # -------------------------------------------------------------
        # INNER selection — uses no outer-val data.
        # -------------------------------------------------------------
        sdir = (
            fdir
            / "selection"
        )
        sdir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            "[STAGE A] INNER-only localizer cross-fit + classifier selection"
        )

        cached_boxes = _load_cached_selection_boxes_if_valid(
            sdir,
            inner_train,
            inner_val,
        )

        if cached_boxes is not None:
            inner_train_boxes, inner_val_boxes = cached_boxes
            print(
                "[RESUME] Reusing validated Stage-A predicted-box caches "
                "(OOF inner-train + final-localizer inner-val)."
            )
        else:
            inner_train_boxes, inner_val_boxes = (
                pred.crossfit_localizer_and_boxes(
                    inner_train,
                    inner_val,
                    args,
                    device,
                    sdir,
                )
            )

        inner_best = pred.train_classifier(
            inner_train,
            inner_val,
            inner_train_boxes,
            inner_val_boxes,
            args,
            device,
            sdir,
        )

        selected_epoch = int(
            inner_best["epoch"]
        )

        print(
            f"[SELECTED] fold={fold} | "
            f"epoch={selected_epoch} | "
            f"INNER Acc="
            f"{inner_best['patient']['accuracy']:.3f}% | "
            f"F1="
            f"{inner_best['patient']['macro_f1']:.3f}% | "
            f"err="
            f"{inner_best['patient']['errors']}"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # -------------------------------------------------------------
        # OUTER-TRAIN refit.
        # -------------------------------------------------------------
        rdir = (
            fdir
            / "refit"
        )
        rdir.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            "[STAGE B] OUTER-TRAIN OOF localizer boxes"
        )

        refit_train_boxes = (
            crossfit_train_boxes_only(
                outer_train,
                args,
                device,
                rdir,
                seed_offset=fold,
            )
        )

        print(
            "[STAGE C] Classifier reinit/refit on ALL OUTER-TRAIN"
        )

        classifier = (
            refit_classifier_fixed_epoch(
                outer_train,
                refit_train_boxes,
                selected_epoch,
                args,
                device,
                seed=(
                    args.outer_seed
                    * 7000
                    + fold
                ),
                outdir=rdir,
            )
        )

        print(
            "[STAGE D] Final localizer on ALL OUTER-TRAIN"
        )

        final_localizer = pred.train_segmenter(
            outer_train,
            args,
            device,
            seed=(
                args.outer_seed
                * 8000
                + fold
            ),
            tag=f"OUTER_FINAL_F{fold}",
        )

        # -------------------------------------------------------------
        # OUTER-VAL image-only localizer inference.
        # This path has no mat_root and cannot access tumorMask.
        # -------------------------------------------------------------
        print(
            "[STAGE E] OUTER-VAL image-only predicted boxes"
        )

        outer_val_boxes = (
            predict_boxes_image_only(
                final_localizer,
                outer_val,
                args,
                device,
                seed=(
                    args.outer_seed
                    * 8100
                    + fold
                ),
                tag=(
                    f"outer_val_image_only_fold_{fold}"
                ),
            )
        )

        pd.DataFrame(
            outer_val_boxes
        ).to_csv(
            rdir / "outer_val_predicted_boxes.csv",
            index=False,
        )

        found_rate = float(
            np.mean(
                [
                    int(r["found"])
                    for r in outer_val_boxes
                ]
            )
        )

        print(
            f"[OUTER LOCALIZER] bbox-found="
            f"{100.0 * found_rate:.2f}% "
            "(no GT mask loaded)"
        )

        # -------------------------------------------------------------
        # OUTER-VAL classification.
        # -------------------------------------------------------------
        print(
            "[STAGE F] OUTER-VAL PredROI classification"
        )

        srows, prows, sm, pm = (
            evaluate_outer(
                classifier,
                outer_val,
                outer_val_boxes,
                args,
                device,
                fold,
            )
        )

        pd.DataFrame(
            srows
        ).to_csv(
            fdir / "outer_slice_predictions.csv",
            index=False,
        )

        pd.DataFrame(
            prows
        ).to_csv(
            fdir / "outer_patient_predictions.csv",
            index=False,
        )

        if args.save_models:
            torch.save(
                {
                    "state_dict": (
                        classifier.state_dict()
                    ),
                    "outer_seed": args.outer_seed,
                    "outer_fold": fold,
                    "selected_epoch": selected_epoch,
                    "method": "PredROI-Swin-v1.1",
                },
                fdir / "final_classifier.pt",
            )

            torch.save(
                {
                    "state_dict": (
                        final_localizer.state_dict()
                    ),
                    "outer_seed": args.outer_seed,
                    "outer_fold": fold,
                    "seg_epochs": args.seg_epochs,
                    "method": "PredROI-localizer-v1",
                },
                fdir / "final_localizer.pt",
            )

        all_slice.extend(
            srows
        )
        all_pat.extend(
            prows
        )

        fold_summary.append(
            {
                "fold": fold,
                "selected_epoch": selected_epoch,
                "inner_patient_acc": (
                    inner_best[
                        "patient"
                    ]["accuracy"]
                ),
                "inner_patient_f1": (
                    inner_best[
                        "patient"
                    ]["macro_f1"]
                ),
                "outer_patient_acc": pm[
                    "accuracy"
                ],
                "outer_patient_f1": pm[
                    "macro_f1"
                ],
                "outer_patient_errors": pm[
                    "errors"
                ],
                "outer_glioma_recall": pm[
                    "glioma_recall"
                ],
                "outer_meningioma_recall": pm[
                    "meningioma_recall"
                ],
                "outer_pituitary_recall": pm[
                    "pituitary_recall"
                ],
                "outer_slice_acc": sm[
                    "accuracy"
                ],
                "outer_slice_f1": sm[
                    "macro_f1"
                ],
                "outer_bbox_found_rate": found_rate,
            }
        )

        print(
            f"[OUTER RESULT] "
            f"Acc={pm['accuracy']:.3f}% | "
            f"F1={pm['macro_f1']:.3f}% | "
            f"G-R={pm['glioma_recall']:.3f}% | "
            f"M-R={pm['meningioma_recall']:.3f}% | "
            f"P-R={pm['pituitary_recall']:.3f}% | "
            f"err={pm['errors']}"
        )

        del classifier
        del final_localizer

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    patient_csv = (
        out_root
        / "patient_oof.csv"
    )

    pd.DataFrame(
        all_pat
    ).to_csv(
        patient_csv,
        index=False,
    )

    pd.DataFrame(
        all_slice
    ).to_csv(
        out_root
        / "slice_oof.csv",
        index=False,
    )

    pd.DataFrame(
        fold_summary
    ).to_csv(
        out_root
        / "fold_summary.csv",
        index=False,
    )

    pm = pred.patient_metrics(
        all_pat
    )

    sm = pred.slice_metrics(
        all_slice
    )

    summary = {
        "method": "PredROI-Swin-v1.1-locked-full5",
        "outer_seed": args.outer_seed,
        "patient_oof": pm,
        "slice_oof": sm,
        "folds": fold_summary,
        "outer_val_ground_truth_masks_loaded": False,
    }

    print(
        "\n"
        + "=" * 156
    )
    print(
        "FINAL LOCKED FULL-5 PredROI-Swin v1.1"
    )
    print(
        "=" * 156
    )

    print(
        f"patient avg-prob | "
        f"Acc={pm['accuracy']:.3f}% | "
        f"F1={pm['macro_f1']:.3f}% | "
        f"G-R={pm['glioma_recall']:.3f}% | "
        f"M-R={pm['meningioma_recall']:.3f}% | "
        f"P-R={pm['pituitary_recall']:.3f}% | "
        f"err={pm['errors']}"
    )

    print(
        f"slice-level      | "
        f"Acc={sm['accuracy']:.3f}% | "
        f"F1={sm['macro_f1']:.3f}% | "
        f"err={sm['errors']}"
    )

    if args.baseline_patient_oof:
        cmp = paired_compare(
            patient_csv,
            Path(args.baseline_patient_oof),
            args.bootstrap,
            seed=args.outer_seed + 9187,
        )

        summary[
            "paired_vs_standard_swin"
        ] = cmp

        b = cmp[
            "bootstrap"
        ]

        print(
            "\n"
            + "=" * 156
        )
        print(
            "PAIRED PredROI-Swin vs STANDARD SWIN"
        )
        print(
            "=" * 156
        )

        print(
            f"Standard Swin | "
            f"Acc={cmp['baseline_acc']:.3f}% | "
            f"F1={cmp['baseline_f1']:.3f}%"
        )

        print(
            f"PredROI-Swin  | "
            f"Acc={cmp['predroi_acc']:.3f}% | "
            f"F1={cmp['predroi_f1']:.3f}%"
        )

        print(
            f"PredROI - Swin | "
            f"ΔAcc={b['delta_acc_pp']:+.3f} pp "
            f"[{b['acc_ci95_pp'][0]:+.3f}, "
            f"{b['acc_ci95_pp'][1]:+.3f}] | "
            f"ΔF1={b['delta_f1_pp']:+.3f} pp "
            f"[{b['f1_ci95_pp'][0]:+.3f}, "
            f"{b['f1_ci95_pp'][1]:+.3f}]"
        )

        print(
            f"McNemar p={cmp['mcnemar_p']:.6f} | "
            f"rescue/lost={cmp['rescue']}/{cmp['lost']}"
        )

    save_json(
        summary,
        out_root / "summary.json",
    )

    print(
        f"\nWritten: "
        f"{out_root / 'summary.json'}"
    )


if __name__ == "__main__":
    main()
