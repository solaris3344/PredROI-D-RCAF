#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
cheng_drcaf_predroi_swin_full5_fixedsplit.py

LOCKED FULL-5 EVALUATION FOR D-RCAF PredROI-Swin
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
import torch.nn as nn
import torch.nn.functional as F
import timm
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
# D-RCAF: disagreement-aware ROI-reliability adaptive fusion
# =============================================================================

DRCAF_GATE_HIDDEN = 256
DRCAF_QUALITY_DIM = 4
DRCAF_QUALITY_EMBED = 32
DRCAF_GATE_DROPOUT = 0.10


def _drcaf_apply_shared_geom(img, do_flip, angle, scale):
    out = img
    if do_flip:
        out = cv2.flip(out, 1)
    h, w = out.shape
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    return cv2.warpAffine(
        out, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _drcaf_to_tensor(img, image_size):
    img = cv2.resize(
        img, (int(image_size), int(image_size)),
        interpolation=cv2.INTER_AREA,
    )
    x = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
    x = x.repeat(3, 1, 1)
    return (x - common.IMAGENET_MEAN) / common.IMAGENET_STD


def _drcaf_quality_cues(box, original_hw, fallback_threshold):
    """Return real, non-oracle reliability cues available at inference time.

    q = [bbox_found, bbox_area_ratio, normalized_center_distance, fallback_used]

    No ground-truth mask and no invented localization confidence are used.
    """
    H, W = int(original_hw[0]), int(original_hw[1])
    found = float(bool(box.get("found", False)))

    x0 = float(box.get("x0", 0.0))
    y0 = float(box.get("y0", 0.0))
    x1 = float(box.get("x1", W - 1.0))
    y1 = float(box.get("y1", H - 1.0))

    bw = max(1.0, x1 - x0 + 1.0)
    bh = max(1.0, y1 - y0 + 1.0)
    area_ratio = float(np.clip((bw * bh) / max(1.0, H * W), 0.0, 1.0))

    cx = 0.5 * (x0 + x1) / max(1.0, W - 1.0)
    cy = 0.5 * (y0 + y1) / max(1.0, H - 1.0)
    center_dist = float(
        np.clip(
            np.sqrt((cx - 0.5) ** 2 + (cy - 0.5) ** 2) / np.sqrt(0.5),
            0.0, 1.0,
        )
    )

    threshold = box.get("threshold", np.nan)
    try:
        threshold = float(threshold)
    except Exception:
        threshold = float("nan")

    fallback_used = float(
        (not found)
        or (np.isfinite(threshold) and abs(threshold - float(fallback_threshold)) < 1e-6)
    )

    return torch.tensor(
        [found, area_ratio, center_dist, fallback_used],
        dtype=torch.float32,
    )


def load_drcaf_views(path, box, image_size, train, rng, margin_factor, fallback_threshold):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read {path}")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    full = clahe.apply(img)
    roi = pred.crop_from_box(full, box, margin_factor)
    quality = _drcaf_quality_cues(box, img.shape, fallback_threshold)

    if train:
        do_flip = bool(rng.random() < 0.5)
        angle = float(rng.uniform(-5.0, 5.0))
        scale = float(rng.uniform(0.97, 1.03))
        full = _drcaf_apply_shared_geom(full, do_flip, angle, scale)
        roi = _drcaf_apply_shared_geom(roi, do_flip, angle, scale)

    return (
        _drcaf_to_tensor(full, image_size),
        _drcaf_to_tensor(roi, image_size),
        quality,
    )


class DRCAFDataset(Dataset):
    def __init__(
        self, rows, box_lookup, image_size, train, seed,
        margin_factor, fallback_threshold,
    ):
        self.rows = list(rows)
        self.box_lookup = box_lookup
        self.image_size = int(image_size)
        self.train = bool(train)
        self.seed = int(seed)
        self.margin_factor = float(margin_factor)
        self.fallback_threshold = float(fallback_threshold)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        path = str(r["path"])
        if path not in self.box_lookup:
            raise RuntimeError(f"Missing predicted box for {path}")

        rng = np.random.default_rng(self.seed + self.epoch * 100003 + idx * 997)
        xg, xr, q = load_drcaf_views(
            path,
            self.box_lookup[path],
            self.image_size,
            self.train,
            rng,
            self.margin_factor,
            self.fallback_threshold,
        )
        return {
            "global_image": xg,
            "roi_image": xr,
            "quality": q,
            "label": torch.tensor(int(r["label"]), dtype=torch.long),
            "patient_id": str(r["patient_id"]),
            "path": path,
        }


class DRCAFSharedSwin(nn.Module):
    """Shared Swin + disagreement-aware, reliability-conditioned channel gate.

    fg = E(x_global), fr = E(x_roi)
    d  = |fg-fr|
    qe = MLP(q), q=[found, bbox area, center distance, fallback-used]
    g  = sigmoid(MLP([fg, fr, d, qe]))  # channel-wise ROI gate
    f  = (1-g)*fg + g*fr

    Final gate layer is zero initialized, therefore g=0.5 initially.
    """
    def __init__(self, backbone, pretrained=True, dropout=0.15):
        super().__init__()
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        dim = int(self.encoder.num_features)
        self.feature_dim = dim

        self.quality = nn.Sequential(
            nn.LayerNorm(DRCAF_QUALITY_DIM),
            nn.Linear(DRCAF_QUALITY_DIM, DRCAF_QUALITY_EMBED),
            nn.GELU(),
        )
        gate_in = 3 * dim + DRCAF_QUALITY_EMBED
        self.gate_norm = nn.LayerNorm(gate_in)
        self.gate_fc1 = nn.Linear(gate_in, DRCAF_GATE_HIDDEN)
        self.gate_fc2 = nn.Linear(DRCAF_GATE_HIDDEN, dim)
        self.gate_drop = nn.Dropout(DRCAF_GATE_DROPOUT)
        nn.init.zeros_(self.gate_fc2.weight)
        nn.init.zeros_(self.gate_fc2.bias)

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(dim, N_CLASSES)

    def forward(self, x_global, x_roi, quality, return_aux=False):
        B = x_global.shape[0]
        both = torch.cat([x_global, x_roi], dim=0)
        feats = self.encoder(both)
        fg = feats[:B]
        fr = feats[B:]
        disagreement = torch.abs(fr - fg)
        qe = self.quality(quality.float())

        z = torch.cat([fg, fr, disagreement, qe], dim=-1)
        h = F.gelu(self.gate_fc1(self.gate_norm(z)))
        h = self.gate_drop(h)
        roi_gate = torch.sigmoid(self.gate_fc2(h))

        fused = (1.0 - roi_gate) * fg + roi_gate * fr
        logits = self.classifier(self.dropout(fused))

        if return_aux:
            return logits, {
                "roi_gate_mean": roi_gate.mean(dim=-1),
                "disagreement_l1": disagreement.mean(dim=-1),
            }
        return logits


def drcaf_set_encoder_trainable(model, trainable):
    for p in model.encoder.parameters():
        p.requires_grad = bool(trainable)


def drcaf_make_optimizer(model, backbone_lr, head_lr, weight_decay):
    bb, head = [], []
    for name, p in model.named_parameters():
        (bb if name.startswith("encoder.") else head).append(p)
    return torch.optim.AdamW(
        [
            {"params": bb, "lr": backbone_lr},
            {"params": head, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


def _drcaf_make_dataset(patients, boxes, args, train, seed):
    rows = pred.slice_rows_from_patients(patients)
    lookup = {str(r["path"]): r for r in boxes}
    expected = {str(r["path"]) for r in rows}
    if set(lookup) != expected:
        raise RuntimeError("D-RCAF predicted-box coverage mismatch.")
    ds = DRCAFDataset(
        rows, lookup, args.image_size, train, seed,
        args.crop_margin_factor, args.box_fallback_threshold,
    )
    return ds


def _drcaf_eval_loader(model, loader, args, device, fold=None):
    model.eval()
    srows = []
    with torch.no_grad():
        for batch in loader:
            xg = batch["global_image"].to(device, non_blocking=True)
            xr = batch["roi_image"].to(device, non_blocking=True)
            q = batch["quality"].to(device, non_blocking=True)
            y = batch["label"].cpu().numpy()

            with common.autocast_ctx(device, args.amp_dtype):
                logits, aux = model(xg, xr, q, return_aux=True)
                prob = F.softmax(logits, dim=-1).float().cpu().numpy()

            pred_id = prob.argmax(axis=1)
            gate = aux["roi_gate_mean"].float().cpu().numpy()
            dis = aux["disagreement_l1"].float().cpu().numpy()
            qcpu = batch["quality"].float().cpu().numpy()

            for i in range(len(y)):
                rec = {
                    "patient_id": str(batch["patient_id"][i]),
                    "path": str(batch["path"][i]),
                    "true_id": int(y[i]),
                    "slice_pred_id": int(pred_id[i]),
                    "roi_gate_mean": float(gate[i]),
                    "disagreement_l1": float(dis[i]),
                    "bbox_found_cue": float(qcpu[i, 0]),
                    "bbox_area_ratio_cue": float(qcpu[i, 1]),
                    "bbox_center_distance_cue": float(qcpu[i, 2]),
                    "bbox_fallback_cue": float(qcpu[i, 3]),
                }
                if fold is not None:
                    rec["fold"] = int(fold)
                for c, cname in enumerate(CLASSES):
                    rec[f"prob_{cname}"] = float(prob[i, c])
                srows.append(rec)

    prows = pred.aggregate_patient_probs(srows)
    if fold is not None:
        for r in prows:
            r["fold"] = int(fold)
    sm = pred.slice_metrics(srows)
    pm = pred.patient_metrics(prows)
    return srows, prows, sm, pm


def train_drcaf_classifier(
    inner_train, inner_val, train_boxes, val_boxes,
    args, device, out_root: Path,
):
    common.seed_all(
        args.outer_seed * 50000 + args.outer_fold,
        args.deterministic,
    )

    tr_ds = _drcaf_make_dataset(
        inner_train, train_boxes, args, True,
        args.outer_seed * 51000 + args.outer_fold,
    )
    va_ds = _drcaf_make_dataset(
        inner_val, val_boxes, args, False,
        args.outer_seed * 52000 + args.outer_fold,
    )

    tr_loader = DataLoader(
        tr_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )
    va_loader = DataLoader(
        va_ds, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )

    model = DRCAFSharedSwin(
        args.backbone,
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
    ).to(device)

    cw = common.patient_class_weights(inner_train, device)
    criterion = common.FocalLoss(
        cw, gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
    )

    drcaf_set_encoder_trainable(model, False)
    opt = drcaf_make_optimizer(
        model, args.backbone_lr, args.head_lr, args.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda" and args.amp_dtype == "fp16"),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.max_select_epochs), eta_min=args.minimum_lr,
    )

    history, best = [], None
    best_slice_rows = best_patient_rows = None

    for epoch in range(1, args.max_select_epochs + 1):
        tr_ds.set_epoch(epoch)
        drcaf_set_encoder_trainable(model, epoch > args.freeze_epochs)
        model.train()
        opt.zero_grad(set_to_none=True)

        losses, gates = [], []
        correct = n = 0

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
                raise FloatingPointError(f"Non-finite D-RCAF loss epoch={epoch}")

            scaler.scale(scaled).backward()
            if step % args.accumulation_steps == 0 or step == len(tr_loader):
                scaler.unscale_(opt)
                grad_params = [
                    p for p in model.parameters()
                    if p.requires_grad and p.grad is not None
                ]
                gnorm = torch.nn.utils.clip_grad_norm_(grad_params, args.grad_clip)
                if not torch.isfinite(gnorm):
                    raise FloatingPointError("Non-finite D-RCAF gradient norm.")
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            losses.append(float(loss.detach().cpu()))
            gates.extend(aux["roi_gate_mean"].detach().float().cpu().tolist())
            correct += int((logits.argmax(1) == y).sum().item())
            n += int(y.numel())

        srows, prows, sm, pm = _drcaf_eval_loader(
            model, va_loader, args, device,
        )
        tracc = 100.0 * correct / max(1, n)
        val_gate = float(np.mean([r["roi_gate_mean"] for r in srows]))
        val_dis = float(np.mean([r["disagreement_l1"] for r in srows]))

        history.append({
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "train_slice_acc": tracc,
            "train_roi_gate_mean": float(np.mean(gates)),
            "inner_patient_acc": pm["accuracy"],
            "inner_patient_f1": pm["macro_f1"],
            "inner_patient_errors": pm["errors"],
            "inner_slice_acc": sm["accuracy"],
            "inner_slice_f1": sm["macro_f1"],
            "inner_roi_gate_mean": val_gate,
            "inner_disagreement_l1": val_dis,
        })

        print(
            f"[D-RCAF] epoch {epoch:02d}/{args.max_select_epochs} | "
            f"loss={np.mean(losses):.4f} train-slice={tracc:6.2f}% | "
            f"INNER patient Acc={pm['accuracy']:6.2f}% "
            f"F1={pm['macro_f1']:6.2f}% err={pm['errors']} | "
            f"ROI-gate={val_gate:.3f} dis={val_dis:.4f}"
        )

        if epoch > args.freeze_epochs:
            key = (pm["macro_f1"], pm["accuracy"], -epoch)
            if best is None or key > best["key"]:
                best = {
                    "key": key,
                    "epoch": int(epoch),
                    "patient": pm,
                    "slice": sm,
                    "roi_gate_mean": val_gate,
                    "disagreement_l1": val_dis,
                }
                best_slice_rows = [dict(r) for r in srows]
                best_patient_rows = [dict(r) for r in prows]

        sched.step()

    if best is None:
        raise RuntimeError("No selectable D-RCAF epoch.")

    pd.DataFrame(history).to_csv(
        out_root / "drcaf_inner_selection_history.csv", index=False,
    )
    pd.DataFrame(best_slice_rows).to_csv(
        out_root / "drcaf_inner_best_slice_predictions.csv", index=False,
    )
    pd.DataFrame(best_patient_rows).to_csv(
        out_root / "drcaf_inner_best_patient_predictions.csv", index=False,
    )
    return best


# =============================================================================
# Classifier fixed-epoch refit
# =============================================================================

def refit_classifier_fixed_epoch(
    outer_train, train_boxes, selected_epoch,
    args, device, seed, outdir: Path,
):
    common.seed_all(seed, args.deterministic)
    ds = _drcaf_make_dataset(
        outer_train, train_boxes, args, True, seed + 1000,
    )
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )

    model = DRCAFSharedSwin(
        args.backbone,
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
    ).to(device)

    cw = common.patient_class_weights(outer_train, device)
    criterion = common.FocalLoss(
        cw, gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
    )
    drcaf_set_encoder_trainable(model, False)
    opt = drcaf_make_optimizer(
        model, args.backbone_lr, args.head_lr, args.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda" and args.amp_dtype == "fp16"),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, int(selected_epoch)), eta_min=args.minimum_lr,
    )

    history = []
    for epoch in range(1, int(selected_epoch) + 1):
        ds.set_epoch(epoch)
        drcaf_set_encoder_trainable(model, epoch > args.freeze_epochs)
        model.train()
        opt.zero_grad(set_to_none=True)
        losses, gates = [], []
        correct = n = 0

        for step, batch in enumerate(loader, start=1):
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
                    f"Non-finite D-RCAF refit loss epoch={epoch}"
                )
            scaler.scale(scaled).backward()

            if step % args.accumulation_steps == 0 or step == len(loader):
                scaler.unscale_(opt)
                grad_params = [
                    p for p in model.parameters()
                    if p.requires_grad and p.grad is not None
                ]
                gnorm = torch.nn.utils.clip_grad_norm_(grad_params, args.grad_clip)
                if not torch.isfinite(gnorm):
                    raise FloatingPointError("Non-finite D-RCAF refit gradient.")
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            losses.append(float(loss.detach().cpu()))
            gates.extend(aux["roi_gate_mean"].detach().float().cpu().tolist())
            correct += int((logits.argmax(1) == y).sum().item())
            n += int(y.numel())

        sched.step()
        tracc = 100.0 * correct / max(1, n)
        history.append({
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "train_slice_acc": tracc,
            "roi_gate_mean": float(np.mean(gates)),
        })
        print(
            f"      D-RCAF REFIT epoch {epoch:02d}/{selected_epoch} | "
            f"loss={np.mean(losses):.4f} | slice-acc={tracc:.2f}% | "
            f"ROI-gate={np.mean(gates):.3f}"
        )

    pd.DataFrame(history).to_csv(
        outdir / "drcaf_classifier_refit_history.csv", index=False,
    )
    return model


# =============================================================================
# OUTER evaluation
# =============================================================================

@torch.no_grad()
@torch.no_grad()
def evaluate_outer(classifier, outer_val, val_boxes, args, device, fold):
    ds = _drcaf_make_dataset(
        outer_val, val_boxes, args, False,
        args.outer_seed * 900000 + fold,
    )
    loader = DataLoader(
        ds, batch_size=args.eval_batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda"),
    )
    return _drcaf_eval_loader(
        classifier, loader, args, device, fold=fold,
    )


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



def _auto_resolve_paths(args):
    """Best-effort Windows/project-local path discovery for one-command use."""
    root = Path(__file__).resolve().parent

    if not args.manifest:
        preferred = root / "cheng_patient_disjoint" / "cheng_manifest.csv"
        cands = []
        if preferred.exists():
            cands.append(preferred)
        cands.extend(root.rglob("cheng_manifest.csv"))
        # stable de-dup
        uniq, seen = [], set()
        for q in cands:
            k = str(q.resolve()).lower()
            if k not in seen:
                seen.add(k); uniq.append(q)
        if len(uniq) == 1:
            args.manifest = str(uniq[0])
            print(f"[AUTO] manifest : {args.manifest}")
        elif len(uniq) == 0:
            raise FileNotFoundError(
                "Could not auto-find cheng_manifest.csv under the brain folder. "
                "Run with --manifest <full path>."
            )
        else:
            raise RuntimeError(
                "Multiple cheng_manifest.csv files found; pass --manifest explicitly:\n" +
                "\n".join(f"  - {q}" for q in uniq[:20])
            )

    if not args.mat_root:
        mdf = pd.read_csv(args.manifest)
        pcol = None
        for c in ["png_path", "path", "image_path", "filepath"]:
            if c in mdf.columns:
                pcol = c; break
        if pcol is None or len(mdf) == 0:
            raise RuntimeError("Manifest has no usable image path column for MAT-root discovery.")

        stems = [Path(str(x)).stem for x in mdf[pcol].head(8).tolist()]
        first = stems[0]
        hits = list(root.rglob(first + ".mat"))
        parents = []
        for h in hits:
            par = h.parent
            if all((par / (st + ".mat")).exists() for st in stems[:min(4, len(stems))]):
                parents.append(par)
        uniq, seen = [], set()
        for q in parents:
            k = str(q.resolve()).lower()
            if k not in seen:
                seen.add(k); uniq.append(q)
        if len(uniq) == 1:
            args.mat_root = str(uniq[0])
            print(f"[AUTO] mat_root : {args.mat_root}")
        elif len(uniq) == 0:
            raise FileNotFoundError(
                "Could not auto-find the Cheng .mat directory from the manifest. "
                "Run with --mat_root <folder containing the .mat files>."
            )
        else:
            raise RuntimeError(
                "Multiple candidate MAT roots found; pass --mat_root explicitly:\n" +
                "\n".join(f"  - {q}" for q in uniq[:20])
            )

    if not args.out_root:
        args.out_root = str(root / f"drcaf_full5_split{args.split_seed}_run{args.run_seed}")
        print(f"[AUTO] out_root : {args.out_root}")

    return args


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--manifest",
        default="",
    )
    ap.add_argument(
        "--mat_root",
        default="",
    )
    ap.add_argument(
        "--out_root",
        default="",
    )

    # Fixed patient split + independent stochastic training seed.
    # This separation is essential for seed-robustness analysis:
    # split_seed controls patient membership only; run_seed controls all
    # stochastic model/localizer training and evaluation RNG.
    ap.add_argument(
        "--split_seed",
        type=int,
        default=5813,
    )
    ap.add_argument(
        "--run_seed",
        type=int,
        default=7253,
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
        default=8,
    )
    ap.add_argument(
        "--eval_batch_size",
        type=int,
        default=16,
    )
    ap.add_argument(
        "--accumulation_steps",
        type=int,
        default=4,
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
        default=True,
        help="Save final D-RCAF and localizer checkpoints (default: on).",
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

    # Imported PredROI helper code expects args.outer_seed as its stochastic
    # training seed. Keep that internal compatibility alias, but NEVER use it
    # for patient split construction in this fixed-split script.
    args.outer_seed = int(args.run_seed)

    args = _auto_resolve_paths(args)

    if args.train_folds > args.n_folds:
        raise ValueError(
            "--train_folds cannot exceed --n_folds."
        )

    if args.split_seed != 5813:
        raise ValueError(
            "Seed-robustness protocol is frozen to split_seed=5813. "
            "Do not change patient fold membership."
        )
    if args.run_seed == args.split_seed:
        print(
            "[NOTE] run_seed equals split_seed; this reproduces the original "
            "5813 stochastic seed on the fixed 5813 patient split."
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
        args.run_seed,
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
        "D-RCAF PredROI-Swin — LOCKED FULL-5 PATIENT-DISJOINT EVALUATION"
    )
    print("=" * 156)
    print(
        f"device                    : {device}"
    )
    print(
        f"FIXED patient split seed  : {args.split_seed}"
    )
    print(
        f"STOCHASTIC run seed       : {args.run_seed}"
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
            "method": "D-RCAF-PredROI-Swin-full5",
            "fixed_patient_split_seed": args.split_seed,
            "stochastic_run_seed": args.run_seed,
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
            "fusion": "D-RCAF channel-wise ROI gate",
            "fusion_inputs": "fg, fr, abs(fr-fg), bbox reliability cues",
            "bbox_reliability_cues": "found, area ratio, center distance, fallback-used",
            "gate_hidden": DRCAF_GATE_HIDDEN,
            "quality_embedding": DRCAF_QUALITY_EMBED,
            "preflight": preflight,
        },
        out_root / "run_config.json",
    )

    splits = common.outer_patient_splits(
        slice_rows,
        args.n_folds,
        args.split_seed,
    )

    # Persist exact patient-to-fold membership so every seed run can be audited
    # for identical split composition.
    split_rows = []
    for audit_fold, (_tr_idx, audit_va_idx) in enumerate(splits, start=1):
        audit_val = common.patient_subset_from_slice_indices(
            slice_rows,
            audit_va_idx,
        )
        for p in audit_val:
            split_rows.append({
                "patient_id": str(p.patient_id),
                "fold": int(audit_fold),
                "label": int(p.label),
            })
    split_df = pd.DataFrame(split_rows).sort_values(
        ["fold", "patient_id"]
    ).reset_index(drop=True)
    split_df.to_csv(
        out_root / "fixed_patient_fold_assignment.csv",
        index=False,
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
            "[STAGE A] INNER-only localizer cross-fit + D-RCAF classifier selection"
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

        inner_best = train_drcaf_classifier(
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
            "[STAGE C] D-RCAF classifier reinit/refit on ALL OUTER-TRAIN"
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
            "[STAGE F] OUTER-VAL D-RCAF global+PredROI classification"
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
                    "split_seed": args.split_seed,
                    "run_seed": args.run_seed,
                    "outer_fold": fold,
                    "selected_epoch": selected_epoch,
                    "method": "D-RCAF-PredROI-Swin",
                },
                fdir / "final_drcaf_classifier.pt",
            )

            torch.save(
                {
                    "state_dict": (
                        final_localizer.state_dict()
                    ),
                    "outer_seed": args.outer_seed,
                    "split_seed": args.split_seed,
                    "run_seed": args.run_seed,
                    "outer_fold": fold,
                    "seg_epochs": args.seg_epochs,
                    "method": "PredROI-localizer-v1 (unchanged for D-RCAF)",
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
        "method": "D-RCAF-PredROI-Swin-full5-fixedsplit",
        "outer_seed": args.outer_seed,
        "split_seed": args.split_seed,
        "run_seed": args.run_seed,
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
        "FINAL LOCKED FULL-5 D-RCAF PredROI-Swin"
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
        f"fixed split/run  | {args.split_seed}/{args.run_seed}"
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
            seed=args.run_seed + 9187,
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



def synthetic_drcaf_audit():
    class TinyEncoder(nn.Module):
        num_features = 8
        def forward(self, x):
            # deterministic [B,8]
            return x.mean(dim=(2, 3))[:, :1].repeat(1, 8)

    # Validate only gate algebra independently of timm/downloads.
    B, D = 3, 8
    fg = torch.randn(B, D)
    fr = torch.randn(B, D)
    q = torch.rand(B, DRCAF_QUALITY_DIM)
    qe = nn.Sequential(
        nn.LayerNorm(DRCAF_QUALITY_DIM),
        nn.Linear(DRCAF_QUALITY_DIM, DRCAF_QUALITY_EMBED),
        nn.GELU(),
    )(q)
    z = torch.cat([fg, fr, torch.abs(fr-fg), qe], dim=-1)
    norm = nn.LayerNorm(3*D + DRCAF_QUALITY_EMBED)
    fc1 = nn.Linear(3*D + DRCAF_QUALITY_EMBED, DRCAF_GATE_HIDDEN)
    fc2 = nn.Linear(DRCAF_GATE_HIDDEN, D)
    nn.init.zeros_(fc2.weight); nn.init.zeros_(fc2.bias)
    g = torch.sigmoid(fc2(F.gelu(fc1(norm(z)))))
    if not torch.allclose(g, torch.full_like(g, 0.5), atol=1e-7):
        raise RuntimeError("D-RCAF zero-init gate audit failed")
    fused = (1-g)*fg + g*fr
    if fused.shape != (B, D):
        raise RuntimeError("D-RCAF fusion shape audit failed")
    return True


if __name__ == "__main__":
    synthetic_drcaf_audit()
    print("[AUDIT] D-RCAF zero-init channel-wise fusion: PASS")
    main()
