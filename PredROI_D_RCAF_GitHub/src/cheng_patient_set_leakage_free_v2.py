#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
cheng_patient_set_leakage_free.py

Leakage-free, patient-level END-TO-END set learning for the Cheng brain MRI dataset.

Why this script exists
----------------------
Previous experiments used cached slice latents from encoder checkpoints selected
with the outer validation fold. That is inappropriate for a final methods paper.

This script fixes that specific problem:

OUTER FOLD
    outer-train patients
        -> INNER patient-disjoint selection split
        -> choose training epoch using INNER validation only
        -> reinitialize model
        -> refit on ALL outer-train patients for exactly the selected epoch
        -> evaluate ONCE on untouched OUTER validation patients

The outer validation patients are never used for:
    - checkpoint/epoch selection,
    - early stopping,
    - architecture selection,
    - hyperparameter selection.

Patient-level training
----------------------
The network is optimized directly with one patient/bag as the training unit.
A random subset of that patient's slices is used during training; all slices are
used for validation/test inference.

Regimes
-------
mean_set
    DeepSets-style patient encoder using the mean of transformed slice embeddings.

moment_set
    Distribution-aware patient encoder using BOTH the mean and standard deviation
    of transformed slice embeddings.  This is not the earlier post-hoc [mean,std]
    classifier: the slice encoder and set representation are trained end-to-end
    with a patient-level loss.

For each regime, the script reports:
    1) neural patient classifier
    2) exact CVA fitted on leakage-free outer-training patient embeddings only

Recommended workflow
--------------------
A) DEBUG ON DEVELOPMENT SPLIT ONLY (do NOT use seed2026 for debugging):

python cheng_patient_set_leakage_free.py ^
  --manifest "C:\PATH\TO\brain_project\cheng_patient_disjoint\cheng_manifest.csv" ^
  --out_root "C:\PATH\TO\brain_project\patient_set_debug" ^
  --outer_seed 42 ^
  --regimes mean_set moment_set ^
  --train_folds 1 ^
  --max_select_epochs 8 ^
  --freeze_epochs 3 ^
  --bag_size 6 ^
  --amp_dtype bf16

B) FINAL UNTOUCHED SPLIT (run only after the pipeline is frozen):

python cheng_patient_set_leakage_free.py ^
  --manifest "C:\PATH\TO\brain_project\cheng_patient_disjoint\cheng_manifest.csv" ^
  --out_root "C:\PATH\TO\brain_project\patient_set_seed2026" ^
  --outer_seed 2026 ^
  --regimes mean_set moment_set ^
  --train_folds 5 ^
  --max_select_epochs 30 ^
  --bag_size 8

Dependencies
------------
pip install timm opencv-python pillow pandas scikit-learn scipy

Notes
-----
- The default backbone is Swin-Base to match the existing global branch.
- The 128-D projection is explicitly defined in this script.
- This script intentionally does NOT use the old validation-selected checkpoints.
- seed42 and seed123 should be treated as DEVELOPMENT splits from this point on.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler

try:
    import timm
except Exception as exc:
    timm = None
    TIMM_IMPORT_ERROR = exc
else:
    TIMM_IMPORT_ERROR = None


CLASSES = ["glioma", "meningioma", "pituitary"]
N_CLASSES = 3
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}


# =============================================================================
# Reproducibility / IO
# =============================================================================

def seed_all(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
    else:
        torch.backends.cudnn.benchmark = True


def save_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# =============================================================================
# Manifest
# =============================================================================

@dataclass
class SliceRec:
    path: str
    patient_id: str
    label: int
    class_name: str


@dataclass
class PatientRec:
    patient_id: str
    label: int
    class_name: str
    paths: list[str]


def _pick_col(cols, candidates):
    lower = {c.lower(): c for c in cols}
    for x in candidates:
        if x.lower() in lower:
            return lower[x.lower()]
    return None


def load_manifest(path: Path) -> tuple[list[SliceRec], list[PatientRec]]:
    df = pd.read_csv(path)
    pcol = _pick_col(df.columns, ["png_path", "path", "image_path", "filepath"])
    pidcol = _pick_col(df.columns, ["patient_id", "pid", "PID"])
    lcol = _pick_col(df.columns, ["label", "class_id", "target"])
    ccol = _pick_col(df.columns, ["class_name", "class", "diagnosis"])

    if pcol is None or pidcol is None:
        raise RuntimeError(
            f"Manifest must contain image path and patient ID columns. Columns={list(df.columns)}"
        )

    rows = []
    for _, r in df.iterrows():
        p = str(r[pcol])
        pid = str(r[pidcol])

        if ccol is not None:
            cname = str(r[ccol]).strip().lower()
            if cname not in CLASS_TO_ID:
                raise RuntimeError(f"Unknown class_name={cname}")
            label = CLASS_TO_ID[cname]
        elif lcol is not None:
            label = int(r[lcol])
            if label not in range(N_CLASSES):
                raise RuntimeError(f"Unexpected label={label}")
            cname = CLASSES[label]
        else:
            raise RuntimeError("Manifest needs either class_name or label.")

        if not os.path.exists(p):
            raise FileNotFoundError(
                f"Image does not exist: {p}\n"
                "Use the Cheng manifest generated on the same machine, with valid png_path entries."
            )
        rows.append(SliceRec(p, pid, label, cname))

    by_pid: dict[str, list[SliceRec]] = defaultdict(list)
    for r in rows:
        by_pid[r.patient_id].append(r)

    patients = []
    for pid, rr in sorted(by_pid.items()):
        labs = {x.label for x in rr}
        if len(labs) != 1:
            raise RuntimeError(f"PID {pid} has multiple class labels: {labs}")
        lab = next(iter(labs))
        patients.append(
            PatientRec(
                patient_id=pid,
                label=lab,
                class_name=CLASSES[lab],
                paths=[x.path for x in rr],
            )
        )

    print(
        f"[manifest] slices={len(rows)} | patients={len(patients)} | "
        f"patient classes={dict(Counter(p.class_name for p in patients))}"
    )
    return rows, patients


# =============================================================================
# Splits: ALL by patient
# =============================================================================

def outer_patient_splits(
    slice_rows: list[SliceRec],
    n_folds: int,
    seed: int,
):
    y = np.asarray([r.label for r in slice_rows], dtype=np.int64)
    groups = np.asarray([r.patient_id for r in slice_rows], dtype=object)

    sgkf = StratifiedGroupKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed,
    )
    return list(sgkf.split(np.zeros(len(slice_rows)), y, groups))


def patient_subset_from_slice_indices(
    slice_rows: list[SliceRec],
    indices: np.ndarray,
) -> list[PatientRec]:
    by_pid = defaultdict(list)
    for idx in indices:
        r = slice_rows[int(idx)]
        by_pid[r.patient_id].append(r)

    out = []
    for pid, rr in sorted(by_pid.items()):
        labs = {x.label for x in rr}
        if len(labs) != 1:
            raise RuntimeError(f"PID {pid} has inconsistent labels")
        lab = next(iter(labs))
        out.append(PatientRec(pid, lab, CLASSES[lab], [x.path for x in rr]))
    return out


def make_inner_selection_split(
    outer_train_patients: list[PatientRec],
    n_splits: int,
    seed: int,
):
    """
    One predeclared inner selection fold.  This is sufficient to ensure the
    outer validation fold is not used for checkpoint selection.
    """
    y = np.asarray([p.label for p in outer_train_patients], dtype=np.int64)
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    itr, iva = next(iter(skf.split(np.zeros(len(y)), y)))
    inner_train = [outer_train_patients[int(i)] for i in itr]
    inner_val = [outer_train_patients[int(i)] for i in iva]
    return inner_train, inner_val


# =============================================================================
# Image preprocessing
# =============================================================================

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load_gray_preprocessed(path: str, size: int, train: bool, rng: np.random.Generator):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read {path}")

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img = clahe.apply(img)

    if train:
        # Fixed, modest augmentation family; no tuning on outer validation.
        if rng.random() < 0.5:
            img = cv2.flip(img, 1)

        angle = float(rng.uniform(-5.0, 5.0))
        scale = float(rng.uniform(0.97, 1.03))
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
        img = cv2.warpAffine(
            img, M, (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
    x = x.repeat(3, 1, 1)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x


# =============================================================================
# Patient datasets
# =============================================================================

class PatientBagDataset(Dataset):
    def __init__(
        self,
        patients: list[PatientRec],
        image_size: int,
        train: bool,
        bag_size: int,
        seed: int,
    ):
        self.patients = list(patients)
        self.image_size = int(image_size)
        self.train = bool(train)
        self.bag_size = int(bag_size)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        p = self.patients[idx]
        rng = np.random.default_rng(
            self.seed + self.epoch * 100003 + idx * 997
        )

        paths = list(p.paths)
        if self.train and self.bag_size > 0 and len(paths) > self.bag_size:
            take = rng.choice(len(paths), size=self.bag_size, replace=False)
            paths = [paths[int(i)] for i in sorted(take)]

        xs = [
            load_gray_preprocessed(
                q, self.image_size, self.train, rng
            )
            for q in paths
        ]
        x = torch.stack(xs, dim=0)  # [S,3,H,W]
        return {
            "images": x,
            "label": torch.tensor(p.label, dtype=torch.long),
            "patient_id": p.patient_id,
            "n_slices": len(p.paths),
            "used_slices": len(paths),
        }


def patient_collate(batch):
    # Default batch size is 1 patient to keep Swin-Base memory safe.
    return batch


# =============================================================================
# Model
# =============================================================================

class PatientSetNet(nn.Module):
    def __init__(
        self,
        backbone: str,
        pretrained: bool,
        latent_dim: int,
        regime: str,
        dropout: float,
    ):
        super().__init__()

        if timm is None:
            raise ImportError(
                "timm could not be imported. Install it with: pip install timm\n"
                f"Original error: {TIMM_IMPORT_ERROR}"
            )

        self.regime = regime
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        enc_dim = int(self.encoder.num_features)

        # Explicit 1024-D (for Swin-Base) -> 128-D penultimate projection.
        self.proj = nn.Sequential(
            nn.LayerNorm(enc_dim),
            nn.Linear(enc_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Permutation-equivariant slice transform.
        self.phi = nn.Sequential(
            nn.LayerNorm(latent_dim),
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim, latent_dim),
        )

        if regime == "mean_set":
            patient_in = latent_dim
        elif regime == "moment_set":
            patient_in = 2 * latent_dim
        else:
            raise ValueError(regime)

        self.rho = nn.Sequential(
            nn.LayerNorm(patient_in),
            nn.Linear(patient_in, latent_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(latent_dim, N_CLASSES)

    def encode_slices(self, images):
        # images: [S,3,H,W]
        f = self.encoder(images)
        z = self.proj(f)
        h = self.phi(z)
        return h

    def aggregate(self, h):
        mu = h.mean(dim=0)
        if self.regime == "mean_set":
            p = self.rho(mu)
        else:
            # Population SD; stable even with small bags.
            var = ((h - mu) ** 2).mean(dim=0)
            sd = torch.sqrt(var + 1e-6)
            p = self.rho(torch.cat([mu, sd], dim=0))
        return p

    def forward_patient(self, images):
        h = self.encode_slices(images)
        p = self.aggregate(h)
        logits = self.classifier(p)
        return logits, p


# =============================================================================
# Loss / optimization
# =============================================================================

class FocalLoss(nn.Module):
    def __init__(self, class_weights, gamma=2.0, label_smoothing=0.03):
        super().__init__()
        self.register_buffer("class_weights", class_weights)
        self.gamma = float(gamma)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()

        ncls = logits.shape[-1]
        with torch.no_grad():
            q = torch.full_like(logp, self.label_smoothing / max(1, ncls - 1))
            q.scatter_(1, target[:, None], 1.0 - self.label_smoothing)

        ce_per_class = -q * logp
        focal = (1.0 - p).pow(self.gamma)
        weighted = ce_per_class * focal * self.class_weights[None, :]
        return weighted.sum(dim=1).mean()


def patient_class_weights(patients, device):
    counts = np.bincount(
        [p.label for p in patients],
        minlength=N_CLASSES,
    ).astype(np.float32)
    w = 1.0 / np.maximum(counts, 1.0)
    w = w / w.sum() * N_CLASSES
    return torch.tensor(w, dtype=torch.float32, device=device)


def make_optimizer(model, backbone_lr, head_lr, weight_decay):
    """
    IMPORTANT:
    Include encoder parameters in the optimizer even while they are temporarily
    frozen. AdamW safely skips parameters with grad=None. When the encoder is
    later unfrozen, the same optimizer will immediately start updating it.

    The previous debug version filtered by requires_grad=False at optimizer
    construction time, so the encoder was never actually added to AdamW.
    """
    bb = []
    head = []
    for name, p in model.named_parameters():
        if name.startswith("encoder."):
            bb.append(p)
        else:
            head.append(p)

    return torch.optim.AdamW(
        [
            {"params": bb, "lr": backbone_lr},
            {"params": head, "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )


def set_encoder_trainable(model, trainable: bool):
    for p in model.encoder.parameters():
        p.requires_grad = bool(trainable)


# =============================================================================
# Train / evaluation
# =============================================================================

def autocast_ctx(device, amp_dtype):
    if device.type != "cuda":
        return torch.autocast("cpu", enabled=False)
    dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
    return torch.autocast("cuda", dtype=dtype)


def train_one_epoch(
    model,
    dataset,
    loader,
    optimizer,
    criterion,
    scaler,
    device,
    epoch,
    freeze_epochs,
    accumulation_steps,
    grad_clip,
    amp_dtype,
):
    dataset.set_epoch(epoch)

    set_encoder_trainable(model, epoch > freeze_epochs)
    model.train()

    optimizer.zero_grad(set_to_none=True)
    losses = []
    correct = 0
    n = 0

    for step, batch_list in enumerate(loader, start=1):
        # Batch can contain >1 patient; process each patient independently,
        # then stack logits. Default patient_batch_size=1.
        logits_list = []
        labels = []

        with autocast_ctx(device, amp_dtype):
            for sample in batch_list:
                images = sample["images"].to(device, non_blocking=True)
                logits, _ = model.forward_patient(images)
                logits_list.append(logits)
                labels.append(sample["label"])

            logits = torch.stack(logits_list, dim=0)
            y = torch.stack(labels).to(device)
            loss = criterion(logits, y)
            loss_scaled = loss / accumulation_steps

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at epoch={epoch}, step={step}: {loss.detach().cpu().item()}"
            )

        scaler.scale(loss_scaled).backward()

        if step % accumulation_steps == 0 or step == len(loader):
            scaler.unscale_(optimizer)

            grad_params = [
                p for p in model.parameters()
                if p.requires_grad and p.grad is not None
            ]
            grad_norm = torch.nn.utils.clip_grad_norm_(grad_params, grad_clip)

            if not torch.isfinite(grad_norm):
                raise FloatingPointError(
                    f"Non-finite gradient norm at epoch={epoch}, step={step}: "
                    f"{float(grad_norm.detach().cpu())}"
                )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        losses.append(float(loss.detach().cpu()))
        correct += int((logits.argmax(dim=1) == y).sum().item())
        n += len(y)

    return float(np.mean(losses)), 100.0 * correct / max(1, n)


@torch.no_grad()
def infer_patients(model, dataset, loader, device, amp_dtype):
    model.eval()
    rows = []

    for batch_list in loader:
        for sample in batch_list:
            images = sample["images"].to(device, non_blocking=True)

            with autocast_ctx(device, amp_dtype):
                logits, emb = model.forward_patient(images)
                prob = F.softmax(logits, dim=-1)

            y = int(sample["label"].item())
            pred = int(prob.argmax().item())

            rec = {
                "patient_id": str(sample["patient_id"]),
                "true_id": y,
                "true_name": CLASSES[y],
                "pred_id": pred,
                "pred_name": CLASSES[pred],
                "correct": int(pred == y),
                "n_slices": int(sample["n_slices"]),
                "used_slices": int(sample["used_slices"]),
                "embedding": emb.float().cpu().numpy(),
            }
            for c in range(N_CLASSES):
                rec[f"prob_{CLASSES[c]}"] = float(prob[c].item())
            rows.append(rec)

    return rows


def metric_dict(rows):
    y = np.asarray([r["true_id"] for r in rows], dtype=np.int64)
    p = np.asarray([r["pred_id"] for r in rows], dtype=np.int64)

    rec = recall_score(
        y, p, labels=list(range(N_CLASSES)),
        average=None, zero_division=0,
    )

    return {
        "n": int(len(y)),
        "accuracy": float(100 * accuracy_score(y, p)),
        "macro_f1": float(
            100 * f1_score(
                y, p,
                labels=list(range(N_CLASSES)),
                average="macro",
                zero_division=0,
            )
        ),
        "glioma_recall": float(100 * rec[0]),
        "errors": int(np.sum(y != p)),
        "confusion_matrix": confusion_matrix(
            y, p, labels=list(range(N_CLASSES))
        ).tolist(),
    }


def select_epoch(
    regime,
    inner_train,
    inner_val,
    args,
    device,
    fold_seed,
    outdir,
):
    """
    Stage A: epoch selection with INNER validation only.
    """
    seed_all(fold_seed, args.deterministic)

    model = PatientSetNet(
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        latent_dim=args.latent_dim,
        regime=regime,
        dropout=args.dropout,
    ).to(device)

    tr_ds = PatientBagDataset(
        inner_train, args.image_size, True, args.bag_size, fold_seed
    )
    va_ds = PatientBagDataset(
        inner_val, args.image_size, False, 0, fold_seed + 10000
    )

    tr_loader = DataLoader(
        tr_ds,
        batch_size=args.patient_batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=patient_collate,
        pin_memory=(device.type == "cuda"),
    )
    va_loader = DataLoader(
        va_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=patient_collate,
        pin_memory=(device.type == "cuda"),
    )

    cw = patient_class_weights(inner_train, device)
    criterion = FocalLoss(
        cw,
        gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
    )

    # Start with encoder frozen, then unfreeze at freeze_epochs+1.
    set_encoder_trainable(model, False)
    opt = make_optimizer(
        model,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda" and args.amp_dtype == "fp16"),
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=max(1, args.max_select_epochs),
        eta_min=args.minimum_lr,
    )

    history = []
    best = None

    for epoch in range(1, args.max_select_epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model,
            tr_ds,
            tr_loader,
            opt,
            criterion,
            scaler,
            device,
            epoch,
            args.freeze_epochs,
            args.accumulation_steps,
            args.grad_clip,
            args.amp_dtype,
        )

        val_rows = infer_patients(
            model, va_ds, va_loader, device, args.amp_dtype
        )
        vm = metric_dict(val_rows)

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "inner_val_acc": vm["accuracy"],
            "inner_val_f1": vm["macro_f1"],
            "inner_val_errors": vm["errors"],
        })

        print(
            f"      epoch {epoch:02d}/{args.max_select_epochs} | "
            f"train loss={tr_loss:.4f} acc={tr_acc:6.2f}% | "
            f"INNER val Acc={vm['accuracy']:6.2f}% "
            f"F1={vm['macro_f1']:6.2f}% err={vm['errors']}"
        )

        # The frozen phase is a mandatory warm-up, not a candidate final model.
        # Epoch selection starts only after the encoder has been unfrozen, so a
        # run described as end-to-end cannot silently select a head-only model.
        if epoch > args.freeze_epochs:
            key = (vm["macro_f1"], vm["accuracy"], -epoch)
            if best is None or key > best["key"]:
                best = {
                    "key": key,
                    "epoch": epoch,
                    "metrics": vm,
                }

        scheduler.step()

    if best is None:
        raise RuntimeError(
            "No selectable epoch exists. Set --max_select_epochs greater than "
            "--freeze_epochs."
        )

    pd.DataFrame(history).to_csv(
        outdir / "inner_selection_history.csv",
        index=False,
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return int(best["epoch"]), best["metrics"]


def refit_outer_train(
    regime,
    outer_train,
    selected_epoch,
    args,
    device,
    fold_seed,
):
    """
    Stage B: reinitialize and train on ALL outer-train patients for exactly E* epochs.
    No validation/early stopping exists in this stage.
    """
    seed_all(fold_seed, args.deterministic)

    model = PatientSetNet(
        backbone=args.backbone,
        pretrained=not args.no_pretrained,
        latent_dim=args.latent_dim,
        regime=regime,
        dropout=args.dropout,
    ).to(device)

    tr_ds = PatientBagDataset(
        outer_train, args.image_size, True, args.bag_size, fold_seed
    )
    tr_loader = DataLoader(
        tr_ds,
        batch_size=args.patient_batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=patient_collate,
        pin_memory=(device.type == "cuda"),
    )

    cw = patient_class_weights(outer_train, device)
    criterion = FocalLoss(
        cw,
        gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
    )

    set_encoder_trainable(model, False)
    opt = make_optimizer(
        model,
        backbone_lr=args.backbone_lr,
        head_lr=args.head_lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda" and args.amp_dtype == "fp16"),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=max(1, selected_epoch),
        eta_min=args.minimum_lr,
    )

    for epoch in range(1, selected_epoch + 1):
        tr_loss, tr_acc = train_one_epoch(
            model,
            tr_ds,
            tr_loader,
            opt,
            criterion,
            scaler,
            device,
            epoch,
            args.freeze_epochs,
            args.accumulation_steps,
            args.grad_clip,
            args.amp_dtype,
        )
        scheduler.step()
        print(
            f"      REFIT epoch {epoch:02d}/{selected_epoch} | "
            f"loss={tr_loss:.4f} acc={tr_acc:6.2f}%"
        )

    return model


# =============================================================================
# Exact CVA on patient embeddings
# =============================================================================

def svd_row_basis(A: np.ndarray):
    A = np.asarray(A, dtype=np.float64)
    if A.size == 0 or np.allclose(A, 0):
        return np.zeros((0, A.shape[1]), dtype=np.float64)
    _, s, vt = np.linalg.svd(A, full_matrices=False)
    if s.size == 0 or s[0] <= 0:
        return np.zeros((0, A.shape[1]), dtype=np.float64)
    tol = max(A.shape) * np.finfo(np.float64).eps * s[0]
    r = int(np.sum(s > tol))
    return vt[:r].copy()


def project_null(X, basis):
    if basis.size == 0:
        return X.copy()
    return X - (X @ basis.T) @ basis


class ExactCVA:
    def __init__(self):
        self.scaler = StandardScaler()
        self.bases = {}
        self.common = {}

    def fit(self, X, y):
        Xs = self.scaler.fit_transform(np.asarray(X, dtype=np.float64))
        y = np.asarray(y, dtype=np.int64)

        for c in range(N_CLASSES):
            Xc = Xs[y == c]
            mu = Xc.mean(axis=0)
            B = svd_row_basis(Xc - mu)
            v = project_null(mu[None, :], B)[0]
            self.bases[c] = B
            self.common[c] = v
        return self

    def predict(self, X):
        Xs = self.scaler.transform(np.asarray(X, dtype=np.float64))
        D = np.zeros((len(Xs), N_CLASSES), dtype=np.float64)
        for c in range(N_CLASSES):
            xp = project_null(Xs, self.bases[c])
            D[:, c] = np.sum((xp - self.common[c]) ** 2, axis=1)
        return np.argmin(D, axis=1).astype(np.int64)


def embeddings_from_rows(rows):
    X = np.stack([r["embedding"] for r in rows]).astype(np.float64)
    y = np.asarray([r["true_id"] for r in rows], dtype=np.int64)
    pids = [r["patient_id"] for r in rows]
    return X, y, pids


def replace_predictions(rows, pred):
    out = []
    for r, p in zip(rows, pred):
        q = {k: v for k, v in r.items() if k != "embedding"}
        q["pred_id"] = int(p)
        q["pred_name"] = CLASSES[int(p)]
        q["correct"] = int(int(p) == q["true_id"])
        out.append(q)
    return out


# =============================================================================
# Paired statistics
# =============================================================================

def exact_mcnemar(correct_a, correct_b):
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    rescue = int(np.sum((~a) & b))
    lost = int(np.sum(a & (~b)))
    n = rescue + lost

    if n == 0:
        return rescue, lost, 1.0

    from math import comb
    k = min(rescue, lost)
    tail = sum(comb(n, i) for i in range(k + 1)) / (2 ** n)
    return rescue, lost, float(min(1.0, 2.0 * tail))


def paired_bootstrap(y, a, b, n_boot=20000, seed=2026):
    rng = np.random.default_rng(seed)
    y = np.asarray(y, dtype=np.int64)
    a = np.asarray(a, dtype=np.int64)
    b = np.asarray(b, dtype=np.int64)
    n = len(y)

    da = np.empty(n_boot)
    df = np.empty(n_boot)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy, aa, bb = y[idx], a[idx], b[idx]
        da[i] = accuracy_score(yy, bb) - accuracy_score(yy, aa)
        df[i] = (
            f1_score(yy, bb, average="macro", zero_division=0)
            - f1_score(yy, aa, average="macro", zero_division=0)
        )

    return {
        "delta_acc_pp": float(
            100 * (accuracy_score(y, b) - accuracy_score(y, a))
        ),
        "acc_ci95_pp": [
            float(100 * np.quantile(da, 0.025)),
            float(100 * np.quantile(da, 0.975)),
        ],
        "delta_f1_pp": float(
            100 * (
                f1_score(y, b, average="macro", zero_division=0)
                - f1_score(y, a, average="macro", zero_division=0)
            )
        ),
        "f1_ci95_pp": [
            float(100 * np.quantile(df, 0.025)),
            float(100 * np.quantile(df, 0.975)),
        ],
    }


# =============================================================================
# Main CV
# =============================================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_root", required=True)

    ap.add_argument(
        "--regimes",
        nargs="+",
        choices=["mean_set", "moment_set"],
        default=["mean_set", "moment_set"],
    )
    ap.add_argument("--backbone", default="swin_base_patch4_window7_224")
    ap.add_argument("--no_pretrained", action="store_true")

    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--train_folds", type=int, default=5)
    ap.add_argument("--outer_seed", type=int, default=2026)

    # Inner patient-disjoint checkpoint selection.
    ap.add_argument("--inner_n_splits", type=int, default=5)
    ap.add_argument("--inner_seed", type=int, default=31415)

    ap.add_argument("--max_select_epochs", type=int, default=30)
    ap.add_argument("--freeze_epochs", type=int, default=3)

    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--bag_size", type=int, default=8)
    ap.add_argument("--patient_batch_size", type=int, default=1)
    ap.add_argument("--accumulation_steps", type=int, default=4)
    ap.add_argument("--workers", type=int, default=0)

    ap.add_argument("--latent_dim", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.15)

    ap.add_argument("--backbone_lr", type=float, default=2e-5)
    ap.add_argument("--head_lr", type=float, default=2e-4)
    ap.add_argument("--minimum_lr", type=float, default=1e-7)
    ap.add_argument("--weight_decay", type=float, default=1e-4)

    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--label_smoothing", type=float, default=0.03)
    ap.add_argument("--grad_clip", type=float, default=3.0)

    ap.add_argument("--amp_dtype", choices=["fp16", "bf16"], default="bf16")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=20000)

    args = ap.parse_args()

    if args.train_folds > args.n_folds:
        raise ValueError("--train_folds cannot exceed --n_folds")
    if args.max_select_epochs <= args.freeze_epochs:
        raise ValueError(
            "--max_select_epochs must be greater than --freeze_epochs so that "
            "at least one end-to-end epoch can be evaluated."
        )

    seed_all(args.outer_seed, args.deterministic)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 132)
    print("LEAKAGE-FREE END-TO-END PATIENT-SET EXPERIMENT")
    print("=" * 132)
    print(f"device            : {device}")
    print(f"manifest          : {args.manifest}")
    print(f"out_root          : {args.out_root}")
    print(f"backbone          : {args.backbone}")
    print(f"regimes           : {args.regimes}")
    print(f"outer seed        : {args.outer_seed}")
    print(f"outer folds       : {args.n_folds}")
    print(f"inner selection   : 1/{args.inner_n_splits} patient-stratified holdout")
    print(f"max select epochs : {args.max_select_epochs}")
    print(f"train bag size    : {args.bag_size}; evaluation uses ALL slices")
    print("OUTER validation  : NEVER used for epoch/checkpoint selection")
    print("=" * 132)

    slice_rows, _ = load_manifest(Path(args.manifest))
    outer_splits = outer_patient_splits(
        slice_rows, args.n_folds, args.outer_seed
    )

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    save_json(
        {
            **vars(args),
            "manifest": str(args.manifest),
            "protocol": (
                "outer patient-disjoint CV; epoch selected on inner patient split; "
                "model reinitialized and refit on all outer-train patients"
            ),
        },
        out_root / "run_config.json",
    )

    all_results = {}

    for regime in args.regimes:
        print("\n" + "#" * 132)
        print(f"REGIME: {regime}")
        print("#" * 132)

        regime_dir = out_root / regime
        regime_dir.mkdir(parents=True, exist_ok=True)

        oof_neural = []
        oof_cva = []
        fold_summary = []

        for fold, (tr_idx, va_idx) in enumerate(outer_splits, start=1):
            if fold > args.train_folds:
                continue

            print("\n" + "=" * 132)
            print(f"{regime} | OUTER FOLD {fold}")
            print("=" * 132)

            outer_train = patient_subset_from_slice_indices(slice_rows, tr_idx)
            outer_val = patient_subset_from_slice_indices(slice_rows, va_idx)

            tr_ids = {p.patient_id for p in outer_train}
            va_ids = {p.patient_id for p in outer_val}
            if tr_ids & va_ids:
                raise RuntimeError("Outer patient leakage detected.")

            inner_train, inner_val = make_inner_selection_split(
                outer_train,
                n_splits=args.inner_n_splits,
                seed=args.inner_seed + fold,
            )
            if {p.patient_id for p in inner_train} & {p.patient_id for p in inner_val}:
                raise RuntimeError("Inner patient leakage detected.")

            print(
                f"outer train={len(outer_train)} patients | "
                f"outer val={len(outer_val)} patients"
            )
            print(
                f"inner train={len(inner_train)} | inner val={len(inner_val)}"
            )

            fold_dir = regime_dir / f"fold_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            # ---------------- Stage A: inner-only epoch selection ----------------
            select_seed = args.outer_seed * 100 + fold
            best_epoch, best_inner = select_epoch(
                regime,
                inner_train,
                inner_val,
                args,
                device,
                fold_seed=select_seed,
                outdir=fold_dir,
            )

            print(
                f"[SELECTED] fold={fold} epoch={best_epoch} from INNER val only | "
                f"Acc={best_inner['accuracy']:.3f}% "
                f"F1={best_inner['macro_f1']:.3f}%"
            )

            # ---------------- Stage B: clean outer-train refit ----------------
            refit_seed = args.outer_seed * 1000 + fold
            model = refit_outer_train(
                regime,
                outer_train,
                best_epoch,
                args,
                device,
                fold_seed=refit_seed,
            )

            # Save only the final clean refit model.
            ckpt = {
                "state_dict": model.state_dict(),
                "regime": regime,
                "backbone": args.backbone,
                "latent_dim": args.latent_dim,
                "selected_epoch": best_epoch,
                "outer_fold": fold,
                "outer_seed": args.outer_seed,
            }
            torch.save(ckpt, fold_dir / "final_refit.pt")

            # Train embeddings use ALL slices and no augmentation.
            tr_eval_ds = PatientBagDataset(
                outer_train, args.image_size, False, 0, refit_seed
            )
            va_eval_ds = PatientBagDataset(
                outer_val, args.image_size, False, 0, refit_seed + 1
            )
            tr_eval_loader = DataLoader(
                tr_eval_ds, batch_size=1, shuffle=False,
                num_workers=args.workers,
                collate_fn=patient_collate,
                pin_memory=(device.type == "cuda"),
            )
            va_eval_loader = DataLoader(
                va_eval_ds, batch_size=1, shuffle=False,
                num_workers=args.workers,
                collate_fn=patient_collate,
                pin_memory=(device.type == "cuda"),
            )

            train_rows = infer_patients(
                model, tr_eval_ds, tr_eval_loader, device, args.amp_dtype
            )
            val_rows = infer_patients(
                model, va_eval_ds, va_eval_loader, device, args.amp_dtype
            )

            # Neural patient classifier.
            neural_rows = [
                {k: v for k, v in r.items() if k != "embedding"}
                for r in val_rows
            ]
            for r in neural_rows:
                r["fold"] = fold
                r["regime"] = regime
                r["head"] = "neural"
            oof_neural.extend(neural_rows)

            # Exact CVA on leakage-free patient embeddings.
            Xtr, ytr, _ = embeddings_from_rows(train_rows)
            Xva, yva, _ = embeddings_from_rows(val_rows)
            cva = ExactCVA().fit(Xtr, ytr)
            pcva = cva.predict(Xva)

            cva_rows = replace_predictions(val_rows, pcva)
            for r in cva_rows:
                r["fold"] = fold
                r["regime"] = regime
                r["head"] = "exact_cva"
            oof_cva.extend(cva_rows)

            mn = metric_dict(neural_rows)
            mc = metric_dict(cva_rows)

            fold_summary.append({
                "fold": fold,
                "outer_train_patients": len(outer_train),
                "outer_val_patients": len(outer_val),
                "inner_train_patients": len(inner_train),
                "inner_val_patients": len(inner_val),
                "selected_epoch": best_epoch,
                "selected_inner_acc": best_inner["accuracy"],
                "selected_inner_f1": best_inner["macro_f1"],
                "outer_neural_acc": mn["accuracy"],
                "outer_neural_f1": mn["macro_f1"],
                "outer_neural_errors": mn["errors"],
                "outer_cva_acc": mc["accuracy"],
                "outer_cva_f1": mc["macro_f1"],
                "outer_cva_errors": mc["errors"],
            })

            print(
                f"[OUTER] neural Acc={mn['accuracy']:.3f}% "
                f"F1={mn['macro_f1']:.3f}% err={mn['errors']} | "
                f"CVA Acc={mc['accuracy']:.3f}% "
                f"F1={mc['macro_f1']:.3f}% err={mc['errors']}"
            )

            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not oof_neural:
            continue

        write_csv(pd.DataFrame(oof_neural), regime_dir / "patient_oof_neural.csv")
        write_csv(pd.DataFrame(oof_cva), regime_dir / "patient_oof_cva.csv")
        write_csv(pd.DataFrame(fold_summary), regime_dir / "fold_summary.csv")

        m_neural = metric_dict(oof_neural)
        m_cva = metric_dict(oof_cva)

        y = np.asarray([r["true_id"] for r in oof_neural], dtype=np.int64)
        pn = np.asarray([r["pred_id"] for r in oof_neural], dtype=np.int64)
        pc = np.asarray([r["pred_id"] for r in oof_cva], dtype=np.int64)

        rescue, lost, pmc = exact_mcnemar(pn == y, pc == y)
        boot = paired_bootstrap(
            y, pn, pc, n_boot=args.bootstrap, seed=args.outer_seed
        )

        regime_summary = {
            "regime": regime,
            "outer_seed": args.outer_seed,
            "completed_folds": args.train_folds,
            "neural_patient_oof": m_neural,
            "exact_cva_patient_oof": m_cva,
            "cva_vs_neural": {
                "rescue": rescue,
                "lost": lost,
                "mcnemar_p": pmc,
                "bootstrap": boot,
            },
            "folds": fold_summary,
        }
        save_json(regime_summary, regime_dir / "summary.json")
        all_results[regime] = regime_summary

        print("\n" + "=" * 132)
        print(f"FINAL {regime}")
        print("=" * 132)
        print(
            f"neural    | Acc={m_neural['accuracy']:.3f}% | "
            f"F1={m_neural['macro_f1']:.3f}% | "
            f"G-R={m_neural['glioma_recall']:.3f}% | err={m_neural['errors']}"
        )
        print(
            f"exact CVA | Acc={m_cva['accuracy']:.3f}% | "
            f"F1={m_cva['macro_f1']:.3f}% | "
            f"G-R={m_cva['glioma_recall']:.3f}% | err={m_cva['errors']}"
        )
        print(
            f"CVA vs neural | ΔAcc={boot['delta_acc_pp']:+.3f} pp "
            f"[{boot['acc_ci95_pp'][0]:+.3f}, {boot['acc_ci95_pp'][1]:+.3f}] | "
            f"R/L={rescue}/{lost} | McNemar p={pmc:.6f}"
        )

    # Regime-vs-regime paired comparison on final neural and CVA heads.
    if len(all_results) == 2 and args.train_folds == args.n_folds:
        comparison = {}
        for head_key, csv_name in [
            ("neural", "patient_oof_neural.csv"),
            ("exact_cva", "patient_oof_cva.csv"),
        ]:
            a_name, b_name = args.regimes[0], args.regimes[1]
            da = pd.read_csv(out_root / a_name / csv_name)
            db = pd.read_csv(out_root / b_name / csv_name)

            ma = da[["patient_id", "true_id", "pred_id"]].rename(
                columns={"pred_id": "pred_a"}
            )
            mb = db[["patient_id", "true_id", "pred_id"]].rename(
                columns={"pred_id": "pred_b"}
            )
            m = ma.merge(
                mb[["patient_id", "pred_b"]],
                on="patient_id",
                how="inner",
            )

            y = m["true_id"].to_numpy(int)
            pa = m["pred_a"].to_numpy(int)
            pb = m["pred_b"].to_numpy(int)

            rescue, lost, pval = exact_mcnemar(pa == y, pb == y)
            boot = paired_bootstrap(
                y, pa, pb, n_boot=args.bootstrap, seed=args.outer_seed + 77
            )

            comparison[head_key] = {
                "a": a_name,
                "b": b_name,
                "n": int(len(y)),
                "rescue_b_over_a": rescue,
                "lost_b_vs_a": lost,
                "mcnemar_p": pval,
                "bootstrap": boot,
            }

            print("\n" + "=" * 132)
            print(f"PAIRED {b_name} vs {a_name} | head={head_key}")
            print("=" * 132)
            print(
                f"ΔAcc={boot['delta_acc_pp']:+.3f} pp "
                f"[{boot['acc_ci95_pp'][0]:+.3f}, {boot['acc_ci95_pp'][1]:+.3f}] | "
                f"ΔF1={boot['delta_f1_pp']:+.3f} pp "
                f"[{boot['f1_ci95_pp'][0]:+.3f}, {boot['f1_ci95_pp'][1]:+.3f}] | "
                f"R/L={rescue}/{lost} | McNemar p={pval:.6f}"
            )

        save_json(comparison, out_root / "paired_regime_comparison.json")


if __name__ == "__main__":
    main()
