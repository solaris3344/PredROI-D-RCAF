#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
bdneuro_external_validation_frozen_ensemble.py

Independent-source EXTERNAL IMAGE-LEVEL validation of the frozen Cheng models
on BDNeuro-MRI v8 test images.

Scientific constraints:
  * NO training / fine-tuning / threshold tuning on BDNeuro-MRI.
  * Use ONLY the published BDNeuro-MRI TEST split.
  * Use ONLY tumor classes shared with Cheng:
        glioma, meningioma, pituitary
    "no_tumor" is deliberately excluded because Cheng is a 3-class model.
  * Baseline: frozen five-fold Cheng full-image Swin ensemble.
  * PredROI: frozen five-fold Cheng localizer + PredROI-Swin ensemble.
  * PredROI thresholds and crop margin remain frozen:
        threshold=0.50, fallback=0.30, margin=1.10
  * Because BDNeuro-MRI does not retain patient IDs, the endpoint is
        IMAGE-LEVEL external validation.
    Do NOT relabel this as independent patient-level validation.

Expected BDNeuro-MRI v8 TEST tumor counts:
  glioma      328
  meningioma  221
  pituitary   222
  total       771

Expected Cheng checkpoint layout:
  baseline_root/
    fold_1/final_refit.pt
    ...
    fold_5/final_refit.pt

  predroi_root/
    fold_1/final_classifier.pt
    fold_1/final_localizer.pt
    ...
    fold_5/final_classifier.pt
    fold_5/final_localizer.pt

If the PredROI run was executed without --save_models, those model files will
not exist. This script will stop before evaluation and list the missing files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
)
from torch.utils.data import Dataset, DataLoader

import cheng_patient_set_leakage_free_v2 as common
import cheng_swin_leakage_free_baseline as base
import cheng_predroi_swin_screen_v1_audited as pred


CLASSES = ["glioma", "meningioma", "pituitary"]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
EXPECTED_V8_TEST_COUNTS = {
    "glioma": 328,
    "meningioma": 221,
    "pituitary": 222,
}
EXPECTED_V8_TUMOR_TOTAL = 771


def save_json(obj, path: Path):
    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def sha256_file(path: Path, chunk_size=1024 * 1024):
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def resolve_test_root(dataset_root: Path) -> Path:
    """
    Accept either:
      .../Brain_Tumor_MRI_Dataset_Final
    or
      .../Brain_Tumor_MRI_Dataset_Final/test
    """
    dataset_root = dataset_root.resolve()

    if dataset_root.name.lower() in {"test", "testing"}:
        return dataset_root

    for name in ["test", "Test", "testing", "Testing"]:
        p = dataset_root / name
        if p.is_dir():
            return p

    raise FileNotFoundError(
        f"Could not find a test/Testing folder under: {dataset_root}"
    )


def _normalize_class_name(name: str):
    s = name.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "glioma": "glioma",
        "glioma_tumor": "glioma",
        "gliomatumor": "glioma",
        "meningioma": "meningioma",
        "meningioma_tumor": "meningioma",
        "meningiomatumor": "meningioma",
        "pituitary": "pituitary",
        "pituitary_tumor": "pituitary",
        "pituitarytumor": "pituitary",
        "pituitary_tumour": "pituitary",
        "no_tumor": "no_tumor",
        "notumor": "no_tumor",
        "no_tumour": "no_tumor",
        "normal": "no_tumor",
    }
    return aliases.get(s, None)


def discover_external_test(
    test_root: Path,
    strict_v8_counts: bool,
):
    image_exts = {
        ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"
    }

    class_dirs = {}
    ignored_dirs = []

    for d in sorted(test_root.iterdir()):
        if not d.is_dir():
            continue
        cls = _normalize_class_name(d.name)
        if cls in CLASS_TO_ID:
            if cls in class_dirs:
                raise RuntimeError(
                    f"Multiple folders map to class {cls}: "
                    f"{class_dirs[cls]} and {d}"
                )
            class_dirs[cls] = d
        else:
            ignored_dirs.append(d)

    missing = [c for c in CLASSES if c not in class_dirs]
    if missing:
        raise RuntimeError(
            f"Missing required tumor class folders in {test_root}: {missing}"
        )

    rows = []
    for cls in CLASSES:
        d = class_dirs[cls]
        files = sorted(
            p for p in d.rglob("*")
            if p.is_file() and p.suffix.lower() in image_exts
        )

        for p in files:
            img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"Unreadable image: {p}")

            h, w = img.shape[:2]
            rows.append({
                "path": str(p.resolve()),
                "class_name": cls,
                "label": CLASS_TO_ID[cls],
                "height": int(h),
                "width": int(w),
                "sha256": sha256_file(p),
            })

    counts = Counter(r["class_name"] for r in rows)

    if strict_v8_counts:
        got = {c: int(counts.get(c, 0)) for c in CLASSES}
        if got != EXPECTED_V8_TEST_COUNTS:
            raise RuntimeError(
                "BDNeuro-MRI v8 test tumor counts do not match the frozen "
                f"expected counts.\nExpected: {EXPECTED_V8_TEST_COUNTS}\n"
                f"Found   : {got}\n"
                "Verify that --dataset_root points to the v8 TEST split."
            )

        if len(rows) != EXPECTED_V8_TUMOR_TOTAL:
            raise RuntimeError(
                f"Expected {EXPECTED_V8_TUMOR_TOTAL} tumor test images, "
                f"found {len(rows)}."
            )

    return rows, class_dirs, ignored_dirs


def audit_checkpoints(
    baseline_root: Path,
    predroi_root: Path,
    n_folds: int,
):
    baseline_ckpts = []
    pred_cls_ckpts = []
    pred_loc_ckpts = []
    missing = []

    for fold in range(1, n_folds + 1):
        b = baseline_root / f"fold_{fold}" / "final_refit.pt"
        c = predroi_root / f"fold_{fold}" / "final_classifier.pt"
        l = predroi_root / f"fold_{fold}" / "final_localizer.pt"

        if not b.exists():
            missing.append(str(b))
        if not c.exists():
            missing.append(str(c))
        if not l.exists():
            missing.append(str(l))

        baseline_ckpts.append(b)
        pred_cls_ckpts.append(c)
        pred_loc_ckpts.append(l)

    if missing:
        txt = "\n".join(f"  - {x}" for x in missing)
        raise FileNotFoundError(
            "Frozen checkpoint audit FAILED. Missing files:\n"
            + txt
            + "\n\nIf PredROI v1.1 was previously run without --save_models, "
              "the locked predictions remain valid, but external validation "
              "requires recreating/saving the frozen fold models before "
              "BDNeuro-MRI is evaluated."
        )

    return baseline_ckpts, pred_cls_ckpts, pred_loc_ckpts


class ExternalImageDataset(Dataset):
    def __init__(self, rows, image_size):
        self.rows = list(rows)
        self.image_size = int(image_size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        rng = np.random.default_rng(123456 + idx)
        x = common.load_gray_preprocessed(
            r["path"],
            self.image_size,
            False,
            rng,
        )
        return {
            "image": x,
            "label": torch.tensor(int(r["label"]), dtype=torch.long),
            "path": str(r["path"]),
        }


class ExternalSegDataset(Dataset):
    """
    Image-only localizer input.
    No patient ID, no MAT file, no ground-truth mask.
    """
    def __init__(self, rows, seg_size):
        self.rows = list(rows)
        self.seg_size = int(seg_size)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img = cv2.imread(r["path"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Could not read {r['path']}")

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
            "label": torch.tensor(int(r["label"]), dtype=torch.long),
        }


class ExternalPredROIDataset(Dataset):
    def __init__(
        self,
        rows,
        box_lookup,
        image_size,
        margin_factor,
    ):
        self.rows = list(rows)
        self.box_lookup = dict(box_lookup)
        self.image_size = int(image_size)
        self.margin_factor = float(margin_factor)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        path = str(r["path"])
        if path not in self.box_lookup:
            raise RuntimeError(f"Missing external predicted box for {path}")

        rng = np.random.default_rng(987654 + idx)
        x = pred.load_predroi_image(
            path,
            self.box_lookup[path],
            self.image_size,
            False,
            rng,
            self.margin_factor,
        )

        return {
            "image": x,
            "path": path,
            "label": torch.tensor(int(r["label"]), dtype=torch.long),
        }


def load_state(path: Path, device):
    obj = torch.load(
        path,
        map_location=device,
        weights_only=False,
    )
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"], obj
    return obj, {}


@torch.no_grad()
def infer_baseline_fold(
    ckpt: Path,
    rows,
    args,
    device,
):
    model = base.SwinSliceClassifier(
        args.backbone,
        pretrained=False,
        dropout=args.dropout,
    ).to(device)

    state, meta = load_state(ckpt, device)
    model.load_state_dict(state, strict=True)
    model.eval()

    ds = ExternalImageDataset(rows, args.image_size)
    loader = DataLoader(
        ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    probs = np.zeros((len(rows), 3), dtype=np.float64)
    seen = 0

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)

        with common.autocast_ctx(device, args.amp_dtype):
            p = F.softmax(model(x), dim=-1).float().cpu().numpy()

        n = p.shape[0]
        probs[seen:seen+n] = p
        seen += n

    if seen != len(rows):
        raise RuntimeError("Baseline external inference row-count mismatch.")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return probs, meta


@torch.no_grad()
def predict_external_boxes(
    localizer_ckpt: Path,
    rows,
    args,
    device,
):
    model = pred.ResNet18UNet(pretrained=False).to(device)

    state, meta = load_state(localizer_ckpt, device)
    model.load_state_dict(state, strict=True)
    model.eval()

    ds = ExternalSegDataset(rows, args.seg_size)
    loader = DataLoader(
        ds,
        batch_size=args.seg_eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    out = []
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)

        with common.autocast_ctx(device, args.amp_dtype):
            prob = torch.sigmoid(
                model(x).float()
            ).cpu().numpy()[:, 0]

        for i in range(len(prob)):
            path = str(batch["path"][i])

            original = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if original is None:
                raise RuntimeError(f"Could not reread {path}")

            rec = pred.prob_to_bbox(
                prob[i],
                original.shape,
                args.box_threshold,
                args.box_fallback_threshold,
            )
            rec.update({
                "path": path,
                "label": int(batch["label"][i]),
            })
            out.append(rec)

    if len(out) != len(rows):
        raise RuntimeError("External localizer row-count mismatch.")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return out, meta


@torch.no_grad()
def infer_predroi_classifier_fold(
    classifier_ckpt: Path,
    rows,
    boxes,
    args,
    device,
):
    model = pred.SwinClassifier(
        args.backbone,
        pretrained=False,
        dropout=args.dropout,
    ).to(device)

    state, meta = load_state(classifier_ckpt, device)
    model.load_state_dict(state, strict=True)
    model.eval()

    box_lookup = {str(r["path"]): r for r in boxes}
    if set(box_lookup) != {str(r["path"]) for r in rows}:
        raise RuntimeError("External PredROI box coverage mismatch.")

    ds = ExternalPredROIDataset(
        rows,
        box_lookup,
        args.image_size,
        args.crop_margin_factor,
    )
    loader = DataLoader(
        ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    probs = np.zeros((len(rows), 3), dtype=np.float64)
    seen = 0

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)

        with common.autocast_ctx(device, args.amp_dtype):
            p = F.softmax(model(x), dim=-1).float().cpu().numpy()

        n = p.shape[0]
        probs[seen:seen+n] = p
        seen += n

    if seen != len(rows):
        raise RuntimeError("PredROI external classifier row-count mismatch.")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return probs, meta


def metrics_from_probs(y, probs):
    pred_id = probs.argmax(axis=1)

    rec = recall_score(
        y,
        pred_id,
        labels=[0, 1, 2],
        average=None,
        zero_division=0,
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y,
        pred_id,
        labels=[0, 1, 2],
        zero_division=0,
    )

    return {
        "n_images": int(len(y)),
        "accuracy": float(100.0 * accuracy_score(y, pred_id)),
        "macro_f1": float(
            100.0
            * f1_score(
                y,
                pred_id,
                labels=[0, 1, 2],
                average="macro",
                zero_division=0,
            )
        ),
        "glioma_recall": float(100.0 * rec[0]),
        "meningioma_recall": float(100.0 * rec[1]),
        "pituitary_recall": float(100.0 * rec[2]),
        "errors": int(np.sum(pred_id != y)),
        "confusion_matrix": confusion_matrix(
            y,
            pred_id,
            labels=[0, 1, 2],
        ).tolist(),
        "per_class": {
            CLASSES[i]: {
                "precision": float(100.0 * precision[i]),
                "recall": float(100.0 * recall[i]),
                "f1": float(100.0 * f1[i]),
                "support": int(support[i]),
            }
            for i in range(3)
        },
    }


def descriptive_paired_bootstrap(
    y,
    p_base,
    p_roi,
    n_boot,
    seed,
):
    """
    IMAGE-level paired bootstrap.
    Because patient IDs are unavailable, this is reported as descriptive
    uncertainty only; it is NOT a patient-clustered inferential CI.
    """
    rng = np.random.default_rng(seed)
    n = len(y)

    base_acc = 100.0 * np.mean(p_base == y)
    roi_acc = 100.0 * np.mean(p_roi == y)
    point = roi_acc - base_acc

    diffs = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[b] = (
            100.0 * np.mean(p_roi[idx] == y[idx])
            - 100.0 * np.mean(p_base[idx] == y[idx])
        )

    lo, hi = np.percentile(diffs, [2.5, 97.5])

    return {
        "delta_acc_pp": float(point),
        "descriptive_image_bootstrap_95ci": [
            float(lo),
            float(hi),
        ],
        "n_boot": int(n_boot),
        "warning": (
            "Image-level bootstrap only; patient IDs are unavailable, "
            "so within-patient dependence cannot be modeled."
        ),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--baseline_root", required=True)
    ap.add_argument("--predroi_root", required=True)
    ap.add_argument("--out_root", required=True)

    ap.add_argument("--n_folds", type=int, default=5)

    # Frozen Cheng model settings.
    ap.add_argument(
        "--backbone",
        default="swin_base_patch4_window7_224",
    )
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--dropout", type=float, default=0.15)

    # Frozen PredROI localizer/crop settings.
    ap.add_argument("--seg_size", type=int, default=256)
    ap.add_argument("--box_threshold", type=float, default=0.50)
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

    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--seg_eval_batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument(
        "--amp_dtype",
        choices=["fp16", "bf16"],
        default="bf16",
    )

    ap.add_argument(
        "--allow_nonstandard_counts",
        action="store_true",
        help=(
            "Disable the frozen BDNeuro-MRI v8 test-count check. "
            "Not recommended for the manuscript run."
        ),
    )
    ap.add_argument(
        "--bootstrap_iterations",
        type=int,
        default=10000,
    )
    ap.add_argument(
        "--bootstrap_seed",
        type=int,
        default=20261009,
    )

    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(args.dataset_root)
    baseline_root = Path(args.baseline_root)
    predroi_root = Path(args.predroi_root)

    test_root = resolve_test_root(dataset_root)

    rows, class_dirs, ignored_dirs = discover_external_test(
        test_root,
        strict_v8_counts=not args.allow_nonstandard_counts,
    )

    # Save exact external test provenance BEFORE model inference.
    manifest_df = pd.DataFrame(rows)
    manifest_df.to_csv(
        out_root / "bdneuro_v8_external_test_manifest.csv",
        index=False,
    )

    counts = (
        manifest_df["class_name"]
        .value_counts()
        .reindex(CLASSES, fill_value=0)
        .to_dict()
    )

    print("=" * 150)
    print("BDNeuro-MRI v8 — FROZEN EXTERNAL IMAGE-LEVEL VALIDATION")
    print("=" * 150)
    print(f"test root               : {test_root}")
    print(f"tumor images            : {len(rows)}")
    print(f"class counts            : {counts}")
    print("excluded class          : no_tumor")
    print("training/fine-tuning    : NONE")
    print("external patient IDs    : unavailable")
    print("primary endpoint here   : IMAGE-LEVEL")
    print(
        "PredROI crop rule       : "
        f"thr={args.box_threshold:.2f}, "
        f"fallback={args.box_fallback_threshold:.2f}, "
        f"margin={args.crop_margin_factor:.2f}"
    )
    print("=" * 150)

    baseline_ckpts, pred_cls_ckpts, pred_loc_ckpts = audit_checkpoints(
        baseline_root,
        predroi_root,
        args.n_folds,
    )
    print("[AUDIT] all frozen five-fold checkpoints: PASS")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"device                  : {device}")

    y = np.asarray([int(r["label"]) for r in rows], dtype=int)

    # ------------------------------------------------------------------
    # Baseline 5-fold frozen ensemble.
    # ------------------------------------------------------------------
    base_sum = np.zeros((len(rows), 3), dtype=np.float64)
    base_fold_meta = []

    for fold, ckpt in enumerate(baseline_ckpts, start=1):
        print(f"[BASELINE] fold {fold}/{args.n_folds}: {ckpt}")
        p, meta = infer_baseline_fold(
            ckpt,
            rows,
            args,
            device,
        )
        base_sum += p
        base_fold_meta.append({
            "fold": fold,
            "checkpoint": str(ckpt),
            "outer_seed": meta.get("outer_seed"),
            "outer_fold": meta.get("outer_fold"),
            "selected_epoch": meta.get("selected_epoch"),
        })

    base_prob = base_sum / float(args.n_folds)

    # ------------------------------------------------------------------
    # PredROI 5-fold frozen ensemble.
    # Each fold uses its own frozen localizer -> own box -> own classifier.
    # ------------------------------------------------------------------
    roi_sum = np.zeros((len(rows), 3), dtype=np.float64)
    pred_fold_meta = []
    bbox_records = []

    for fold, (loc_ckpt, cls_ckpt) in enumerate(
        zip(pred_loc_ckpts, pred_cls_ckpts),
        start=1,
    ):
        print(f"[PREDROI] fold {fold}/{args.n_folds}: localizer")
        boxes, loc_meta = predict_external_boxes(
            loc_ckpt,
            rows,
            args,
            device,
        )

        found_rate = float(
            100.0 * np.mean([int(r["found"]) for r in boxes])
        )
        fallback_rate = float(
            100.0
            * np.mean(
                [
                    int(
                        bool(r["found"])
                        and float(r.get("threshold", args.box_threshold))
                        < args.box_threshold
                    )
                    for r in boxes
                ]
            )
        )
        full_fallback_rate = float(
            100.0 * np.mean([int(not bool(r["found"])) for r in boxes])
        )

        for rec in boxes:
            rr = dict(rec)
            rr["fold"] = fold
            bbox_records.append(rr)

        print(
            f"[PREDROI] fold {fold}: bbox-found={found_rate:.2f}% | "
            f"low-threshold fallback={fallback_rate:.2f}% | "
            f"full-image fallback={full_fallback_rate:.2f}%"
        )

        print(f"[PREDROI] fold {fold}/{args.n_folds}: classifier")
        p, cls_meta = infer_predroi_classifier_fold(
            cls_ckpt,
            rows,
            boxes,
            args,
            device,
        )
        roi_sum += p

        pred_fold_meta.append({
            "fold": fold,
            "localizer_checkpoint": str(loc_ckpt),
            "classifier_checkpoint": str(cls_ckpt),
            "localizer_outer_seed": loc_meta.get("outer_seed"),
            "localizer_outer_fold": loc_meta.get("outer_fold"),
            "classifier_outer_seed": cls_meta.get("outer_seed"),
            "classifier_outer_fold": cls_meta.get("outer_fold"),
            "classifier_selected_epoch": cls_meta.get("selected_epoch"),
            "bbox_found_rate_pct": found_rate,
            "low_threshold_fallback_rate_pct": fallback_rate,
            "full_image_fallback_rate_pct": full_fallback_rate,
        })

    roi_prob = roi_sum / float(args.n_folds)

    # ------------------------------------------------------------------
    # Metrics and paired image-level transitions.
    # ------------------------------------------------------------------
    base_m = metrics_from_probs(y, base_prob)
    roi_m = metrics_from_probs(y, roi_prob)

    base_pred = base_prob.argmax(axis=1)
    roi_pred = roi_prob.argmax(axis=1)

    rescue = int(np.sum((base_pred != y) & (roi_pred == y)))
    lost = int(np.sum((base_pred == y) & (roi_pred != y)))
    shared_wrong = int(np.sum((base_pred != y) & (roi_pred != y)))
    shared_correct = int(np.sum((base_pred == y) & (roi_pred == y)))

    paired = descriptive_paired_bootstrap(
        y,
        base_pred,
        roi_pred,
        args.bootstrap_iterations,
        args.bootstrap_seed,
    )
    paired.update({
        "rescues": rescue,
        "losses": lost,
        "shared_wrong": shared_wrong,
        "shared_correct": shared_correct,
    })

    pred_df = manifest_df.copy()
    pred_df["baseline_pred_id"] = base_pred
    pred_df["baseline_pred_class"] = [CLASSES[i] for i in base_pred]
    pred_df["predroi_pred_id"] = roi_pred
    pred_df["predroi_pred_class"] = [CLASSES[i] for i in roi_pred]
    pred_df["baseline_correct"] = (base_pred == y).astype(int)
    pred_df["predroi_correct"] = (roi_pred == y).astype(int)

    for c, cname in enumerate(CLASSES):
        pred_df[f"baseline_prob_{cname}"] = base_prob[:, c]
        pred_df[f"predroi_prob_{cname}"] = roi_prob[:, c]

    pred_df.to_csv(
        out_root / "external_predictions.csv",
        index=False,
    )
    pd.DataFrame(bbox_records).to_csv(
        out_root / "external_predroi_boxes_all_folds.csv",
        index=False,
    )

    summary = {
        "dataset": {
            "name": "BDNeuro-MRI v8",
            "evaluation_level": "image",
            "patient_ids_available": False,
            "test_root": str(test_root),
            "tumor_classes_used": CLASSES,
            "no_tumor_excluded": True,
            "n_images": int(len(rows)),
            "class_counts": {k: int(v) for k, v in counts.items()},
            "strict_v8_count_check": not args.allow_nonstandard_counts,
        },
        "frozen_protocol": {
            "no_external_training_or_finetuning": True,
            "baseline_ensemble_folds": args.n_folds,
            "predroi_ensemble_folds": args.n_folds,
            "box_threshold": args.box_threshold,
            "box_fallback_threshold": args.box_fallback_threshold,
            "crop_margin_factor": args.crop_margin_factor,
        },
        "baseline_full_image_swin": base_m,
        "predroi_swin": roi_m,
        "paired_image_level": paired,
        "baseline_fold_metadata": base_fold_meta,
        "predroi_fold_metadata": pred_fold_meta,
        "interpretation_warning": (
            "This is independent-source external IMAGE-level validation. "
            "BDNeuro-MRI filenames do not retain patient identifiers; therefore "
            "these results must not be described as independent patient-level "
            "external validation."
        ),
    }

    save_json(
        summary,
        out_root / "external_summary.json",
    )

    flat_rows = [
        {
            "method": "Frozen Cheng 5-fold full-image Swin ensemble",
            "accuracy": base_m["accuracy"],
            "macro_f1": base_m["macro_f1"],
            "glioma_recall": base_m["glioma_recall"],
            "meningioma_recall": base_m["meningioma_recall"],
            "pituitary_recall": base_m["pituitary_recall"],
            "errors": base_m["errors"],
        },
        {
            "method": "Frozen Cheng 5-fold PredROI-Swin ensemble",
            "accuracy": roi_m["accuracy"],
            "macro_f1": roi_m["macro_f1"],
            "glioma_recall": roi_m["glioma_recall"],
            "meningioma_recall": roi_m["meningioma_recall"],
            "pituitary_recall": roi_m["pituitary_recall"],
            "errors": roi_m["errors"],
        },
    ]
    pd.DataFrame(flat_rows).to_csv(
        out_root / "external_method_summary.csv",
        index=False,
    )

    print("\n" + "=" * 150)
    print("FINAL BDNeuro-MRI v8 EXTERNAL IMAGE-LEVEL RESULTS")
    print("=" * 150)
    print(
        "Full-image Swin ensemble | "
        f"Acc={base_m['accuracy']:.3f}% | "
        f"F1={base_m['macro_f1']:.3f}% | "
        f"G-R={base_m['glioma_recall']:.3f}% | "
        f"M-R={base_m['meningioma_recall']:.3f}% | "
        f"P-R={base_m['pituitary_recall']:.3f}% | "
        f"err={base_m['errors']}"
    )
    print(
        "PredROI-Swin ensemble    | "
        f"Acc={roi_m['accuracy']:.3f}% | "
        f"F1={roi_m['macro_f1']:.3f}% | "
        f"G-R={roi_m['glioma_recall']:.3f}% | "
        f"M-R={roi_m['meningioma_recall']:.3f}% | "
        f"P-R={roi_m['pituitary_recall']:.3f}% | "
        f"err={roi_m['errors']}"
    )
    print(
        f"Delta accuracy            | {paired['delta_acc_pp']:+.3f} pp "
        f"| descriptive image-bootstrap 95% CI "
        f"[{paired['descriptive_image_bootstrap_95ci'][0]:+.3f}, "
        f"{paired['descriptive_image_bootstrap_95ci'][1]:+.3f}]"
    )
    print(
        f"Rescue / lost             | {rescue} / {lost} | "
        f"shared wrong={shared_wrong}"
    )
    mean_bbox = np.mean(
        [x["bbox_found_rate_pct"] for x in pred_fold_meta]
    )
    print(
        f"Mean external bbox-found  | {mean_bbox:.2f}% across frozen localizers"
    )
    print(
        "INTERPRETATION            | independent-source EXTERNAL IMAGE-LEVEL validation"
    )
    print(
        "PATIENT-LEVEL CLAIM       | NOT ALLOWED: BDNeuro-MRI patient IDs are unavailable"
    )
    print("=" * 150)


if __name__ == "__main__":
    main()
