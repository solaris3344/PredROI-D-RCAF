#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bdneuro_external_validation_fixed05_ensemble.py

Frozen independent-source EXTERNAL IMAGE-LEVEL validation on BDNeuro-MRI v8.
Compares three frozen Cheng ensembles:
  1) full-image Swin
  2) PredROI-Swin
  3) Fixed-0.5 shared-Swin dual-view (global + PredROI)

Scientific constraints:
  * NO training / fine-tuning / threshold tuning on BDNeuro-MRI.
  * Use only glioma / meningioma / pituitary (no_tumor excluded).
  * Fixed-0.5 classifier checkpoints come from the locked full-5 run.
  * Fixed-0.5 localizers come from the D-RCAF locked run because the Fixed-0.5
    ablation reused those exact predicted-box/localizer conditions and did not
    save a new localizer.
  * BDNeuro-MRI has no patient IDs here; all endpoints are IMAGE-LEVEL.
  * Any bootstrap CI is descriptive image-level uncertainty only.

Place next to:
  bdneuro_external_validation_frozen_ensemble.py
  cheng_fixed05_predroi_swin_full5.py
  cheng_patient_set_leakage_free_v2.py
  cheng_swin_leakage_free_baseline.py
  cheng_predroi_swin_screen_v1_audited.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset, DataLoader

import bdneuro_external_validation_frozen_ensemble as ext
import cheng_fixed05_predroi_swin_full5 as fixed
import cheng_patient_set_leakage_free_v2 as common

CLASSES = ext.CLASSES


def save_json(obj, path: Path):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_state(path: Path, device):
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"], obj
    return obj, {}



def _checkpoint_meta_cpu(path: Path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj if isinstance(obj, dict) else {}


def _root_has_folds(root: Path, filenames, n_folds: int):
    for fold in range(1, n_folds + 1):
        for fn in filenames:
            if not (root / f"fold_{fold}" / fn).exists():
                return False
    return True


def _discover_unique_root(
    brain_root: Path,
    search_filename: str,
    required_filenames,
    n_folds: int,
    target_seed=None,
    label="checkpoint root",
):
    candidates = []
    seen = set()
    for hit in brain_root.rglob(search_filename):
        if hit.parent.name.lower() != "fold_1":
            continue
        root = hit.parent.parent.resolve()
        key = str(root).lower()
        if key in seen:
            continue
        seen.add(key)
        if not _root_has_folds(root, required_filenames, n_folds):
            continue
        if target_seed is not None:
            meta = _checkpoint_meta_cpu(root / "fold_1" / search_filename)
            seed = meta.get("outer_seed")
            if seed is None or int(seed) != int(target_seed):
                continue
        candidates.append(root)

    if len(candidates) == 1:
        print(f"[AUTO] {label:<22}: {candidates[0]}")
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"Could not auto-find {label} under {brain_root}. "
            f"Pass the corresponding --*_root argument explicitly."
        )
    raise RuntimeError(
        f"Multiple {label} candidates found under {brain_root}; pass explicitly:\n"
        + "\n".join(f"  - {x}" for x in candidates[:20])
    )


def auto_resolve_model_roots(args):
    brain_root = Path(args.brain_root or Path(__file__).resolve().parent).resolve()
    if not brain_root.exists():
        raise FileNotFoundError(f"brain_root not found: {brain_root}")
    print(f"[AUTO] brain_root            : {brain_root}")

    # Find Fixed first; its metadata defines the locked seed to match everywhere.
    if args.fixed05_root:
        fixed_root = Path(args.fixed05_root).resolve()
    else:
        fixed_root = _discover_unique_root(
            brain_root,
            "final_fixed05_classifier.pt",
            ["final_fixed05_classifier.pt"],
            args.n_folds,
            label="Fixed-0.5 root",
        )
    fixed_meta = _checkpoint_meta_cpu(fixed_root / "fold_1" / "final_fixed05_classifier.pt")
    target_seed = fixed_meta.get("outer_seed")
    if target_seed is None:
        raise RuntimeError("Fixed-0.5 checkpoint has no outer_seed metadata.")
    target_seed = int(target_seed)
    print(f"[AUTO] locked outer seed     : {target_seed}")

    if args.fixed_localizer_root:
        fixed_loc_root = Path(args.fixed_localizer_root).resolve()
    else:
        # D-RCAF root is identifiable by final_drcaf_classifier.pt + final_localizer.pt.
        fixed_loc_root = _discover_unique_root(
            brain_root,
            "final_drcaf_classifier.pt",
            ["final_drcaf_classifier.pt", "final_localizer.pt"],
            args.n_folds,
            target_seed=target_seed,
            label="Fixed localizer root",
        )

    if args.baseline_root:
        baseline_root = Path(args.baseline_root).resolve()
    else:
        baseline_root = _discover_unique_root(
            brain_root,
            "final_refit.pt",
            ["final_refit.pt"],
            args.n_folds,
            target_seed=target_seed,
            label="baseline root",
        )

    if args.predroi_root:
        predroi_root = Path(args.predroi_root).resolve()
    else:
        predroi_root = _discover_unique_root(
            brain_root,
            "final_classifier.pt",
            ["final_classifier.pt", "final_localizer.pt"],
            args.n_folds,
            target_seed=target_seed,
            label="PredROI root",
        )

    return brain_root, baseline_root, predroi_root, fixed_root, fixed_loc_root, target_seed


def audit_all_checkpoints(
    baseline_root: Path,
    predroi_root: Path,
    fixed05_root: Path,
    fixed_localizer_root: Path,
    n_folds: int,
):
    base_ckpts, pred_cls_ckpts, pred_loc_ckpts = ext.audit_checkpoints(
        baseline_root, predroi_root, n_folds
    )

    fixed_cls_ckpts = []
    fixed_loc_ckpts = []
    missing = []
    for fold in range(1, n_folds + 1):
        c = fixed05_root / f"fold_{fold}" / "final_fixed05_classifier.pt"
        l = fixed_localizer_root / f"fold_{fold}" / "final_localizer.pt"
        if not c.exists():
            missing.append(str(c))
        if not l.exists():
            missing.append(str(l))
        fixed_cls_ckpts.append(c)
        fixed_loc_ckpts.append(l)

    if missing:
        raise FileNotFoundError(
            "Fixed-0.5 external checkpoint audit FAILED. Missing files:\n"
            + "\n".join(f"  - {x}" for x in missing)
        )

    # Metadata compatibility audit without loading tensors to GPU.
    for fold, (c, l) in enumerate(zip(fixed_cls_ckpts, fixed_loc_ckpts), start=1):
        cobj = torch.load(c, map_location="cpu", weights_only=False)
        lobj = torch.load(l, map_location="cpu", weights_only=False)
        cseed = cobj.get("outer_seed") if isinstance(cobj, dict) else None
        lseed = lobj.get("outer_seed") if isinstance(lobj, dict) else None
        cfold = cobj.get("outer_fold") if isinstance(cobj, dict) else None
        lfold = lobj.get("outer_fold") if isinstance(lobj, dict) else None
        method = cobj.get("method", "") if isinstance(cobj, dict) else ""

        if "Fixed-0.5" not in str(method):
            raise RuntimeError(
                f"Fold {fold}: unexpected Fixed classifier method metadata: {method!r}"
            )
        if cfold is not None and int(cfold) != fold:
            raise RuntimeError(f"Fold {fold}: classifier outer_fold metadata={cfold}")
        if lfold is not None and int(lfold) != fold:
            raise RuntimeError(f"Fold {fold}: localizer outer_fold metadata={lfold}")
        if cseed is not None and lseed is not None and int(cseed) != int(lseed):
            raise RuntimeError(
                f"Fold {fold}: classifier/localizer outer_seed mismatch: {cseed} vs {lseed}"
            )

    return (
        base_ckpts,
        pred_cls_ckpts,
        pred_loc_ckpts,
        fixed_cls_ckpts,
        fixed_loc_ckpts,
    )


class ExternalFixed05Dataset(Dataset):
    def __init__(
        self,
        rows,
        box_lookup,
        image_size,
        margin_factor,
        fallback_threshold,
    ):
        self.rows = list(rows)
        self.box_lookup = dict(box_lookup)
        self.image_size = int(image_size)
        self.margin_factor = float(margin_factor)
        self.fallback_threshold = float(fallback_threshold)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        path = str(r["path"])
        if path not in self.box_lookup:
            raise RuntimeError(f"Missing Fixed-0.5 external predicted box for {path}")

        rng = np.random.default_rng(246810 + idx)
        xg, xr, q = fixed.load_drcaf_views(
            path,
            self.box_lookup[path],
            self.image_size,
            False,
            rng,
            self.margin_factor,
            self.fallback_threshold,
        )
        return {
            "global_image": xg,
            "roi_image": xr,
            "quality": q,
            "path": path,
            "label": torch.tensor(int(r["label"]), dtype=torch.long),
        }


@torch.no_grad()
def infer_fixed05_classifier_fold(
    classifier_ckpt: Path,
    rows,
    boxes,
    args,
    device,
):
    model = fixed.DRCAFSharedSwin(
        args.backbone,
        pretrained=False,
        dropout=args.dropout,
    ).to(device)

    state, meta = load_state(classifier_ckpt, device)
    model.load_state_dict(state, strict=True)
    model.eval()

    box_lookup = {str(r["path"]): r for r in boxes}
    expected = {str(r["path"]) for r in rows}
    if set(box_lookup) != expected:
        raise RuntimeError("Fixed-0.5 external box coverage mismatch.")

    ds = ExternalFixed05Dataset(
        rows,
        box_lookup,
        args.image_size,
        args.crop_margin_factor,
        args.box_fallback_threshold,
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
        xg = batch["global_image"].to(device, non_blocking=True)
        xr = batch["roi_image"].to(device, non_blocking=True)
        q = batch["quality"].to(device, non_blocking=True)

        with common.autocast_ctx(device, args.amp_dtype):
            logits = model(xg, xr, q)
            p = F.softmax(logits, dim=-1).float().cpu().numpy()

        n = p.shape[0]
        probs[seen:seen+n] = p
        seen += n

    if seen != len(rows):
        raise RuntimeError("Fixed-0.5 external classifier row-count mismatch.")

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probs, meta


def paired_descriptive(y, pred_a, pred_b, n_boot, seed, name_a, name_b):
    """Descriptive paired IMAGE-level differences B-A, including Acc and Macro-F1."""
    y = np.asarray(y, dtype=int)
    a = np.asarray(pred_a, dtype=int)
    b = np.asarray(pred_b, dtype=int)

    acc_a = 100.0 * accuracy_score(y, a)
    acc_b = 100.0 * accuracy_score(y, b)
    f1_a = 100.0 * f1_score(y, a, labels=[0, 1, 2], average="macro", zero_division=0)
    f1_b = 100.0 * f1_score(y, b, labels=[0, 1, 2], average="macro", zero_division=0)

    rescue = int(np.sum((a != y) & (b == y)))
    lost = int(np.sum((a == y) & (b != y)))
    shared_wrong = int(np.sum((a != y) & (b != y)))

    rng = np.random.default_rng(seed)
    n = len(y)
    da = np.empty(n_boot, dtype=np.float64)
    df = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy, aa, bb = y[idx], a[idx], b[idx]
        da[i] = 100.0 * (np.mean(bb == yy) - np.mean(aa == yy))
        df[i] = 100.0 * (
            f1_score(yy, bb, labels=[0, 1, 2], average="macro", zero_division=0)
            - f1_score(yy, aa, labels=[0, 1, 2], average="macro", zero_division=0)
        )

    return {
        "comparison": f"{name_b} minus {name_a}",
        "delta_accuracy_pp": float(acc_b - acc_a),
        "delta_macro_f1_pp": float(f1_b - f1_a),
        "accuracy_delta_descriptive_image_bootstrap_95ci": [
            float(x) for x in np.percentile(da, [2.5, 97.5])
        ],
        "macro_f1_delta_descriptive_image_bootstrap_95ci": [
            float(x) for x in np.percentile(df, [2.5, 97.5])
        ],
        "rescues": rescue,
        "losses": lost,
        "shared_wrong": shared_wrong,
        "n_boot": int(n_boot),
        "warning": (
            "Descriptive IMAGE-level paired bootstrap only. Patient IDs are unavailable; "
            "within-patient dependence cannot be modeled and no patient-level inference is allowed."
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--brain_root", default="", help="Default: folder containing this script.")
    ap.add_argument("--baseline_root", default="")
    ap.add_argument("--predroi_root", default="")
    ap.add_argument("--fixed05_root", default="")
    ap.add_argument(
        "--fixed_localizer_root",
        default="",
        help="Optional D-RCAF locked full-5 root; auto-discovered when omitted.",
    )

    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--backbone", default="swin_base_patch4_window7_224")
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--seg_size", type=int, default=256)
    ap.add_argument("--box_threshold", type=float, default=0.50)
    ap.add_argument("--box_fallback_threshold", type=float, default=0.30)
    ap.add_argument("--crop_margin_factor", type=float, default=1.10)
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--seg_eval_batch_size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--amp_dtype", choices=["fp16", "bf16"], default="bf16")
    ap.add_argument("--allow_nonstandard_counts", action="store_true")
    ap.add_argument("--bootstrap_iterations", type=int, default=10000)
    ap.add_argument("--bootstrap_seed", type=int, default=20261009)
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(args.dataset_root)
    (
        brain_root, baseline_root, predroi_root, fixed05_root,
        fixed_localizer_root, locked_seed,
    ) = auto_resolve_model_roots(args)

    test_root = ext.resolve_test_root(dataset_root)
    rows, class_dirs, ignored_dirs = ext.discover_external_test(
        test_root,
        strict_v8_counts=not args.allow_nonstandard_counts,
    )
    manifest_df = pd.DataFrame(rows)
    manifest_df.to_csv(out_root / "bdneuro_v8_external_test_manifest.csv", index=False)

    counts = (
        manifest_df["class_name"]
        .value_counts()
        .reindex(CLASSES, fill_value=0)
        .to_dict()
    )

    print("=" * 156)
    print("BDNeuro-MRI v8 — FROZEN EXTERNAL IMAGE-LEVEL: Full / PredROI / Fixed-0.5")
    print("=" * 156)
    print(f"test root                 : {test_root}")
    print(f"tumor images              : {len(rows)}")
    print(f"class counts              : {counts}")
    print("excluded class            : no_tumor")
    print("training/fine-tuning      : NONE")
    print("external patient IDs      : unavailable")
    print("primary endpoint          : IMAGE-LEVEL")
    print("Fixed-0.5 fusion          : 0.5*global + 0.5*PredROI features")
    print(
        f"PredROI rule              : thr={args.box_threshold:.2f}, "
        f"fallback={args.box_fallback_threshold:.2f}, margin={args.crop_margin_factor:.2f}"
    )
    print("=" * 156)

    (
        baseline_ckpts,
        pred_cls_ckpts,
        pred_loc_ckpts,
        fixed_cls_ckpts,
        fixed_loc_ckpts,
    ) = audit_all_checkpoints(
        baseline_root,
        predroi_root,
        fixed05_root,
        fixed_localizer_root,
        args.n_folds,
    )
    print("[AUDIT] all frozen five-fold checkpoints + Fixed classifier/localizer pairing: PASS")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device                    : {device}")
    y = np.asarray([int(r["label"]) for r in rows], dtype=int)

    # Full-image ensemble.
    base_sum = np.zeros((len(rows), 3), dtype=np.float64)
    base_meta = []
    for fold, ckpt in enumerate(baseline_ckpts, start=1):
        print(f"[BASELINE] fold {fold}/{args.n_folds}")
        p, meta = ext.infer_baseline_fold(ckpt, rows, args, device)
        base_sum += p
        base_meta.append({"fold": fold, "checkpoint": str(ckpt), **{k: meta.get(k) for k in ["outer_seed", "outer_fold", "selected_epoch"]}})
    base_prob = base_sum / float(args.n_folds)

    # Original PredROI ensemble with its own frozen localizers.
    pred_sum = np.zeros((len(rows), 3), dtype=np.float64)
    pred_meta = []
    pred_bbox_records = []
    for fold, (loc_ckpt, cls_ckpt) in enumerate(zip(pred_loc_ckpts, pred_cls_ckpts), start=1):
        print(f"[PREDROI] fold {fold}/{args.n_folds}: localizer")
        boxes, lmeta = ext.predict_external_boxes(loc_ckpt, rows, args, device)
        found = 100.0 * np.mean([int(r["found"]) for r in boxes])
        for r in boxes:
            rr = dict(r); rr["fold"] = fold; rr["method"] = "PredROI"; pred_bbox_records.append(rr)
        print(f"[PREDROI] fold {fold}: bbox-found={found:.2f}%")
        print(f"[PREDROI] fold {fold}/{args.n_folds}: classifier")
        p, cmeta = ext.infer_predroi_classifier_fold(cls_ckpt, rows, boxes, args, device)
        pred_sum += p
        pred_meta.append({
            "fold": fold,
            "localizer_checkpoint": str(loc_ckpt),
            "classifier_checkpoint": str(cls_ckpt),
            "bbox_found_rate_pct": float(found),
            "localizer_outer_seed": lmeta.get("outer_seed"),
            "classifier_outer_seed": cmeta.get("outer_seed"),
            "classifier_selected_epoch": cmeta.get("selected_epoch"),
        })
    pred_prob = pred_sum / float(args.n_folds)

    # Fixed-0.5 ensemble with the D-RCAF localizers used by the locked Fixed ablation.
    fixed_sum = np.zeros((len(rows), 3), dtype=np.float64)
    fixed_meta = []
    fixed_bbox_records = []
    for fold, (loc_ckpt, cls_ckpt) in enumerate(zip(fixed_loc_ckpts, fixed_cls_ckpts), start=1):
        print(f"[FIXED-0.5] fold {fold}/{args.n_folds}: frozen localizer")
        boxes, lmeta = ext.predict_external_boxes(loc_ckpt, rows, args, device)
        found = 100.0 * np.mean([int(r["found"]) for r in boxes])
        for r in boxes:
            rr = dict(r); rr["fold"] = fold; rr["method"] = "Fixed-0.5"; fixed_bbox_records.append(rr)
        print(f"[FIXED-0.5] fold {fold}: bbox-found={found:.2f}%")
        print(f"[FIXED-0.5] fold {fold}/{args.n_folds}: global+PredROI classifier")
        p, cmeta = infer_fixed05_classifier_fold(cls_ckpt, rows, boxes, args, device)
        fixed_sum += p
        fixed_meta.append({
            "fold": fold,
            "localizer_checkpoint": str(loc_ckpt),
            "classifier_checkpoint": str(cls_ckpt),
            "bbox_found_rate_pct": float(found),
            "localizer_outer_seed": lmeta.get("outer_seed"),
            "classifier_outer_seed": cmeta.get("outer_seed"),
            "classifier_selected_epoch": cmeta.get("selected_epoch"),
        })
    fixed_prob = fixed_sum / float(args.n_folds)

    base_m = ext.metrics_from_probs(y, base_prob)
    pred_m = ext.metrics_from_probs(y, pred_prob)
    fixed_m = ext.metrics_from_probs(y, fixed_prob)

    base_pred = base_prob.argmax(1)
    pred_pred = pred_prob.argmax(1)
    fixed_pred = fixed_prob.argmax(1)

    paired_pred_fixed = paired_descriptive(
        y, pred_pred, fixed_pred,
        args.bootstrap_iterations, args.bootstrap_seed + 1,
        "PredROI-Swin", "Fixed-0.5 Dual-View",
    )
    paired_base_fixed = paired_descriptive(
        y, base_pred, fixed_pred,
        args.bootstrap_iterations, args.bootstrap_seed + 2,
        "Full-image Swin", "Fixed-0.5 Dual-View",
    )

    pred_df = manifest_df.copy()
    methods = {
        "baseline": (base_prob, base_pred),
        "predroi": (pred_prob, pred_pred),
        "fixed05": (fixed_prob, fixed_pred),
    }
    for prefix, (prob, pr) in methods.items():
        pred_df[f"{prefix}_pred_id"] = pr
        pred_df[f"{prefix}_pred_class"] = [CLASSES[i] for i in pr]
        pred_df[f"{prefix}_correct"] = (pr == y).astype(int)
        for c, cname in enumerate(CLASSES):
            pred_df[f"{prefix}_prob_{cname}"] = prob[:, c]
    pred_df.to_csv(out_root / "external_predictions_three_methods.csv", index=False)

    pd.DataFrame(pred_bbox_records).to_csv(out_root / "external_predroi_boxes_all_folds.csv", index=False)
    pd.DataFrame(fixed_bbox_records).to_csv(out_root / "external_fixed05_boxes_all_folds.csv", index=False)

    method_rows = []
    for name, m in [
        ("Frozen Cheng 5-fold full-image Swin ensemble", base_m),
        ("Frozen Cheng 5-fold PredROI-Swin ensemble", pred_m),
        ("Frozen Cheng 5-fold Fixed-0.5 Dual-View ensemble", fixed_m),
    ]:
        method_rows.append({
            "method": name,
            "accuracy": m["accuracy"],
            "macro_f1": m["macro_f1"],
            "glioma_recall": m["glioma_recall"],
            "meningioma_recall": m["meningioma_recall"],
            "pituitary_recall": m["pituitary_recall"],
            "errors": m["errors"],
        })
    pd.DataFrame(method_rows).to_csv(out_root / "external_method_summary_three_methods.csv", index=False)

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
            "locked_outer_seed": int(locked_seed),
            "ensemble_folds": args.n_folds,
            "fixed05_fusion": "0.5*global_feature + 0.5*PredROI_feature",
            "box_threshold": args.box_threshold,
            "box_fallback_threshold": args.box_fallback_threshold,
            "crop_margin_factor": args.crop_margin_factor,
            "fixed05_localizer_source": str(fixed_localizer_root),
            "fixed05_classifier_source": str(fixed05_root),
        },
        "baseline_full_image_swin": base_m,
        "predroi_swin": pred_m,
        "fixed05_dual_view": fixed_m,
        "paired_predroi_to_fixed05_image_level": paired_pred_fixed,
        "paired_baseline_to_fixed05_image_level": paired_base_fixed,
        "baseline_fold_metadata": base_meta,
        "predroi_fold_metadata": pred_meta,
        "fixed05_fold_metadata": fixed_meta,
        "interpretation_warning": (
            "Independent-source external IMAGE-level validation only. BDNeuro-MRI patient IDs are unavailable; "
            "results must not be described as independent patient-level external validation."
        ),
    }
    save_json(summary, out_root / "external_fixed05_summary.json")

    print("\n" + "=" * 156)
    print("FINAL BDNeuro-MRI v8 EXTERNAL IMAGE-LEVEL RESULTS — THREE FROZEN METHODS")
    print("=" * 156)
    for label, m in [
        ("Full-image Swin", base_m),
        ("PredROI-Swin", pred_m),
        ("Fixed-0.5 Dual-View", fixed_m),
    ]:
        print(
            f"{label:<23} | Acc={m['accuracy']:.3f}% | F1={m['macro_f1']:.3f}% | "
            f"G-R={m['glioma_recall']:.3f}% | M-R={m['meningioma_recall']:.3f}% | "
            f"P-R={m['pituitary_recall']:.3f}% | err={m['errors']}"
        )

    pp = paired_pred_fixed
    print(
        f"Fixed - PredROI          | Acc={pp['delta_accuracy_pp']:+.3f} pp | "
        f"F1={pp['delta_macro_f1_pp']:+.3f} pp | rescue/lost={pp['rescues']}/{pp['losses']}"
    )
    print(
        "Acc descriptive 95% CI   | "
        f"[{pp['accuracy_delta_descriptive_image_bootstrap_95ci'][0]:+.3f}, "
        f"{pp['accuracy_delta_descriptive_image_bootstrap_95ci'][1]:+.3f}] pp"
    )
    print(
        "F1 descriptive 95% CI    | "
        f"[{pp['macro_f1_delta_descriptive_image_bootstrap_95ci'][0]:+.3f}, "
        f"{pp['macro_f1_delta_descriptive_image_bootstrap_95ci'][1]:+.3f}] pp"
    )
    mean_fixed_bbox = np.mean([x["bbox_found_rate_pct"] for x in fixed_meta])
    print(f"Mean Fixed bbox-found    | {mean_fixed_bbox:.2f}% across frozen localizers")
    print("INTERPRETATION           | independent-source EXTERNAL IMAGE-LEVEL validation")
    print("PATIENT-LEVEL CLAIM      | NOT ALLOWED: BDNeuro-MRI patient IDs are unavailable")
    print("=" * 156)


if __name__ == "__main__":
    main()
