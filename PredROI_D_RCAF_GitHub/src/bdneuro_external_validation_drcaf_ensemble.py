#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
bdneuro_external_validation_drcaf_ensemble.py

Frozen independent-source EXTERNAL IMAGE-LEVEL evaluation of the already-trained
D-RCAF Cheng ensemble on BDNeuro-MRI v8.

This script is intentionally an ABLATION/DIAGNOSTIC external evaluation only:
  * NO training, fine-tuning, threshold tuning, or checkpoint selection.
  * Uses only glioma / meningioma / pituitary; no_tumor is excluded.
  * Uses the same frozen D-RCAF localizers and crop rule as the locked Cheng run.
  * Reuses the prior Full-image / PredROI / Fixed-0.5 external predictions when
    available, so only D-RCAF inference is newly executed.
  * BDNeuro-MRI patient IDs are unavailable, therefore all endpoints and all
    bootstrap intervals are descriptive IMAGE-level quantities only.
  * D-RCAF is NOT promoted to a confirmatory primary method based on this run.

Place next to:
  bdneuro_external_validation_frozen_ensemble.py
  cheng_drcaf_predroi_swin_full5.py
  cheng_patient_set_leakage_free_v2.py
  cheng_predroi_swin_screen_v1_audited.py

Recommended use after the previous Fixed-0.5 external run:
  python .\\bdneuro_external_validation_drcaf_ensemble.py `
    --dataset_root "C:\\...\\BDNeuro_MRI_v8" `
    --reference_root "C:\\PATH\\TO\\brain_project\\bdneuro_fixed05_external" `
    --out_root "C:\\PATH\\TO\\brain_project\\bdneuro_drcaf_external"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, recall_score
from torch.utils.data import Dataset, DataLoader

import bdneuro_external_validation_frozen_ensemble as ext
import cheng_drcaf_predroi_swin_full5 as drcaf
import cheng_patient_set_leakage_free_v2 as common

CLASSES = ext.CLASSES


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def _checkpoint_meta_cpu(path: Path):
    obj = torch.load(path, map_location="cpu", weights_only=False)
    return obj if isinstance(obj, dict) else {}


def _load_state(path: Path, device):
    obj = torch.load(path, map_location=device, weights_only=False)
    if isinstance(obj, dict) and "state_dict" in obj:
        return obj["state_dict"], obj
    return obj, {}


def _root_has_folds(root: Path, filenames, n_folds: int):
    return all(
        (root / f"fold_{fold}" / fn).exists()
        for fold in range(1, n_folds + 1)
        for fn in filenames
    )


def discover_drcaf_root(brain_root: Path, n_folds: int, explicit="") -> Path:
    if explicit:
        root = Path(explicit).resolve()
        if not _root_has_folds(
            root, ["final_drcaf_classifier.pt", "final_localizer.pt"], n_folds
        ):
            raise FileNotFoundError(
                f"D-RCAF root does not contain all frozen fold checkpoints: {root}"
            )
        return root

    candidates = []
    seen = set()
    for hit in brain_root.rglob("final_drcaf_classifier.pt"):
        if hit.parent.name.lower() != "fold_1":
            continue
        root = hit.parent.parent.resolve()
        key = str(root).lower()
        if key in seen:
            continue
        seen.add(key)
        if _root_has_folds(
            root, ["final_drcaf_classifier.pt", "final_localizer.pt"], n_folds
        ):
            candidates.append(root)

    # Prefer the already-used locked seed 5813 when metadata permits.
    seed5813 = []
    for root in candidates:
        meta = _checkpoint_meta_cpu(root / "fold_1" / "final_drcaf_classifier.pt")
        if meta.get("outer_seed") is not None and int(meta["outer_seed"]) == 5813:
            seed5813.append(root)
    if len(seed5813) == 1:
        return seed5813[0]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates:
        raise FileNotFoundError(
            f"Could not auto-find frozen D-RCAF full-5 root under {brain_root}. "
            "Pass --drcaf_root explicitly."
        )
    raise RuntimeError(
        "Multiple D-RCAF roots found. Pass --drcaf_root explicitly:\n"
        + "\n".join(f"  - {x}" for x in candidates[:20])
    )


def audit_drcaf_checkpoints(root: Path, n_folds: int):
    cls_ckpts, loc_ckpts = [], []
    seed = None
    for fold in range(1, n_folds + 1):
        c = root / f"fold_{fold}" / "final_drcaf_classifier.pt"
        l = root / f"fold_{fold}" / "final_localizer.pt"
        if not c.exists() or not l.exists():
            raise FileNotFoundError(
                f"Missing frozen D-RCAF fold {fold} checkpoint(s):\n  {c}\n  {l}"
            )
        cm = _checkpoint_meta_cpu(c)
        lm = _checkpoint_meta_cpu(l)
        method = str(cm.get("method", ""))
        if "D-RCAF" not in method:
            raise RuntimeError(
                f"Fold {fold}: unexpected classifier metadata method={method!r}"
            )
        if cm.get("outer_fold") is not None and int(cm["outer_fold"]) != fold:
            raise RuntimeError(f"Fold {fold}: classifier outer_fold mismatch")
        if lm.get("outer_fold") is not None and int(lm["outer_fold"]) != fold:
            raise RuntimeError(f"Fold {fold}: localizer outer_fold mismatch")
        cs = cm.get("outer_seed")
        ls = lm.get("outer_seed")
        if cs is not None and ls is not None and int(cs) != int(ls):
            raise RuntimeError(f"Fold {fold}: classifier/localizer outer_seed mismatch")
        if cs is not None:
            if seed is None:
                seed = int(cs)
            elif seed != int(cs):
                raise RuntimeError("D-RCAF fold checkpoints do not share one outer_seed")
        cls_ckpts.append(c)
        loc_ckpts.append(l)
    return cls_ckpts, loc_ckpts, seed


class ExternalDRCAFDataset(Dataset):
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
            raise RuntimeError(f"Missing D-RCAF external predicted box for {path}")
        rng = np.random.default_rng(97531 + idx)
        xg, xr, q = drcaf.load_drcaf_views(
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


def _forward_drcaf_with_diagnostics(model, xg, xr, q):
    """Exact eval forward plus channel-gate diagnostics, without changing the model."""
    B = xg.shape[0]
    both = torch.cat([xg, xr], dim=0)
    feats = model.encoder(both)
    fg, fr = feats[:B], feats[B:]
    disagreement = torch.abs(fr - fg)
    qe = model.quality(q.float())
    z = torch.cat([fg, fr, disagreement, qe], dim=-1)
    h = F.gelu(model.gate_fc1(model.gate_norm(z)))
    h = model.gate_drop(h)
    gate = torch.sigmoid(model.gate_fc2(h))
    fused = (1.0 - gate) * fg + gate * fr
    logits = model.classifier(model.dropout(fused))
    return logits, {
        "gate_mean": gate.mean(dim=-1),
        "gate_std": gate.std(dim=-1, unbiased=False),
        "gate_min": gate.min(dim=-1).values,
        "gate_max": gate.max(dim=-1).values,
        "disagreement_l1": disagreement.mean(dim=-1),
    }


@torch.no_grad()
def infer_drcaf_fold(classifier_ckpt, rows, boxes, args, device, fold):
    model = drcaf.DRCAFSharedSwin(
        args.backbone,
        pretrained=False,
        dropout=args.dropout,
    ).to(device)
    state, meta = _load_state(classifier_ckpt, device)
    model.load_state_dict(state, strict=True)
    model.eval()

    box_lookup = {str(r["path"]): r for r in boxes}
    expected = {str(r["path"]) for r in rows}
    if set(box_lookup) != expected:
        raise RuntimeError("D-RCAF external box coverage mismatch")

    ds = ExternalDRCAFDataset(
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
    diagnostics = []
    seen = 0
    for batch in loader:
        xg = batch["global_image"].to(device, non_blocking=True)
        xr = batch["roi_image"].to(device, non_blocking=True)
        q = batch["quality"].to(device, non_blocking=True)
        with common.autocast_ctx(device, args.amp_dtype):
            logits, aux = _forward_drcaf_with_diagnostics(model, xg, xr, q)
            p = F.softmax(logits, dim=-1).float().cpu().numpy()

        n = len(p)
        probs[seen:seen+n] = p
        gm = aux["gate_mean"].float().cpu().numpy()
        gs = aux["gate_std"].float().cpu().numpy()
        gmin = aux["gate_min"].float().cpu().numpy()
        gmax = aux["gate_max"].float().cpu().numpy()
        dis = aux["disagreement_l1"].float().cpu().numpy()
        qcpu = batch["quality"].float().cpu().numpy()
        for i in range(n):
            diagnostics.append({
                "fold": int(fold),
                "path": str(batch["path"][i]),
                "true_id": int(batch["label"][i]),
                "gate_mean": float(gm[i]),
                "gate_channel_std": float(gs[i]),
                "gate_min": float(gmin[i]),
                "gate_max": float(gmax[i]),
                "disagreement_l1": float(dis[i]),
                "bbox_found_cue": float(qcpu[i, 0]),
                "bbox_area_ratio_cue": float(qcpu[i, 1]),
                "bbox_center_distance_cue": float(qcpu[i, 2]),
                "bbox_fallback_cue": float(qcpu[i, 3]),
            })
        seen += n

    if seen != len(rows):
        raise RuntimeError("D-RCAF external classifier row-count mismatch")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probs, diagnostics, meta


def metrics_from_probs(y, probs):
    pred = np.asarray(probs).argmax(axis=1)
    rec = recall_score(
        y, pred, labels=[0, 1, 2], average=None, zero_division=0
    )
    return {
        "n": int(len(y)),
        "accuracy": float(100.0 * accuracy_score(y, pred)),
        "macro_f1": float(
            100.0 * f1_score(
                y, pred, labels=[0, 1, 2], average="macro", zero_division=0
            )
        ),
        "glioma_recall": float(100.0 * rec[0]),
        "meningioma_recall": float(100.0 * rec[1]),
        "pituitary_recall": float(100.0 * rec[2]),
        "errors": int(np.sum(pred != y)),
    }


def paired_descriptive(y, pred_a, pred_b, n_boot, seed, name_a, name_b):
    """Descriptive paired IMAGE-level differences B-A."""
    y = np.asarray(y, dtype=int)
    a = np.asarray(pred_a, dtype=int)
    b = np.asarray(pred_b, dtype=int)
    acc_a = 100.0 * accuracy_score(y, a)
    acc_b = 100.0 * accuracy_score(y, b)
    f1_a = 100.0 * f1_score(y, a, labels=[0,1,2], average="macro", zero_division=0)
    f1_b = 100.0 * f1_score(y, b, labels=[0,1,2], average="macro", zero_division=0)
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
            f1_score(yy, bb, labels=[0,1,2], average="macro", zero_division=0)
            - f1_score(yy, aa, labels=[0,1,2], average="macro", zero_division=0)
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
            "within-patient dependence cannot be modeled."
        ),
    }


def _resolve_reference_root(brain_root: Path, explicit: str) -> Path:
    if explicit:
        root = Path(explicit).resolve()
        f = root / "external_predictions_three_methods.csv"
        if not f.exists():
            raise FileNotFoundError(f"Reference prediction CSV not found: {f}")
        return root
    preferred = brain_root / "bdneuro_fixed05_external"
    if (preferred / "external_predictions_three_methods.csv").exists():
        return preferred
    hits = list(brain_root.rglob("external_predictions_three_methods.csv"))
    if len(hits) == 1:
        return hits[0].parent.resolve()
    if not hits:
        raise FileNotFoundError(
            "Could not find prior Fixed-0.5 external_predictions_three_methods.csv. "
            "Pass --reference_root explicitly."
        )
    raise RuntimeError(
        "Multiple prior external reference runs found. Pass --reference_root explicitly:\n"
        + "\n".join(f"  - {h.parent}" for h in hits[:20])
    )


def _load_reference_predictions(reference_root: Path, rows):
    path = reference_root / "external_predictions_three_methods.csv"
    df = pd.read_csv(path)
    required = {
        "path", "label",
        "baseline_prob_glioma", "baseline_prob_meningioma", "baseline_prob_pituitary",
        "predroi_prob_glioma", "predroi_prob_meningioma", "predroi_prob_pituitary",
        "fixed05_prob_glioma", "fixed05_prob_meningioma", "fixed05_prob_pituitary",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Reference CSV missing columns: {missing}")

    expected_paths = [str(Path(r["path"]).resolve()) for r in rows]
    df["_norm_path"] = df["path"].map(lambda x: str(Path(str(x)).resolve()))
    if df["_norm_path"].duplicated().any():
        raise RuntimeError("Reference CSV contains duplicate paths")
    lookup = df.set_index("_norm_path")
    if set(expected_paths) != set(lookup.index):
        only_now = sorted(set(expected_paths) - set(lookup.index))[:5]
        only_ref = sorted(set(lookup.index) - set(expected_paths))[:5]
        raise RuntimeError(
            "Current BDNeuro manifest does not exactly match the prior reference run.\n"
            f"Only current (sample): {only_now}\nOnly reference (sample): {only_ref}"
        )
    aligned = lookup.loc[expected_paths].reset_index(drop=True)
    y = np.asarray([int(r["label"]) for r in rows], dtype=int)
    if not np.array_equal(y, aligned["label"].to_numpy(dtype=int)):
        raise RuntimeError("Reference labels do not match current BDNeuro manifest")

    def P(prefix):
        return aligned[[
            f"{prefix}_prob_glioma",
            f"{prefix}_prob_meningioma",
            f"{prefix}_prob_pituitary",
        ]].to_numpy(dtype=np.float64)
    return aligned, P("baseline"), P("predroi"), P("fixed05")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--brain_root", default="", help="Default: folder containing this script")
    ap.add_argument("--reference_root", default="", help="Prior Fixed-0.5 external output root")
    ap.add_argument("--drcaf_root", default="", help="Frozen D-RCAF full-5 root")
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
    brain_root = Path(args.brain_root or Path(__file__).resolve().parent).resolve()
    reference_root = _resolve_reference_root(brain_root, args.reference_root)
    drcaf_root = discover_drcaf_root(brain_root, args.n_folds, args.drcaf_root)
    cls_ckpts, loc_ckpts, locked_seed = audit_drcaf_checkpoints(drcaf_root, args.n_folds)

    print(f"[AUTO] brain_root          : {brain_root}")
    print(f"[AUTO] reference_root      : {reference_root}")
    print(f"[AUTO] D-RCAF root         : {drcaf_root}")
    print(f"[AUTO] locked outer seed   : {locked_seed}")

    test_root = ext.resolve_test_root(Path(args.dataset_root))
    rows, class_dirs, ignored_dirs = ext.discover_external_test(
        test_root,
        strict_v8_counts=not args.allow_nonstandard_counts,
    )
    manifest_df = pd.DataFrame(rows)
    manifest_df.to_csv(out_root / "bdneuro_v8_external_test_manifest.csv", index=False)
    counts = (
        manifest_df["class_name"].value_counts().reindex(CLASSES, fill_value=0).to_dict()
    )
    ref_df, base_prob, predroi_prob, fixed_prob = _load_reference_predictions(
        reference_root, rows
    )

    y = np.asarray([int(r["label"]) for r in rows], dtype=int)
    base_m = metrics_from_probs(y, base_prob)
    predroi_m = metrics_from_probs(y, predroi_prob)
    fixed_m = metrics_from_probs(y, fixed_prob)

    print("=" * 156)
    print("BDNeuro-MRI v8 — FROZEN EXTERNAL IMAGE-LEVEL D-RCAF ABLATION")
    print("=" * 156)
    print(f"test root                 : {test_root}")
    print(f"tumor images              : {len(rows)}")
    print(f"class counts              : {counts}")
    print("excluded class            : no_tumor")
    print("training/fine-tuning      : NONE")
    print("external patient IDs      : unavailable")
    print("primary endpoint          : IMAGE-LEVEL")
    print("reference predictions     : reused from prior frozen external run")
    print(
        f"PredROI rule              : thr={args.box_threshold:.2f}, "
        f"fallback={args.box_fallback_threshold:.2f}, margin={args.crop_margin_factor:.2f}"
    )
    print("=" * 156)
    print("[AUDIT] frozen D-RCAF 5-fold classifier + localizer checkpoints: PASS")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device                    : {device}")

    dr_sum = np.zeros((len(rows), 3), dtype=np.float64)
    all_diag = []
    all_boxes = []
    fold_meta = []

    for fold, (loc_ckpt, cls_ckpt) in enumerate(zip(loc_ckpts, cls_ckpts), start=1):
        print(f"[D-RCAF] fold {fold}/{args.n_folds}: frozen localizer")
        boxes, lmeta = ext.predict_external_boxes(loc_ckpt, rows, args, device)
        found = 100.0 * np.mean([int(r["found"]) for r in boxes])
        for r in boxes:
            rr = dict(r)
            rr["fold"] = fold
            rr["method"] = "D-RCAF"
            all_boxes.append(rr)
        print(f"[D-RCAF] fold {fold}: bbox-found={found:.2f}%")
        print(f"[D-RCAF] fold {fold}/{args.n_folds}: adaptive fusion classifier")
        p, diag, cmeta = infer_drcaf_fold(cls_ckpt, rows, boxes, args, device, fold)
        dr_sum += p
        all_diag.extend(diag)
        fold_meta.append({
            "fold": int(fold),
            "localizer_checkpoint": str(loc_ckpt),
            "classifier_checkpoint": str(cls_ckpt),
            "bbox_found_rate_pct": float(found),
            "localizer_outer_seed": lmeta.get("outer_seed"),
            "classifier_outer_seed": cmeta.get("outer_seed"),
            "classifier_selected_epoch": cmeta.get("selected_epoch"),
        })

    dr_prob = dr_sum / float(args.n_folds)
    dr_m = metrics_from_probs(y, dr_prob)
    base_pred = base_prob.argmax(1)
    predroi_pred = predroi_prob.argmax(1)
    fixed_pred = fixed_prob.argmax(1)
    dr_pred = dr_prob.argmax(1)

    cmp_pred = paired_descriptive(
        y, predroi_pred, dr_pred,
        args.bootstrap_iterations, args.bootstrap_seed + 11,
        "PredROI-Swin", "D-RCAF",
    )
    cmp_fixed = paired_descriptive(
        y, fixed_pred, dr_pred,
        args.bootstrap_iterations, args.bootstrap_seed + 12,
        "Fixed-0.5 Dual-View", "D-RCAF",
    )
    cmp_base = paired_descriptive(
        y, base_pred, dr_pred,
        args.bootstrap_iterations, args.bootstrap_seed + 13,
        "Full-image Swin", "D-RCAF",
    )

    diag_df = pd.DataFrame(all_diag)
    diag_df.to_csv(out_root / "external_drcaf_gate_diagnostics_all_folds.csv", index=False)
    pd.DataFrame(all_boxes).to_csv(
        out_root / "external_drcaf_boxes_all_folds.csv", index=False
    )

    diag_by_path = diag_df.groupby("path", as_index=False).agg(
        drcaf_gate_mean=("gate_mean", "mean"),
        drcaf_gate_channel_std=("gate_channel_std", "mean"),
        drcaf_gate_min=("gate_min", "mean"),
        drcaf_gate_max=("gate_max", "mean"),
        drcaf_disagreement_l1=("disagreement_l1", "mean"),
    )
    diag_lookup = diag_by_path.set_index("path")

    pred_df = manifest_df.copy()
    pred_df["drcaf_pred_id"] = dr_pred
    pred_df["drcaf_pred_class"] = [CLASSES[i] for i in dr_pred]
    pred_df["drcaf_correct"] = (dr_pred == y).astype(int)
    for c, cname in enumerate(CLASSES):
        pred_df[f"drcaf_prob_{cname}"] = dr_prob[:, c]
    for col in [
        "drcaf_gate_mean", "drcaf_gate_channel_std", "drcaf_gate_min",
        "drcaf_gate_max", "drcaf_disagreement_l1",
    ]:
        pred_df[col] = [
            float(diag_lookup.loc[str(r["path"]), col]) for r in rows
        ]
    pred_df.to_csv(out_root / "external_drcaf_predictions.csv", index=False)

    gate_summary = {
        "overall_gate_mean": float(diag_df["gate_mean"].mean()),
        "overall_gate_mean_sd_across_fold_images": float(diag_df["gate_mean"].std(ddof=0)),
        "mean_within_sample_channel_std": float(diag_df["gate_channel_std"].mean()),
        "mean_gate_min": float(diag_df["gate_min"].mean()),
        "mean_gate_max": float(diag_df["gate_max"].mean()),
        "mean_disagreement_l1": float(diag_df["disagreement_l1"].mean()),
        "by_true_class": {},
    }
    for cid, cname in enumerate(CLASSES):
        d = diag_df[diag_df["true_id"] == cid]
        gate_summary["by_true_class"][cname] = {
            "gate_mean": float(d["gate_mean"].mean()),
            "gate_channel_std": float(d["gate_channel_std"].mean()),
            "disagreement_l1": float(d["disagreement_l1"].mean()),
        }

    summary = {
        "dataset": {
            "name": "BDNeuro-MRI v8",
            "evaluation_level": "image",
            "patient_ids_available": False,
            "n_images": int(len(rows)),
            "class_counts": {k: int(v) for k, v in counts.items()},
            "no_tumor_excluded": True,
        },
        "frozen_protocol": {
            "no_external_training_or_finetuning": True,
            "drcaf_locked_outer_seed": locked_seed,
            "ensemble_folds": int(args.n_folds),
            "drcaf_root": str(drcaf_root),
            "reference_root": str(reference_root),
            "box_threshold": float(args.box_threshold),
            "box_fallback_threshold": float(args.box_fallback_threshold),
            "crop_margin_factor": float(args.crop_margin_factor),
            "status": "external exploratory ablation of a model frozen before external inspection",
        },
        "full_image_swin": base_m,
        "predroi_swin": predroi_m,
        "fixed05_dual_view": fixed_m,
        "drcaf": dr_m,
        "paired_predroi_to_drcaf_image_level": cmp_pred,
        "paired_fixed05_to_drcaf_image_level": cmp_fixed,
        "paired_full_to_drcaf_image_level": cmp_base,
        "drcaf_gate_diagnostics": gate_summary,
        "fold_metadata": fold_meta,
        "interpretation_warning": (
            "Independent-source external IMAGE-level validation only. Patient IDs are unavailable. "
            "D-RCAF must be treated as an exploratory frozen-model ablation here, not selected or tuned on BDNeuro."
        ),
    }
    save_json(summary, out_root / "external_drcaf_summary.json")

    pd.DataFrame([
        {"method": "Full-image Swin", **base_m},
        {"method": "PredROI-Swin", **predroi_m},
        {"method": "Fixed-0.5 Dual-View", **fixed_m},
        {"method": "D-RCAF", **dr_m},
    ]).to_csv(out_root / "external_four_method_summary.csv", index=False)

    print("\n" + "=" * 156)
    print("FINAL BDNeuro-MRI v8 EXTERNAL IMAGE-LEVEL RESULTS — FROZEN D-RCAF ABLATION")
    print("=" * 156)
    for label, m in [
        ("Full-image Swin", base_m),
        ("PredROI-Swin", predroi_m),
        ("Fixed-0.5 Dual-View", fixed_m),
        ("D-RCAF", dr_m),
    ]:
        print(
            f"{label:<23} | Acc={m['accuracy']:.3f}% | F1={m['macro_f1']:.3f}% | "
            f"G-R={m['glioma_recall']:.3f}% | M-R={m['meningioma_recall']:.3f}% | "
            f"P-R={m['pituitary_recall']:.3f}% | err={m['errors']}"
        )

    for tag, cmp in [("D-RCAF - PredROI", cmp_pred), ("D-RCAF - Fixed", cmp_fixed)]:
        print(
            f"{tag:<23} | Acc={cmp['delta_accuracy_pp']:+.3f} pp | "
            f"F1={cmp['delta_macro_f1_pp']:+.3f} pp | "
            f"rescue/lost={cmp['rescues']}/{cmp['losses']}"
        )
        print(
            f"  Acc descriptive 95% CI | "
            f"[{cmp['accuracy_delta_descriptive_image_bootstrap_95ci'][0]:+.3f}, "
            f"{cmp['accuracy_delta_descriptive_image_bootstrap_95ci'][1]:+.3f}] pp"
        )
        print(
            f"  F1 descriptive 95% CI  | "
            f"[{cmp['macro_f1_delta_descriptive_image_bootstrap_95ci'][0]:+.3f}, "
            f"{cmp['macro_f1_delta_descriptive_image_bootstrap_95ci'][1]:+.3f}] pp"
        )

    mean_bbox = float(np.mean([x["bbox_found_rate_pct"] for x in fold_meta]))
    print(f"Mean D-RCAF bbox-found   | {mean_bbox:.2f}% across frozen localizers")
    print(
        f"D-RCAF gate diagnostics | mean={gate_summary['overall_gate_mean']:.4f} | "
        f"mean channel-std={gate_summary['mean_within_sample_channel_std']:.4f} | "
        f"mean[min,max]=[{gate_summary['mean_gate_min']:.4f}, {gate_summary['mean_gate_max']:.4f}]"
    )
    print("INTERPRETATION           | exploratory frozen-model EXTERNAL IMAGE-LEVEL ablation")
    print("PATIENT-LEVEL CLAIM      | NOT ALLOWED: BDNeuro-MRI patient IDs are unavailable")
    print("MODEL-SELECTION CLAIM    | NOT ALLOWED: do not select/tune D-RCAF based on this external result")
    print("=" * 156)


if __name__ == "__main__":
    main()
