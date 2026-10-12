#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
cheng_drcaf_component_ablation_cached_boxes.py

Post-lock MECHANISM ablation for the already-defined D-RCAF architecture.

Variants:
  no_d   : remove sample-specific disagreement d from the gate
           (same tensor dimensions; d is replaced by zeros)
  no_q   : remove sample-specific localization reliability q from the gate
           (quality embedding replaced by zeros)
  scalar : replace channel-wise ROI gate with ONE learned scalar gate per image

Reference:
  full D-RCAF patient_oof.csv is REUSED from the existing locked seed-5813 run.
  It is not retrained by default.

Critical speed safeguard:
  ALL localizer work is reused from the existing D-RCAF locked run:
    - Stage-A inner-train OOF boxes
    - Stage-A inner-val boxes
    - outer-train OOF refit boxes
    - outer-val image-only boxes
  Therefore this script trains classifiers only. It does not retrain any localizer.

Scientific scope:
  This is a post-lock descriptive mechanism analysis. It must not be described
  as an independent confirmatory experiment or used to tune D-RCAF after seeing
  the locked seed-5813 outcome.

Pairing:
  no_d and no_q have exactly the same module shapes as full D-RCAF.
  scalar constructs the original full model first, then swaps only gate_fc2
  while restoring the PyTorch RNG state so common encoder/classifier
  initialization and the subsequent training RNG stream remain aligned.

Expected companion files in the same folder:
  cheng_drcaf_predroi_swin_full5.py
  cheng_patient_set_leakage_free_v2.py
  cheng_predroi_swin_screen_v1_audited.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, recall_score

import cheng_patient_set_leakage_free_v2 as common
import cheng_predroi_swin_screen_v1_audited as pred
import cheng_drcaf_predroi_swin_full5 as base


CLASSES = common.CLASSES
VARIANTS = ("no_d", "no_q", "scalar")


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


class AblatedDRCAF(base.DRCAFSharedSwin):
    def __init__(self, backbone, pretrained=True, dropout=0.15, variant="no_d"):
        super().__init__(backbone, pretrained=pretrained, dropout=dropout)
        self.variant = variant

        if variant == "scalar":
            # Preserve common-model initialization and downstream RNG stream:
            # build full model first, save RNG, replace only final gate layer,
            # zero-init it, then restore RNG.
            cpu_state = torch.random.get_rng_state()
            self.gate_fc2 = nn.Linear(base.DRCAF_GATE_HIDDEN, 1)
            nn.init.zeros_(self.gate_fc2.weight)
            nn.init.zeros_(self.gate_fc2.bias)
            torch.random.set_rng_state(cpu_state)

        elif variant not in ("no_d", "no_q"):
            raise ValueError(f"Unknown ablation variant: {variant}")

    def forward(self, x_global, x_roi, quality, return_aux=False):
        B = x_global.shape[0]
        both = torch.cat([x_global, x_roi], dim=0)
        feats = self.encoder(both)
        fg, fr = feats[:B], feats[B:]

        raw_disagreement = torch.abs(fr - fg)
        gate_disagreement = (
            torch.zeros_like(raw_disagreement)
            if self.variant == "no_d"
            else raw_disagreement
        )

        if self.variant == "no_q":
            qe = torch.zeros(
                (B, base.DRCAF_QUALITY_EMBED),
                dtype=fg.dtype,
                device=fg.device,
            )
        else:
            qe = self.quality(quality.float()).to(dtype=fg.dtype)

        z = torch.cat([fg, fr, gate_disagreement, qe], dim=-1)
        h = F.gelu(self.gate_fc1(self.gate_norm(z)))
        h = self.gate_drop(h)

        if self.variant == "scalar":
            g_scalar = torch.sigmoid(self.gate_fc2(h))  # [B,1]
            roi_gate = g_scalar.expand(-1, fg.shape[1])
        else:
            roi_gate = torch.sigmoid(self.gate_fc2(h))

        fused = (1.0 - roi_gate) * fg + roi_gate * fr
        logits = self.classifier(self.dropout(fused))

        if return_aux:
            return logits, {
                "roi_gate_mean": roi_gate.mean(dim=-1),
                # Keep diagnostic disagreement from actual features, even no_d.
                "disagreement_l1": raw_disagreement.mean(dim=-1),
            }
        return logits


def model_factory(variant):
    def _ctor(backbone, pretrained=True, dropout=0.15):
        return AblatedDRCAF(
            backbone,
            pretrained=pretrained,
            dropout=dropout,
            variant=variant,
        )
    return _ctor


def expected_map(patients):
    return {
        str(q): (str(p.patient_id), int(p.label))
        for p in patients for q in p.paths
    }


def load_box_csv_exact(path: Path, patients, label):
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path, dtype={"patient_id": str})
    req = {"path","patient_id","label","found","x0","y0","x1","y1"}
    miss = req - set(df.columns)
    if miss:
        raise RuntimeError(f"{label}: missing columns {sorted(miss)} in {path}")
    if df["path"].duplicated().any():
        raise RuntimeError(f"{label}: duplicate path in {path}")
    exp = expected_map(patients)
    if set(df["path"].astype(str)) != set(exp):
        raise RuntimeError(f"{label}: path coverage mismatch in {path}")
    for r in df.itertuples(index=False):
        pid, y = exp[str(r.path)]
        if str(r.patient_id) != pid or int(r.label) != y:
            raise RuntimeError(f"{label}: patient/label mismatch in {path}")
    return df.to_dict("records")


def metrics(df):
    y = df["true_id"].to_numpy(int)
    p = df["pred_id"].to_numpy(int)
    rec = recall_score(y,p,labels=[0,1,2],average=None,zero_division=0)
    return {
        "n": int(len(y)),
        "accuracy": float(100*accuracy_score(y,p)),
        "macro_f1": float(100*f1_score(y,p,labels=[0,1,2],average="macro",zero_division=0)),
        "errors": int(np.sum(y!=p)),
        "glioma_recall": float(100*rec[0]),
        "meningioma_recall": float(100*rec[1]),
        "pituitary_recall": float(100*rec[2]),
    }


def pair_compare(ref, cand, n_boot, seed):
    a = ref[["patient_id","true_id","pred_id"]].copy()
    b = cand[["patient_id","true_id","pred_id"]].copy()
    a["patient_id"] = a["patient_id"].astype(str)
    b["patient_id"] = b["patient_id"].astype(str)
    z = a.rename(columns={"true_id":"true_a","pred_id":"pred_a"}).merge(
        b.rename(columns={"true_id":"true_b","pred_id":"pred_b"}),
        on="patient_id", validate="one_to_one"
    )
    if len(z) != len(a) or len(z) != len(b):
        raise RuntimeError("Paired patient coverage mismatch.")
    if not np.array_equal(z.true_a.to_numpy(int), z.true_b.to_numpy(int)):
        raise RuntimeError("Paired true labels mismatch.")
    y=z.true_a.to_numpy(int); pa=z.pred_a.to_numpy(int); pb=z.pred_b.to_numpy(int)
    rescue=int(np.sum((pa!=y)&(pb==y)))
    lost=int(np.sum((pa==y)&(pb!=y)))
    ndis=rescue+lost
    p=1.0 if ndis==0 else float(binomtest(min(rescue,lost),ndis,0.5).pvalue)

    rng=np.random.default_rng(seed)
    da=np.empty(n_boot); df=np.empty(n_boot)
    n=len(y)
    for i in range(n_boot):
        idx=rng.integers(0,n,size=n)
        yy,aa,bb=y[idx],pa[idx],pb[idx]
        da[i]=100*(np.mean(bb==yy)-np.mean(aa==yy))
        df[i]=100*(
            f1_score(yy,bb,labels=[0,1,2],average="macro",zero_division=0)
            - f1_score(yy,aa,labels=[0,1,2],average="macro",zero_division=0)
        )
    return {
        "rescues": rescue,
        "losses": lost,
        "discordant": ndis,
        "exact_mcnemar_p": p,
        "delta_accuracy_pp": float(100*(accuracy_score(y,pb)-accuracy_score(y,pa))),
        "delta_macro_f1_pp": float(100*(
            f1_score(y,pb,labels=[0,1,2],average="macro",zero_division=0)
            - f1_score(y,pa,labels=[0,1,2],average="macro",zero_division=0)
        )),
        "accuracy_delta_95ci": [float(x) for x in np.percentile(da,[2.5,97.5])],
        "macro_f1_delta_95ci": [float(x) for x in np.percentile(df,[2.5,97.5])],
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument(
        "--cache_root",
        default=r"C:\PATH\TO\brain_project\drcaf_full5_seed5813",
        help="Existing locked D-RCAF seed5813 root containing saved predicted-box caches and full patient_oof.csv.",
    )
    ap.add_argument(
        "--out_root",
        default=r"C:\PATH\TO\brain_project\drcaf_component_ablation_seed5813",
    )
    ap.add_argument("--outer_seed", type=int, default=5813)
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--train_folds", type=int, default=5)
    ap.add_argument("--inner_n_splits", type=int, default=5)
    ap.add_argument("--inner_seed", type=int, default=31415)
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))

    # Classifier settings must match locked D-RCAF.
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
    ap.add_argument("--crop_margin_factor", type=float, default=1.10)
    ap.add_argument("--box_fallback_threshold", type=float, default=0.30)
    ap.add_argument("--amp_dtype", choices=["fp16","bf16"], default="bf16")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=20000)
    args=ap.parse_args()

    if args.outer_seed != 5813:
        raise ValueError("Reviewer mechanism ablation is locked to outer_seed=5813.")
    if args.max_select_epochs != 15 or args.freeze_epochs != 3:
        raise ValueError("Keep the locked D-RCAF 15-epoch / 3-epoch warm-up settings.")
    if abs(args.crop_margin_factor-1.10)>1e-12 or abs(args.box_fallback_threshold-0.30)>1e-12:
        raise ValueError("PredROI crop settings are locked.")

    cache_root=Path(args.cache_root)
    out_root=Path(args.out_root)
    out_root.mkdir(parents=True,exist_ok=True)
    full_csv=cache_root/"patient_oof.csv"
    if not full_csv.exists():
        raise FileNotFoundError(full_csv)
    full_df=pd.read_csv(full_csv,dtype={"patient_id":str})

    common.seed_all(args.outer_seed,args.deterministic)
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    slice_rows,_=common.load_manifest(Path(args.manifest))
    splits=common.outer_patient_splits(slice_rows,args.n_folds,args.outer_seed)

    print("="*156)
    print("D-RCAF COMPONENT ABLATION — CACHED-BOX CLASSIFIER-ONLY FULL-5")
    print("="*156)
    print(f"device        : {device}")
    print(f"cache root    : {cache_root}")
    print(f"variants      : {args.variants}")
    print("localizers    : NOT RETRAINED; locked box caches reused")
    print("scope         : post-lock descriptive mechanism analysis")
    print("="*156)

    original_class = base.DRCAFSharedSwin
    variant_summaries = {}

    try:
        for variant in args.variants:
            print("\n"+"#"*156)
            print(f"VARIANT: {variant}")
            print("#"*156)
            base.DRCAFSharedSwin = model_factory(variant)

            vroot=out_root/variant
            vroot.mkdir(parents=True,exist_ok=True)
            all_pat=[]; all_slice=[]; folds=[]

            for fold,(tr_idx,va_idx) in enumerate(splits,start=1):
                if fold>args.train_folds:
                    continue
                args.outer_fold=fold
                outer_train=common.patient_subset_from_slice_indices(slice_rows,tr_idx)
                outer_val=common.patient_subset_from_slice_indices(slice_rows,va_idx)
                inner_train,inner_val=common.make_inner_selection_split(
                    outer_train,args.inner_n_splits,args.inner_seed+fold
                )

                cached=base._load_cached_selection_boxes_if_valid(
                    cache_root/f"fold_{fold}"/"selection",
                    inner_train,inner_val
                )
                if cached is None:
                    raise RuntimeError(
                        f"Fold {fold}: locked Stage-A box cache could not be validated. "
                        "Do NOT regenerate for this reviewer ablation."
                    )
                inner_train_boxes,inner_val_boxes=cached

                refit_boxes=load_box_csv_exact(
                    cache_root/f"fold_{fold}"/"refit"/"refit_oof_train_boxes.csv",
                    outer_train,
                    f"fold{fold} outer-train OOF boxes",
                )
                outer_val_boxes=load_box_csv_exact(
                    cache_root/f"fold_{fold}"/"refit"/"outer_val_predicted_boxes.csv",
                    outer_val,
                    f"fold{fold} outer-val boxes",
                )

                fdir=vroot/f"fold_{fold}"
                sdir=fdir/"selection"; rdir=fdir/"refit"
                sdir.mkdir(parents=True,exist_ok=True)
                rdir.mkdir(parents=True,exist_ok=True)

                print(f"\n[F{fold}] selection — {variant}")
                inner_best=base.train_drcaf_classifier(
                    inner_train,inner_val,
                    inner_train_boxes,inner_val_boxes,
                    args,device,sdir
                )
                selected_epoch=int(inner_best["epoch"])

                print(f"[F{fold}] refit — {variant} | epoch={selected_epoch}")
                clf=base.refit_classifier_fixed_epoch(
                    outer_train,refit_boxes,selected_epoch,
                    args,device,
                    seed=args.outer_seed*7000+fold,
                    outdir=rdir,
                )

                print(f"[F{fold}] outer-val — cached image-only boxes")
                srows,prows,sm,pm=base.evaluate_outer(
                    clf,outer_val,outer_val_boxes,args,device,fold
                )
                pd.DataFrame(srows).to_csv(fdir/"outer_slice_predictions.csv",index=False)
                pd.DataFrame(prows).to_csv(fdir/"outer_patient_predictions.csv",index=False)
                all_slice.extend(srows); all_pat.extend(prows)
                folds.append({
                    "fold":fold,
                    "selected_epoch":selected_epoch,
                    "inner_patient_acc":inner_best["patient"]["accuracy"],
                    "inner_patient_f1":inner_best["patient"]["macro_f1"],
                    "outer_patient_acc":pm["accuracy"],
                    "outer_patient_f1":pm["macro_f1"],
                    "outer_patient_errors":pm["errors"],
                    "outer_glioma_recall":pm["glioma_recall"],
                    "outer_meningioma_recall":pm["meningioma_recall"],
                    "outer_pituitary_recall":pm["pituitary_recall"],
                    "outer_slice_acc":sm["accuracy"],
                    "outer_slice_f1":sm["macro_f1"],
                })
                print(
                    f"[F{fold}] RESULT | Acc={pm['accuracy']:.3f}% "
                    f"F1={pm['macro_f1']:.3f}% err={pm['errors']}"
                )
                del clf
                if device.type=="cuda":
                    torch.cuda.empty_cache()

            pdf=pd.DataFrame(all_pat)
            sdf=pd.DataFrame(all_slice)
            pdf.to_csv(vroot/"patient_oof.csv",index=False)
            sdf.to_csv(vroot/"slice_oof.csv",index=False)
            pd.DataFrame(folds).to_csv(vroot/"fold_summary.csv",index=False)
            vm=metrics(pdf)
            cmp=pair_compare(
                full_df,pdf,args.bootstrap,args.outer_seed+10000+len(variant_summaries)
            )
            variant_summaries[variant]={
                "metrics":vm,
                "full_drcaf_minus_or_ablation_direction": "candidate ablation minus full D-RCAF",
                "paired_vs_full_drcaf":cmp,
                "folds":folds,
            }
            save_json(variant_summaries[variant],vroot/"summary.json")

            print(
                f"[FINAL {variant}] Acc={vm['accuracy']:.3f}% "
                f"F1={vm['macro_f1']:.3f}% err={vm['errors']} | "
                f"vs full ΔAcc={cmp['delta_accuracy_pp']:+.3f} pp "
                f"rescue/lost={cmp['rescues']}/{cmp['losses']} "
                f"p={cmp['exact_mcnemar_p']:.6g}"
            )

    finally:
        base.DRCAFSharedSwin = original_class

    full_m=metrics(full_df)
    rows=[{
        "variant":"full_existing",
        **full_m,
        "delta_accuracy_pp_vs_full":0.0,
        "delta_macro_f1_pp_vs_full":0.0,
        "rescues_vs_full":0,
        "losses_vs_full":0,
        "mcnemar_p_vs_full":1.0,
    }]
    for v,s in variant_summaries.items():
        c=s["paired_vs_full_drcaf"]
        rows.append({
            "variant":v,
            **s["metrics"],
            "delta_accuracy_pp_vs_full":c["delta_accuracy_pp"],
            "delta_macro_f1_pp_vs_full":c["delta_macro_f1_pp"],
            "rescues_vs_full":c["rescues"],
            "losses_vs_full":c["losses"],
            "mcnemar_p_vs_full":c["exact_mcnemar_p"],
        })
    summary_df=pd.DataFrame(rows)
    summary_df.to_csv(out_root/"component_ablation_summary.csv",index=False)

    payload={
        "scope":"post-lock descriptive mechanism analysis",
        "reference_full_drcaf_patient_oof":str(full_csv),
        "localizer_training_performed":False,
        "box_cache_reuse":"locked seed5813 D-RCAF Stage-A, refit OOF, and outer-val image-only boxes",
        "full_drcaf":full_m,
        "variants":variant_summaries,
    }
    save_json(payload,out_root/"component_ablation_summary.json")

    print("\n"+"="*156)
    print("FINAL D-RCAF COMPONENT ABLATION SUMMARY")
    print("="*156)
    print(summary_df.to_string(index=False))
    print("="*156)
    print(f"Outputs: {out_root}")


if __name__=="__main__":
    main()
