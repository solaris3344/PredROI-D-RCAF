#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
compare_drcaf_predroi_paired.py

Patient-level paired comparison for the SAME locked outer split.
Compares PredROI-Swin vs D-RCAF using patient_oof.csv files.
Reports exact McNemar and paired patient bootstrap CIs for Accuracy and Macro-F1.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score
from scipy.stats import binomtest

CLASSES = ["glioma", "meningioma", "pituitary"]


def load_patient_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"patient_id": str})
    req = {"patient_id", "true_id", "pred_id"}
    miss = req - set(df.columns)
    if miss:
        raise RuntimeError(f"{path} missing columns: {sorted(miss)}")
    if df["patient_id"].duplicated().any():
        raise RuntimeError(f"Duplicate patient_id in {path}")
    out = df[["patient_id", "true_id", "pred_id"]].copy()
    out["patient_id"] = out["patient_id"].astype(str)
    out["true_id"] = out["true_id"].astype(int)
    out["pred_id"] = out["pred_id"].astype(int)
    return out


def metrics(y, p):
    rec = recall_score(y, p, labels=[0,1,2], average=None, zero_division=0)
    return {
        "accuracy": 100.0 * accuracy_score(y, p),
        "macro_f1": 100.0 * f1_score(y, p, labels=[0,1,2], average="macro", zero_division=0),
        "glioma_recall": 100.0 * rec[0],
        "meningioma_recall": 100.0 * rec[1],
        "pituitary_recall": 100.0 * rec[2],
        "errors": int(np.sum(np.asarray(y) != np.asarray(p))),
    }


def exact_mcnemar(y, p_base, p_cand):
    b_ok = p_base == y
    c_ok = p_cand == y
    rescue = int(np.sum((~b_ok) & c_ok))
    lost = int(np.sum(b_ok & (~c_ok)))
    discordant = rescue + lost
    p = 1.0 if discordant == 0 else float(binomtest(min(rescue, lost), discordant, 0.5, alternative="two-sided").pvalue)
    return rescue, lost, p


def paired_bootstrap(y, p_base, p_cand, n_boot=20000, seed=5813):
    y = np.asarray(y, dtype=int)
    pb = np.asarray(p_base, dtype=int)
    pc = np.asarray(p_cand, dtype=int)
    n = len(y)
    rng = np.random.default_rng(seed)
    da = np.empty(n_boot, dtype=np.float64)
    df1 = np.empty(n_boot, dtype=np.float64)
    dr = np.empty((n_boot, 3), dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy, bb, cc = y[idx], pb[idx], pc[idx]
        da[i] = 100.0 * (accuracy_score(yy, cc) - accuracy_score(yy, bb))
        df1[i] = 100.0 * (
            f1_score(yy, cc, labels=[0,1,2], average="macro", zero_division=0)
            - f1_score(yy, bb, labels=[0,1,2], average="macro", zero_division=0)
        )
        rb = recall_score(yy, bb, labels=[0,1,2], average=None, zero_division=0)
        rc = recall_score(yy, cc, labels=[0,1,2], average=None, zero_division=0)
        dr[i] = 100.0 * (rc-rb)
    def s(x):
        return {
            "mean": float(np.mean(x)),
            "median": float(np.median(x)),
            "ci95_low": float(np.percentile(x, 2.5)),
            "ci95_high": float(np.percentile(x, 97.5)),
            "p_delta_le_0": float(np.mean(x <= 0.0)),
        }
    return {
        "delta_accuracy_pp": s(da),
        "delta_macro_f1_pp": s(df1),
        "delta_glioma_recall_pp": s(dr[:,0]),
        "delta_meningioma_recall_pp": s(dr[:,1]),
        "delta_pituitary_recall_pp": s(dr[:,2]),
    }


def valid_pair(base_path: Path, cand_df: pd.DataFrame):
    try:
        b = load_patient_csv(base_path)
    except Exception:
        return False
    if len(b) != len(cand_df):
        return False
    z = cand_df[["patient_id","true_id"]].merge(
        b[["patient_id","true_id"]], on="patient_id", how="inner", suffixes=("_c","_b")
    )
    if len(z) != len(cand_df):
        return False
    return np.array_equal(z["true_id_c"].to_numpy(int), z["true_id_b"].to_numpy(int))


def auto_find_predroi(root: Path, cand_path: Path, cand_df: pd.DataFrame):
    cands=[]
    for p in root.rglob("patient_oof.csv"):
        try:
            if p.resolve() == cand_path.resolve():
                continue
        except Exception:
            pass
        low=str(p).lower()
        if "fixed05" in low or "drcaf" in low:
            continue
        if not valid_pair(p, cand_df):
            continue
        score=0
        if "predroi" in low: score += 10
        if "full5" in low or "full_5" in low: score += 4
        if "5813" in low: score += 4
        if "locked" in low: score += 2
        cands.append((score,p))
    if not cands:
        raise FileNotFoundError(
            "Could not auto-find a patient-matched PredROI patient_oof.csv. "
            "Pass --predroi <full path>."
        )
    cands.sort(key=lambda x:(x[0], str(x[1])), reverse=True)
    top_score=cands[0][0]
    top=[p for s,p in cands if s==top_score]
    if len(top)>1:
        print("[AUTO] Multiple equally scored matched candidates; selecting first:")
        for p in top[:10]: print("   ",p)
    return top[0]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--drcaf", default="")
    ap.add_argument("--predroi", default="")
    ap.add_argument("--brain_root", default="")
    ap.add_argument("--outdir", default="")
    ap.add_argument("--bootstrap", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=5813)
    args=ap.parse_args()

    root=Path(args.brain_root) if args.brain_root else Path(__file__).resolve().parent
    cand_path=Path(args.drcaf) if args.drcaf else root / "drcaf_full5_seed5813" / "patient_oof.csv"
    if not cand_path.exists():
        hits=[p for p in root.rglob("patient_oof.csv") if "drcaf" in str(p).lower()]
        if len(hits)==1:
            cand_path=hits[0]
        elif not hits:
            raise FileNotFoundError(f"D-RCAF patient_oof.csv not found under {root}")
        else:
            hits.sort(key=lambda p:("5813" in str(p), p.stat().st_mtime), reverse=True)
            cand_path=hits[0]

    c=load_patient_csv(cand_path)
    base_path=Path(args.predroi) if args.predroi else auto_find_predroi(root, cand_path, c)
    b=load_patient_csv(base_path)

    z=b.rename(columns={"true_id":"true_b","pred_id":"pred_predroi"}).merge(
        c.rename(columns={"true_id":"true_c","pred_id":"pred_drcaf"}),
        on="patient_id", how="inner", validate="one_to_one"
    )
    if len(z)!=len(b) or len(z)!=len(c):
        raise RuntimeError(f"Patient coverage mismatch: PredROI={len(b)} D-RCAF={len(c)} matched={len(z)}")
    if not np.array_equal(z.true_b.to_numpy(int), z.true_c.to_numpy(int)):
        raise RuntimeError("True labels differ between paired files.")

    y=z.true_b.to_numpy(int)
    pb=z.pred_predroi.to_numpy(int)
    pc=z.pred_drcaf.to_numpy(int)
    mb=metrics(y,pb); mc=metrics(y,pc)
    rescue,lost,p=exact_mcnemar(y,pb,pc)
    boot=paired_bootstrap(y,pb,pc,args.bootstrap,args.seed+9187)

    summary={
        "n":int(len(y)),
        "predroi_csv":str(base_path),
        "drcaf_csv":str(cand_path),
        "predroi":mb,
        "drcaf":mc,
        "delta_accuracy_pp":mc["accuracy"]-mb["accuracy"],
        "delta_macro_f1_pp":mc["macro_f1"]-mb["macro_f1"],
        "error_reduction_count":mb["errors"]-mc["errors"],
        "relative_error_reduction_pct":100.0*(mb["errors"]-mc["errors"])/mb["errors"] if mb["errors"] else 0.0,
        "mcnemar":{"rescue":rescue,"lost":lost,"discordant":rescue+lost,"exact_p":p},
        "paired_patient_bootstrap":boot,
    }

    outdir=Path(args.outdir) if args.outdir else root / "drcaf_vs_predroi_paired_seed5813"
    outdir.mkdir(parents=True,exist_ok=True)
    (outdir/"paired_summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding="utf-8")
    z["predroi_correct"]=(pb==y).astype(int)
    z["drcaf_correct"]=(pc==y).astype(int)
    z["status"]=np.where((pb!=y)&(pc==y),"rescued_by_drcaf",np.where((pb==y)&(pc!=y),"lost_by_drcaf",np.where((pb==y)&(pc==y),"both_correct","both_wrong")))
    z.to_csv(outdir/"paired_patients.csv",index=False)
    rows=[]
    for k,v in boot.items(): rows.append({"metric":k,**v})
    pd.DataFrame(rows).to_csv(outdir/"bootstrap_ci.csv",index=False)

    print("="*120)
    print("PAIRED PATIENT-LEVEL: PredROI-Swin vs D-RCAF")
    print("="*120)
    print(f"PredROI : Acc={mb['accuracy']:.3f}% | F1={mb['macro_f1']:.3f}% | err={mb['errors']}")
    print(f"D-RCAF  : Acc={mc['accuracy']:.3f}% | F1={mc['macro_f1']:.3f}% | err={mc['errors']}")
    print(f"Delta   : Acc={summary['delta_accuracy_pp']:+.3f} pp | F1={summary['delta_macro_f1_pp']:+.3f} pp")
    print(f"McNemar : rescue={rescue} | lost={lost} | exact p={p:.6g}")
    a=boot['delta_accuracy_pp']; f=boot['delta_macro_f1_pp']
    print(f"Bootstrap Acc delta 95% CI: [{a['ci95_low']:+.3f}, {a['ci95_high']:+.3f}] pp")
    print(f"Bootstrap F1  delta 95% CI: [{f['ci95_low']:+.3f}, {f['ci95_high']:+.3f}] pp")
    print(f"Outputs : {outdir}")
    print("="*120)

if __name__=="__main__":
    main()
