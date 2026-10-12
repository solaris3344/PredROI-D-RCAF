#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
reviewer_missing_paired_stats.py

Training-free reviewer-closure statistics.

A) Internal Cheng patient-level:
   Fixed-0.5 versus D-RCAF on the same patient_oof.csv.
   Reports:
     - rescue / lost / shared wrong
     - exact two-sided McNemar p
     - paired patient bootstrap 95% CIs for delta Accuracy and Macro-F1

B) External BDNeuro image-level:
   Fixed-0.5 versus reconstructed PredROI using the already-saved
   external_predictions_three_methods.csv.
   Reports:
     - rescue / lost / shared wrong
     - descriptive paired image bootstrap 95% CIs for delta Accuracy/Macro-F1
   IMPORTANT: external patient IDs are unavailable, so this is descriptive only.

No model is trained and no threshold is tuned.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score, recall_score


CLASSES = ["glioma", "meningioma", "pituitary"]


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def metrics(y, p):
    y = np.asarray(y, dtype=int)
    p = np.asarray(p, dtype=int)
    rec = recall_score(y, p, labels=[0, 1, 2], average=None, zero_division=0)
    return {
        "n": int(len(y)),
        "accuracy": float(100.0 * accuracy_score(y, p)),
        "macro_f1": float(
            100.0 * f1_score(y, p, labels=[0,1,2], average="macro", zero_division=0)
        ),
        "errors": int(np.sum(y != p)),
        "glioma_recall": float(100.0 * rec[0]),
        "meningioma_recall": float(100.0 * rec[1]),
        "pituitary_recall": float(100.0 * rec[2]),
    }


def paired_status(y, base, cand):
    y = np.asarray(y, int)
    b = np.asarray(base, int)
    c = np.asarray(cand, int)
    b_ok = b == y
    c_ok = c == y
    rescue = int(np.sum((~b_ok) & c_ok))
    lost = int(np.sum(b_ok & (~c_ok)))
    shared_wrong = int(np.sum((~b_ok) & (~c_ok)))
    both_correct = int(np.sum(b_ok & c_ok))
    n_dis = rescue + lost
    p = 1.0 if n_dis == 0 else float(
        binomtest(min(rescue, lost), n_dis, 0.5, alternative="two-sided").pvalue
    )
    return {
        "both_correct": both_correct,
        "rescues": rescue,
        "losses": lost,
        "shared_wrong": shared_wrong,
        "discordant": n_dis,
        "exact_mcnemar_p": p,
    }


def paired_bootstrap(y, base, cand, n_boot, seed):
    y = np.asarray(y, int)
    b = np.asarray(base, int)
    c = np.asarray(cand, int)
    n = len(y)
    rng = np.random.default_rng(seed)
    dacc = np.empty(n_boot, np.float64)
    df1 = np.empty(n_boot, np.float64)
    for k in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy, bb, cc = y[idx], b[idx], c[idx]
        dacc[k] = 100.0 * (np.mean(cc == yy) - np.mean(bb == yy))
        df1[k] = 100.0 * (
            f1_score(yy, cc, labels=[0,1,2], average="macro", zero_division=0)
            - f1_score(yy, bb, labels=[0,1,2], average="macro", zero_division=0)
        )
    return {
        "delta_accuracy_pp": float(
            100.0 * (accuracy_score(y, c) - accuracy_score(y, b))
        ),
        "delta_macro_f1_pp": float(
            100.0 * (
                f1_score(y, c, labels=[0,1,2], average="macro", zero_division=0)
                - f1_score(y, b, labels=[0,1,2], average="macro", zero_division=0)
            )
        ),
        "accuracy_delta_95ci": [float(x) for x in np.percentile(dacc, [2.5, 97.5])],
        "macro_f1_delta_95ci": [float(x) for x in np.percentile(df1, [2.5, 97.5])],
        "n_boot": int(n_boot),
    }


def load_patient_oof(path: Path):
    df = pd.read_csv(path, dtype={"patient_id": str})
    req = {"patient_id", "true_id", "pred_id"}
    miss = req - set(df.columns)
    if miss:
        raise RuntimeError(f"{path} missing columns: {sorted(miss)}")
    if df["patient_id"].duplicated().any():
        raise RuntimeError(f"Duplicate patient_id in {path}")
    return df


def find_root_with_file(brain_root: Path, filename: str, keyword: str):
    hits = []
    for p in brain_root.rglob(filename):
        low = str(p).lower()
        if keyword.lower() in low:
            hits.append(p.parent)
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise FileNotFoundError(
            f"Could not auto-find {filename!r} with keyword {keyword!r} under {brain_root}"
        )
    # Prefer seed5813 exact-looking folder.
    hits.sort(
        key=lambda p: ("5813" in str(p), keyword.lower() in p.name.lower(), p.stat().st_mtime),
        reverse=True,
    )
    print(f"[AUTO] multiple {keyword} candidates; using {hits[0]}")
    return hits[0]


def internal_fixed_vs_drcaf(fixed_csv: Path, drcaf_csv: Path, n_boot, seed):
    f = load_patient_oof(fixed_csv)
    d = load_patient_oof(drcaf_csv)

    z = f[["patient_id","true_id","pred_id"]].rename(
        columns={"true_id":"true_f","pred_id":"fixed_pred"}
    ).merge(
        d[["patient_id","true_id","pred_id"]].rename(
            columns={"true_id":"true_d","pred_id":"drcaf_pred"}
        ),
        on="patient_id",
        how="inner",
        validate="one_to_one",
    )
    if len(z) != len(f) or len(z) != len(d):
        raise RuntimeError(
            f"Internal patient coverage mismatch: Fixed={len(f)} D-RCAF={len(d)} matched={len(z)}"
        )
    if not np.array_equal(z["true_f"].to_numpy(int), z["true_d"].to_numpy(int)):
        raise RuntimeError("Internal true labels differ between paired files.")

    y = z["true_f"].to_numpy(int)
    pf = z["fixed_pred"].to_numpy(int)
    pdrc = z["drcaf_pred"].to_numpy(int)

    status = paired_status(y, pf, pdrc)
    boot = paired_bootstrap(y, pf, pdrc, n_boot, seed)

    z["fixed_correct"] = (pf == y).astype(int)
    z["drcaf_correct"] = (pdrc == y).astype(int)
    z["paired_status"] = np.where(
        (pf != y) & (pdrc == y), "rescued_by_drcaf",
        np.where(
            (pf == y) & (pdrc != y), "lost_by_drcaf",
            np.where((pf == y) & (pdrc == y), "both_correct", "both_wrong")
        )
    )

    return {
        "comparison": "D-RCAF minus Fixed-0.5",
        "fixed05": metrics(y, pf),
        "drcaf": metrics(y, pdrc),
        "paired": status,
        "bootstrap": boot,
        "warning": (
            "Post-lock architecture comparison on the same patient folds; "
            "interpret descriptively rather than as a preregistered confirmatory test."
        ),
    }, z


def external_fixed_vs_predroi(ref_csv: Path, n_boot, seed):
    df = pd.read_csv(ref_csv)
    req = {
        "path","label",
        "predroi_prob_glioma","predroi_prob_meningioma","predroi_prob_pituitary",
        "fixed05_prob_glioma","fixed05_prob_meningioma","fixed05_prob_pituitary",
    }
    miss = req - set(df.columns)
    if miss:
        raise RuntimeError(f"{ref_csv} missing columns: {sorted(miss)}")
    if df["path"].astype(str).duplicated().any():
        raise RuntimeError("Duplicate path in external reference CSV.")

    y = df["label"].to_numpy(int)
    predroi_p = df[
        ["predroi_prob_glioma","predroi_prob_meningioma","predroi_prob_pituitary"]
    ].to_numpy(float)
    fixed_p = df[
        ["fixed05_prob_glioma","fixed05_prob_meningioma","fixed05_prob_pituitary"]
    ].to_numpy(float)

    pp = predroi_p.argmax(1)
    pf = fixed_p.argmax(1)

    # Direction is PredROI minus Fixed, making a positive delta easy to interpret.
    status = paired_status(y, pf, pp)
    boot = paired_bootstrap(y, pf, pp, n_boot, seed)

    outdf = df[["path","label"]].copy()
    outdf["fixed05_pred_id"] = pf
    outdf["predroi_pred_id"] = pp
    outdf["fixed05_correct"] = (pf == y).astype(int)
    outdf["predroi_correct"] = (pp == y).astype(int)
    outdf["paired_status"] = np.where(
        (pf != y) & (pp == y), "rescued_by_predroi",
        np.where(
            (pf == y) & (pp != y), "lost_by_predroi",
            np.where((pf == y) & (pp == y), "both_correct", "both_wrong")
        )
    )

    return {
        "comparison": "PredROI minus Fixed-0.5",
        "fixed05": metrics(y, pf),
        "predroi": metrics(y, pp),
        "paired": status,
        "bootstrap": boot,
        "warning": (
            "DESCRIPTIVE IMAGE-level paired bootstrap only. BDNeuro patient IDs "
            "are unavailable, so within-patient dependence cannot be modeled."
        ),
    }, outdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--brain_root", default=r"C:\PATH\TO\brain_project")
    ap.add_argument("--fixed_patient_oof", default="")
    ap.add_argument("--drcaf_patient_oof", default="")
    ap.add_argument("--external_reference_csv", default="")
    ap.add_argument("--out_root", default="")
    ap.add_argument("--bootstrap", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20261011)
    args = ap.parse_args()

    brain = Path(args.brain_root)
    out = Path(args.out_root) if args.out_root else brain / "reviewer_closure_paired_stats"
    out.mkdir(parents=True, exist_ok=True)

    if args.fixed_patient_oof:
        fixed_csv = Path(args.fixed_patient_oof)
    else:
        preferred = brain / "fixed05_full5_seed5813" / "patient_oof.csv"
        fixed_csv = preferred if preferred.exists() else find_root_with_file(
            brain, "patient_oof.csv", "fixed05"
        ) / "patient_oof.csv"

    if args.drcaf_patient_oof:
        drcaf_csv = Path(args.drcaf_patient_oof)
    else:
        preferred = brain / "drcaf_full5_seed5813" / "patient_oof.csv"
        drcaf_csv = preferred if preferred.exists() else find_root_with_file(
            brain, "patient_oof.csv", "drcaf"
        ) / "patient_oof.csv"

    if args.external_reference_csv:
        ext_csv = Path(args.external_reference_csv)
    else:
        preferred = brain / "bdneuro_fixed05_external" / "external_predictions_three_methods.csv"
        if preferred.exists():
            ext_csv = preferred
        else:
            hits = list(brain.rglob("external_predictions_three_methods.csv"))
            if len(hits) != 1:
                raise FileNotFoundError(
                    "Could not uniquely auto-find external_predictions_three_methods.csv. "
                    "Pass --external_reference_csv."
                )
            ext_csv = hits[0]

    internal, iz = internal_fixed_vs_drcaf(
        fixed_csv, drcaf_csv, args.bootstrap, args.seed + 1
    )
    external, ez = external_fixed_vs_predroi(
        ext_csv, args.bootstrap, args.seed + 2
    )

    payload = {
        "internal_fixed05_vs_drcaf": internal,
        "external_fixed05_vs_predroi": external,
        "files": {
            "fixed_patient_oof": str(fixed_csv),
            "drcaf_patient_oof": str(drcaf_csv),
            "external_reference_csv": str(ext_csv),
        },
    }
    save_json(payload, out / "reviewer_missing_paired_stats.json")
    iz.to_csv(out / "internal_fixed05_vs_drcaf_patients.csv", index=False)
    ez.to_csv(out / "external_fixed05_vs_predroi_images.csv", index=False)

    rows = []
    for scope, block in [
        ("internal_patient", internal),
        ("external_image_descriptive", external),
    ]:
        b = block["bootstrap"]
        s = block["paired"]
        rows.append({
            "scope": scope,
            "comparison": block["comparison"],
            "delta_accuracy_pp": b["delta_accuracy_pp"],
            "acc_ci95_low": b["accuracy_delta_95ci"][0],
            "acc_ci95_high": b["accuracy_delta_95ci"][1],
            "delta_macro_f1_pp": b["delta_macro_f1_pp"],
            "f1_ci95_low": b["macro_f1_delta_95ci"][0],
            "f1_ci95_high": b["macro_f1_delta_95ci"][1],
            "rescues": s["rescues"],
            "losses": s["losses"],
            "shared_wrong": s["shared_wrong"],
            "mcnemar_p": s["exact_mcnemar_p"] if scope == "internal_patient" else np.nan,
        })
    pd.DataFrame(rows).to_csv(out / "reviewer_paired_stats_summary.csv", index=False)

    print("=" * 140)
    print("REVIEWER CLOSURE — PAIRED STATISTICS")
    print("=" * 140)
    s = internal["paired"]; b = internal["bootstrap"]
    print("[INTERNAL] D-RCAF minus Fixed-0.5")
    print(
        f"  Fixed: Acc={internal['fixed05']['accuracy']:.3f}% "
        f"F1={internal['fixed05']['macro_f1']:.3f}% err={internal['fixed05']['errors']}"
    )
    print(
        f"  D-RCAF: Acc={internal['drcaf']['accuracy']:.3f}% "
        f"F1={internal['drcaf']['macro_f1']:.3f}% err={internal['drcaf']['errors']}"
    )
    print(
        f"  rescue/lost={s['rescues']}/{s['losses']} | "
        f"exact McNemar p={s['exact_mcnemar_p']:.6g}"
    )
    print(
        f"  delta Acc={b['delta_accuracy_pp']:+.3f} pp "
        f"[{b['accuracy_delta_95ci'][0]:+.3f}, {b['accuracy_delta_95ci'][1]:+.3f}]"
    )
    print(
        f"  delta F1 ={b['delta_macro_f1_pp']:+.3f} pp "
        f"[{b['macro_f1_delta_95ci'][0]:+.3f}, {b['macro_f1_delta_95ci'][1]:+.3f}]"
    )

    s = external["paired"]; b = external["bootstrap"]
    print("-" * 140)
    print("[EXTERNAL IMAGE-LEVEL, DESCRIPTIVE] PredROI minus Fixed-0.5")
    print(
        f"  Fixed: Acc={external['fixed05']['accuracy']:.3f}% "
        f"F1={external['fixed05']['macro_f1']:.3f}% err={external['fixed05']['errors']}"
    )
    print(
        f"  PredROI: Acc={external['predroi']['accuracy']:.3f}% "
        f"F1={external['predroi']['macro_f1']:.3f}% err={external['predroi']['errors']}"
    )
    print(f"  rescue/lost={s['rescues']}/{s['losses']} | shared_wrong={s['shared_wrong']}")
    print(
        f"  delta Acc={b['delta_accuracy_pp']:+.3f} pp "
        f"[{b['accuracy_delta_95ci'][0]:+.3f}, {b['accuracy_delta_95ci'][1]:+.3f}]"
    )
    print(
        f"  delta F1 ={b['delta_macro_f1_pp']:+.3f} pp "
        f"[{b['macro_f1_delta_95ci'][0]:+.3f}, {b['macro_f1_delta_95ci'][1]:+.3f}]"
    )
    print("  WARNING: image-level descriptive interval; patient IDs unavailable.")
    print("=" * 140)
    print(f"Outputs: {out}")


if __name__ == "__main__":
    main()
