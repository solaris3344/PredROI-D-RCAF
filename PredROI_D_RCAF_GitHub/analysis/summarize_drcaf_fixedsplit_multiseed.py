#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, recall_score


CLASS_NAMES = ["glioma", "meningioma", "pituitary"]


def load_patient_oof(run_dir: Path):
    p = run_dir / "patient_oof.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing patient_oof.csv: {p}")
    df = pd.read_csv(p)
    required = ["patient_id", "true_id", "pred_id"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise RuntimeError(f"{p} missing columns: {missing}")
    df["patient_id"] = df["patient_id"].astype(str)
    return df


def metrics(df: pd.DataFrame):
    y = df["true_id"].astype(int).to_numpy()
    p = df["pred_id"].astype(int).to_numpy()
    labels = sorted(np.unique(y).tolist())
    rec = recall_score(y, p, labels=labels, average=None, zero_division=0)

    out = {
        "n_patients": int(len(df)),
        "accuracy": float(100.0 * accuracy_score(y, p)),
        "macro_f1": float(
            100.0 * f1_score(y, p, labels=labels, average="macro", zero_division=0)
        ),
        "errors": int(np.sum(y != p)),
    }
    for i, lab in enumerate(labels):
        name = CLASS_NAMES[lab] if 0 <= lab < len(CLASS_NAMES) else f"class_{lab}"
        out[f"recall_{name}"] = float(100.0 * rec[i])
    return out


def mapping(df: pd.DataFrame):
    cols = ["patient_id", "true_id"]
    if "fold" in df.columns:
        cols.append("fold")
    return df[cols].sort_values("patient_id").reset_index(drop=True)


def assert_same_patient_partition(ref: pd.DataFrame, cur: pd.DataFrame, ref_name: str, cur_name: str):
    a = mapping(ref)
    b = mapping(cur)

    if list(a["patient_id"]) != list(b["patient_id"]):
        raise RuntimeError(
            f"Patient-ID set/order mismatch between {ref_name} and {cur_name}."
        )
    if not np.array_equal(a["true_id"].to_numpy(), b["true_id"].to_numpy()):
        raise RuntimeError(
            f"Patient label mismatch between {ref_name} and {cur_name}."
        )

    if "fold" in a.columns and "fold" in b.columns:
        if not np.array_equal(a["fold"].to_numpy(), b["fold"].to_numpy()):
            diff = a.loc[a["fold"].to_numpy() != b["fold"].to_numpy()].head(10)
            raise RuntimeError(
                f"FIXED-SPLIT AUDIT FAILED: fold membership differs between "
                f"{ref_name} and {cur_name}. Examples:\n{diff}"
            )
        return "PASS: identical patient IDs, labels, and fold membership"

    return (
        "PARTIAL PASS: identical patient IDs and labels; at least one legacy "
        "patient_oof.csv lacks a fold column, so exact fold membership could "
        "not be re-audited from that file."
    )


def sample_sd(vals):
    vals = np.asarray(vals, dtype=float)
    if vals.size <= 1:
        return 0.0
    return float(np.std(vals, ddof=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--original", required=True)
    ap.add_argument("--fresh", nargs="+", required=True)
    ap.add_argument("--out_dir", default="drcaf_multiseed_fixedsplit_summary")
    args = ap.parse_args()

    root = Path(args.root)
    run_names = [args.original] + list(args.fresh)
    run_dirs = [root / n for n in run_names]

    frames = []
    rows = []

    for idx, (name, run_dir) in enumerate(zip(run_names, run_dirs)):
        df = load_patient_oof(run_dir)
        frames.append(df)
        m = metrics(df)

        if idx == 0:
            run_seed = 5813
            role = "original_locked"
        else:
            # Parse final integer from folder name.
            digits = "".join(ch if ch.isdigit() else " " for ch in name).split()
            run_seed = int(digits[-1]) if digits else -1
            role = "fresh_fixedsplit_repeat"

        rows.append({
            "run": name,
            "role": role,
            "split_seed": 5813,
            "run_seed": run_seed,
            **m,
        })

    audit_notes = []
    ref = frames[0]
    for name, df in zip(run_names[1:], frames[1:]):
        audit_notes.append({
            "comparison": f"{run_names[0]} vs {name}",
            "result": assert_same_patient_partition(
                ref, df, run_names[0], name
            ),
        })

    out_df = pd.DataFrame(rows)

    metric_cols = [
        "accuracy",
        "macro_f1",
        "errors",
        "recall_glioma",
        "recall_meningioma",
        "recall_pituitary",
    ]
    aggregate = {}
    for c in metric_cols:
        if c not in out_df.columns:
            continue
        vals = out_df[c].astype(float).to_numpy()
        aggregate[c] = {
            "mean": float(np.mean(vals)),
            "sd_sample": sample_sd(vals),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }

    out_dir = root / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_df.to_csv(out_dir / "multiseed_runs.csv", index=False)

    payload = {
        "protocol": {
            "fixed_patient_split_seed": 5813,
            "runs": run_names,
            "n_runs": len(run_names),
            "interpretation": (
                "Patient split membership is held fixed; stochastic pipeline "
                "training seed varies. This estimates seed sensitivity without "
                "confounding it with split-policy variation."
            ),
        },
        "split_audit": audit_notes,
        "aggregate": aggregate,
        "runs": rows,
    }
    (out_dir / "multiseed_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("=" * 120)
    print("D-RCAF FIXED-PATIENT-SPLIT MULTI-SEED ROBUSTNESS")
    print("=" * 120)
    for r in rows:
        print(
            f"run_seed={r['run_seed']:5d} | "
            f"Acc={r['accuracy']:.3f}% | "
            f"F1={r['macro_f1']:.3f}% | "
            f"err={r['errors']:2d} | "
            f"G-R={r.get('recall_glioma', float('nan')):.3f}% | "
            f"M-R={r.get('recall_meningioma', float('nan')):.3f}% | "
            f"P-R={r.get('recall_pituitary', float('nan')):.3f}%"
        )

    print("-" * 120)
    for a in audit_notes:
        print(f"SPLIT AUDIT | {a['comparison']} | {a['result']}")

    print("-" * 120)
    for c in ("accuracy", "macro_f1", "errors"):
        a = aggregate[c]
        unit = "%" if c != "errors" else ""
        print(
            f"{c:10s} | mean={a['mean']:.3f}{unit} "
            f"| SD={a['sd_sample']:.3f}{unit} "
            f"| range=[{a['min']:.3f}, {a['max']:.3f}]{unit}"
        )
    print("=" * 120)
    print(f"Saved: {out_dir / 'multiseed_runs.csv'}")
    print(f"Saved: {out_dir / 'multiseed_summary.json'}")


if __name__ == "__main__":
    main()
