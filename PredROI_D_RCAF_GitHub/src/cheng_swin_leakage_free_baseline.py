#!/usr/bin/env python
# -*- coding: utf-8 -*-

r"""
Leakage-free STANDARD Swin-Base baseline.

Dependency (same folder):
    cheng_patient_set_leakage_free_v2.py

Protocol:
  outer patient-disjoint fold
    -> inner patient-disjoint epoch selection
    -> slice-level Swin training
    -> INNER patient prediction by mean slice softmax probability
    -> reinitialize
    -> refit on all outer-train slices for selected epoch
    -> evaluate untouched OUTER patients once

Primary metric unit: patient.
Outer validation labels are never used for checkpoint selection.

Example:
python cheng_swin_leakage_free_baseline.py ^
  --manifest "C:\PATH\TO\brain_project\cheng_patient_disjoint\cheng_manifest.csv" ^
  --out_root "C:\PATH\TO\brain_project\swin_baseline_seed2026_FINAL" ^
  --outer_seed 2026 ^
  --train_folds 5 ^
  --max_select_epochs 30 ^
  --freeze_epochs 3 ^
  --amp_dtype bf16 ^
  --compare_mean_set_csv "C:\PATH\TO\brain_project\patient_set_seed2026_FINAL\mean_set\patient_oof_neural.csv"
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, recall_score, confusion_matrix

import timm
import cheng_patient_set_leakage_free_v2 as common


CLASSES = common.CLASSES
N_CLASSES = common.N_CLASSES


def slice_rows_from_patients(patients):
    rows = []
    for p in patients:
        for q in p.paths:
            rows.append({
                "path": q,
                "patient_id": p.patient_id,
                "label": p.label,
            })
    return rows


class SliceDataset(Dataset):
    def __init__(self, rows, image_size, train, seed):
        self.rows = list(rows)
        self.image_size = int(image_size)
        self.train = bool(train)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        rng = np.random.default_rng(self.seed + self.epoch * 100003 + idx * 997)
        x = common.load_gray_preprocessed(
            r["path"], self.image_size, self.train, rng
        )
        return {
            "image": x,
            "label": torch.tensor(int(r["label"]), dtype=torch.long),
            "patient_id": str(r["patient_id"]),
            "path": str(r["path"]),
        }


class SwinSliceClassifier(nn.Module):
    def __init__(self, backbone, pretrained=True, dropout=0.15):
        super().__init__()
        self.encoder = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        dim = int(self.encoder.num_features)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(dim, N_CLASSES)

    def forward(self, x):
        f = self.encoder(x)
        return self.classifier(self.dropout(f))


def make_optimizer(model, backbone_lr, head_lr, weight_decay):
    # Keep encoder in optimizer even during frozen warm-up.
    bb, head = [], []
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


def set_encoder_trainable(model, trainable):
    for p in model.encoder.parameters():
        p.requires_grad = bool(trainable)


def train_one_epoch(
    model, dataset, loader, optimizer, criterion, scaler, device,
    epoch, freeze_epochs, accumulation_steps, grad_clip, amp_dtype
):
    dataset.set_epoch(epoch)
    set_encoder_trainable(model, epoch > freeze_epochs)
    model.train()
    optimizer.zero_grad(set_to_none=True)

    losses, correct, n = [], 0, 0

    for step, batch in enumerate(loader, start=1):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        with common.autocast_ctx(device, amp_dtype):
            logits = model(x)
            loss = criterion(logits, y)
            scaled_loss = loss / accumulation_steps

        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss epoch={epoch}, step={step}: "
                f"{float(loss.detach().cpu())}"
            )

        scaler.scale(scaled_loss).backward()

        if step % accumulation_steps == 0 or step == len(loader):
            scaler.unscale_(optimizer)
            grad_params = [
                p for p in model.parameters()
                if p.requires_grad and p.grad is not None
            ]
            gnorm = torch.nn.utils.clip_grad_norm_(grad_params, grad_clip)

            if not torch.isfinite(gnorm):
                raise FloatingPointError(
                    f"Non-finite grad norm epoch={epoch}, step={step}"
                )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        losses.append(float(loss.detach().cpu()))
        correct += int((logits.argmax(1) == y).sum().item())
        n += int(y.numel())

    return float(np.mean(losses)), 100.0 * correct / max(n, 1)


@torch.no_grad()
def infer_slices(model, loader, device, amp_dtype):
    model.eval()
    out = []

    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].cpu().numpy()

        with common.autocast_ctx(device, amp_dtype):
            prob = F.softmax(model(x), dim=-1).float().cpu().numpy()

        pred = prob.argmax(1)

        for i in range(len(y)):
            r = {
                "patient_id": str(batch["patient_id"][i]),
                "path": str(batch["path"][i]),
                "true_id": int(y[i]),
                "slice_pred_id": int(pred[i]),
            }
            for c, cname in enumerate(CLASSES):
                r[f"prob_{cname}"] = float(prob[i, c])
            out.append(r)

    return out


def aggregate_patient_probs(slice_rows):
    by_pid = defaultdict(list)
    for r in slice_rows:
        by_pid[str(r["patient_id"])].append(r)

    out = []
    for pid, rr in sorted(by_pid.items()):
        ys = {int(x["true_id"]) for x in rr}
        if len(ys) != 1:
            raise RuntimeError(f"Inconsistent labels for patient {pid}")
        y = next(iter(ys))

        P = np.asarray(
            [[x[f"prob_{c}"] for c in CLASSES] for x in rr],
            dtype=np.float64,
        )
        p = P.mean(0)
        pred = int(np.argmax(p))

        rec = {
            "patient_id": pid,
            "true_id": y,
            "pred_id": pred,
            "correct": int(pred == y),
            "n_slices": len(rr),
        }
        for c, cname in enumerate(CLASSES):
            rec[f"mean_prob_{cname}"] = float(p[c])
        out.append(rec)

    return out


def patient_metrics(rows):
    y = np.asarray([r["true_id"] for r in rows], dtype=int)
    p = np.asarray([r["pred_id"] for r in rows], dtype=int)
    rec = recall_score(
        y, p, labels=list(range(N_CLASSES)),
        average=None, zero_division=0
    )
    return {
        "n": int(len(y)),
        "accuracy": float(100 * accuracy_score(y, p)),
        "macro_f1": float(
            100 * f1_score(
                y, p, labels=list(range(N_CLASSES)),
                average="macro", zero_division=0
            )
        ),
        "glioma_recall": float(100 * rec[0]),
        "errors": int(np.sum(y != p)),
        "confusion_matrix": confusion_matrix(
            y, p, labels=list(range(N_CLASSES))
        ).tolist(),
    }


def slice_metrics(rows):
    y = np.asarray([r["true_id"] for r in rows], dtype=int)
    p = np.asarray([r["slice_pred_id"] for r in rows], dtype=int)
    return {
        "n": int(len(y)),
        "accuracy": float(100 * accuracy_score(y, p)),
        "macro_f1": float(
            100 * f1_score(y, p, average="macro", zero_division=0)
        ),
        "errors": int(np.sum(y != p)),
    }


def build_model_and_training(patients, args, device, seed, select_horizon):
    common.seed_all(seed, args.deterministic)

    model = SwinSliceClassifier(
        args.backbone,
        pretrained=not args.no_pretrained,
        dropout=args.dropout,
    ).to(device)

    rows = slice_rows_from_patients(patients)
    ds = SliceDataset(rows, args.image_size, True, seed)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    cw = common.patient_class_weights(patients, device)
    criterion = common.FocalLoss(
        cw,
        gamma=args.focal_gamma,
        label_smoothing=args.label_smoothing,
    )

    set_encoder_trainable(model, False)
    opt = make_optimizer(
        model, args.backbone_lr, args.head_lr, args.weight_decay
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=(device.type == "cuda" and args.amp_dtype == "fp16"),
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=max(1, select_horizon),
        eta_min=args.minimum_lr,
    )

    return model, ds, loader, criterion, opt, scaler, sched


def select_epoch(inner_train, inner_val, args, device, seed, outdir):
    model, tr_ds, tr_loader, criterion, opt, scaler, sched = \
        build_model_and_training(
            inner_train, args, device, seed, args.max_select_epochs
        )

    val_rows = slice_rows_from_patients(inner_val)
    va_ds = SliceDataset(val_rows, args.image_size, False, seed + 10000)
    va_loader = DataLoader(
        va_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    history, best = [], None

    for epoch in range(1, args.max_select_epochs + 1):
        loss, tracc = train_one_epoch(
            model, tr_ds, tr_loader, opt, criterion, scaler, device,
            epoch, args.freeze_epochs, args.accumulation_steps,
            args.grad_clip, args.amp_dtype
        )

        srows = infer_slices(model, va_loader, device, args.amp_dtype)
        prows = aggregate_patient_probs(srows)
        sm, pm = slice_metrics(srows), patient_metrics(prows)

        history.append({
            "epoch": epoch,
            "train_loss": loss,
            "train_slice_acc": tracc,
            "inner_patient_acc": pm["accuracy"],
            "inner_patient_f1": pm["macro_f1"],
            "inner_patient_errors": pm["errors"],
            "inner_slice_acc": sm["accuracy"],
            "inner_slice_f1": sm["macro_f1"],
        })

        print(
            f"      epoch {epoch:02d}/{args.max_select_epochs} | "
            f"loss={loss:.4f} train-slice={tracc:6.2f}% | "
            f"INNER patient Acc={pm['accuracy']:6.2f}% "
            f"F1={pm['macro_f1']:6.2f}% err={pm['errors']} | "
            f"slice Acc={sm['accuracy']:6.2f}%"
        )

        # Frozen warm-up epochs cannot be selected.
        if epoch > args.freeze_epochs:
            key = (pm["macro_f1"], pm["accuracy"], -epoch)
            if best is None or key > best["key"]:
                best = {"key": key, "epoch": epoch, "patient": pm}

        sched.step()

    if best is None:
        raise RuntimeError("No selectable epoch.")

    pd.DataFrame(history).to_csv(
        outdir / "inner_selection_history.csv", index=False
    )

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return int(best["epoch"]), best["patient"]


def refit(outer_train, selected_epoch, args, device, seed):
    model, ds, loader, criterion, opt, scaler, sched = \
        build_model_and_training(
            outer_train, args, device, seed, selected_epoch
        )

    for epoch in range(1, selected_epoch + 1):
        loss, acc = train_one_epoch(
            model, ds, loader, opt, criterion, scaler, device,
            epoch, args.freeze_epochs, args.accumulation_steps,
            args.grad_clip, args.amp_dtype
        )
        sched.step()
        print(
            f"      REFIT epoch {epoch:02d}/{selected_epoch} | "
            f"loss={loss:.4f} slice-acc={acc:6.2f}%"
        )

    return model


def paired_compare(baseline_csv, mean_set_csv, n_boot, seed):
    b = pd.read_csv(baseline_csv)
    m = pd.read_csv(mean_set_csv)

    b = b[["patient_id", "true_id", "pred_id"]].rename(
        columns={"pred_id": "base_pred"}
    )
    m = m[["patient_id", "true_id", "pred_id"]].rename(
        columns={"true_id": "true2", "pred_id": "mean_pred"}
    )

    z = b.merge(m, on="patient_id", how="inner")
    if len(z) != len(b) or len(z) != len(m):
        raise RuntimeError(
            f"Patient mismatch: baseline={len(b)}, mean_set={len(m)}, merged={len(z)}"
        )
    if not np.array_equal(z["true_id"].to_numpy(), z["true2"].to_numpy()):
        raise RuntimeError("True labels differ between paired files.")

    y = z["true_id"].to_numpy(int)
    pb = z["base_pred"].to_numpy(int)
    pm = z["mean_pred"].to_numpy(int)

    rescue, lost, p = common.exact_mcnemar(pb == y, pm == y)
    boot = common.paired_bootstrap(y, pb, pm, n_boot=n_boot, seed=seed)

    return {
        "n": int(len(y)),
        "baseline_acc": float(100 * accuracy_score(y, pb)),
        "baseline_f1": float(
            100 * f1_score(y, pb, average="macro", zero_division=0)
        ),
        "mean_set_acc": float(100 * accuracy_score(y, pm)),
        "mean_set_f1": float(
            100 * f1_score(y, pm, average="macro", zero_division=0)
        ),
        "mean_set_rescue": rescue,
        "mean_set_lost": lost,
        "mcnemar_p": p,
        "bootstrap": boot,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out_root", required=True)

    ap.add_argument("--backbone", default="swin_base_patch4_window7_224")
    ap.add_argument("--no_pretrained", action="store_true")

    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--train_folds", type=int, default=5)
    ap.add_argument("--outer_seed", type=int, default=2026)

    ap.add_argument("--inner_n_splits", type=int, default=5)
    ap.add_argument("--inner_seed", type=int, default=31415)

    ap.add_argument("--max_select_epochs", type=int, default=30)
    ap.add_argument("--freeze_epochs", type=int, default=3)

    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--eval_batch_size", type=int, default=32)
    ap.add_argument("--accumulation_steps", type=int, default=2)
    ap.add_argument("--workers", type=int, default=0)

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

    ap.add_argument("--compare_mean_set_csv", default="")
    args = ap.parse_args()

    if args.max_select_epochs <= args.freeze_epochs:
        raise ValueError("--max_select_epochs must exceed --freeze_epochs.")
    if args.train_folds > args.n_folds:
        raise ValueError("--train_folds cannot exceed --n_folds.")

    common.seed_all(args.outer_seed, args.deterministic)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 128)
    print("LEAKAGE-FREE STANDARD SWIN SLICE BASELINE")
    print("=" * 128)
    print(f"device            : {device}")
    print(f"outer seed        : {args.outer_seed}")
    print(f"backbone          : {args.backbone}")
    print("training unit     : slice")
    print("selection metric  : INNER patient macro-F1")
    print("patient inference : mean slice softmax probability")
    print("outer validation  : never used for checkpoint selection")
    print("=" * 128)

    slice_rows, _ = common.load_manifest(Path(args.manifest))
    splits = common.outer_patient_splits(
        slice_rows, args.n_folds, args.outer_seed
    )

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    all_pat, all_slice, folds = [], [], []

    for fold, (tr_idx, va_idx) in enumerate(splits, start=1):
        if fold > args.train_folds:
            continue

        print("\n" + "=" * 128)
        print(f"OUTER FOLD {fold}")
        print("=" * 128)

        outer_train = common.patient_subset_from_slice_indices(slice_rows, tr_idx)
        outer_val = common.patient_subset_from_slice_indices(slice_rows, va_idx)

        if {p.patient_id for p in outer_train} & {p.patient_id for p in outer_val}:
            raise RuntimeError("Outer patient leakage.")

        inner_train, inner_val = common.make_inner_selection_split(
            outer_train, args.inner_n_splits, args.inner_seed + fold
        )

        print(
            f"outer train={len(outer_train)} | outer val={len(outer_val)} | "
            f"inner train={len(inner_train)} | inner val={len(inner_val)}"
        )

        fdir = out_root / f"fold_{fold}"
        fdir.mkdir(parents=True, exist_ok=True)

        selected_epoch, inner_best = select_epoch(
            inner_train, inner_val, args, device,
            seed=args.outer_seed * 100 + fold,
            outdir=fdir,
        )

        print(
            f"[SELECTED] fold={fold} epoch={selected_epoch} "
            f"from INNER patient val only | "
            f"Acc={inner_best['accuracy']:.3f}% "
            f"F1={inner_best['macro_f1']:.3f}%"
        )

        model = refit(
            outer_train, selected_epoch, args, device,
            seed=args.outer_seed * 1000 + fold
        )

        torch.save(
            {
                "state_dict": model.state_dict(),
                "outer_seed": args.outer_seed,
                "outer_fold": fold,
                "selected_epoch": selected_epoch,
                "backbone": args.backbone,
            },
            fdir / "final_refit.pt",
        )

        vrows = slice_rows_from_patients(outer_val)
        vds = SliceDataset(
            vrows, args.image_size, False,
            args.outer_seed * 1000 + fold + 1
        )
        vloader = DataLoader(
            vds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=(device.type == "cuda"),
        )

        srows = infer_slices(model, vloader, device, args.amp_dtype)
        prows = aggregate_patient_probs(srows)

        for r in srows:
            r["fold"] = fold
        for r in prows:
            r["fold"] = fold

        sm, pm = slice_metrics(srows), patient_metrics(prows)
        all_slice.extend(srows)
        all_pat.extend(prows)

        folds.append({
            "fold": fold,
            "selected_epoch": selected_epoch,
            "outer_patient_acc": pm["accuracy"],
            "outer_patient_f1": pm["macro_f1"],
            "outer_patient_errors": pm["errors"],
            "outer_slice_acc": sm["accuracy"],
            "outer_slice_f1": sm["macro_f1"],
        })

        print(
            f"[OUTER] patient Acc={pm['accuracy']:.3f}% "
            f"F1={pm['macro_f1']:.3f}% err={pm['errors']} | "
            f"slice Acc={sm['accuracy']:.3f}%"
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    patient_csv = out_root / "patient_oof.csv"
    pd.DataFrame(all_pat).to_csv(patient_csv, index=False)
    pd.DataFrame(all_slice).to_csv(out_root / "slice_oof.csv", index=False)
    pd.DataFrame(folds).to_csv(out_root / "fold_summary.csv", index=False)

    pm = patient_metrics(all_pat)
    sm = slice_metrics(all_slice)

    summary = {
        "patient_oof": pm,
        "slice_oof": sm,
        "folds": folds,
    }

    print("\n" + "=" * 128)
    print("FINAL LEAKAGE-FREE STANDARD SWIN BASELINE")
    print("=" * 128)
    print(
        f"patient avg-prob | Acc={pm['accuracy']:.3f}% | "
        f"F1={pm['macro_f1']:.3f}% | "
        f"G-R={pm['glioma_recall']:.3f}% | err={pm['errors']}"
    )
    print(
        f"slice-level      | Acc={sm['accuracy']:.3f}% | "
        f"F1={sm['macro_f1']:.3f}% | err={sm['errors']}"
    )

    if args.compare_mean_set_csv:
        cmp = paired_compare(
            patient_csv,
            Path(args.compare_mean_set_csv),
            args.bootstrap,
            args.outer_seed + 77,
        )
        summary["paired_mean_set_vs_baseline"] = cmp
        b = cmp["bootstrap"]

        print("\n" + "=" * 128)
        print("PAIRED mean_set vs STANDARD SWIN BASELINE")
        print("=" * 128)
        print(
            f"baseline | Acc={cmp['baseline_acc']:.3f}% | "
            f"F1={cmp['baseline_f1']:.3f}%"
        )
        print(
            f"mean_set | Acc={cmp['mean_set_acc']:.3f}% | "
            f"F1={cmp['mean_set_f1']:.3f}%"
        )
        print(
            f"mean_set - baseline | "
            f"ΔAcc={b['delta_acc_pp']:+.3f} pp "
            f"[{b['acc_ci95_pp'][0]:+.3f}, {b['acc_ci95_pp'][1]:+.3f}] | "
            f"ΔF1={b['delta_f1_pp']:+.3f} pp "
            f"[{b['f1_ci95_pp'][0]:+.3f}, {b['f1_ci95_pp'][1]:+.3f}]"
        )
        print(
            f"McNemar p={cmp['mcnemar_p']:.6f} | "
            f"mean_set rescue/lost={cmp['mean_set_rescue']}/{cmp['mean_set_lost']}"
        )

    common.save_json(summary, out_root / "summary.json")
    print(f"\nWritten: {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
