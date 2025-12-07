#!/usr/bin/env python
import os
import sys
import csv
import argparse

import numpy as np
import matplotlib.pyplot as plt


def load_csv(path: str):
    if not os.path.exists(path):
        print(f"[WARN] {path} not found, skipping.")
        return []
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    print(f"[INFO] Loaded {len(rows)} rows from {path}")
    return rows


def summarize_cls(rows):
    if not rows:
        return
    print("\n=== Classification Ablation (CLS) ===")
    for r in rows:
        print(
            f"{r['experiment']:>16s} | "
            f"AUC={float(r['auc_macro']):.4f} | "
            f"ACC={float(r['accuracy']):.4f} | "
            f"F1={float(r['f1_macro']):.4f}"
        )

    names = [r["experiment"] for r in rows]
    aucs = np.array([float(r["auc_macro"]) for r in rows])
    accs = np.array([float(r["accuracy"]) for r in rows])
    f1s = np.array([float(r["f1_macro"]) for r in rows])

    x = np.arange(len(names))
    width = 0.25

    plt.figure(figsize=(8, 5))
    plt.bar(x - width, aucs, width, label="AUC (macro)")
    plt.bar(x, accs, width, label="Accuracy")
    plt.bar(x + width, f1s, width, label="F1 (macro)")
    plt.xticks(x, names, rotation=20)
    plt.ylabel("Score")
    plt.title("Classification Ablation Summary")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join("outputs", "ablation_cls_summary.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved classification summary plot to {out_path}")


def summarize_sr(rows):
    if not rows:
        return
    print("\n=== Super-Resolution Ablation (SR) ===")
    for r in rows:
        print(
            f"{r['experiment']:>16s} | "
            f"MSE={float(r['mse']):.6f} | "
            f"PSNR={float(r['psnr']):.4f} | "
            f"SSIM={float(r['ssim']):.4f}"
        )

    names = [r["experiment"] for r in rows]
    mse = np.array([float(r["mse"]) for r in rows])
    psnr = np.array([float(r["psnr"]) for r in rows])
    ssim = np.array([float(r["ssim"]) for r in rows])

    x = np.arange(len(names))
    width = 0.25

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.bar(x - width, mse, width, label="MSE")
    ax1.set_ylabel("MSE")
    ax2 = ax1.twinx()
    ax2.bar(x, psnr, width, label="PSNR", alpha=0.7)
    ax2.bar(x + width, ssim, width, label="SSIM", alpha=0.7)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=20)
    ax1.set_title("Super-Resolution Ablation Summary")

    lines, labels = [], []
    for ax in [ax1, ax2]:
        for l in ax.get_legend_handles_labels()[0]:
            lines.append(l)
        for lab in ax.get_legend_handles_labels()[1]:
            labels.append(lab)
    ax1.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)

    plt.tight_layout()
    out_path = os.path.join("outputs", "ablation_sr_summary.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved SR summary plot to {out_path}")


def summarize_mask_ratio(rows):
    if not rows:
        return
    print("\n=== MAE Mask-Ratio Ablation ===")
    for r in rows:
        print(
            f"mask={float(r['mask_ratio']):.2f} | "
            f"MAE_loss={float(r['mae_final_loss']):.6f} | "
            f"CLS AUC={float(r['cls_auc_macro']):.4f} | "
            f"PSNR={float(r['sr_psnr']):.4f} | SSIM={float(r['sr_ssim']):.4f}"
        )

    mask = np.array([float(r["mask_ratio"]) for r in rows])
    mae_loss = np.array([float(r["mae_final_loss"]) for r in rows])
    cls_auc = np.array([float(r["cls_auc_macro"]) for r in rows])
    psnr = np.array([float(r["sr_psnr"]) for r in rows])
    ssim = np.array([float(r["sr_ssim"]) for r in rows])

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(mask, mae_loss, marker="o", label="MAE final loss")
    ax1.set_xlabel("Mask ratio")
    ax1.set_ylabel("MAE loss")

    ax2 = ax1.twinx()
    ax2.plot(mask, cls_auc, marker="s", label="CLS AUC (macro)")
    ax2.plot(mask, psnr, marker="^", label="SR PSNR")
    ax2.plot(mask, ssim, marker="d", label="SR SSIM")

    lines, labels = [], []
    for ax in [ax1, ax2]:
        h, l = ax.get_legend_handles_labels()
        lines.extend(h)
        labels.extend(l)
    ax1.legend(lines, labels, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)

    ax1.set_title("Mask-Ratio Ablation Summary")
    plt.tight_layout()
    out_path = os.path.join("outputs", "mask_ratio_ablation_summary.png")
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved mask-ratio summary plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Summarize and plot DeepLense ablation CSV results."
    )
    parser.add_argument("--results_dir", type=str, default="outputs",
                        help="Directory containing ablation_*.csv files.")

    # Notebook-safe
    if hasattr(sys, "argv"):
        args, unknown = parser.parse_known_args()
        if unknown:
            print(f"[WARN] Ignoring unknown arguments: {unknown}")
    else:
        args = parser.parse_args([])

    os.makedirs(args.results_dir, exist_ok=True)

    cls_rows = load_csv(os.path.join(args.results_dir, "ablation_cls.csv"))
    sr_rows = load_csv(os.path.join(args.results_dir, "ablation_sr.csv"))
    mask_rows = load_csv(os.path.join(args.results_dir, "mask_ratio_ablation.csv"))

    if not any([cls_rows, sr_rows, mask_rows]):
        print("[WARN] No ablation CSVs found, nothing to summarize.")
        return

    if cls_rows:
        summarize_cls(cls_rows)
    if sr_rows:
        summarize_sr(sr_rows)
    if mask_rows:
        summarize_mask_ratio(mask_rows)


if __name__ == "__main__":
    main()
