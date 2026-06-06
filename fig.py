import numpy as np
import matplotlib.pyplot as plt

# ===== Labels =====
variants = ["NSIK", "-Gating", "-Diffusion", "-Source"]
datasets = ["WN18RR-ind", "FB15k237-ind", "NELL995-ind"]

# ===== Hits@10 Avg =====
hits = np.array([
    [92.86, 94.64, 95.18],
    [88.75, 90.28, 90.25],
    [87.50, 88.88, 88.68],
    [89.40, 91.33, 91.63],
])

# ===== AUC-PR Avg =====
auc = np.array([
    [97.80, 96.58, 96.45],
    [95.15, 94.05, 93.73],
    [94.50, 93.33, 93.03],
    [94.83, 93.70, 93.48],
])

# ===== Soft colors for bars =====
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

width = 0.2

# ======================================================
# Two panels:
# Left  — Hits@10 (grouped by dataset, 4 variants together)
# Right — AUC-PR  (same grouping)
# ======================================================

x = np.arange(len(datasets))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# --------------------------
# LEFT: Hits@10
# --------------------------
for i in range(len(variants)):
    axes[0].bar(
        x + (i - 1.5) * width,
        hits[i],
        width,
        label=variants[i],
        color=colors[i],
        alpha=0.85
    )

axes[0].set_title("Ablation — Hits@10")
axes[0].set_xticks(x)
axes[0].set_xticklabels(datasets)
axes[0].set_ylabel("Performance (%)")
axes[0].set_ylim(0, 100)
axes[0].grid(axis='y', linestyle='--', alpha=0.5)
axes[0].legend()

# --------------------------
# RIGHT: AUC-PR
# --------------------------
for i in range(len(variants)):
    axes[1].bar(
        x + (i - 1.5) * width,
        auc[i],
        width,
        label=variants[i],
        color=colors[i],
        alpha=0.75
    )

axes[1].set_title("Ablation — AUC-PR")
axes[1].set_xticks(x)
axes[1].set_xticklabels(datasets)
axes[1].set_ylim(0, 100)
axes[1].grid(axis='y', linestyle='--', alpha=0.5)
axes[1].legend()

plt.tight_layout()
plt.savefig("ablation_dual_metrics.pdf")
