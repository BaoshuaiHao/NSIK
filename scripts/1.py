import matplotlib.pyplot as plt
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parents[1] / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# ======================
# 数据
# ======================

roles = [4, 8, 12, 16]

# Hits@10
wn_hits = [91.20, 92.86, 92.55, 92.10]
fb_hits = [93.05, 94.64, 94.30, 93.95]
nell_hits = [93.40, 95.18, 94.90, 94.35]

# AUC-PR
wn_auc = [96.20, 97.80, 97.40, 96.90]
fb_auc = [94.90, 96.58, 96.10, 95.70]
nell_auc = [95.10, 96.45, 96.00, 95.60]

# ======================
# 横向双图
# ======================

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

# ===== 左图：Hits@10 =====
axes[0].plot(roles, wn_hits, marker='o', linewidth=2, label='WN18RR-ind')
axes[0].plot(roles, fb_hits, marker='s', linewidth=2, label='FB15k237-ind')
axes[0].plot(roles, nell_hits, marker='^', linewidth=2, label='NELL995-ind')

axes[0].set_title("Hits@10 Sensitivity")
axes[0].set_xlabel("Structural Roles")
axes[0].set_ylabel("Hits@10 (%)")
axes[0].grid(True, linestyle='--', alpha=0.6)
axes[0].legend()
axes[0].set_ylim(80, 100)

# ===== 右图：AUC-PR =====
axes[1].plot(roles, wn_auc, marker='o', linewidth=2, label='WN18RR-ind')
axes[1].plot(roles, fb_auc, marker='s', linewidth=2, label='FB15k237-ind')
axes[1].plot(roles, nell_auc, marker='^', linewidth=2, label='NELL995-ind')

axes[1].set_title("AUC-PR Sensitivity")
axes[1].set_xlabel("Structural Roles")
axes[1].set_ylabel("AUC-PR (%)")
axes[1].grid(True, linestyle='--', alpha=0.6)
axes[1].legend()
axes[1].set_ylim(80, 100)

plt.tight_layout()

# 保存论文图
plt.savefig(FIGURES_DIR / "role_sensitivity_side_by_side.pdf")

plt.close()

print("✅ role_sensitivity_side_by_side.pdf generated")
