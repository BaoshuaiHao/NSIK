#!/usr/bin/env python3
"""Summarize reviewer-oriented experiment CSV files."""

import argparse
import csv
import math
from collections import defaultdict

try:
    from scipy.stats import ttest_rel
except Exception:
    ttest_rel = None


def mean_std(values):
    values = [float(v) for v in values]
    mean = sum(values) / len(values)
    if len(values) == 1:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def fmt(mean, std):
    return f"{mean:.6f} +/- {std:.6f}"


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("auc") or not row.get("auc_pr"):
                continue
            rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Summarize NSIK reviewer experiment results.")
    parser.add_argument("csv_path")
    args = parser.parse_args()

    rows = load_rows(args.csv_path)
    groups = defaultdict(list)
    for row in rows:
        key = (
            row["dataset"],
            row["test_dataset"],
            row["role_dim"],
            row["variant"],
            row["diffusion_steps"],
            row["top_global"],
            row["per_role"],
            row["lambda_ratio"],
        )
        groups[key].append(row)

    print("Summary")
    print("dataset,test_dataset,role_dim,variant,diffusion_steps,top_global,per_role,lambda_ratio,n,auc_mean_std,auc_pr_mean_std")
    for key in sorted(groups):
        auc = [r["auc"] for r in groups[key]]
        auc_pr = [r["auc_pr"] for r in groups[key]]
        auc_m, auc_s = mean_std(auc)
        pr_m, pr_s = mean_std(auc_pr)
        print(",".join(key + (str(len(auc)), fmt(auc_m, auc_s), fmt(pr_m, pr_s))))

    print("\nPaired tests vs full")
    print("dataset,test_dataset,role_dim,variant,diffusion_steps,top_global,per_role,lambda_ratio,metric,n,p_value")
    indexed = defaultdict(dict)
    for row in rows:
        base = (
            row["dataset"],
            row["test_dataset"],
            row["role_dim"],
            row["diffusion_steps"],
            row["top_global"],
            row["per_role"],
            row["lambda_ratio"],
        )
        indexed[(base, row["variant"])][row["seed"]] = row

    for (base, variant), seed_rows in sorted(indexed.items()):
        if variant == "full":
            continue
        full_rows = indexed.get((base, "full"), {})
        seeds = sorted(set(seed_rows) & set(full_rows))
        if len(seeds) < 2 or ttest_rel is None:
            p_auc = "NA"
            p_pr = "NA"
        else:
            p_auc = f"{ttest_rel([float(full_rows[s]['auc']) for s in seeds], [float(seed_rows[s]['auc']) for s in seeds]).pvalue:.6g}"
            p_pr = f"{ttest_rel([float(full_rows[s]['auc_pr']) for s in seeds], [float(seed_rows[s]['auc_pr']) for s in seeds]).pvalue:.6g}"
        prefix = ",".join(base[:3] + (variant,) + base[3:])
        print(f"{prefix},auc,{len(seeds)},{p_auc}")
        print(f"{prefix},auc_pr,{len(seeds)},{p_pr}")


if __name__ == "__main__":
    main()
