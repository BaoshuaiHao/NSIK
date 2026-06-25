"""
compute_global_diffusion.py

Global structure-aware diffusion module for NSIK-style GraIL enhancement.

Pipeline (per dataset):

1) Load:
   - node_role_ids.npy
   - relation_role_tfidf.npy
   - relation2id_rr.json
   - triplets (train/valid/test)

2) Compute node importance score using TF-IDF prior

3) Diffusion source selection:
   A) global top nodes by score
   B) role-wise sampling
   → combine using lambda ratio

4) Build weighted graph using TF-IDF as soft structural prior

5) Run K-step diffusion (power iteration style)

6) Save:
   - diffusion_sources.npy
   - global_emb.npy

This module runs offline and feeds global structural priors into GraIL dataset.
"""

import argparse
import os
import json
import numpy as np
from collections import defaultdict


# ================= CONFIG =================
DATA_ROOT = "./data"

DATASETS = [
    # FB237 transductive
    "fb237_v1", "fb237_v2", "fb237_v3", "fb237_v4",
    # FB237 inductive
    "fb237_v1_ind", "fb237_v2_ind", "fb237_v3_ind", "fb237_v4_ind",

    # NELL transductive
    "nell_v1", "nell_v2", "nell_v3", "nell_v4",
    # NELL inductive
    "nell_v1_ind", "nell_v2_ind", "nell_v3_ind", "nell_v4_ind",

    # WN18RR transductive
    "WN18RR_v1", "WN18RR_v2", "WN18RR_v3", "WN18RR_v4",
    # WN18RR inductive
    "WN18RR_v1_ind", "WN18RR_v2_ind", "WN18RR_v3_ind", "WN18RR_v4_ind",
]

DIFFUSION_STEPS = 5
LAMBDA_RATIO = 0.5   # A:B mixing
TOP_GLOBAL = 100     # global top importance
PER_ROLE = 10        # per-role sampling
# ==========================================


def load_triplets(data_dir):
    triplets = []
    for split in ["train.txt", "valid.txt", "test.txt"]:
        path = os.path.join(data_dir, split)
        if not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                h, r, t = line.strip().split()
                triplets.append((h, r, t))
    return triplets


def compute_node_scores(triplets, node2id, role_ids, tfidf, rel2id):
    scores = np.zeros(len(role_ids))

    for h, r, t in triplets:
        if r not in rel2id:
            continue
        r_id = rel2id[r]

        for node in [h, t]:
            if node not in node2id:
                continue
            nid = node2id[node]
            role = role_ids[nid]
            scores[nid] += tfidf[r_id, role]

    return scores


def select_sources(scores, role_ids):
    # A: global top
    global_top = np.argsort(scores)[-TOP_GLOBAL:]

    # B: role-wise sampling
    role_sources = []
    for role in np.unique(role_ids):
        nodes = np.where(role_ids == role)[0]
        role_scores = scores[nodes]
        top = nodes[np.argsort(role_scores)[-PER_ROLE:]]
        role_sources.extend(top)

    role_sources = np.array(role_sources)

    # mix
    total = len(global_top) + len(role_sources)
    A_num = min(len(global_top), int(LAMBDA_RATIO * total))
    B_num = min(len(role_sources), total - A_num)

    global_part = global_top[-A_num:] if A_num > 0 else np.array([], dtype=int)
    role_part = role_sources[:B_num] if B_num > 0 else np.array([], dtype=int)
    sources = np.concatenate([global_part, role_part])

    return np.unique(sources)


def build_weighted_graph(triplets, node2id, role_ids, tfidf, rel2id):
    adj = defaultdict(list)
    weights = defaultdict(list)

    for h, r, t in triplets:
        if r not in rel2id:
            continue

        r_id = rel2id[r]

        if h in node2id and t in node2id:
            u = node2id[h]
            v = node2id[t]

            w = (
                tfidf[r_id, role_ids[u]] +
                tfidf[r_id, role_ids[v]]
            ) / 2

            adj[u].append(v)
            weights[u].append(w)

            adj[v].append(u)
            weights[v].append(w)

    return adj, weights


def run_rolewise_diffusion(adj, weights,
                           role_ids,
                           num_roles,
                           num_nodes,
                           diffusion_sources=None):

    global_emb = np.zeros((num_nodes, num_roles))

    for role in range(num_roles):

        print(f"Diffusion for role {role}")

        emb = np.zeros(num_nodes)

        role_nodes = np.where(role_ids == role)[0]
        if diffusion_sources is None:
            sources = role_nodes
        else:
            sources = np.intersect1d(diffusion_sources, role_nodes, assume_unique=False)
            if len(sources) == 0:
                sources = role_nodes
        if len(sources) == 0:
            continue

        emb[sources] = 1.0 / len(sources)

        for _ in range(DIFFUSION_STEPS):

            new_emb = np.zeros(num_nodes)

            for u in adj:

                neigh = adj[u]
                w = weights[u]

                if len(neigh) == 0:
                    continue

                w = np.array(w)
                w = w / (w.sum() + 1e-9)

                for v, weight in zip(neigh, w):
                    new_emb[v] += emb[u] * weight

            emb = new_emb

        global_emb[:, role] = emb

    return global_emb


def process_dataset(dataset):
    print(f"\n=== Diffusion: {dataset} ===")

    data_dir = os.path.join(DATA_ROOT, dataset)

    role_path = os.path.join(data_dir, "node_role_ids.npy")
    tfidf_path = os.path.join(data_dir, "relation_role_tfidf.npy")
    relmap_path = os.path.join(data_dir, "relation2id_rr.json")
    node_map_path = os.path.join(data_dir, "role_node2id.json")

    if not os.path.exists(role_path):
        print("Missing role data — skip")
        return

    role_ids = np.load(role_path)
    tfidf = np.load(tfidf_path)

    with open(relmap_path) as f:
        rel2id = json.load(f)

    with open(node_map_path) as f:
        node2id = json.load(f)

    triplets = load_triplets(data_dir)

    scores = compute_node_scores(
        triplets, node2id, role_ids, tfidf, rel2id
    )

    sources = select_sources(scores, role_ids)

    adj, weights = build_weighted_graph(
        triplets, node2id, role_ids, tfidf, rel2id
    )

    num_roles = tfidf.shape[1]

    global_emb = run_rolewise_diffusion(
        adj,
        weights,
        role_ids,
        num_roles,
        len(role_ids),
        diffusion_sources=sources
    )

    np.save(os.path.join(data_dir, "diffusion_sources.npy"), sources)
    np.save(os.path.join(data_dir, "global_emb.npy"), global_emb)
    with open(os.path.join(data_dir, "diffusion_config.json"), "w") as f:
        json.dump({
            "diffusion_steps": DIFFUSION_STEPS,
            "lambda_ratio": LAMBDA_RATIO,
            "top_global": TOP_GLOBAL,
            "per_role": PER_ROLE,
        }, f, indent=2)

    print("Saved diffusion outputs.")


def main():
    global DIFFUSION_STEPS, LAMBDA_RATIO, TOP_GLOBAL, PER_ROLE

    parser = argparse.ArgumentParser(description="Compute global structural diffusion embeddings.")
    parser.add_argument("--datasets", nargs="*", default=DATASETS,
                        help="Dataset folders to process. Defaults to all configured datasets.")
    parser.add_argument("--diffusion-steps", type=int, default=DIFFUSION_STEPS,
                        help="Number of diffusion propagation steps.")
    parser.add_argument("--top-global", type=int, default=TOP_GLOBAL,
                        help="Number of globally important source nodes.")
    parser.add_argument("--per-role", type=int, default=PER_ROLE,
                        help="Number of source candidates selected per structural role.")
    parser.add_argument("--lambda-ratio", type=float, default=LAMBDA_RATIO,
                        help="Approximate fraction of global top sources in the mixed source pool.")
    args = parser.parse_args()

    DIFFUSION_STEPS = args.diffusion_steps
    TOP_GLOBAL = args.top_global
    PER_ROLE = args.per_role
    LAMBDA_RATIO = args.lambda_ratio

    for dataset in args.datasets:
        process_dataset(dataset)

    print("\nGlobal diffusion complete.")


if __name__ == "__main__":
    main()
