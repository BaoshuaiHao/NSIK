

"""
build_rR_matrix.py

Build relation–role TF‑IDF matrix for GraIL structural prior.

Pipeline:

1) Load structural role features (continuous 4‑dim vectors)
2) Discretize roles using KMeans clustering
3) Count relation ↔ role co‑occurrence
4) Compute TF‑IDF weights
5) Save matrices for downstream use

Outputs per dataset:

- relation_role_counts.npy
- relation_role_tfidf.npy
- node_role_ids.npy
- role_cluster_centers.npy

Author intent: structural prior modeling for inductive reasoning.
"""

import argparse
import os
import json
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans


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

K_ROLES = 8  # default number of structural roles
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


def cluster_roles(role_features, k_roles):
    print("Clustering structural roles...")
    kmeans = KMeans(n_clusters=k_roles, random_state=0, n_init=10)
    role_ids = kmeans.fit_predict(role_features)
    return role_ids, kmeans.cluster_centers_


def build_relation_role_matrix(triplets, role_ids, node2id, k_roles):
    rel2id = {}
    counts = defaultdict(lambda: np.zeros(k_roles))

    for h, r, t in triplets:
        if r not in rel2id:
            rel2id[r] = len(rel2id)

        r_id = rel2id[r]

        if h in node2id:
            counts[r_id][role_ids[node2id[h]]] += 1
        if t in node2id:
            counts[r_id][role_ids[node2id[t]]] += 1

    num_rel = len(rel2id)
    matrix = np.zeros((num_rel, K_ROLES))

    for r_id in counts:
        matrix[r_id] = counts[r_id]

    return matrix, rel2id


def compute_tfidf(matrix):
    print("Computing TF‑IDF weights...")

    tf = matrix / (matrix.sum(axis=1, keepdims=True) + 1e-9)

    role_presence = (matrix > 0).sum(axis=0)
    idf = np.log((matrix.shape[0] + 1) / (role_presence + 1))

    tfidf = tf * idf
    return tfidf


def process_dataset(dataset):
    print(f"\n=== Processing {dataset} ===")

    data_dir = os.path.join(DATA_ROOT, dataset)

    role_path = os.path.join(data_dir, "role_features.npy")
    mapping_path = os.path.join(data_dir, "role_node2id.json")

    if not os.path.exists(role_path):
        print("No role features — skip")
        return

    role_features = np.load(role_path)
    node2id = json.load(open(mapping_path))

    role_ids, centers = cluster_roles(role_features, K_ROLES)

    triplets = load_triplets(data_dir)

    rr_matrix, rel2id = build_relation_role_matrix(triplets, role_ids, node2id, K_ROLES)

    tfidf = compute_tfidf(rr_matrix)

    np.save(os.path.join(data_dir, "relation_role_counts.npy"), rr_matrix)
    np.save(os.path.join(data_dir, "relation_role_tfidf.npy"), tfidf)
    np.save(os.path.join(data_dir, "node_role_ids.npy"), role_ids)
    np.save(os.path.join(data_dir, "role_cluster_centers.npy"), centers)

    with open(os.path.join(data_dir, "relation2id_rr.json"), "w") as f:
        json.dump(rel2id, f)

    print("Saved matrices.")


def main():
    global K_ROLES

    parser = argparse.ArgumentParser(description="Build relation-role TF-IDF matrices.")
    parser.add_argument("--k-roles", type=int, default=K_ROLES,
                        help="Number of structural roles for KMeans clustering.")
    parser.add_argument("--datasets", nargs="*", default=DATASETS,
                        help="Dataset folders to process. Defaults to all configured datasets.")
    args = parser.parse_args()

    K_ROLES = args.k_roles

    for dataset in args.datasets:
        process_dataset(dataset)

    print("\nAll datasets complete.")


if __name__ == "__main__":
    main()
