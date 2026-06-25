"""
compute_roles.py

Compute structural role features for each entity in a knowledge graph.

For every entity (node), we compute four structural features:

1. Degree
2. PageRank
3. Clustering coefficient
4. Relation diversity

These features are saved as a numpy matrix:

    role_features[node_id] = [degree, pagerank, clustering, rel_diversity]

This file is intentionally standalone so you can run it once and reuse
the saved features inside GraIL's dataset pipeline.
"""

import argparse
import os
import json
import numpy as np
import networkx as nx
from collections import defaultdict


# ========= CONFIG =========
# List all dataset folders you want to process
DATA_ROOT = "./data"

DATASETS = [
    # FB237 train + inductive
    "fb237_v1", "fb237_v2", "fb237_v3", "fb237_v4",
    "fb237_v1_ind", "fb237_v2_ind", "fb237_v3_ind", "fb237_v4_ind",

    # NELL train + inductive
    "nell_v1", "nell_v2", "nell_v3", "nell_v4",
    "nell_v1_ind", "nell_v2_ind", "nell_v3_ind", "nell_v4_ind",

    # WN18RR train + inductive
    "WN18RR_v1", "WN18RR_v2", "WN18RR_v3", "WN18RR_v4",
    "WN18RR_v1_ind", "WN18RR_v2_ind", "WN18RR_v3_ind", "WN18RR_v4_ind",
]
# ==========================


def load_triplets(data_dir):
    """
    Load triplets from train/valid/test files.
    Assumes format: head relation tail (space-separated).
    """
    triplets = []
    for split in ["train.txt", "valid.txt", "test.txt"]:
        path = os.path.join(data_dir, split)
        if not os.path.exists(path):
            continue

        with open(path, "r") as f:
            for line in f:
                h, r, t = line.strip().split()
                triplets.append((h, r, t))

    return triplets


def build_graph(triplets):
    """
    Build an undirected graph for structural feature computation.
    """
    G = nx.Graph()
    rel_map = defaultdict(set)

    for h, r, t in triplets:
        G.add_edge(h, t)
        rel_map[h].add(r)
        rel_map[t].add(r)

    return G, rel_map


def compute_structural_features(G, rel_map):
    print("Computing structural features...")

    # degree
    degree_dict = dict(G.degree())

    # pagerank
    pagerank_dict = nx.pagerank(G)

    # clustering coefficient
    clustering_dict = nx.clustering(G)

    # relation diversity
    rel_div_dict = {
        node: len(rel_map[node])
        for node in G.nodes()
    }

    # normalize features
    def normalize(values):
        arr = np.array(list(values.values()), dtype=float)
        if arr.max() - arr.min() < 1e-9:
            return {k: 0.0 for k in values}
        arr = (arr - arr.min()) / (arr.max() - arr.min())
        return {k: v for k, v in zip(values.keys(), arr)}

    degree_dict = normalize(degree_dict)
    pagerank_dict = normalize(pagerank_dict)
    clustering_dict = normalize(clustering_dict)
    rel_div_dict = normalize(rel_div_dict)

    nodes = list(G.nodes())
    node2id = {n: i for i, n in enumerate(nodes)}

    features = np.zeros((len(nodes), 4), dtype=np.float32)

    for n in nodes:
        i = node2id[n]
        features[i] = [
            degree_dict[n],
            pagerank_dict[n],
            clustering_dict[n],
            rel_div_dict[n],
        ]

    return features, node2id


def save_outputs(features, node2id, data_dir):
    output_file = os.path.join(data_dir, "role_features.npy")
    np.save(output_file, features)

    mapping_path = os.path.join(data_dir, "role_node2id.json")
    with open(mapping_path, "w") as f:
        json.dump(node2id, f)

    print(f"Saved role features to: {output_file}")
    print(f"Saved node mapping to: {mapping_path}")


def main():
    parser = argparse.ArgumentParser(description="Compute structural role features.")
    parser.add_argument("--datasets", nargs="*", default=DATASETS,
                        help="Dataset folders to process. Defaults to all configured datasets.")
    args = parser.parse_args()

    for dataset in args.datasets:
        data_dir = os.path.join(DATA_ROOT, dataset)

        if not os.path.isdir(data_dir):
            print(f"[Skip] {dataset} not found.")
            continue

        print(f"\n=== Processing dataset: {dataset} ===")

        triplets = load_triplets(data_dir)
        if len(triplets) == 0:
            print(f"[Skip] No triplets found in {dataset}")
            continue

        G, rel_map = build_graph(triplets)
        features, node2id = compute_structural_features(G, rel_map)
        save_outputs(features, node2id, data_dir)

        print(f"[Done] {dataset}")


if __name__ == "__main__":
    main()
