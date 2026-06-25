#!/usr/bin/env python3
"""Create noisy training splits for robustness experiments.

The script copies a dataset folder and rewrites only train.txt. Validation and
test files stay unchanged so robustness is evaluated on the original target
distribution.
"""

import argparse
import os
import random
import shutil
from collections import Counter, defaultdict


def read_triplets(path):
    triples = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if parts:
                triples.append(tuple(parts))
    return triples


def write_triplets(path, triples):
    with open(path, "w") as f:
        for h, r, t in triples:
            f.write(f"{h} {r} {t}\n")


def weighted_choice(items, weights):
    total = float(sum(weights))
    if total <= 0:
        return random.choice(items)
    threshold = random.random() * total
    acc = 0.0
    for item, weight in zip(items, weights):
        acc += weight
        if acc >= threshold:
            return item
    return items[-1]


def corrupt_tail(triple, entities):
    h, r, t = triple
    new_t = random.choice(entities)
    while new_t == t:
        new_t = random.choice(entities)
    return h, r, new_t


def corrupt_degree_tail(triple, entities, degree):
    h, r, t = triple
    weights = [degree[e] + 1 for e in entities]
    new_t = weighted_choice(entities, weights)
    while new_t == t:
        new_t = weighted_choice(entities, weights)
    return h, r, new_t


def corrupt_relation(triple, relations):
    h, r, t = triple
    new_r = random.choice(relations)
    while new_r == r:
        new_r = random.choice(relations)
    return h, new_r, t


def inject_edges(triples, entities, relations, count):
    existing = set(triples)
    injected = []
    attempts = 0
    while len(injected) < count and attempts < count * 50 + 100:
        attempts += 1
        h = random.choice(entities)
        r = random.choice(relations)
        t = random.choice(entities)
        if h == t:
            continue
        triple = (h, r, t)
        if triple in existing:
            continue
        existing.add(triple)
        injected.append(triple)
    return triples + injected


def make_noisy_train(triples, ratio, mode):
    entities = sorted({h for h, _, _ in triples} | {t for _, _, t in triples})
    relations = sorted({r for _, r, _ in triples})
    degree = Counter()
    for h, _, t in triples:
        degree[h] += 1
        degree[t] += 1

    n_corrupt = int(round(len(triples) * ratio))
    indices = set(random.sample(range(len(triples)), n_corrupt))

    if mode == "edge_injection":
        return inject_edges(list(triples), entities, relations, n_corrupt)

    out = []
    for idx, triple in enumerate(triples):
        if idx not in indices:
            out.append(triple)
            continue

        if mode == "tail":
            out.append(corrupt_tail(triple, entities))
        elif mode == "degree_tail":
            out.append(corrupt_degree_tail(triple, entities, degree))
        elif mode == "relation":
            out.append(corrupt_relation(triple, relations))
        else:
            raise ValueError(f"Unknown noise mode: {mode}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Create noisy KG training datasets.")
    parser.add_argument("--source", required=True, help="Source dataset folder under data/, e.g. WN18RR_v1")
    parser.add_argument("--ratio", type=float, required=True, help="Noise ratio, e.g. 0.1")
    parser.add_argument("--mode", choices=["tail", "degree_tail", "relation", "edge_injection"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--suffix", default=None,
                        help="Optional output suffix. Defaults to noise mode and ratio.")
    parser.add_argument("--data-root", default="data")
    args = parser.parse_args()

    random.seed(args.seed)

    source_dir = os.path.join(args.data_root, args.source)
    if not os.path.isdir(source_dir):
        raise FileNotFoundError(source_dir)

    ratio_tag = str(args.ratio).replace(".", "p")
    suffix = args.suffix or f"{args.mode}_{ratio_tag}_s{args.seed}"
    target_name = f"{args.source}_noise_{suffix}"
    target_dir = os.path.join(args.data_root, target_name)

    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)
    for name in os.listdir(target_dir):
        path = os.path.join(target_dir, name)
        if os.path.isdir(path) and (name.startswith("subgraphs") or name.startswith("test_subgraphs")):
            shutil.rmtree(path)

    train_path = os.path.join(target_dir, "train.txt")
    triples = read_triplets(train_path)
    noisy = make_noisy_train(triples, args.ratio, args.mode)
    write_triplets(train_path, noisy)

    with open(os.path.join(target_dir, "noise_config.txt"), "w") as f:
        f.write(f"source={args.source}\n")
        f.write(f"target={target_name}\n")
        f.write(f"mode={args.mode}\n")
        f.write(f"ratio={args.ratio}\n")
        f.write(f"seed={args.seed}\n")
        f.write(f"original_train_triples={len(triples)}\n")
        f.write(f"noisy_train_triples={len(noisy)}\n")

    print(target_name)


if __name__ == "__main__":
    main()
