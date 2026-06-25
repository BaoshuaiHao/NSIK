# NSIK — Noise-aware Structural Inductive Knowledge Graph Completion
This repository contains the official implementation of NSIK (Noise-aware Structural Inductive Knowledge Graph Completion), a structure-prior-driven framework for inductive knowledge graph completion under noisy and sparse conditions. The implementation extends the GraIL subgraph reasoning pipeline with structural prior modeling, relation-aware gating, and global diffusion mechanisms.

This implementation builds upon and extends the original GraIL framework:

Teru, K. K., Denis, E., & Hamilton, W. L. (2020). *Inductive Relation Prediction by Subgraph Reasoning*. arXiv.

---

## Requirements

All required packages can be installed via:

```
pip install -r requirements.txt
```

---

## Inductive Relation Prediction Experiments

All train-graph and inductive test-graph pairs are stored in the `data` folder. We use `WN18RR_v1` as an example to illustrate the workflow.

---

### Preprocessing Pipeline (Must Run Before Training)

Before training NSIK, structural priors must be computed for each dataset. These scripts generate node roles, relation–role TF‑IDF statistics, and global diffusion embeddings, which are automatically loaded during training and evaluation.

Run the following scripts **in order**:

```
python compute_roles.py
python build_rR_matrix.py
python compute_global_diffusion.py
```

Pipeline overview:

```
Structural features → Role assignment → Relation–Role TF‑IDF → Global diffusion embedding
```

Each script processes all datasets (both training and inductive versions) and saves results into their corresponding data folders. Training should only begin after this preprocessing pipeline completes. Key hyperparameters and processing settings can be adjusted directly within the corresponding scripts to suit different datasets or experimental needs.

---

### NSIK Training

To train an NSIK model:

```
python3.7 train.py -d WN18RR_v1 -e grail_wn_v1
```

To evaluate NSIK:

```
python3.7 test_auc.py -d WN18RR_v1_ind -e grail_wn_v1
python3.7 test_ranking.py -d WN18RR_v1_ind -e grail_wn_v1
```

Trained models and logs are stored in the `experiments` folder. For fair comparison, all models use the same sampled negative triplets during evaluation.

---


## Citation

This implementation is built upon and extends the original GraIL codebase. If you use this repository in your research, please cite the GraIL paper as the foundational work that this implementation is derived from:

```
@article{Teru2020InductiveRP,
  title={Inductive Relation Prediction by Subgraph Reasoning},
  author={Komal K. Teru and Etienne Denis and William L. Hamilton},
  journal={arXiv},
  year={2020}
}
```