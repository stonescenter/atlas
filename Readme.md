# Atlas: Learning Temporal Transition Policies for Dynamic Graph Representation Learning

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red.svg)
![PyTorch Geometric](https://img.shields.io/badge/PyTorch_Geometric-latest-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**A multi-task framework for Temporal Link Prediction by learning temporal transition policies on dynamic graphs.**

</div>

---

# Overview

Atlas is the part of my PhD research. Unlike conventional temporal graph models that directly predict future links, Atlas explicitly **learns how interactions evolve over time** by modeling the probability distribution over candidate temporal continuations of a graph trajectory.

---

# Main Contributions

- Learning temporal transition policies
- Multi-task learning for temporal graphs
- Attention-based candidate continuation modeling
- Time-aware graph representation learning
- End-to-end optimization
- Dynamic neighborhood reasoning
- Temporal random walk supervision

---

# Model Architecture

```
                Seed Edge
             (u, v, t)

                 │
                 ▼

        Candidate Future Nodes

                 │
                 ▼

      ┌───────────────────────┐
      │ Temporal Encoder       │
      │ Structural Features    │
      │ Time Encoding          │
      │ Semantic Features      │
      └───────────────────────┘

                 │
       ┌─────────┴─────────┐
       ▼                   ▼

 Transition Policy      Link Prediction
      Head                    Head

       │                      │
       ▼                      ▼

 Walk Policy Loss        BCE Loss

       └─────────┬───────────┘
                 ▼

      Multi-task Optimization
```

---

# Methodology

Atlas jointly optimizes two complementary objectives.

## 1. Walk Policy Learning

The first objective learns the probability distribution over candidate future nodes

```
π(x | u,v,t)
```

This auxiliary task encourages the model to understand temporal graph evolution.

Different supervision strategies are supported:

- Earliest continuation
- Sampled continuation
- Soft supervision

---

## 2. Link Prediction

The second objective predicts whether an edge will appear in the future.

The learned transition policy acts as an auxiliary supervision signal that regularizes the temporal encoder.

---

## Multi-task Objective

```
L = λ_walk L_walk + λ_link L_link
```

where

- Walk Policy Loss learns graph evolution
- Link Prediction Loss learns future interactions

---

# Features

- Dynamic graph representation learning
- Temporal random walks
- Attention pooling
- Time encoding
- Multi-task learning
- Continuous-time graphs
- Negative sampling
- Temporal neighborhood sampling
- PyTorch Geometric implementation

---

# Installation

Clone the repository

```bash
git clone https://github.com/stonescenter/atlas.git

cd atlas
```

---

# Requirements

Create a conda env and install all dependencies from file ``env.yml``

---

# Datasets

## Download our preprocessed datasets

- Click [here](https://drive.google.com/file/d/1MNIAoA3eI5C7ysfCvmtJt1-_xz-Ggmym/view?usp=sharing) to download our preprocessed datasets.
- Or you can use `gdow 1MNIAoA3eI5C7ysfCvmtJt1-_xz-Ggmym` 
- Unzip the downloaded file
- Place all dataset files under the ./data directory

---

# Training

Run

```bash
python train_policy_distribution.py
```

or

```bash
python train_policy_multi-task.py \
    --dataset wikipedia \
    --epochs 20 \
    --batch-size 64 
```

---

# Evaluation

Evaluate a trained model

```bash
python test_link_prediction.py --dataset wikipedia --model "bin/temporal_walk_policy_model.pt"
```

Metrics

- ROC-AUC
- Average Precision (AP)

---

# Experimental Settings

Atlas supports

- Single-task learning
- Multi-task learning
- Earliest supervision
- Sampled supervision
- Soft supervision
- Attention pooling
- Ablation studies

---

# Citation

If you use Atlas in your research, please cite the corresponding publication (to be released).

```bibtex
@misc{atlas2026,
  title={Atlas: A multi-task framework for Temporal Link Prediction by learning temporal transition policies on dynamic graphs.},
  author={L. F. Steve Ataucuri Cruz},
  year={2026},
  note={PhD Research Project},
}
```

# License

MIT License and creative commons

---

## Acknowledgements

This repository contains the implementation developed during my PhD research on Temporal Graph Representation Learning. The project investigates how learning temporal transition policies can improve representation learning and temporal link prediction in evolving networks.
