# FL Shapley 20 Newsgroups

A federated learning research project that studies **class-level Shapley
value contribution**, **free-rider attacks**, and **poisoning attacks** on
the 20 Newsgroups text classification dataset.

> **No defense mechanism is implemented.**
> The goal is purely *observation*: how do attacks shift class-level
> Shapley contribution values across communication rounds?

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Folder Structure](#folder-structure)
3. [Environment Setup](#environment-setup)
4. [How to Run](#how-to-run)
5. [Experiment Descriptions](#experiment-descriptions)
6. [Output Descriptions](#output-descriptions)
7. [Shapley Approximation Notes](#shapley-approximation-notes)
8. [Attack Settings](#attack-settings)
9. [Reproducibility](#reproducibility)
10. [中文说明 (Chinese)](#中文说明)

---

## Project Overview

| Item | Choice |
|---|---|
| Dataset | [20 Newsgroups](https://scikit-learn.org/stable/datasets/real_world.html#the-20-newsgroups-text-dataset) |
| Features | TF-IDF (10 000 vocabulary, fitted on training split) |
| Model | Logistic Regression via `SGDClassifier(loss='log_loss')` |
| FL Algorithm | FedAvg (McMahan et al., 2017) |
| Data Partition | Dirichlet non-IID (`α = 0.1` default) |
| Shapley Method | Monte Carlo permutation sampling with coalition caching |
| Tracking granularity | **round × client × class → shapley_value** |

The experiment answers three questions:

- **(A) Clean** – what is the natural class-level contribution pattern?
- **(B) Free-rider** – how do lazy clients affect others' Shapley values?
- **(C) Poisoning** – how does label-flipping degrade class contributions?

---

## Folder Structure

```
fl_shapley_20news/
│
├── config.yaml             ← All hyperparameters (edit here)
├── requirements.txt
├── .gitignore
├── README.md
│
├── main.py                 ← Run ONE experiment
├── run_experiments.py      ← Run ALL three experiments (A/B/C)
├── analysis.py             ← Post-hoc textual analysis
│
├── src/
│   ├── __init__.py
│   ├── utils.py            ← Config, logging, seeds
│   ├── data_utils.py       ← Load 20 Newsgroups + TF-IDF
│   ├── partition.py        ← Dirichlet partitioning
│   ├── model_utils.py      ← SGDClassifier wrapper + FedAvg
│   ├── federated.py        ← FLServer + FLClient classes
│   ├── attacks.py          ← Free-rider + poisoning attacks
│   ├── shapley.py          ← Monte Carlo Shapley estimation
│   ├── evaluation.py       ← Accuracy + macro-F1
│   └── plotting.py         ← All matplotlib figures
│
├── data/                   ← sklearn caches dataset here automatically
├── outputs/
│   ├── accuracy_vs_round.png          ← Combined comparison
│   ├── client_data_distribution.csv
│   ├── clean/
│   ├── freerider/
│   └── poisoning/
│
└── notebooks/
    └── quick_analysis.ipynb
```

---

## Environment Setup

### Option A – conda (recommended)

```bash
conda create -n fl_shapley python=3.10 -y
conda activate fl_shapley
pip install -r requirements.txt
```

### Option B – virtualenv

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Dependencies

```
numpy>=1.23   pandas>=1.5   scipy>=1.9
scikit-learn>=1.1   matplotlib>=3.6
pyyaml>=6.0   tqdm>=4.64
notebook>=6.5  ipykernel>=6.0   (for the notebook only)
```

---

## How to Run

### 1. Run all three experiments at once (recommended)

```bash
cd fl_shapley_20news
python run_experiments.py
```

This generates outputs in `outputs/clean/`, `outputs/freerider/`, and
`outputs/poisoning/`, plus the combined comparison chart
`outputs/accuracy_vs_round.png`.

### 2. Run a single experiment

```bash
python main.py --attack clean
python main.py --attack freerider
python main.py --attack poisoning
```

### 3. Run post-hoc analysis

```bash
python analysis.py
```

### 4. Interactive notebook

```bash
jupyter notebook notebooks/quick_analysis.ipynb
```

### 5. Key config knobs

Edit `config.yaml` before running:

| Key | Default | Description |
|---|---|---|
| `num_clients` | 10 | Number of FL clients |
| `dirichlet_alpha` | 0.1 | Non-IID degree (smaller = more skewed) |
| `num_rounds` | 20 | Communication rounds |
| `client_fraction` | 0.5 | Fraction of clients per round |
| `local_epochs` | 3 | Local SGD epochs |
| `learning_rate` | 0.01 | SGD step size |
| `shapley_num_permutations` | 10 | Monte Carlo samples for Shapley |
| `free_rider_ratio` | 0.2 | Fraction of free-rider clients |
| `poisoning_ratio` | 0.2 | Fraction of poisoning clients |
| `attack_type` | clean | Used by `main.py` (ignored by `run_experiments.py`) |

---

## Experiment Descriptions

### Experiment A – Clean
All clients train honestly.  Provides the baseline Shapley pattern.

### Experiment B – Free-rider Attack
A fraction of clients (`free_rider_ratio`) upload fake parameter updates
instead of performing real local training.

Two strategies (set via `free_rider_strategy`):
- `random` – add small Gaussian noise to the global parameters.
- `stale`  – re-upload the previous round's global parameters unchanged.

The key observation: **free-rider clients have positive Shapley potential**
(their data would be useful) but contribute zero real gradient to the global
model, causing slower convergence for honest clients.

### Experiment C – Poisoning Attack
A fraction of clients (`poisoning_ratio`) flip their training labels before
local SGD.

Two strategies (set via `poisoning_strategy`):
- `targeted` – all samples of `poison_source_class` → `poison_target_class`.
- `random`   – a fraction of labels flipped to a random different class.

The key observation: **the poisoned class shows a drop (often negative) in
Shapley value**, while other classes on the same client may remain positive.

---

## Output Descriptions

### CSVs

| File | Description |
|---|---|
| `client_data_distribution.csv` | client_id, class_id, class_name, sample_count |
| `round_metrics.csv` | round, global_accuracy, global_macro_f1, test_accuracy, test_macro_f1, attack_type |
| `class_shapley_by_round.csv` | **round, client_id, class_id, class_name, shapley_value, attack_type, client_role** |
| `client_summary.csv` | client_id, attack_role, total_samples, dominant_classes, notes |

### Figures

| File | Description |
|---|---|
| `accuracy_vs_round.png` | Overlaid accuracy/F1 curves for all conditions |
| `shapley_heatmap_<attack>.png` | Class × Round heatmap of mean Shapley |
| `top_class_contributions_<attack>.png` | Horizontal bar chart of top/bottom classes |
| `client_contribution_trends.png` | Per-client total Shapley trend over rounds |

---

## Shapley Approximation Notes

**What is the Shapley value here?**

For each FL client *i* and communication round *t*:

- **Players**: the distinct classes present in client *i*'s local data
  (e.g., classes {2, 5, 11} if client *i* holds data from those three
  20-Newsgroups categories).

- **Coalition S**: any subset of those classes.

- **Utility v(S)**: validation-set accuracy after *fine-tuning the current
  global model for one epoch* on the subset of client *i*'s data whose
  labels belong to S.  `v({}) = accuracy of the global model with no
  local training at all`.

- **Marginal contribution** of class *k* given coalition *S*:
  ```
  mc(k | S) = v(S ∪ {k}) − v(S)
  ```

- **Shapley value** φ(k):
  ```
  φ(k) = (1/M) × Σ_{m=1}^{M} mc(k | S_k(π_m))
  ```
  where π_m is the m-th random permutation and S_k(π_m) is the set of
  players appearing before *k* in that permutation.

**Why Monte Carlo?**
Exact Shapley requires summing over all *n!* permutations.  With up to
20 classes, this is 2.43 × 10¹⁸ evaluations – infeasible.  Instead,
we sample *M = 10* permutations (configurable via `shapley_num_permutations`)
and average the marginal contributions.  Coalition utility values are
cached in a `frozenset → float` dict to avoid redundant fine-tuning.

**Interpretation:**
- φ(k) > 0 → class *k*'s data helps the global model.
- φ(k) < 0 → class *k*'s data hurts (e.g., due to label poisoning).
- φ(k) ≈ 0 → class *k*'s data has negligible impact (very few samples,
  or a free-rider withholding their updates).

---

## Attack Settings

### Free-rider attack

```yaml
free_rider_ratio:    0.2      # 20 % of clients are free-riders
free_rider_strategy: random   # "random" or "stale"
free_rider_noise_scale: 0.01  # Gaussian noise std (random strategy)
```

### Poisoning attack

```yaml
poisoning_ratio:      0.2     # 20 % of clients poison their labels
poisoning_strategy:   targeted
poison_source_class:  0       # alt.atheism → flipped to...
poison_target_class:  1       # comp.graphics
poison_fraction:      0.30    # (random strategy only)
```

---

## Reproducibility

1. All random operations are seeded by `random_seed` in `config.yaml`.
2. Client selection per round uses `random_seed + round_num` so it is
   deterministic but varied across rounds.
3. Shapley permutations use `random_seed + round * 1000 + client_id`.
4. The dataset download is cached by sklearn under `~/scikit_learn_data/`.
5. To exactly reproduce results: keep `config.yaml` unchanged and run
   `python run_experiments.py`.

---

---

# 中文说明

## 项目概览

本项目在 **20 Newsgroups** 文本分类数据集上实现了联邦学习（FedAvg），并研究：

1. **类级别 Shapley 值贡献**：每个客户端的每个类别数据对全局模型的贡献
2. **搭便车攻击（Free-rider）**：恶意客户端不做真实训练，上传假更新
3. **投毒攻击（Poisoning）**：恶意客户端在本地训练前翻转标签

> **本项目不含任何防御机制。** 目标是纯粹的观测：攻击如何改变类级 Shapley 贡献值。

---

## 模型与技术栈

| 组件 | 选择 |
|---|---|
| 数据集 | 20 Newsgroups（sklearn 内置） |
| 特征 | TF-IDF（10000 维，全局拟合） |
| 模型 | Logistic Regression（SGDClassifier，log loss） |
| 联邦算法 | FedAvg |
| 数据划分 | Dirichlet 非IID（α=0.1） |
| Shapley 方法 | 蒙特卡洛排列采样 + 联盟缓存 |

---

## 环境配置

```bash
conda create -n fl_shapley python=3.10 -y
conda activate fl_shapley
pip install -r requirements.txt
```

---

## 运行方式

```bash
cd fl_shapley_20news

# 一次运行全部三个实验（推荐）
python run_experiments.py

# 单独运行某个实验
python main.py --attack clean
python main.py --attack freerider
python main.py --attack poisoning

# 事后分析
python analysis.py
```

---

## Shapley 值说明

**玩家（Players）**：客户端 i 本地数据集中出现的各类别数据子集。

**联盟价值 v(S)**：用全局模型在联盟 S 的数据上微调后，在验证集上的准确率。

**Shapley 值 φ(k)**：类别 k 在随机排列中的平均边际贡献。

- φ(k) > 0：该类数据对全局模型有帮助
- φ(k) < 0：该类数据有害（如被投毒）
- φ(k) ≈ 0：贡献可忽略（样本太少，或搭便车客户端未提交真实梯度）

---

## 输出文件说明

| 文件 | 说明 |
|---|---|
| `client_data_distribution.csv` | 各客户端的类别样本分布 |
| `round_metrics.csv` | 每轮全局准确率和 F1 |
| `class_shapley_by_round.csv` | **核心输出**：round × client × class → shapley_value |
| `client_summary.csv` | 客户端角色与主导类别汇总 |
| `accuracy_vs_round.png` | 三种实验的准确率对比曲线 |
| `shapley_heatmap_<attack>.png` | 类别×轮次的 Shapley 热图 |
| `top_class_contributions_<attack>.png` | 贡献最高/最低的类别条形图 |
| `client_contribution_trends.png` | 各客户端随轮次的贡献趋势 |

---

## 攻击参数说明

### 搭便车攻击
- `free_rider_ratio`：恶意客户端比例（默认 0.2）
- `free_rider_strategy`：`random`（随机噪声）或 `stale`（返回旧参数）

### 投毒攻击
- `poisoning_ratio`：恶意客户端比例（默认 0.2）
- `poisoning_strategy`：`targeted`（指定类别翻转）或 `random`（随机翻转）
- `poison_source_class`：被翻转的源类别 ID
- `poison_target_class`：目标类别 ID

---

## 复现说明

所有随机操作均由 `config.yaml` 中的 `random_seed` 控制，保证完全可复现。
保持 `config.yaml` 不变，运行 `python run_experiments.py` 即可复现全部结果。
