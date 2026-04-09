"""
src/model_utils.py
==================
PyTorch-based logistic regression model for federated learning.

Device support
--------------
* CPU   – all platforms (default fallback)
* CUDA  – NVIDIA GPUs (Linux / Windows)
* MPS   – Apple Silicon M-series chips via Metal Performance Shaders

Why PyTorch instead of sklearn SGDClassifier?
---------------------------------------------
sklearn's SGDClassifier is CPU-only.  By reimplementing the same model
(linear layer + cross-entropy loss + SGD) in PyTorch we get transparent
GPU/MPS acceleration with a minimal code change.

MPS constraints (Apple Silicon)
--------------------------------
* float64 tensors are **not** supported on MPS → all tensors use float32.
* Sparse tensor ops are not supported on MPS → dense conversion happens
  inside :func:`sparse_to_tensor` before any data touches the device.

Parameter interface
-------------------
:func:`get_model_params` / :func:`set_model_params` always use float64
NumPy arrays so that FedAvg aggregation (CPU numpy) stays device-independent
and the parameter format is identical to the original sklearn version.
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from scipy.sparse import issparse


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------

class TorchLRModel(nn.Module):
    """
    Single-layer logistic regression:  scores = X @ W.T + b

    Equivalent to sklearn's ``SGDClassifier(loss='log_loss')`` but
    runs on any torch.device (CPU / CUDA / MPS).
    """

    def __init__(self, num_classes: int, num_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(num_features, num_classes, bias=True)
        # Zero init so rounds start from the same global params
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def create_model(
    num_classes: int,
    num_features: int,
    learning_rate: float = 0.01,      # kept for API compatibility
    device: torch.device | None = None,
    random_seed: int = 42,
) -> TorchLRModel:
    """
    Create a zero-initialised :class:`TorchLRModel` on *device*.

    Args:
        num_classes:   Number of output classes (20 for 20 Newsgroups).
        num_features:  TF-IDF vocabulary dimension.
        learning_rate: Not used here; consumed by :func:`train_on_data`.
        device:        Target torch device.  Defaults to CPU.
        random_seed:   Seed for any internal random ops.

    Returns:
        Initialised :class:`TorchLRModel` with all-zero weights.
    """
    if device is None:
        device = torch.device("cpu")
    torch.manual_seed(random_seed)
    model = TorchLRModel(num_classes, num_features).to(device)
    return model


# ---------------------------------------------------------------------------
# Parameter serialization
# ---------------------------------------------------------------------------

def get_model_params(model: TorchLRModel) -> Dict[str, np.ndarray]:
    """
    Return a **copy** of model weights as float64 NumPy arrays.

    Returns:
        ``{"coef": ndarray(num_classes, num_features),
           "intercept": ndarray(num_classes)}``
    """
    return {
        "coef":      model.linear.weight.detach().cpu().numpy().astype(np.float64),
        "intercept": model.linear.bias.detach().cpu().numpy().astype(np.float64),
    }


def set_model_params(
    model: TorchLRModel,
    params: Dict[str, np.ndarray],
    device: torch.device | None = None,
) -> TorchLRModel:
    """
    Load NumPy parameter arrays into *model* in-place (no grad).

    Args:
        model:  Already-created :class:`TorchLRModel`.
        params: Dict with keys ``"coef"`` and ``"intercept"``.
        device: Device to move tensors to (uses model's current device if None).

    Returns:
        The same *model* object (mutated).
    """
    dev = device or next(model.parameters()).device
    with torch.no_grad():
        model.linear.weight.copy_(
            torch.tensor(params["coef"], dtype=torch.float32, device=dev)
        )
        model.linear.bias.copy_(
            torch.tensor(params["intercept"], dtype=torch.float32, device=dev)
        )
    return model


def clone_model(
    model: TorchLRModel,
    device: torch.device | None = None,
    params: Dict[str, np.ndarray] | None = None,
) -> TorchLRModel:
    """
    Create a new :class:`TorchLRModel` with the same architecture as *model*
    and optionally load *params* into it.

    Args:
        model:   Source model (used for shape inference).
        device:  Target device for the clone.
        params:  If given, replaces source model's parameters in the clone.

    Returns:
        Independent :class:`TorchLRModel`.
    """
    dev = device or next(model.parameters()).device
    nc = model.linear.out_features
    nf = model.linear.in_features
    new_model = TorchLRModel(nc, nf).to(dev)
    src_params = params if params is not None else get_model_params(model)
    set_model_params(new_model, src_params, dev)
    return new_model


# ---------------------------------------------------------------------------
# Sparse → Tensor conversion  (MPS / CUDA safe)
# ---------------------------------------------------------------------------

def sparse_to_tensor(X, device: torch.device) -> torch.Tensor:
    """
    Convert a scipy sparse matrix (or numpy array) to a float32 torch Tensor.

    The dense conversion always happens on CPU first, then the tensor is
    moved to *device*.  This is required because MPS and CUDA do not support
    sparse tensor operations.

    Args:
        X:       scipy sparse matrix or numpy array  (n_samples × n_features).
        device:  Target device.

    Returns:
        float32 Tensor of shape (n_samples, n_features) on *device*.
    """
    if issparse(X):
        arr = X.toarray().astype(np.float32)
    else:
        arr = np.asarray(X, dtype=np.float32)
    # torch.from_numpy shares memory; .to(device) copies to GPU/MPS if needed
    return torch.from_numpy(arr).to(device)


# ---------------------------------------------------------------------------
# Local training
# ---------------------------------------------------------------------------

def train_on_data(
    model: TorchLRModel,
    X,
    y: np.ndarray,
    num_classes: int,           # kept for API compatibility
    local_epochs: int,
    batch_size: int,
    learning_rate: float,
    device: torch.device | None = None,
    random_seed: int = 42,
) -> TorchLRModel:
    """
    Run *local_epochs* passes of mini-batch SGD on *(X, y)*.

    Equivalent to the previous sklearn ``partial_fit`` loop, but accelerated
    by PyTorch on CUDA / MPS.

    Args:
        model:         :class:`TorchLRModel` pre-loaded with global params.
        X:             Sparse feature matrix  (n_samples × n_features).
        y:             Integer label array    (n_samples,).
        num_classes:   Unused; kept for call-site compatibility.
        local_epochs:  Number of full passes over the local dataset.
        batch_size:    Mini-batch size.
        learning_rate: SGD step size.
        device:        Inference device; falls back to model's current device.
        random_seed:   Seed for mini-batch shuffling.

    Returns:
        The updated *model* (same object, parameters modified in-place).
    """
    n = X.shape[0]
    if n == 0:
        return model

    dev = device or next(model.parameters()).device
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    rng = np.random.RandomState(random_seed)

    for _ in range(local_epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            X_b = sparse_to_tensor(X[idx], dev)
            y_b = torch.tensor(y[idx], dtype=torch.long, device=dev)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()

    return model


# ---------------------------------------------------------------------------
# FedAvg aggregation  (always CPU / numpy)
# ---------------------------------------------------------------------------

def fedavg_aggregate(
    client_params_list: List[Dict[str, np.ndarray]],
    client_weights: List[float],
) -> Dict[str, np.ndarray]:
    """
    Federated Averaging: weighted mean of client parameter dicts.

    Aggregation always runs on CPU with NumPy so it is device-agnostic.
    The weight for each client is proportional to its local sample count
    (standard FedAvg, McMahan et al. 2017).

    Args:
        client_params_list: List of ``{"coef": …, "intercept": …}`` dicts.
        client_weights:     Corresponding sample counts (FedAvg weights).

    Returns:
        Aggregated parameter dict.
    """
    total = float(sum(client_weights))
    if total == 0.0:
        total = float(len(client_params_list))
        client_weights = [1.0] * len(client_params_list)

    coef_shape      = client_params_list[0]["coef"].shape
    intercept_shape = client_params_list[0]["intercept"].shape

    avg_coef      = np.zeros(coef_shape,      dtype=np.float64)
    avg_intercept = np.zeros(intercept_shape, dtype=np.float64)

    for params, w in zip(client_params_list, client_weights):
        frac = w / total
        avg_coef      += frac * params["coef"]
        avg_intercept += frac * params["intercept"]

    return {"coef": avg_coef, "intercept": avg_intercept}
