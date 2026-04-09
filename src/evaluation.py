"""
src/evaluation.py
=================
Model evaluation utilities – device-aware.

Two entry-points are provided:

* :func:`evaluate_model`  – takes a fitted :class:`~model_utils.TorchLRModel`.
* :func:`evaluate_params` – takes a raw parameter dict; used inside the
  Shapley utility loop.  Runs on the selected device (CUDA / MPS / CPU).
"""

from __future__ import annotations

from typing import Dict

import numpy as np
import torch
from scipy.sparse import issparse
from sklearn.metrics import accuracy_score, f1_score

from .model_utils import TorchLRModel, sparse_to_tensor


# ---------------------------------------------------------------------------
# Evaluate a fitted TorchLRModel
# ---------------------------------------------------------------------------

def evaluate_model(
    model: TorchLRModel,
    X,
    y: np.ndarray,
    device: torch.device | None = None,
) -> Dict[str, float]:
    """
    Compute accuracy and macro-F1 for a :class:`~model_utils.TorchLRModel`.

    Args:
        model:   A trained model.
        X:       Feature matrix (sparse or dense).
        y:       Ground-truth integer labels.
        device:  Inference device; defaults to model's current device.

    Returns:
        ``{"accuracy": float, "macro_f1": float}``
    """
    if X.shape[0] == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0}

    dev = device or next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X_t   = sparse_to_tensor(X, dev)
        logits = model(X_t)
        y_pred = logits.argmax(dim=1).cpu().numpy()

    acc = float(accuracy_score(y, y_pred))
    f1  = float(f1_score(y, y_pred, average="macro", zero_division=0))
    return {"accuracy": acc, "macro_f1": f1}


# ---------------------------------------------------------------------------
# Evaluate raw parameters  (used inside Shapley loop for speed)
# ---------------------------------------------------------------------------

def evaluate_params(
    params: Dict[str, np.ndarray],
    X,
    y: np.ndarray,
    device: torch.device | None = None,
) -> float:
    """
    Compute validation accuracy directly from raw ``coef`` / ``intercept``
    arrays without constructing a full model object.

    This is the **utility function v(S)** used in the Shapley computation.

    * On CPU: uses NumPy matrix multiply (lightweight).
    * On CUDA / MPS: converts to float32 tensors for GPU-accelerated inference.

    Args:
        params: ``{"coef": ndarray(C, F), "intercept": ndarray(C)}``.
        X:      Feature matrix (sparse or dense).
        y:      Ground-truth labels.
        device: Compute device; defaults to CPU if None.

    Returns:
        Accuracy as a float in [0, 1].
    """
    if X.shape[0] == 0:
        return 0.0

    dev = device or torch.device("cpu")

    if dev.type == "cpu":
        # ---------- fast numpy path (no tensor overhead on CPU) ----------
        coef      = params["coef"]        # (C, F)
        intercept = params["intercept"]   # (C,)
        scores    = X.dot(coef.T)
        if issparse(scores):
            scores = scores.toarray()
        scores = scores + intercept
        y_pred = np.argmax(scores, axis=1)

    else:
        # ---------- torch path (CUDA / MPS) ------------------------------
        coef_t = torch.tensor(params["coef"],      dtype=torch.float32, device=dev)
        int_t  = torch.tensor(params["intercept"], dtype=torch.float32, device=dev)
        X_t    = sparse_to_tensor(X, dev)
        # (n, F) @ (F, C) + (C,)  →  (n, C)
        scores = X_t @ coef_t.T + int_t
        y_pred = scores.argmax(dim=1).cpu().numpy()

    return float(accuracy_score(y, y_pred))
