"""
src/evaluation.py
=================
Model evaluation utilities.

Two entry-points are provided:

* :func:`evaluate_model`  – takes a fitted ``SGDClassifier``.
* :func:`evaluate_params` – takes a raw parameter dict, avoiding the need to
  build a full model object inside the Shapley utility loop.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.sparse import csr_matrix, issparse
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, f1_score


# ---------------------------------------------------------------------------
# Evaluate a fitted model object
# ---------------------------------------------------------------------------

def evaluate_model(
    model: SGDClassifier,
    X: csr_matrix,
    y: np.ndarray,
) -> Dict[str, float]:
    """
    Compute accuracy and macro-F1 for a fitted ``SGDClassifier``.

    Args:
        model: A trained model that supports ``model.predict(X)``.
        X:     Feature matrix (sparse or dense).
        y:     Ground-truth integer labels.

    Returns:
        ``{"accuracy": float, "macro_f1": float}``
    """
    if X.shape[0] == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0}

    y_pred = model.predict(X)
    acc    = float(accuracy_score(y, y_pred))
    f1     = float(f1_score(y, y_pred, average="macro", zero_division=0))

    return {"accuracy": acc, "macro_f1": f1}


# ---------------------------------------------------------------------------
# Evaluate raw parameters (used inside Shapley loop for speed)
# ---------------------------------------------------------------------------

def evaluate_params(
    params: Dict[str, np.ndarray],
    X: csr_matrix,
    y: np.ndarray,
) -> float:
    """
    Compute validation accuracy directly from raw ``coef_`` / ``intercept_``
    without constructing a full ``SGDClassifier``.

    This function is the **utility function** v(S) used in the Shapley
    computation.  It avoids the overhead of sklearn's ``predict`` wrapper
    by computing the class scores manually:

        scores = X @ coef.T + intercept

    The predicted class is ``argmax(scores, axis=1)``.

    Args:
        params: Dict with keys ``"coef"``  (num_classes × num_features)
                            and ``"intercept"`` (num_classes,).
        X:      Sparse feature matrix  (n_samples × num_features).
        y:      Ground-truth labels   (n_samples,).

    Returns:
        Accuracy as a float in [0, 1].
    """
    if X.shape[0] == 0:
        return 0.0

    coef      = params["coef"]        # (num_classes, num_features)
    intercept = params["intercept"]   # (num_classes,)

    # Sparse matrix-vector multiply then add bias
    # Result shape: (n_samples, num_classes)
    scores = X.dot(coef.T)
    if issparse(scores):
        scores = scores.toarray()
    scores = scores + intercept  # broadcast over samples

    y_pred = np.argmax(scores, axis=1)
    return float(accuracy_score(y, y_pred))
