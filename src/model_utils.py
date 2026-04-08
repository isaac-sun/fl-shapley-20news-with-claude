"""
src/model_utils.py
==================
Thin wrappers around scikit-learn's SGDClassifier for use in federated
learning.

Why SGDClassifier?
------------------
* ``partial_fit`` allows incremental / mini-batch training – directly
  analogous to one local SGD step per communication round.
* The model parameters (``coef_``, ``intercept_``) are plain NumPy arrays
  that can be averaged across clients (FedAvg).
* With ``loss='log_loss'`` the model is logistic regression, which is
  simple, interpretable, and unlikely to converge instantly on 20 NewsGroups,
  making it ideal for observing the effect of attacks over many rounds.

Initialization trick
--------------------
sklearn's SGDClassifier sets many internal attributes lazily on the first
``partial_fit`` call.  We trigger that initialization with a tiny dummy
dataset, then reset ``coef_`` and ``intercept_`` to zero so the model
effectively starts from scratch (or from whatever params we inject via
:func:`set_model_params`).
"""

from __future__ import annotations

import copy
from typing import Dict, List

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.linear_model import SGDClassifier


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

def create_model(
    num_classes: int,
    num_features: int,
    learning_rate: float = 0.01,
    random_seed: int = 42,
) -> SGDClassifier:
    """
    Create a fully-initialized SGDClassifier (logistic regression).

    The model is initialised by calling ``partial_fit`` once on a tiny dummy
    dataset.  This ensures all sklearn internal attributes are present before
    we start injecting arbitrary parameter dicts.  The weights are then reset
    to zero.

    Args:
        num_classes:   Number of output classes (20 for 20 Newsgroups).
        num_features:  TF-IDF feature dimension.
        learning_rate: Constant SGD step size.
        random_seed:   Seed for weight initialisation.

    Returns:
        Initialised ``SGDClassifier`` with zero weights.
    """
    rng = np.random.RandomState(random_seed)

    model = SGDClassifier(
        loss="log_loss",
        learning_rate="constant",  # constant LR → predictable FL dynamics
        eta0=learning_rate,
        fit_intercept=True,
        tol=None,                  # disable convergence stopping
        max_iter=1,                # we drive epochs ourselves via partial_fit
        warm_start=False,
        random_state=random_seed,
        n_jobs=1,
    )

    # --- Trigger sklearn lazy initialization --------------------------------
    # Use small non-zero values so log-loss gradient is well-defined.
    dummy_X = csr_matrix(
        rng.rand(num_classes, num_features).astype(np.float32) * 0.1
    )
    dummy_y = np.arange(num_classes, dtype=int)
    model.partial_fit(dummy_X, dummy_y, classes=dummy_y)

    # Reset weights to zero after the dummy pass
    model.coef_      = np.zeros((num_classes, num_features), dtype=np.float64)
    model.intercept_ = np.zeros(num_classes, dtype=np.float64)

    return model


# ---------------------------------------------------------------------------
# Parameter serialization helpers
# ---------------------------------------------------------------------------

def get_model_params(model: SGDClassifier) -> Dict[str, np.ndarray]:
    """
    Return a **copy** of the model's trainable parameters.

    Returns:
        ``{"coef": ndarray(num_classes, num_features),
           "intercept": ndarray(num_classes)}``
    """
    return {
        "coef":      model.coef_.copy().astype(np.float64),
        "intercept": model.intercept_.copy().astype(np.float64),
    }


def set_model_params(
    model: SGDClassifier,
    params: Dict[str, np.ndarray],
) -> SGDClassifier:
    """
    Overwrite ``model.coef_`` and ``model.intercept_`` in-place.

    Args:
        model:  An already-initialised SGDClassifier.
        params: Dict with keys ``"coef"`` and ``"intercept"``.

    Returns:
        The same ``model`` object (mutated in-place).
    """
    model.coef_      = params["coef"].astype(np.float64).copy()
    model.intercept_ = params["intercept"].astype(np.float64).copy()
    return model


def clone_model(
    model: SGDClassifier,
    params: Dict[str, np.ndarray] | None = None,
) -> SGDClassifier:
    """
    Deep-copy *model* and optionally inject new *params*.

    Args:
        model:  Source model.
        params: If given, replaces the cloned model's parameters.

    Returns:
        A new ``SGDClassifier`` independent of the original.
    """
    new_model = copy.deepcopy(model)
    if params is not None:
        set_model_params(new_model, params)
    return new_model


# ---------------------------------------------------------------------------
# FedAvg aggregation
# ---------------------------------------------------------------------------

def fedavg_aggregate(
    client_params_list: List[Dict[str, np.ndarray]],
    client_weights: List[float],
) -> Dict[str, np.ndarray]:
    """
    Federated Averaging: compute a weighted mean of client parameters.

    The weight for each client is proportional to its number of local
    training samples (standard FedAvg as in McMahan et al. 2017).

    Args:
        client_params_list: List of ``{"coef": …, "intercept": …}`` dicts.
        client_weights:     Corresponding non-negative weights (sample counts).

    Returns:
        Aggregated parameter dict with same structure as inputs.
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


# ---------------------------------------------------------------------------
# Local training
# ---------------------------------------------------------------------------

def train_on_data(
    model: SGDClassifier,
    X: csr_matrix,
    y: np.ndarray,
    num_classes: int,
    local_epochs: int,
    batch_size: int,
    random_seed: int = 42,
) -> SGDClassifier:
    """
    Run ``local_epochs`` passes of mini-batch SGD on ``(X, y)``.

    ``partial_fit`` is called once per mini-batch so that the model
    parameters are updated incrementally (exactly as in standard SGD-based
    federated learning).  The ``classes`` argument is always the full set
    ``[0, …, num_classes-1]`` so that ``coef_`` keeps its original shape
    even when a batch contains only a subset of classes.

    Args:
        model:        SGDClassifier initialised with current (global) params.
        X:            Sparse feature matrix for this client.
        y:            Label array for this client.
        num_classes:  Total number of classes in the problem.
        local_epochs: Number of full passes over ``(X, y)``.
        batch_size:   Mini-batch size.
        random_seed:  Seed for per-epoch shuffling.

    Returns:
        The updated model (same object, mutated in-place).
    """
    n = X.shape[0]
    if n == 0:
        return model

    all_classes = np.arange(num_classes, dtype=int)
    rng = np.random.RandomState(random_seed)

    for _ in range(local_epochs):
        order = rng.permutation(n)
        for start in range(0, n, batch_size):
            idx = order[start : start + batch_size]
            model.partial_fit(X[idx], y[idx], classes=all_classes)

    return model
