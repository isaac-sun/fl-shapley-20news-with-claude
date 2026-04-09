"""
src/shapley.py
==============
Monte Carlo approximation of **class-level Shapley values** for each
FL client.

Conceptual Design
-----------------
The goal is to answer: *"How much does each class-specific data subset owned
by client i contribute to improving the global model?"*

Shapley game formulation
~~~~~~~~~~~~~~~~~~~~~~~~
* **Players**  : the distinct classes present in client *i*'s local dataset.
  E.g. if client 3 holds data from classes {2, 5, 11, 14}, those four classes
  are the four players.
* **Coalition S**: any subset of the client's classes.
* **Utility v(S)**: validation-set accuracy achieved after fine-tuning the
  *current global model* for one epoch on the data from client *i* that
  belongs to the classes in S.  ``v({}) = accuracy of the global model
  without any fine-tuning``.
* **Marginal contribution** of class k given coalition S:
    ``mc(k | S) = v(S ∪ {k}) − v(S)``
* **Shapley value** for class k:
    ``φ(k) = E_{permutation π}[ mc(k | S_k(π)) ]``
  where ``S_k(π)`` is the set of classes appearing before k in permutation π.

Monte Carlo approximation
~~~~~~~~~~~~~~~~~~~~~~~~~
Instead of summing over all n! permutations (expensive for n=20), we sample
``num_permutations`` random orderings and average the marginal contributions.
Coalition utility values are cached in a ``frozenset → float`` dict so that
the same coalition is never evaluated twice within a single client/round call.

Device support
~~~~~~~~~~~~~~
The device (CUDA / MPS / CPU) is threaded through model creation, training,
and the evaluate_params call inside the utility function, so all computation
runs on the selected accelerator.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, Tuple

import numpy as np
import torch
from scipy.sparse import csr_matrix, vstack as sp_vstack

from .model_utils import (
    TorchLRModel,
    create_model,
    set_model_params,
    get_model_params,
    train_on_data,
)
from .evaluation import evaluate_params


# ---------------------------------------------------------------------------
# Internal: coalition utility function
# ---------------------------------------------------------------------------

def _coalition_utility(
    coalition: FrozenSet[int],
    global_params: Dict[str, np.ndarray],
    class_data: Dict[int, Tuple[csr_matrix, np.ndarray]],
    X_val: csr_matrix,
    y_val: np.ndarray,
    num_classes: int,
    num_features: int,
    shapley_local_epochs: int,
    batch_size: int,
    learning_rate: float,
    random_seed: int,
    device: torch.device,
) -> float:
    """
    Compute v(coalition) = val accuracy after fine-tuning global model on
    the data from the given coalition of classes.

    Args:
        coalition:            Frozenset of class IDs to include.
        global_params:        Current global model parameters (starting point).
        class_data:           ``{class_id: (X_c, y_c)}`` – sparse data slices.
        X_val, y_val:         Global held-out validation set.
        num_classes:          Total number of classes in the problem.
        num_features:         TF-IDF feature dimension.
        shapley_local_epochs: Training epochs inside the utility function.
        batch_size:           Mini-batch size.
        learning_rate:        SGD step size.
        random_seed:          For reproducibility inside this evaluation.
        device:               Compute device (CUDA / MPS / CPU).

    Returns:
        Validation accuracy as a float.
    """
    if len(coalition) == 0:
        # v(∅) – evaluate the global model as-is (no local training)
        return evaluate_params(global_params, X_val, y_val, device)

    # Merge all data belonging to classes in the coalition
    X_parts: List[csr_matrix] = []
    y_parts: List[np.ndarray] = []
    for c in sorted(coalition):
        Xc, yc = class_data[c]
        if Xc.shape[0] > 0:
            X_parts.append(Xc)
            y_parts.append(yc)

    if not X_parts:
        return evaluate_params(global_params, X_val, y_val, device)

    X_coal = sp_vstack(X_parts)
    y_coal = np.concatenate(y_parts)

    # Fine-tune a *fresh* copy of the global model on coalition data
    local_model = create_model(
        num_classes, num_features, learning_rate, device, random_seed
    )
    set_model_params(local_model, global_params, device)
    train_on_data(
        local_model, X_coal, y_coal,
        num_classes, shapley_local_epochs, batch_size, learning_rate,
        device, random_seed,
    )

    return evaluate_params(get_model_params(local_model), X_val, y_val, device)


# ---------------------------------------------------------------------------
# Public API: per-client class-level Shapley
# ---------------------------------------------------------------------------

def compute_class_shapley(
    global_params: Dict[str, np.ndarray],
    client_data: Dict,
    X_val: csr_matrix,
    y_val: np.ndarray,
    num_classes: int,
    num_features: int,
    num_permutations: int = 10,
    shapley_local_epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    random_seed: int = 42,
    device: torch.device | None = None,
) -> Dict[int, float]:
    """
    Estimate class-level Shapley values for a single FL client.

    For each class k in the client's local dataset, the Shapley value φ(k)
    measures the *average marginal improvement in global validation accuracy*
    from including class k's data in local fine-tuning.

    * φ(k) > 0  →  class k's data helps the global model.
    * φ(k) < 0  →  class k's data hurts (e.g. due to label poisoning).
    * φ(k) ≈ 0  →  class k's data has negligible effect
                   (e.g. free-rider or very few samples).

    Args:
        global_params:        Current global model parameters.
        client_data:          Client dict from :mod:`partition`; must contain
                              ``"X"``, ``"y"``, and ``"class_counts"``.
        X_val, y_val:         Global held-out validation set.
        num_classes:          Total classes in the problem (e.g. 20).
        num_features:         TF-IDF vocabulary size.
        num_permutations:     Number of Monte Carlo permutation samples.
        shapley_local_epochs: Epochs for the fine-tuning step in v(S).
        batch_size:           Mini-batch size for fine-tuning.
        learning_rate:        SGD learning rate for fine-tuning.
        random_seed:          Base RNG seed.
        device:               Compute device; defaults to CPU.

    Returns:
        ``{class_id: shapley_value}`` for every class owned by this client.
    """
    dev = device or torch.device("cpu")

    # ------------------------------------------------------------------
    # Identify which classes this client actually has data for
    # ------------------------------------------------------------------
    classes_owned: List[int] = sorted(client_data["class_counts"].keys())

    if not classes_owned:
        return {}

    X_local: csr_matrix = client_data["X"]
    y_local: np.ndarray = client_data["y"]

    # Build per-class data slices (reused across permutations)
    class_data: Dict[int, Tuple[csr_matrix, np.ndarray]] = {}
    for c in classes_owned:
        mask = y_local == c
        class_data[c] = (X_local[mask], y_local[mask])

    num_features_actual = X_local.shape[1]

    # ------------------------------------------------------------------
    # Special case: single class → Shapley = v({k}) − v({})
    # ------------------------------------------------------------------
    if len(classes_owned) == 1:
        c = classes_owned[0]
        v_empty = _coalition_utility(
            frozenset(), global_params, class_data,
            X_val, y_val, num_classes, num_features_actual,
            shapley_local_epochs, batch_size, learning_rate, random_seed, dev,
        )
        v_full = _coalition_utility(
            frozenset([c]), global_params, class_data,
            X_val, y_val, num_classes, num_features_actual,
            shapley_local_epochs, batch_size, learning_rate, random_seed, dev,
        )
        return {c: v_full - v_empty}

    # ------------------------------------------------------------------
    # Monte Carlo permutation sampling
    # ------------------------------------------------------------------
    shapley_values: Dict[int, float] = {c: 0.0 for c in classes_owned}
    cache: Dict[FrozenSet[int], float] = {}
    rng = np.random.RandomState(random_seed)

    for perm_idx in range(num_permutations):
        perm_seed  = random_seed + perm_idx + 1
        permutation: List[int] = rng.permutation(classes_owned).tolist()
        coalition: FrozenSet[int] = frozenset()

        for c in permutation:
            if coalition not in cache:
                cache[coalition] = _coalition_utility(
                    coalition, global_params, class_data,
                    X_val, y_val, num_classes, num_features_actual,
                    shapley_local_epochs, batch_size, learning_rate, perm_seed, dev,
                )
            v_without = cache[coalition]

            new_coalition = coalition | frozenset([c])
            if new_coalition not in cache:
                cache[new_coalition] = _coalition_utility(
                    new_coalition, global_params, class_data,
                    X_val, y_val, num_classes, num_features_actual,
                    shapley_local_epochs, batch_size, learning_rate, perm_seed, dev,
                )
            v_with = cache[new_coalition]

            shapley_values[c] += v_with - v_without
            coalition = new_coalition

    for c in shapley_values:
        shapley_values[c] /= num_permutations

    return shapley_values
