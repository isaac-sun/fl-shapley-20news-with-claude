"""
src/attacks.py
==============
Attack implementations for federated learning experiments.

Two attack families are supported:

1. **Free-rider attack** – a malicious client does NOT perform real local
   training but instead uploads a fake parameter update.  Two strategies:

   * ``"random"``  – add small Gaussian noise to the global parameters.
     The update looks superficially legitimate but carries no real gradient.
   * ``"stale"``   – re-upload the global parameters unchanged (or the
     parameters from the previous round).  The client contributes zero net
     gradient to the aggregation.

2. **Poisoning attack** (label-flipping) – a malicious client corrupts its
   local training labels *before* running local SGD.  The global model
   therefore learns from mislabelled data.  Two strategies:

   * ``"targeted"``  – all instances of a chosen *source* class are
     relabelled as a chosen *target* class.
   * ``"random"``    – a random fraction of labels are flipped to a
     uniformly-chosen different class.

Note: **No defense is implemented here.**  The goal is purely observation
of how attacks alter class-level Shapley contributions.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Free-rider attack
# ---------------------------------------------------------------------------

def apply_freerider_attack(
    global_params: Dict[str, np.ndarray],
    strategy: str = "random",
    prev_params: Optional[Dict[str, np.ndarray]] = None,
    noise_scale: float = 0.01,
    random_seed: int = 42,
) -> Dict[str, np.ndarray]:
    """
    Return fake model parameters for a free-rider client.

    The client **does not** perform local training; instead it uploads
    parameters crafted to look benign while contributing no real learning.

    Args:
        global_params: Current global model parameters broadcast by the server.
        strategy:      ``"random"`` or ``"stale"``.
        prev_params:   Parameters from the *previous* round (used by ``"stale"``).
                       If ``None`` on the first round, the global params are returned.
        noise_scale:   Standard deviation of Gaussian noise for ``"random"``
                       strategy (relative to parameter scale).
        random_seed:   RNG seed for noise generation.

    Returns:
        Fake parameter dict with the same structure as ``global_params``.

    Raises:
        ValueError: If ``strategy`` is not recognised.
    """
    rng = np.random.RandomState(random_seed)

    if strategy == "random":
        # Add small Gaussian noise so the update is not identical to the global
        # model (which would be too obvious), but carries no real information.
        fake_coef = global_params["coef"] + rng.normal(
            0.0, noise_scale, global_params["coef"].shape
        )
        fake_intercept = global_params["intercept"] + rng.normal(
            0.0, noise_scale, global_params["intercept"].shape
        )
        return {"coef": fake_coef, "intercept": fake_intercept}

    elif strategy == "stale":
        # Return the *previous* global parameters unchanged.
        # On the very first round there are no previous params, so fall back
        # to the current global params (effectively a zero-update client).
        if prev_params is not None:
            return {
                "coef":      prev_params["coef"].copy(),
                "intercept": prev_params["intercept"].copy(),
            }
        else:
            return {
                "coef":      global_params["coef"].copy(),
                "intercept": global_params["intercept"].copy(),
            }

    else:
        raise ValueError(
            f"Unknown free-rider strategy '{strategy}'. "
            "Choose 'random' or 'stale'."
        )


# ---------------------------------------------------------------------------
# Poisoning attack (label flipping)
# ---------------------------------------------------------------------------

def apply_poisoning_attack(
    y: np.ndarray,
    attack_config: Dict,
    random_seed: int = 42,
) -> np.ndarray:
    """
    Flip labels in ``y`` according to *attack_config*.

    A copy of ``y`` is returned; the original array is never modified.

    Strategies
    ----------
    ``"targeted"``
        Every sample whose label equals ``poison_source_class`` is
        relabelled as ``poison_target_class``.  The effect is concentrated
        on one class, making it easy to track via Shapley values.

    ``"random"``
        A random fraction (``poison_fraction``) of all samples have their
        labels replaced by a uniformly-drawn *different* class.

    Args:
        y:             Original integer label array (not mutated).
        attack_config: Dict containing:

            * ``"poisoning_strategy"``  – ``"targeted"`` or ``"random"``
            * ``"poison_source_class"`` – int (targeted mode)
            * ``"poison_target_class"`` – int (targeted mode)
            * ``"poison_fraction"``     – float in [0,1] (random mode)
            * ``"num_classes"``         – total number of classes

        random_seed:   RNG seed.

    Returns:
        Poisoned label array (copy of ``y`` with flipped entries).

    Raises:
        ValueError: If ``poisoning_strategy`` is not recognised.
    """
    rng = np.random.RandomState(random_seed)
    y_poisoned = y.copy()
    num_classes = int(attack_config.get("num_classes", 20))
    strategy    = attack_config.get("poisoning_strategy", "targeted")

    if strategy == "targeted":
        src = int(attack_config.get("poison_source_class", 0))
        tgt = int(attack_config.get("poison_target_class", 1))

        mask = y_poisoned == src
        if mask.sum() > 0:
            y_poisoned[mask] = tgt

    elif strategy == "random":
        fraction = float(attack_config.get("poison_fraction", 0.3))
        n_flip   = int(len(y_poisoned) * fraction)
        if n_flip > 0:
            flip_idx = rng.choice(len(y_poisoned), size=n_flip, replace=False)
            for idx in flip_idx:
                orig = int(y_poisoned[idx])
                candidates = [c for c in range(num_classes) if c != orig]
                y_poisoned[idx] = rng.choice(candidates)

    else:
        raise ValueError(
            f"Unknown poisoning strategy '{strategy}'. "
            "Choose 'targeted' or 'random'."
        )

    return y_poisoned


# ---------------------------------------------------------------------------
# Helper: assign malicious client IDs
# ---------------------------------------------------------------------------

def assign_malicious_clients(
    num_clients: int,
    malicious_ratio: float,
    random_seed: int = 42,
) -> list[int]:
    """
    Randomly select ``floor(num_clients * malicious_ratio)`` client IDs
    to be malicious.

    Args:
        num_clients:     Total number of FL clients.
        malicious_ratio: Fraction of clients that are malicious.
        random_seed:     RNG seed.

    Returns:
        Sorted list of malicious client IDs.
    """
    rng = np.random.RandomState(random_seed)
    n_malicious = max(0, int(num_clients * malicious_ratio))
    malicious = sorted(
        rng.choice(num_clients, size=n_malicious, replace=False).tolist()
    )
    return malicious
