"""
src/federated.py
================
Federated Averaging (FedAvg) server and client logic.

Architecture
------------
* :class:`FLServer`  – holds the global model; selects clients each round;
  aggregates updates via weighted average (FedAvg).
* :class:`FLClient`  – holds local data; performs local SGD training;
  optionally applies an attack before or instead of training.

The server knows nothing about whether individual clients are malicious.
All attack logic is encapsulated inside :class:`FLClient`.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from .attacks import apply_freerider_attack, apply_poisoning_attack
from .model_utils import (
    clone_model,
    create_model,
    fedavg_aggregate,
    get_model_params,
    set_model_params,
    train_on_data,
)

logger = logging.getLogger("fl_shapley")


# ---------------------------------------------------------------------------
# FL Server
# ---------------------------------------------------------------------------

class FLServer:
    """
    Central server for FedAvg federated learning.

    Responsibilities:
    * Maintain the global model parameters.
    * Broadcast parameters to selected clients each round.
    * Aggregate client updates via weighted FedAvg.
    * Select participating clients per round.
    """

    def __init__(
        self,
        num_classes: int,
        num_features: int,
        learning_rate: float,
        random_seed: int,
    ) -> None:
        """
        Args:
            num_classes:   Number of output classes (20 for 20 Newsgroups).
            num_features:  TF-IDF vocabulary size.
            learning_rate: SGD learning rate (used when constructing the model).
            random_seed:   Seed used for client selection.
        """
        self.num_classes   = num_classes
        self.num_features  = num_features
        self.learning_rate = learning_rate
        self.random_seed   = random_seed

        # Initialise global model
        self._global_model = create_model(
            num_classes, num_features, learning_rate, random_seed
        )
        self.global_params: Dict[str, np.ndarray] = get_model_params(
            self._global_model
        )

    # ------------------------------------------------------------------
    # Parameter access
    # ------------------------------------------------------------------

    def get_global_params(self) -> Dict[str, np.ndarray]:
        """Return a **copy** of the current global parameters."""
        return {
            "coef":      self.global_params["coef"].copy(),
            "intercept": self.global_params["intercept"].copy(),
        }

    def set_global_params(self, params: Dict[str, np.ndarray]) -> None:
        """Overwrite the global parameters (e.g. after loading a checkpoint)."""
        self.global_params = {
            "coef":      params["coef"].copy(),
            "intercept": params["intercept"].copy(),
        }
        set_model_params(self._global_model, self.global_params)

    # ------------------------------------------------------------------
    # Client selection
    # ------------------------------------------------------------------

    def select_clients(
        self,
        all_clients: List["FLClient"],
        client_fraction: float,
        round_num: int,
    ) -> List["FLClient"]:
        """
        Randomly select a fraction of clients to participate in this round.

        A different RNG seed is used each round (``random_seed + round_num``)
        so the same clients are not always chosen, while the selection is
        still deterministic given the seed.

        Args:
            all_clients:      Full list of :class:`FLClient` objects.
            client_fraction:  Fraction in (0, 1] to select.
            round_num:        Current round index (0-based).

        Returns:
            Sub-list of selected :class:`FLClient` objects.
        """
        n_select = max(1, int(len(all_clients) * client_fraction))
        rng = np.random.RandomState(self.random_seed + round_num)
        chosen_idx = rng.choice(len(all_clients), size=n_select, replace=False)
        return [all_clients[i] for i in sorted(chosen_idx)]

    # ------------------------------------------------------------------
    # FedAvg aggregation
    # ------------------------------------------------------------------

    def aggregate(
        self,
        client_updates: List[Dict[str, np.ndarray]],
        client_sample_counts: List[int],
    ) -> None:
        """
        Aggregate client parameter updates with FedAvg and update the
        global model in-place.

        Args:
            client_updates:       List of ``{"coef": …, "intercept": …}`` dicts.
            client_sample_counts: Number of local training samples per client
                                  (used as FedAvg weights).
        """
        if not client_updates:
            logger.warning("[server] No client updates received – skipping aggregation.")
            return

        new_params = fedavg_aggregate(client_updates, client_sample_counts)
        self.global_params = new_params
        set_model_params(self._global_model, new_params)


# ---------------------------------------------------------------------------
# FL Client
# ---------------------------------------------------------------------------

class FLClient:
    """
    Federated learning client.

    Each client holds a local dataset and an *attack configuration* that
    determines its behaviour during local training.

    Attack roles
    ~~~~~~~~~~~~
    * ``"clean"``     – perform honest local SGD training.
    * ``"freerider"`` – upload fake parameters without real training.
    * ``"poisoning"`` – flip labels locally before honest SGD training.
    """

    def __init__(
        self,
        client_id: int,
        client_data: Dict,
        num_classes: int,
        local_epochs: int,
        batch_size: int,
        learning_rate: float,
        random_seed: int,
        attack_role: str = "clean",
        attack_config: Optional[Dict] = None,
    ) -> None:
        """
        Args:
            client_id:     Integer client identifier.
            client_data:   Dict from :mod:`partition` with keys
                           ``"X"``, ``"y"``, ``"class_counts"``, etc.
            num_classes:   Total number of classes.
            local_epochs:  SGD epochs per FL round.
            batch_size:    Mini-batch size.
            learning_rate: SGD step size.
            random_seed:   Base RNG seed.
            attack_role:   One of ``"clean"``, ``"freerider"``, ``"poisoning"``.
            attack_config: Additional attack parameters (see :mod:`attacks`).
        """
        self.client_id     = client_id
        self.data          = client_data        # original, unmodified
        self.num_classes   = num_classes
        self.local_epochs  = local_epochs
        self.batch_size    = batch_size
        self.learning_rate = learning_rate
        self.random_seed   = random_seed
        self.attack_role   = attack_role
        self.attack_config = attack_config or {}

        # Stale-update free-rider remembers the previous global params
        self._prev_global_params: Optional[Dict[str, np.ndarray]] = None

    # ------------------------------------------------------------------
    # Local training  (called by the server each round)
    # ------------------------------------------------------------------

    def local_train(
        self,
        global_params: Dict[str, np.ndarray],
        round_num: int,
    ) -> Tuple[Dict[str, np.ndarray], int]:
        """
        Perform one round of local training and return updated parameters.

        Returns:
            Tuple of ``(updated_params, num_local_samples)``.
            ``num_local_samples`` is used as the FedAvg weight.
        """
        seed = self.random_seed + round_num * 100 + self.client_id

        # ---- Free-rider attack: skip real training ---------------------
        if self.attack_role == "freerider":
            strategy   = self.attack_config.get("free_rider_strategy", "random")
            noise_scale = self.attack_config.get("free_rider_noise_scale", 0.01)
            fake_params = apply_freerider_attack(
                global_params,
                strategy=strategy,
                prev_params=self._prev_global_params,
                noise_scale=noise_scale,
                random_seed=seed,
            )
            # Store current global params for potential stale-update next round
            self._prev_global_params = {
                "coef":      global_params["coef"].copy(),
                "intercept": global_params["intercept"].copy(),
            }
            # Return 1 as weight so free-rider is not entirely ignored
            # (the server has no way to verify local sample counts)
            return fake_params, 1

        # ---- Poisoning attack: flip labels before training -------------
        X_local = self.data["X"]
        y_local = self.data["y"]

        if self.attack_role == "poisoning":
            cfg = dict(self.attack_config)
            cfg["num_classes"] = self.num_classes
            y_local = apply_poisoning_attack(y_local, cfg, random_seed=seed)

        # ---- Honest local SGD training ---------------------------------
        local_model = create_model(
            self.num_classes,
            global_params["coef"].shape[1],
            self.learning_rate,
            seed,
        )
        set_model_params(local_model, global_params)
        train_on_data(
            local_model, X_local, y_local,
            self.num_classes, self.local_epochs, self.batch_size, seed,
        )

        # Store global params for potential stale update
        self._prev_global_params = {
            "coef":      global_params["coef"].copy(),
            "intercept": global_params["intercept"].copy(),
        }

        updated_params = get_model_params(local_model)
        return updated_params, int(self.data["num_samples"])
