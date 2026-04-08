"""
main.py
=======
Run a **single** FL experiment (clean / freerider / poisoning) as
specified by ``config.yaml`` (or a config dict passed programmatically).

Usage
-----
    # use config.yaml defaults (attack_type: clean)
    python main.py

    # override attack_type from command line
    python main.py --attack freerider
    python main.py --attack poisoning

    # point to a custom config file
    python main.py --config my_config.yaml --attack poisoning

Outputs (inside ``outputs/<attack_type>/``)
-------------------------------------------
* ``client_data_distribution.csv``
* ``round_metrics.csv``
* ``class_shapley_by_round.csv``
* ``client_summary.csv``
* ``shapley_heatmap_<attack>.png``
* ``top_class_contributions_<attack>.png``
* ``client_contribution_trends.png``
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# ---- project imports -------------------------------------------------------
sys.path.insert(0, os.path.dirname(__file__))

from src.utils        import load_config, set_seed, get_logger, experiment_output_dir
from src.data_utils   import load_dataset
from src.partition    import dirichlet_partition, save_client_distribution
from src.model_utils  import set_model_params
from src.attacks      import assign_malicious_clients
from src.evaluation   import evaluate_model, evaluate_params
from src.shapley      import compute_class_shapley
from src.federated    import FLServer, FLClient
from src.plotting     import (
    plot_shapley_heatmap,
    plot_top_class_contributions,
    plot_client_contribution_trends,
)


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------

def run_experiment(
    cfg: Dict,
    attack_type: str,
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    class_names: List[str],
    clients_data: List[Dict],
    output_dir: str,
    logger,
) -> Dict:
    """
    Execute one complete FL experiment and persist all outputs.

    Args:
        cfg:          Full configuration dictionary.
        attack_type:  One of ``"clean"``, ``"freerider"``, ``"poisoning"``.
        X_train … y_test: Pre-split feature matrices and label arrays.
        class_names:  List of 20 class name strings.
        clients_data: Client partition list from :func:`dirichlet_partition`.
        output_dir:   Root output directory for this experiment.
        logger:       Python Logger instance.

    Returns:
        Dict with keys ``"round_metrics_df"`` and ``"shapley_df"`` for
        downstream comparison plotting.
    """
    logger.info("=" * 60)
    logger.info(f"  Starting experiment: {attack_type.upper()}")
    logger.info("=" * 60)

    num_clients      = cfg["num_clients"]
    num_rounds       = cfg["num_rounds"]
    client_fraction  = cfg["client_fraction"]
    local_epochs     = cfg["local_epochs"]
    learning_rate    = cfg["learning_rate"]
    batch_size       = cfg["batch_size"]
    random_seed      = cfg["random_seed"]
    num_classes      = cfg["num_classes"]
    num_features     = X_train.shape[1]

    shapley_n_perm   = cfg["shapley_num_permutations"]
    shapley_epochs   = cfg["shapley_local_epochs"]
    shapley_every    = cfg.get("shapley_every_n_rounds", 1)

    # ------------------------------------------------------------------
    # Determine malicious clients
    # ------------------------------------------------------------------
    malicious_ids: List[int] = []
    attack_config: Dict = {}

    if attack_type == "freerider":
        malicious_ids = assign_malicious_clients(
            num_clients, cfg["free_rider_ratio"], random_seed
        )
        attack_config = {
            "free_rider_strategy":   cfg["free_rider_strategy"],
            "free_rider_noise_scale": cfg.get("free_rider_noise_scale", 0.01),
        }
        logger.info(f"Free-rider clients: {malicious_ids}")

    elif attack_type == "poisoning":
        malicious_ids = assign_malicious_clients(
            num_clients, cfg["poisoning_ratio"], random_seed
        )
        attack_config = {
            "poisoning_strategy":  cfg["poisoning_strategy"],
            "poison_source_class": cfg["poison_source_class"],
            "poison_target_class": cfg["poison_target_class"],
            "poison_fraction":     cfg.get("poison_fraction", 0.3),
            "num_classes":         num_classes,
        }
        logger.info(
            f"Poisoning clients: {malicious_ids}  |  "
            f"strategy={cfg['poisoning_strategy']}  "
            f"src={cfg['poison_source_class']} → tgt={cfg['poison_target_class']}"
        )

    # ------------------------------------------------------------------
    # Build FL clients
    # ------------------------------------------------------------------
    client_role_map: Dict[int, str] = {}
    fl_clients: List[FLClient] = []

    for cdata in clients_data:
        cid  = cdata["client_id"]
        role = "clean"
        if cid in malicious_ids:
            role = attack_type  # "freerider" or "poisoning"
        client_role_map[cid] = role

        fl_clients.append(
            FLClient(
                client_id=cid,
                client_data=cdata,
                num_classes=num_classes,
                local_epochs=local_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                random_seed=random_seed,
                attack_role=role,
                attack_config=attack_config,
            )
        )

    # ------------------------------------------------------------------
    # Initialise server
    # ------------------------------------------------------------------
    server = FLServer(num_classes, num_features, learning_rate, random_seed)

    # ------------------------------------------------------------------
    # Accumulators
    # ------------------------------------------------------------------
    round_metrics_rows: List[Dict]  = []
    shapley_rows:       List[Dict]  = []

    # ------------------------------------------------------------------
    # FL training loop
    # ------------------------------------------------------------------
    for rnd in tqdm(range(1, num_rounds + 1), desc=f"[{attack_type}] rounds"):
        t0 = time.time()

        # 1. Broadcast global params & select clients
        global_params    = server.get_global_params()
        selected_clients = server.select_clients(fl_clients, client_fraction, rnd)

        # 2. Local training on each selected client
        client_updates:  List[Dict] = []
        client_weights:  List[int]  = []

        for client in selected_clients:
            updated_params, n_samples = client.local_train(global_params, rnd)
            client_updates.append(updated_params)
            client_weights.append(n_samples)

        # 3. FedAvg aggregation
        server.aggregate(client_updates, client_weights)
        new_global_params = server.get_global_params()

        # 4. Evaluate global model on validation and test sets
        set_model_params(server._global_model, new_global_params)
        val_metrics  = evaluate_model(server._global_model, X_val,  y_val)
        test_metrics = evaluate_model(server._global_model, X_test, y_test)

        round_metrics_rows.append(
            {
                "round":          rnd,
                "global_accuracy": val_metrics["accuracy"],
                "global_macro_f1": val_metrics["macro_f1"],
                "test_accuracy":   test_metrics["accuracy"],
                "test_macro_f1":   test_metrics["macro_f1"],
                "attack_type":     attack_type,
            }
        )

        logger.info(
            f"Round {rnd:>3} | val_acc={val_metrics['accuracy']:.4f} "
            f"test_acc={test_metrics['accuracy']:.4f} "
            f"({time.time()-t0:.1f}s)"
        )

        # 5. Shapley value estimation
        if rnd % shapley_every == 0:
            for client in selected_clients:
                cid = client.client_id

                sv_map = compute_class_shapley(
                    global_params=new_global_params,
                    client_data=client.data,
                    X_val=X_val,
                    y_val=y_val,
                    num_classes=num_classes,
                    num_features=num_features,
                    num_permutations=shapley_n_perm,
                    shapley_local_epochs=shapley_epochs,
                    batch_size=batch_size,
                    learning_rate=learning_rate,
                    random_seed=random_seed + rnd * 1000 + cid,
                )

                for class_id, sv in sv_map.items():
                    shapley_rows.append(
                        {
                            "round":         rnd,
                            "client_id":     cid,
                            "class_id":      class_id,
                            "class_name":    class_names[class_id],
                            "shapley_value": sv,
                            "attack_type":   attack_type,
                            "client_role":   client_role_map.get(cid, "clean"),
                        }
                    )

    # ------------------------------------------------------------------
    # Persist outputs
    # ------------------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)

    # round_metrics.csv
    round_df = pd.DataFrame(round_metrics_rows)
    round_csv = os.path.join(output_dir, "round_metrics.csv")
    round_df.to_csv(round_csv, index=False)
    logger.info(f"Saved {round_csv}")

    # class_shapley_by_round.csv
    shapley_df = pd.DataFrame(shapley_rows)
    shapley_csv = os.path.join(output_dir, "class_shapley_by_round.csv")
    shapley_df.to_csv(shapley_csv, index=False)
    logger.info(f"Saved {shapley_csv}")

    # client_summary.csv
    _save_client_summary(clients_data, client_role_map, class_names, output_dir)

    # ------------------------------------------------------------------
    # Plots for this experiment
    # ------------------------------------------------------------------
    if not shapley_df.empty:
        plot_shapley_heatmap(
            shapley_df, class_names, attack_type,
            os.path.join(output_dir, f"shapley_heatmap_{attack_type}.png"),
        )
        plot_top_class_contributions(
            shapley_df, attack_type,
            os.path.join(output_dir, f"top_class_contributions_{attack_type}.png"),
        )
        plot_client_contribution_trends(
            shapley_df, attack_type,
            os.path.join(output_dir, "client_contribution_trends.png"),
            client_roles=client_role_map,
        )

    return {"round_metrics_df": round_df, "shapley_df": shapley_df}


# ---------------------------------------------------------------------------
# Helper: save client summary
# ---------------------------------------------------------------------------

def _save_client_summary(
    clients_data: List[Dict],
    client_role_map: Dict[int, str],
    class_names: List[str],
    output_dir: str,
) -> None:
    """Write client_summary.csv."""
    rows = []
    for cdata in clients_data:
        cid = cdata["client_id"]
        cc  = cdata["class_counts"]

        # Dominant classes: top-3 by sample count
        dominant = sorted(cc.items(), key=lambda x: -x[1])[:3]
        dom_str  = "; ".join(
            f"{class_names[c]} ({n})" for c, n in dominant
        )
        rows.append(
            {
                "client_id":       cid,
                "attack_role":     client_role_map.get(cid, "clean"),
                "total_samples":   cdata["num_samples"],
                "num_classes_owned": len(cc),
                "dominant_classes": dom_str,
                "notes":           (
                    "No real gradient upload"
                    if client_role_map.get(cid) == "freerider"
                    else "Labels flipped"
                    if client_role_map.get(cid) == "poisoning"
                    else ""
                ),
            }
        )

    df = pd.DataFrame(rows)
    path = os.path.join(output_dir, "client_summary.csv")
    df.to_csv(path, index=False)
    print(f"[main] Saved {path}")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run a single FL Shapley experiment."
    )
    p.add_argument(
        "--config", default="config.yaml",
        help="Path to YAML config file (default: config.yaml)"
    )
    p.add_argument(
        "--attack",
        choices=["clean", "freerider", "poisoning"],
        default=None,
        help="Override the attack_type from config."
    )
    return p.parse_args()


def main() -> None:
    args   = _parse_args()
    cfg    = load_config(args.config)
    attack = args.attack or cfg.get("attack_type", "clean")
    cfg["attack_type"] = attack

    set_seed(cfg["random_seed"])

    out_base = cfg.get("output_dir", "outputs")
    out_dir  = experiment_output_dir(out_base, attack)

    logger = get_logger(
        log_file=os.path.join(out_dir, "experiment.log"),
        log_level=cfg.get("log_level", "INFO"),
    )
    logger.info(f"Config: {cfg}")

    # ---- Load data (shared across experiments) --------------------------
    X_train, X_val, X_test, y_train, y_val, y_test, vectorizer, class_names = (
        load_dataset(
            test_size=cfg["test_size"],
            val_size=cfg["val_size"],
            max_tfidf_features=cfg["max_tfidf_features"],
            random_seed=cfg["random_seed"],
        )
    )

    # ---- Partition data -------------------------------------------------
    clients_data = dirichlet_partition(
        X_train, y_train,
        num_clients=cfg["num_clients"],
        alpha=cfg["dirichlet_alpha"],
        random_seed=cfg["random_seed"],
    )

    # ---- Save distribution CSV ------------------------------------------
    dist_path = os.path.join(out_dir, "client_data_distribution.csv")
    save_client_distribution(clients_data, class_names, dist_path)

    # ---- Run experiment -------------------------------------------------
    run_experiment(
        cfg=cfg,
        attack_type=attack,
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
        class_names=class_names,
        clients_data=clients_data,
        output_dir=out_dir,
        logger=logger,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
