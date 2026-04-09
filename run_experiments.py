"""
run_experiments.py
==================
Convenience script that runs **all three** FL experiments (clean,
freerider, poisoning) sequentially and then generates comparison
plots that overlay results from all conditions.

Usage
-----
    python run_experiments.py                    # uses config.yaml
    python run_experiments.py --config cfg.yaml  # custom config

Outputs (inside ``outputs/``)
------------------------------
Per-experiment sub-directories (clean/, freerider/, poisoning/) are
created by :mod:`main`, plus the combined comparison figure:

    outputs/accuracy_vs_round.png

All per-experiment CSV and PNG files are also preserved in their
respective sub-directories.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict

import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))

from src.utils      import load_config, set_seed, get_logger, experiment_output_dir
from src.device_utils import get_device, device_info
from src.data_utils import load_dataset
from src.partition  import dirichlet_partition, save_client_distribution
from src.plotting   import plot_accuracy_vs_round
from main           import run_experiment


# ---------------------------------------------------------------------------
# Entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all three FL Shapley experiments sequentially."
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Path to YAML config (default: config.yaml)"
    )
    args = parser.parse_args()
    cfg  = load_config(args.config)
    set_seed(cfg["random_seed"])

    out_base = cfg.get("output_dir", "outputs")
    os.makedirs(out_base, exist_ok=True)

    logger = get_logger(
        name="fl_shapley_all",
        log_file=os.path.join(out_base, "all_experiments.log"),
        log_level=cfg.get("log_level", "INFO"),
    )

    logger.info("=" * 60)
    logger.info("  FL SHAPLEY EXPERIMENT SUITE")
    logger.info(f"  Rounds: {cfg['num_rounds']}  |  Clients: {cfg['num_clients']}")
    logger.info("=" * 60)

    # Detect device once; share across all experiments
    device = get_device(cfg.get("device", "auto"))
    logger.info(f"  Device: {device_info(device)}")

    # ---------------------------------------------------------------
    # Load data ONCE (shared across all three experiments)
    # ---------------------------------------------------------------
    logger.info("Loading & vectorising 20 Newsgroups …")
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     vectorizer, class_names) = load_dataset(
        test_size=cfg["test_size"],
        val_size=cfg["val_size"],
        max_tfidf_features=cfg["max_tfidf_features"],
        random_seed=cfg["random_seed"],
    )

    # ---------------------------------------------------------------
    # Partition data ONCE (same partition for all experiments so
    # comparisons are fair)
    # ---------------------------------------------------------------
    logger.info("Partitioning data with Dirichlet …")
    clients_data = dirichlet_partition(
        X_train, y_train,
        num_clients=cfg["num_clients"],
        alpha=cfg["dirichlet_alpha"],
        random_seed=cfg["random_seed"],
    )

    # Save the common distribution CSV at the top-level output folder
    dist_path = os.path.join(out_base, "client_data_distribution.csv")
    save_client_distribution(clients_data, class_names, dist_path)

    # ---------------------------------------------------------------
    # Run all three experiments
    # ---------------------------------------------------------------
    ATTACKS = ["clean", "freerider", "poisoning"]
    results: Dict[str, Dict] = {}

    for attack in ATTACKS:
        logger.info(f"\n{'='*60}")
        logger.info(f"  Running: {attack.upper()}")
        logger.info(f"{'='*60}\n")

        # Each experiment gets its own output sub-directory
        out_dir = experiment_output_dir(out_base, attack)

        # Also save the distribution CSV inside each sub-dir for convenience
        save_client_distribution(
            clients_data, class_names,
            os.path.join(out_dir, "client_data_distribution.csv")
        )

        t_start = time.time()
        results[attack] = run_experiment(
            cfg=cfg,
            attack_type=attack,
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test,
            class_names=class_names,
            clients_data=clients_data,
            output_dir=out_dir,
            logger=logger,
            device=device,
        )
        elapsed = time.time() - t_start
        logger.info(f"  Finished {attack} in {elapsed:.1f}s")

    # ---------------------------------------------------------------
    # Combined accuracy comparison plot
    # ---------------------------------------------------------------
    logger.info("\nGenerating combined accuracy comparison plot …")
    metrics_dfs = {
        attack: results[attack]["round_metrics_df"]
        for attack in ATTACKS
        if attack in results
    }
    plot_accuracy_vs_round(
        metrics_dfs,
        output_path=os.path.join(out_base, "accuracy_vs_round.png"),
    )

    # ---------------------------------------------------------------
    # Print final summary table
    # ---------------------------------------------------------------
    logger.info("\n" + "=" * 60)
    logger.info("  FINAL SUMMARY  (test accuracy at last round)")
    logger.info("=" * 60)
    for attack, res in results.items():
        df = res["round_metrics_df"]
        last = df.iloc[-1]
        logger.info(
            f"  {attack:12s} | val_acc={last['global_accuracy']:.4f} "
            f"| test_acc={last['test_accuracy']:.4f}"
        )

    logger.info("\nAll experiments complete.")
    logger.info(f"Outputs saved to: {out_base}/")


if __name__ == "__main__":
    main()
