"""
analysis.py
===========
Post-hoc analysis script.

Reads the CSV outputs produced by ``run_experiments.py`` and prints:

1. Top 5 most contributive classes  (highest mean Shapley across all rounds)
2. Bottom 5 least contributive classes
3. Suspicious clients   (mean Shapley ≈ 0 or consistently near zero)
4. Clean vs attack comparison: how much do Shapley values shift?

Usage
-----
    # Run all experiments first
    python run_experiments.py

    # Then analyse
    python analysis.py
    python analysis.py --output_dir outputs
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_shapley(path: str) -> pd.DataFrame | None:
    if os.path.exists(path):
        return pd.read_csv(path)
    print(f"  [warn] File not found: {path}")
    return None


def _section(title: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {title}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Analysis functions
# ---------------------------------------------------------------------------

def top_bottom_classes(df: pd.DataFrame, n: int = 5) -> None:
    """Print top-n and bottom-n classes by mean Shapley value."""
    mean_sv = (
        df.groupby("class_name")["shapley_value"]
        .mean()
        .sort_values(ascending=False)
    )

    print(f"\n  Top {n} most contributive classes:")
    for cls, val in mean_sv.head(n).items():
        print(f"    {cls:<40s}  {val:+.5f}")

    print(f"\n  Bottom {n} least contributive classes:")
    for cls, val in mean_sv.tail(n).items():
        print(f"    {cls:<40s}  {val:+.5f}")


def suspicious_clients(df: pd.DataFrame, threshold: float = 0.005) -> None:
    """
    Identify clients whose mean absolute Shapley value is below ``threshold``
    or whose values are consistently near zero (likely free-riders or clients
    with uninformative data).
    """
    # Per client: mean absolute Shapley across all rounds and classes
    client_stats = (
        df.groupby("client_id")["shapley_value"]
        .agg(mean="mean", mean_abs=lambda x: np.abs(x).mean(), std="std")
        .reset_index()
    )

    suspicious = client_stats[client_stats["mean_abs"] < threshold]

    if suspicious.empty:
        print(f"\n  No suspicious clients found (threshold={threshold})")
    else:
        print(f"\n  Suspicious clients (|mean Shapley| < {threshold}):")
        print(
            suspicious.to_string(
                index=False,
                float_format=lambda v: f"{v:+.6f}",
            )
        )

    # Also flag clients with very negative mean (possible poisoners)
    very_neg = client_stats[client_stats["mean"] < -threshold]
    if not very_neg.empty:
        print(f"\n  Clients with negative mean Shapley (possible poisoners):")
        print(
            very_neg.to_string(
                index=False,
                float_format=lambda v: f"{v:+.6f}",
            )
        )


def compare_clean_vs_attacks(
    clean_df: pd.DataFrame,
    attack_dfs: dict[str, pd.DataFrame],
) -> None:
    """
    Show per-class Shapley difference between clean and each attack condition.
    """
    clean_mean = (
        clean_df.groupby("class_name")["shapley_value"].mean()
    )

    for attack_name, adf in attack_dfs.items():
        attack_mean = adf.groupby("class_name")["shapley_value"].mean()
        delta = (attack_mean - clean_mean).dropna().sort_values()

        print(f"\n  Δ Shapley  (clean → {attack_name})  — largest negative shifts:")
        for cls, d in delta.head(5).items():
            print(f"    {cls:<40s}  {d:+.5f}")

        print(f"\n  Δ Shapley  (clean → {attack_name})  — largest positive shifts:")
        for cls, d in delta.tail(5).items():
            print(f"    {cls:<40s}  {d:+.5f}")


def round_trend(df: pd.DataFrame) -> None:
    """Show how mean Shapley changes across rounds."""
    round_mean = (
        df.groupby("round")["shapley_value"].mean()
    )
    print(f"\n  Round-wise mean Shapley (first 5 / last 5):")
    combined = pd.concat([round_mean.head(5), round_mean.tail(5)]).drop_duplicates()
    for rnd, val in combined.items():
        print(f"    Round {rnd:>3}:  {val:+.6f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Post-hoc FL Shapley analysis")
    parser.add_argument(
        "--output_dir", default="outputs",
        help="Root output directory produced by run_experiments.py"
    )
    args = parser.parse_args()
    base = args.output_dir

    # ------------------------------------------------------------------
    # Load CSVs
    # ------------------------------------------------------------------
    clean_sv      = _load_shapley(os.path.join(base, "clean",     "class_shapley_by_round.csv"))
    freerider_sv  = _load_shapley(os.path.join(base, "freerider", "class_shapley_by_round.csv"))
    poisoning_sv  = _load_shapley(os.path.join(base, "poisoning", "class_shapley_by_round.csv"))

    clean_metrics = None
    fr_metrics    = None
    po_metrics    = None
    for label, fname in [
        ("clean",     "clean/round_metrics.csv"),
        ("freerider", "freerider/round_metrics.csv"),
        ("poisoning", "poisoning/round_metrics.csv"),
    ]:
        p = os.path.join(base, fname)
        if os.path.exists(p):
            df = pd.read_csv(p)
            if label == "clean":     clean_metrics = df
            elif label == "freerider": fr_metrics = df
            else:                      po_metrics  = df

    # ------------------------------------------------------------------
    # 1. Round accuracy summary
    # ------------------------------------------------------------------
    _section("1. ACCURACY SUMMARY (last round)")
    for label, mdf in [("clean", clean_metrics), ("freerider", fr_metrics),
                       ("poisoning", po_metrics)]:
        if mdf is not None:
            last = mdf.iloc[-1]
            print(
                f"  {label:12s} | val_acc={last['global_accuracy']:.4f} "
                f"| test_acc={last['test_accuracy']:.4f}"
                if "test_accuracy" in last
                else f"  {label:12s} | val_acc={last['global_accuracy']:.4f}"
            )

    # ------------------------------------------------------------------
    # 2. Top / bottom classes – CLEAN
    # ------------------------------------------------------------------
    if clean_sv is not None:
        _section("2. TOP / BOTTOM CLASSES  –  CLEAN")
        top_bottom_classes(clean_sv, n=5)

    # ------------------------------------------------------------------
    # 3. Suspicious clients
    # ------------------------------------------------------------------
    for label, sv_df in [
        ("CLEAN",     clean_sv),
        ("FREERIDER", freerider_sv),
        ("POISONING", poisoning_sv),
    ]:
        if sv_df is not None:
            _section(f"3. SUSPICIOUS CLIENTS  –  {label}")
            suspicious_clients(sv_df)

    # ------------------------------------------------------------------
    # 4. Cross-condition class comparison
    # ------------------------------------------------------------------
    if clean_sv is not None:
        attack_dfs = {}
        if freerider_sv is not None: attack_dfs["freerider"] = freerider_sv
        if poisoning_sv is not None: attack_dfs["poisoning"] = poisoning_sv
        if attack_dfs:
            _section("4. CLASS-LEVEL SHAPLEY SHIFT  (clean → attack)")
            compare_clean_vs_attacks(clean_sv, attack_dfs)

    # ------------------------------------------------------------------
    # 5. Round trend (clean)
    # ------------------------------------------------------------------
    if clean_sv is not None:
        _section("5. ROUND-WISE MEAN SHAPLEY TREND  –  CLEAN")
        round_trend(clean_sv)

    # ------------------------------------------------------------------
    # 6. Top classes under attack (poisoning)
    # ------------------------------------------------------------------
    if poisoning_sv is not None:
        _section("6. TOP / BOTTOM CLASSES  –  POISONING")
        top_bottom_classes(poisoning_sv, n=5)

    print("\n[analysis] Done.\n")


if __name__ == "__main__":
    main()
