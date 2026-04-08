"""
src/plotting.py
===============
All matplotlib visualisations for the FL Shapley experiment.

Functions
---------
* :func:`plot_accuracy_vs_round`          – overlaid accuracy curves for all
  three experiment conditions.
* :func:`plot_shapley_heatmap`            – heatmap of mean per-class Shapley
  values across rounds.
* :func:`plot_top_class_contributions`    – horizontal bar chart of the top-N
  contributing classes.
* :func:`plot_client_contribution_trends` – per-client total Shapley over
  rounds (optional).

All plots use ``matplotlib`` only (no seaborn).
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Colour palette (consistent across plots)
# ---------------------------------------------------------------------------
_PALETTE = {
    "clean":      "#2196F3",   # blue
    "freerider":  "#FF9800",   # orange
    "poisoning":  "#F44336",   # red
}

_ALPHA_LINE = 0.9
_FIG_DPI    = 150


# ---------------------------------------------------------------------------
# 1. Accuracy vs Round
# ---------------------------------------------------------------------------

def plot_accuracy_vs_round(
    metrics_dfs: Dict[str, pd.DataFrame],
    output_path: str,
) -> None:
    """
    Plot global accuracy over FL communication rounds for one or more
    experiment conditions.

    Args:
        metrics_dfs:  ``{"clean": df, "freerider": df, "poisoning": df}``
                      Each DataFrame has columns ``["round", "global_accuracy",
                      "global_macro_f1", "attack_type"]``.
        output_path:  Path to save the PNG.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), dpi=_FIG_DPI)

    for label, df in metrics_dfs.items():
        colour = _PALETTE.get(label, "grey")
        rounds = df["round"].values

        axes[0].plot(
            rounds, df["global_accuracy"].values,
            label=label, color=colour, linewidth=2, alpha=_ALPHA_LINE,
            marker="o", markersize=4,
        )
        axes[1].plot(
            rounds, df["global_macro_f1"].values,
            label=label, color=colour, linewidth=2, alpha=_ALPHA_LINE,
            marker="s", markersize=4,
        )

    for ax, metric in zip(axes, ["Accuracy", "Macro-F1"]):
        ax.set_xlabel("Communication Round", fontsize=11)
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f"Global {metric} vs Round", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.set_ylim(0, 1)

    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# 2. Shapley Heatmap
# ---------------------------------------------------------------------------

def plot_shapley_heatmap(
    shapley_df: pd.DataFrame,
    class_names: List[str],
    attack_type: str,
    output_path: str,
    max_rounds_shown: int = 20,
) -> None:
    """
    Plot a heatmap of mean Shapley value per class per round.

    The DataFrame is aggregated by ``(round, class_name)`` → mean Shapley
    across all clients that own that class in that round.

    Args:
        shapley_df:       DataFrame with columns ``["round", "class_id",
                          "class_name", "shapley_value", "client_id"]``.
        class_names:      Ordered list of class name strings (20 entries).
        attack_type:      Label for the plot title.
        output_path:      Path to save the PNG.
        max_rounds_shown: Cap on the number of rounds displayed.
    """
    # Pivot: rows = class_name, columns = round, values = mean shapley
    agg = (
        shapley_df
        .groupby(["round", "class_name"])["shapley_value"]
        .mean()
        .reset_index()
    )
    pivot = agg.pivot(index="class_name", columns="round", values="shapley_value")

    # Limit rounds
    if pivot.shape[1] > max_rounds_shown:
        pivot = pivot.iloc[:, :max_rounds_shown]

    # Sort classes by their mean Shapley (descending)
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(
        figsize=(max(8, pivot.shape[1] * 0.55), max(6, pivot.shape[0] * 0.35)),
        dpi=_FIG_DPI,
    )

    vmax = max(abs(pivot.values[~np.isnan(pivot.values)]).max(), 1e-6)
    vmin = -vmax

    im = ax.imshow(
        pivot.fillna(0).values,
        aspect="auto",
        cmap="RdBu_r",
        vmin=vmin,
        vmax=vmax,
    )

    # Axes labels
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns.astype(int), fontsize=8, rotation=45)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_xlabel("Communication Round", fontsize=10)
    ax.set_ylabel("Class", fontsize=10)
    ax.set_title(
        f"Class-Level Shapley Values – {attack_type.capitalize()}",
        fontsize=12, fontweight="bold",
    )

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("Mean Shapley Value", fontsize=9)

    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# 3. Top class contributions bar chart
# ---------------------------------------------------------------------------

def plot_top_class_contributions(
    shapley_df: pd.DataFrame,
    attack_type: str,
    output_path: str,
    top_n: int = 10,
) -> None:
    """
    Horizontal bar chart showing the top-N and bottom-N contributing classes
    (by mean Shapley value across all rounds and clients).

    Args:
        shapley_df:  DataFrame with ``["class_name", "shapley_value"]`` columns.
        attack_type: Label for the plot title.
        output_path: Path to save the PNG.
        top_n:       Number of classes to show at each end.
    """
    mean_sv = (
        shapley_df
        .groupby("class_name")["shapley_value"]
        .mean()
        .sort_values()
    )

    # Show top_n and bottom_n
    n_show = min(top_n, len(mean_sv))
    top    = mean_sv.tail(n_show)
    bottom = mean_sv.head(n_show)

    combined = pd.concat([bottom, top]).drop_duplicates()
    combined = combined.sort_values()

    colours = [
        _PALETTE["poisoning"] if v < 0 else _PALETTE["clean"]
        for v in combined.values
    ]

    fig, ax = plt.subplots(figsize=(8, max(4, len(combined) * 0.38)), dpi=_FIG_DPI)
    bars = ax.barh(combined.index, combined.values, color=colours, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mean Shapley Value (all rounds)", fontsize=10)
    ax.set_title(
        f"Top / Bottom Class Contributions – {attack_type.capitalize()}",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, axis="x", linestyle="--", alpha=0.4)

    # Annotate values
    for bar, val in zip(bars, combined.values):
        ax.text(
            val + (0.0005 if val >= 0 else -0.0005),
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=7,
        )

    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# 4. Client contribution trends  (optional)
# ---------------------------------------------------------------------------

def plot_client_contribution_trends(
    shapley_df: pd.DataFrame,
    attack_type: str,
    output_path: str,
    client_roles: Optional[Dict[int, str]] = None,
) -> None:
    """
    Line plot of the summed Shapley value per client across rounds.

    Useful for spotting clients whose total contribution drops to zero
    (free-riders) or turns negative (poisoners).

    Args:
        shapley_df:   DataFrame with ``["round", "client_id", "shapley_value"]``.
        attack_type:  Label for the title.
        output_path:  Save path.
        client_roles: Optional ``{client_id: "clean"/"freerider"/"poisoning"}``
                      dict for colour-coding.
    """
    client_round = (
        shapley_df
        .groupby(["round", "client_id"])["shapley_value"]
        .sum()
        .reset_index()
    )

    clients = sorted(client_round["client_id"].unique())
    fig, ax = plt.subplots(figsize=(11, 5), dpi=_FIG_DPI)

    for cid in clients:
        sub = client_round[client_round["client_id"] == cid]
        role   = (client_roles or {}).get(cid, "clean")
        colour = _PALETTE.get(role, "grey")
        ls     = "--" if role != "clean" else "-"
        lw     = 1.5 if role == "clean" else 2.0
        ax.plot(
            sub["round"].values, sub["shapley_value"].values,
            label=f"C{cid}({role[0].upper()})",
            color=colour, linewidth=lw, linestyle=ls, alpha=0.75,
        )

    ax.axhline(0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Communication Round", fontsize=11)
    ax.set_ylabel("Total Shapley Contribution", fontsize=11)
    ax.set_title(
        f"Per-Client Contribution Trends – {attack_type.capitalize()}",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=7, ncol=max(1, len(clients) // 5), loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.35)

    fig.tight_layout()
    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save(fig: plt.Figure, path: str) -> None:
    """Save *fig* to *path*, creating parent directories as needed."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"[plot] Saved → {path}")
