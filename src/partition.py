"""
src/partition.py
================
Dirichlet-based non-IID data partitioning for federated learning.

The Dirichlet distribution controls how "non-IID" the partition is:
  - Small  alpha (e.g. 0.1) → highly skewed; each client sees only 1–3 classes.
  - Large  alpha (e.g. 10 ) → near-IID;  each client's distribution ≈ global.
"""

from __future__ import annotations

import os
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


# ---------------------------------------------------------------------------
# Core partitioning function
# ---------------------------------------------------------------------------

def dirichlet_partition(
    X: csr_matrix,
    y: np.ndarray,
    num_clients: int,
    alpha: float,
    random_seed: int = 42,
) -> List[Dict]:
    """
    Partition ``(X, y)`` among ``num_clients`` clients using a Dirichlet draw.

    For each class ``c``, we draw a Dirichlet vector of length ``num_clients``
    and use it as the proportion of class-``c`` samples assigned to each client.
    This naturally produces heterogeneous (non-IID) distributions.

    Args:
        X:           Full sparse feature matrix  (n_samples × n_features).
        y:           Full integer label array    (n_samples,).
        num_clients: Number of FL clients.
        alpha:       Dirichlet concentration parameter.
        random_seed: RNG seed for reproducibility.

    Returns:
        A list of ``num_clients`` dicts, each containing:

        * ``client_id``   – integer client index
        * ``X``           – sparse feature matrix for this client
        * ``y``           – label array for this client
        * ``indices``     – original row indices in the full matrix
        * ``class_counts``– ``{class_id: count}`` (only classes with >0 samples)
        * ``num_samples`` – total number of samples
    """
    rng = np.random.RandomState(random_seed)
    num_classes = int(y.max()) + 1

    # Group sample indices by class
    class_indices: Dict[int, List[int]] = {
        c: list(np.where(y == c)[0]) for c in range(num_classes)
    }
    for c in class_indices:
        rng.shuffle(class_indices[c])

    # Draw Dirichlet proportions: shape (num_classes, num_clients)
    proportions = rng.dirichlet(alpha=[alpha] * num_clients, size=num_classes)

    client_index_lists: List[List[int]] = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        idx_c = class_indices[c]
        n_c = len(idx_c)

        # Convert proportions → integer counts, fix rounding errors
        counts = (proportions[c] * n_c).astype(int)
        diff = n_c - counts.sum()
        if diff > 0:
            # Distribute leftover samples to top-proportion clients
            top_k = np.argsort(-proportions[c])[:diff]
            counts[top_k] += 1
        elif diff < 0:
            # Remove excess from bottom-proportion clients (only where count > 0)
            bot_k = np.argsort(proportions[c])
            for k in bot_k:
                if counts[k] > 0 and diff < 0:
                    counts[k] -= 1
                    diff += 1

        # Slice the class sample list and append to each client's index list
        start = 0
        for cid in range(num_clients):
            end = start + counts[cid]
            client_index_lists[cid].extend(idx_c[start:end])
            start = end

    # Build per-client data dicts
    clients: List[Dict] = []
    for cid in range(num_clients):
        idx = np.array(client_index_lists[cid], dtype=int)

        if idx.size == 0:
            # Guarantee at least one sample (edge-case safety net)
            idx = np.array([rng.randint(0, len(y))], dtype=int)

        client_y = y[idx]
        client_X = X[idx]

        class_counts = {
            int(c): int((client_y == c).sum())
            for c in range(num_classes)
            if (client_y == c).sum() > 0
        }

        clients.append(
            {
                "client_id":    cid,
                "X":            client_X,
                "y":            client_y,
                "indices":      idx,
                "class_counts": class_counts,
                "num_samples":  int(idx.size),
            }
        )

    # Print a quick summary
    sizes = [c["num_samples"] for c in clients]
    print(
        f"[partition] {num_clients} clients | "
        f"alpha={alpha} | "
        f"samples per client: min={min(sizes)} max={max(sizes)} mean={np.mean(sizes):.0f}"
    )

    return clients


# ---------------------------------------------------------------------------
# Distribution CSV export
# ---------------------------------------------------------------------------

def save_client_distribution(
    clients: List[Dict],
    class_names: List[str],
    output_path: str,
) -> pd.DataFrame:
    """
    Write the per-client class distribution to a CSV file.

    The resulting CSV has columns:
        client_id, class_id, class_name, sample_count

    Args:
        clients:     List of client dicts returned by :func:`dirichlet_partition`.
        class_names: Ordered list of class name strings.
        output_path: Destination CSV path (parent dirs are created if needed).

    Returns:
        The DataFrame that was written to disk.
    """
    rows = []
    for client in clients:
        for class_id, count in client["class_counts"].items():
            rows.append(
                {
                    "client_id":   client["client_id"],
                    "class_id":    class_id,
                    "class_name":  class_names[class_id],
                    "sample_count": count,
                }
            )

    df = pd.DataFrame(rows).sort_values(["client_id", "class_id"])
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[partition] Distribution saved → {output_path}")
    return df
