"""
src/data_utils.py
=================
Load the 20 Newsgroups text dataset, build a global TF-IDF feature matrix,
and return consistent train / validation / test splits.

Design note
-----------
The TF-IDF vectorizer is fitted on the **training** split only and then used
to transform the val and test splits.  In a real FL setting each client would
fit its own vectorizer, but using a shared, globally-fitted vectorizer is the
standard simplification in text-FL research because it gives a consistent
feature space without increasing experimental complexity.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Class meta-data
# ---------------------------------------------------------------------------

#: The 20 canonical newsgroup category names (same order as sklearn target).
NEWSGROUP_NAMES: List[str] = [
    "alt.atheism",
    "comp.graphics",
    "comp.os.ms-windows.misc",
    "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware",
    "comp.windows.x",
    "misc.forsale",
    "rec.autos",
    "rec.motorcycles",
    "rec.sport.baseball",
    "rec.sport.hockey",
    "sci.crypt",
    "sci.electronics",
    "sci.med",
    "sci.space",
    "soc.religion.christian",
    "talk.politics.guns",
    "talk.politics.mideast",
    "talk.politics.misc",
    "talk.religion.misc",
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(
    test_size: float = 0.15,
    val_size: float = 0.15,
    max_tfidf_features: int = 10_000,
    random_seed: int = 42,
) -> Tuple[
    csr_matrix, csr_matrix, csr_matrix,   # X_train, X_val, X_test
    np.ndarray, np.ndarray, np.ndarray,    # y_train, y_val,  y_test
    TfidfVectorizer,                        # fitted vectorizer
    List[str],                              # class_names
]:
    """
    Download (or load from cache) 20 Newsgroups, build TF-IDF features, and
    return stratified train / val / test splits.

    Args:
        test_size:           Proportion of the full dataset reserved for test.
        val_size:            Proportion reserved for validation (Shapley utility).
        max_tfidf_features:  Maximum vocabulary size for TF-IDF.
        random_seed:         Seed for all random operations.

    Returns:
        X_train, X_val, X_test  – sparse float matrices (n_samples × n_features)
        y_train, y_val,  y_test – integer label arrays
        vectorizer               – TfidfVectorizer fitted on train split
        class_names              – list of 20 category name strings
    """
    print("[data] Fetching 20 Newsgroups dataset …")
    news = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes"),   # strip metadata leakage
        random_state=random_seed,
    )
    X_raw: List[str] = news.data
    y: np.ndarray = news.target
    class_names: List[str] = list(news.target_names)

    print(f"[data] Total samples: {len(X_raw)}  |  Classes: {len(class_names)}")

    # ---- Split: train+val  vs  test ----------------------------------------
    X_tv, X_test_raw, y_tv, y_test = train_test_split(
        X_raw, y,
        test_size=test_size,
        stratify=y,
        random_state=random_seed,
    )

    # ---- Split: train  vs  val  --------------------------------------------
    # val_ratio is relative to the train+val pool
    val_ratio_relative = val_size / (1.0 - test_size)
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_tv, y_tv,
        test_size=val_ratio_relative,
        stratify=y_tv,
        random_state=random_seed,
    )

    print(
        f"[data] Split → train: {len(y_train)}  "
        f"val: {len(y_val)}  test: {len(y_test)}"
    )

    # ---- TF-IDF  -----------------------------------------------------------
    print(f"[data] Building TF-IDF (max_features={max_tfidf_features}) …")
    vectorizer = TfidfVectorizer(
        max_features=max_tfidf_features,
        sublinear_tf=True,   # apply 1 + log(tf) scaling
        min_df=2,            # ignore very rare terms
        max_df=0.95,         # ignore near-universal terms
        stop_words="english",
        dtype=np.float64,   # SGDClassifier requires float64
    )

    X_train: csr_matrix = vectorizer.fit_transform(X_train_raw)
    X_val:   csr_matrix = vectorizer.transform(X_val_raw)
    X_test:  csr_matrix = vectorizer.transform(X_test_raw)

    print(f"[data] Feature dimension: {X_train.shape[1]}")

    return X_train, X_val, X_test, y_train, y_val, y_test, vectorizer, class_names
