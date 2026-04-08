"""
src/utils.py
============
General-purpose utilities: config loading, seed setting, logging, and
filesystem helpers used across the entire project.
"""

import os
import random
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load YAML configuration file and return as a plain Python dict.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Dictionary of configuration values.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return cfg


def save_config(cfg: Dict[str, Any], path: str) -> None:
    """Persist a config dictionary back to a YAML file."""
    with open(path, "w", encoding="utf-8") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """
    Set random seeds for Python, NumPy, and the OS hash seed so that all
    random operations in the project are reproducible.

    Args:
        seed: Integer seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def get_logger(
    name: str = "fl_shapley",
    log_level: str = "INFO",
    log_file: str | None = None,
) -> logging.Logger:
    """
    Create (or retrieve) a named logger with a console handler and an
    optional file handler.

    Args:
        name:      Logger name (reuses an existing logger if already created).
        log_level: Logging level string, e.g. "INFO", "DEBUG".
        log_file:  Optional path to a log file.

    Returns:
        Configured Logger instance.
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if log_file:
        ensure_dir(os.path.dirname(log_file))
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def ensure_dir(path: str | None) -> None:
    """
    Create a directory (and any missing parents) if it does not already exist.
    Silently ignores None or empty paths.

    Args:
        path: Directory path to create.
    """
    if path:
        Path(path).mkdir(parents=True, exist_ok=True)


def experiment_output_dir(base_dir: str, attack_type: str) -> str:
    """
    Return (and create) a per-experiment sub-directory inside *base_dir*.

    Example: ``outputs/clean/``, ``outputs/freerider/``.

    Args:
        base_dir:    Root output directory from config.
        attack_type: One of ``"clean"``, ``"freerider"``, ``"poisoning"``.

    Returns:
        Path string of the created sub-directory.
    """
    out = os.path.join(base_dir, attack_type)
    ensure_dir(out)
    return out
