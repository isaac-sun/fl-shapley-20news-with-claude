"""
src/device_utils.py
===================
Cross-platform device detection for CPU / CUDA / Apple MPS.

Priority order (auto mode):  CUDA  >  MPS  >  CPU

Supported environments
----------------------
* Linux / Windows + NVIDIA GPU  → CUDA
* macOS Apple Silicon (M1/M2/M3/M4) → MPS (Metal Performance Shaders)
* Any platform fallback            → CPU

Usage
-----
    from src.device_utils import get_device, device_info

    device = get_device("auto")          # let the code decide
    device = get_device("mps")           # force MPS (M-series Mac)
    device = get_device("cuda")          # force CUDA
    device = get_device("cpu")           # force CPU

    print(device_info(device))           # human-readable description

MPS notes
---------
* Requires macOS 12.3+ and PyTorch >= 1.12.
* float64 (double) tensors are NOT supported on MPS → always use float32.
* Sparse tensor operations are not supported on MPS → convert to dense first
  (handled automatically by src/model_utils.sparse_to_tensor).
"""

from __future__ import annotations

import logging
import platform
import sys

import torch

logger = logging.getLogger("fl_shapley")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_device(preference: str = "auto") -> torch.device:
    """
    Select and return the best available torch.device.

    Args:
        preference: One of ``"auto"``, ``"cuda"``, ``"mps"``, ``"cpu"``.

    Returns:
        A ``torch.device`` object.
    """
    pref = preference.strip().lower()

    # ---- Explicit CPU request ------------------------------------------
    if pref == "cpu":
        logger.info("[device] Using CPU (forced)")
        return torch.device("cpu")

    # ---- Explicit CUDA request -----------------------------------------
    if pref == "cuda":
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            logger.info(f"[device] Using CUDA: {name}")
            return torch.device("cuda")
        logger.warning("[device] CUDA requested but unavailable → CPU")
        return torch.device("cpu")

    # ---- Explicit MPS request ------------------------------------------
    if pref == "mps":
        if _mps_available():
            logger.info("[device] Using Apple MPS (M-series chip)")
            return torch.device("mps")
        logger.warning("[device] MPS requested but unavailable → CPU")
        return torch.device("cpu")

    # ---- Auto selection: CUDA > MPS > CPU ------------------------------
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        logger.info(f"[device] Auto → CUDA: {name}")
        return torch.device("cuda")

    if _mps_available():
        logger.info("[device] Auto → Apple MPS")
        return torch.device("mps")

    logger.info("[device] Auto → CPU")
    return torch.device("cpu")


def device_info(device: torch.device) -> str:
    """
    Return a human-readable one-line description of *device*.

    Examples::

        "CUDA  – NVIDIA GeForce RTX 4090  (VRAM 24 GB)"
        "MPS   – Apple Silicon (Metal Performance Shaders)"
        "CPU   – macOS Darwin arm64"
    """
    d = device.type

    if d == "cuda":
        idx  = device.index or 0
        name = torch.cuda.get_device_name(idx)
        vram = torch.cuda.get_device_properties(idx).total_memory // (1024 ** 3)
        return f"CUDA  – {name}  (VRAM {vram} GB)"

    if d == "mps":
        return (
            f"MPS   – Apple Silicon  "
            f"(macOS {platform.mac_ver()[0]}, PyTorch {torch.__version__})"
        )

    # CPU
    arch = platform.machine()
    os_  = platform.system()
    return f"CPU   – {os_} {arch}  (PyTorch {torch.__version__})"


def print_device_summary() -> torch.device:
    """
    Detect the best device, print a summary table, and return it.

    Intended for use at the top of training scripts and notebooks.
    """
    device = get_device("auto")

    print("=" * 55)
    print("  Device Summary")
    print("=" * 55)
    print(f"  Python      : {sys.version.split()[0]}")
    print(f"  PyTorch     : {torch.__version__}")
    print(f"  Platform    : {platform.system()} {platform.machine()}")
    print(f"  CUDA avail  : {torch.cuda.is_available()}")

    try:
        mps_ok = torch.backends.mps.is_available() and torch.backends.mps.is_built()
    except AttributeError:
        mps_ok = False
    print(f"  MPS  avail  : {mps_ok}")
    print(f"  ➜  Selected : {device_info(device)}")
    print("=" * 55)

    return device


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _mps_available() -> bool:
    """Return True iff Apple MPS is both built and available at runtime."""
    try:
        return (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_built()
            and torch.backends.mps.is_available()
        )
    except Exception:
        return False
