"""Validated, hashed dataset loading and frozen train/validation/test partitions."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from ml.model import RESOLUTION


def load_dataset(path: Path) -> tuple[torch.Tensor, torch.Tensor, dict]:
    if not path.is_file() or not path.with_suffix(".json").is_file():
        raise FileNotFoundError("Dataset missing. Run python scripts/fetch_pdebench.py first.")
    try:
        with np.load(path, allow_pickle=False) as data:
            x, y = data["x"].copy(), data["y"].copy()
            coordinates = {key: data[key].tolist() for key in ["x_coordinates", "y_coordinates"]}
        metadata = json.loads(path.with_suffix(".json").read_text())
    except (OSError, ValueError, KeyError) as error:
        raise ValueError("Dataset is corrupt or missing required numeric arrays") from error
    if (
        x.ndim != 3
        or x.shape != y.shape
        or x.shape[1:] != (RESOLUTION, RESOLUTION)
        or not 16 <= len(x) <= 2048
    ):
        raise ValueError("Dataset must contain 16–2048 paired 32×32 fields")
    if x.dtype != np.float32 or y.dtype != np.float32:
        raise ValueError("Dataset must use float32 arrays")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Dataset contains non-finite values")
    if not np.isin(x, np.asarray([0.1, 1.0], dtype=np.float32)).all():
        raise ValueError("Dataset coefficients must be the documented phases 0.1 and 1.0")
    for key, array in [("nu", x), ("tensor", y)]:
        if hashlib.sha256(array.tobytes()).hexdigest() != metadata["array_sha256"][key]:
            raise ValueError("Dataset checksum mismatch; prepare the subset again")
    for coordinate in coordinates.values():
        expected = (np.arange(0, 128, 4, dtype=np.float32) + 0.5) / 128
        if (
            len(coordinate) != RESOLUTION
            or not np.isfinite(coordinate).all()
            or not np.array_equal(np.asarray(coordinate, dtype=np.float32), expected)
        ):
            raise ValueError("Invalid dataset coordinates")
    metadata["coordinates"] = coordinates
    return torch.from_numpy(x[:, None]), torch.from_numpy(y[:, None]), metadata


def make_split(count: int, train_size: int, val_size: int, seed: int) -> dict[str, list[int]]:
    if train_size < 2 or val_size < 1 or train_size + val_size >= count:
        raise ValueError("Split needs at least 2 training, 1 validation and 1 test example")
    order = np.random.default_rng(seed).permutation(count).tolist()
    return {
        "train": order[:train_size],
        "validation": order[train_size : train_size + val_size],
        "test": order[train_size + val_size :],
    }


def assert_no_duplicate_leakage(x: torch.Tensor, split: dict[str, list[int]]) -> None:
    hashes = [hashlib.sha256(sample.numpy().tobytes()).hexdigest() for sample in x]
    groups = {key: {hashes[i] for i in indices} for key, indices in split.items()}
    if (
        groups["train"] & groups["validation"]
        or groups["train"] & groups["test"]
        or groups["validation"] & groups["test"]
    ):
        raise ValueError("Duplicate coefficient fields cross partitions; use a group-aware split")
