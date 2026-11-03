"""Independent invariants for transformations, metrics, splits and trusted reload."""

import hashlib
import json
import math

import numpy as np
import pytest
import torch

from ml.data import load_dataset, make_split
from ml.model import (
    Normalization,
    create_model,
    file_sha256,
    load_artifact,
    save_weights,
)
from ml.pipeline import metrics


def test_normalization_round_trip_and_train_only_statistics():
    x = torch.tensor([1.0, 3.0])
    y = torch.tensor([2.0, 6.0])
    norm = Normalization.fit(x, y)
    assert norm.x_mean == 2 and norm.x_std == 1
    assert norm.y_mean == 4 and norm.y_std == 2
    assert torch.equal(norm.encode_x(x), torch.tensor([-1.0, 1.0]))
    unseen = torch.tensor([10.0])
    assert torch.equal(norm.decode_y(norm.encode_y(unseen)), unseen)
    assert norm.y_mean == 4


def test_metrics_average_sample_ratios_not_batch_norms():
    p = np.array([[3.0, 4.0], [0.0, 2.0]])
    y = np.array([[0.0, 5.0], [0.0, 1.0]])
    result = metrics(p, y)
    assert result["mse"] == pytest.approx(2.75)
    assert result["mae"] == pytest.approx(1.25)
    assert result["relative_l2"] == pytest.approx((math.sqrt(10) / 5 + 1) / 2)


def test_partitions_are_deterministic_disjoint_and_exhaustive():
    split = make_split(1024, 800, 112, 42)
    assert split == make_split(1024, 800, 112, 42)
    assert [len(v) for v in split.values()] == [800, 112, 112]
    flat = sum(split.values(), [])
    assert sorted(flat) == list(range(1024))
    assert len(set(flat)) == len(flat)


@pytest.fixture
def artifact(tmp_path):
    """Random FNO weights for engineering tests, not trained benchmark evidence."""
    torch.manual_seed(1)
    torch.set_num_threads(2)
    architecture = {"n_modes": [4, 4], "hidden_channels": 8, "n_layers": 2}
    model = create_model(architecture).eval()
    save_weights(model, tmp_path / "best.pt")
    config = {
        "format_version": 1,
        "resolution": 32,
        "architecture": architecture,
        "normalization": {"x_mean": 0.5, "x_std": 0.4, "y_mean": 0.2, "y_std": 0.1},
        "checkpoint_sha256": file_sha256(tmp_path / "best.pt"),
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path, model


def test_tensor_only_checkpoint_reloads_identical_predictions(artifact):
    path, original = artifact
    loaded = load_artifact(path)
    field = torch.full((1, 1, 32, 32), 0.7)
    with torch.inference_mode():
        expected = original(loaded.normalization.encode_x(field))
        expected = loaded.normalization.decode_y(expected)
    actual = loaded.predict(field)
    assert actual.shape == (1, 1, 32, 32)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_corrupt_checkpoint_is_rejected_before_loading(artifact):
    path, _ = artifact
    with (path / "best.pt").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        load_artifact(path)


def test_missing_dataset_has_a_useful_preparation_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="fetch_pdebench"):
        load_dataset(tmp_path / "missing.npz")


def test_dataset_hash_and_coordinate_integrity(tmp_path):
    # Synthetic transformation fixture only; not benchmark evidence.
    rng = np.random.default_rng(1)
    x = rng.choice(np.asarray([0.1, 1.0], dtype=np.float32), size=(16, 32, 32))
    y = rng.random((16, 32, 32), dtype=np.float32)
    coordinates = (np.arange(0, 128, 4, dtype=np.float32) + 0.5) / 128
    path = tmp_path / "fixture.npz"
    np.savez(path, x=x, y=y, x_coordinates=coordinates, y_coordinates=coordinates)
    metadata = {
        "source": "synthetic unit fixture",
        "array_sha256": {
            "nu": hashlib.sha256(x.tobytes()).hexdigest(),
            "tensor": hashlib.sha256(y.tobytes()).hexdigest(),
        },
    }
    path.with_suffix(".json").write_text(json.dumps(metadata))
    actual_x, actual_y, _ = load_dataset(path)
    assert actual_x.shape == actual_y.shape == (16, 1, 32, 32)
    y[0, 0, 0] += 1
    np.savez(path, x=x, y=y, x_coordinates=coordinates, y_coordinates=coordinates)
    with pytest.raises(ValueError, match="checksum"):
        load_dataset(path)
    y[0, 0, 0] -= 1
    # Restore the exact original bytes, then corrupt only coordinates.
    y = rng.random((16, 32, 32), dtype=np.float32)
    metadata["array_sha256"]["tensor"] = hashlib.sha256(y.tobytes()).hexdigest()
    path.with_suffix(".json").write_text(json.dumps(metadata))
    np.savez(path, x=x, y=y, x_coordinates=coordinates[::-1], y_coordinates=coordinates)
    with pytest.raises(ValueError, match="coordinates"):
        load_dataset(path)
