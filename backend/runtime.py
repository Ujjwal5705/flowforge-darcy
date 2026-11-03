"""Load once, keep trusted artifacts local, and serialize bounded CPU inference."""

import json
import threading
import time
from pathlib import Path

import numpy as np
import torch

from ml.model import load_artifact
from ml.pipeline import metrics


class ReferenceMismatch(ValueError):
    pass


class ModelBusy(RuntimeError):
    pass


class Runtime:
    def __init__(self, directory: Path, threads: int = 2):
        if not 1 <= threads <= 8:
            raise ValueError("Inference threads must be between 1 and 8")
        torch.set_num_threads(threads)
        self.surrogate = load_artifact(directory)
        self.config = self.surrogate.config
        self.evaluation = json.loads((directory / "evaluation.json").read_text())
        if self.evaluation["checkpoint_sha256"] != self.config["checkpoint_sha256"]:
            raise ValueError("Evaluation report belongs to another checkpoint")
        self.provenance = json.loads((directory / "dataset.json").read_text())
        self.history = json.loads((directory / "history.json").read_text())
        examples = json.loads((directory / "examples.json").read_text())
        self.examples = {example["id"]: example for example in examples}
        for example in self.examples.values():
            if example["split"] != "test":
                raise ValueError("Only held-out examples may be served with references")
            for key in ["coefficient", "reference"]:
                array = np.asarray(example[key], dtype=np.float32)
                if array.shape != (32, 32) or not np.isfinite(array).all():
                    raise ValueError("Bundled example has invalid fields")
        self.version = "darcy-fno-" + self.config["checkpoint_sha256"][:12]
        self.lock = threading.Lock()
        self.surrogate.predict(torch.full((1, 1, 32, 32), 0.55))

    def info(self) -> dict:
        config, source = self.config, self.provenance
        training_keys = [
            "epochs",
            "best_epoch",
            "seed",
            "batch_size",
            "learning_rate",
            "training_seconds",
            "parameters",
            "split_sizes",
            "hardware",
            "versions",
            "trained_at_utc",
        ]
        return {
            "model_version": self.version,
            "checkpoint_sha256": config["checkpoint_sha256"],
            "pde": "2D Darcy flow",
            "equation": "−∇·(a(x,y) ∇u(x,y)) = 1",
            "domain": "Unit square [0,1]²; sampled interior cell centers",
            "boundary": (
                "Dataset generator uses zero-valued exterior ghost cells; "
                "interior edge cells need not be zero."
            ),
            "forcing": 1.0,
            "resolution": [32, 32],
            "input_shape": [1, 32, 32],
            "output_shape": [1, 32, 32],
            "supported_input_range": [0.1, 1.0],
            "training_coefficient_values": [0.1, 1.0],
            "array_layout": "field[x_index][y_index]; increasing y is displayed upward",
            "solution_units": "PDEBench dataset units; no SI pressure calibration",
            "boundary_enforced": False,
            "architecture": config["architecture"],
            "normalization": config["normalization"],
            "training": {key: config[key] for key in training_keys},
            "evaluation": self.evaluation["model"],
            "baseline": self.evaluation["baseline"],
            "latency": self.evaluation["latency"],
            "provenance": {
                **{
                    key: source[key]
                    for key in [
                        "source",
                        "doi",
                        "filename",
                        "url",
                        "license",
                        "source_resolution",
                        "subsampling",
                        "array_sha256",
                        "full_source_checksum_verified",
                    ]
                },
                "subset": (
                    f"Source samples {min(source['sample_indices'])}–"
                    f"{max(source['sample_indices'])}, with frozen custom partitions"
                ),
                "sample_count": len(source["sample_indices"]),
            },
            "coordinates": source["coordinates"],
            "limitations": [
                "A learned surrogate for one fixed PDE, forcing, geometry and 32×32 grid.",
                (
                    "Range-valid custom fields, including binary presets, "
                    "may be outside the training distribution."
                ),
                (
                    "Intermediate coefficients were not used for training; "
                    "physical boundary behavior and positivity are not enforced."
                ),
                (
                    "References are sampled fine-grid dataset solutions, "
                    "not numerical solves of arbitrary coarse input."
                ),
                "Metrics describe a small custom held-out cohort, not the full PDEBench benchmark.",
            ],
        }

    def predict(self, coefficient: list[list[float]], example_id: str | None) -> dict:
        field = np.asarray(coefficient, dtype=np.float32)
        example = None
        if example_id:
            if example_id not in self.examples:
                raise KeyError(example_id)
            example = self.examples[example_id]
            if not np.array_equal(field, np.asarray(example["coefficient"], dtype=np.float32)):
                raise ReferenceMismatch(
                    "The field was edited; omit example_id to predict without a reference."
                )
        if not self.lock.acquire(timeout=5):
            raise ModelBusy("The model is busy. Please retry shortly.")
        try:
            started = time.perf_counter()
            prediction = self.surrogate.predict(torch.from_numpy(field)[None, None])[0, 0].numpy()
            inference_ms = (time.perf_counter() - started) * 1000
        finally:
            self.lock.release()
        warnings = (
            []
            if example
            else [
                "Custom input: no reference solution is available, "
                "and accuracy outside the training distribution is unknown."
            ]
        )
        if not np.isin(field, np.asarray([0.1, 1.0], dtype=np.float32)).all():
            warnings.append(
                "This field contains intermediate coefficients; training used only 0.1 and 1.0."
            )
        if float(prediction.min()) < 0:
            warnings.append(
                "The unconstrained surrogate predicted negative values; outputs are not clipped."
            )
        result = {
            "prediction": prediction.tolist(),
            "shape": [32, 32],
            "array_layout": "field[x_index][y_index]",
            "inference_ms": inference_ms,
            "model_version": self.version,
            "checkpoint_sha256": self.config["checkpoint_sha256"],
            "device": "cpu",
            "summary": {
                "minimum": float(prediction.min()),
                "maximum": float(prediction.max()),
                "mean": float(prediction.mean()),
            },
            "warnings": warnings,
            "example_id": example_id,
            "reference": None,
            "absolute_error": None,
            "comparison": None,
        }
        if example:
            reference = np.asarray(example["reference"], dtype=np.float32)
            result.update(
                reference=reference.tolist(),
                absolute_error=np.abs(prediction - reference).tolist(),
                comparison=metrics(prediction[None], reference[None]),
            )
        return result
