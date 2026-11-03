"""The single preprocessing and inference implementation used everywhere."""

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from neuralop.models import FNO

RESOLUTION = 32
INPUT_MIN = 0.1
INPUT_MAX = 1.0


@dataclass(frozen=True)
class Normalization:
    x_mean: float
    x_std: float
    y_mean: float
    y_std: float

    def __post_init__(self) -> None:
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Normalization must be finite")
        if self.x_std <= 0 or self.y_std <= 0:
            raise ValueError("Normalization standard deviations must be positive")

    @classmethod
    def fit(cls, x: torch.Tensor, y: torch.Tensor) -> "Normalization":
        return cls(
            x.mean().item(),
            x.std(unbiased=False).item(),
            y.mean().item(),
            y.std(unbiased=False).item(),
        )

    def encode_x(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.x_mean) / self.x_std

    def encode_y(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.y_mean) / self.y_std

    def decode_y(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.y_std + self.y_mean


def create_model(architecture: dict) -> FNO:
    return FNO(
        n_modes=tuple(architecture["n_modes"]),
        hidden_channels=architecture["hidden_channels"],
        n_layers=architecture["n_layers"],
        in_channels=1,
        out_channels=1,
        positional_embedding="grid",
        projection_channel_ratio=2,
    )


def parameter_counts(model: torch.nn.Module) -> dict[str, int]:
    parameters = [p for p in model.parameters() if p.requires_grad]
    return {
        "tensor_elements": sum(p.numel() for p in parameters),
        "real_scalars": sum(p.numel() * (2 if p.is_complex() else 1) for p in parameters),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_weights(model: torch.nn.Module, path: Path) -> None:
    # neuraloperator adds callable constructor metadata to state_dict. Keep only
    # tensors, and save our explicit JSON constructor config separately. This
    # permits weights_only=True without unsafe pickle allowlists.
    state = model.state_dict()
    tensors = {
        key: value.detach().cpu().clone()
        for key, value in state.items()
        if isinstance(value, torch.Tensor)
    }
    if any(
        key != "_metadata" and not isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise ValueError("Unexpected non-tensor model state")
    temporary = path.with_suffix(".tmp")
    torch.save(tensors, temporary)
    temporary.replace(path)


@dataclass
class Surrogate:
    model: torch.nn.Module
    normalization: Normalization
    config: dict

    @torch.inference_mode()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, RESOLUTION, RESOLUTION):
            raise ValueError("Expected [batch,1,32,32] coefficient tensor")
        if not torch.isfinite(x).all():
            raise ValueError("Coefficient tensor must be finite")
        prediction = self.normalization.decode_y(self.model(self.normalization.encode_x(x)))
        if prediction.shape != x.shape or not torch.isfinite(prediction).all():
            raise RuntimeError("Model produced an invalid solution")
        return prediction


def load_artifact(directory: Path) -> Surrogate:
    config_path, weights_path = directory / "config.json", directory / "best.pt"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError("Model artifact is missing config.json or best.pt")
    config = json.loads(config_path.read_text())
    if config.get("format_version") != 1 or config.get("resolution") != RESOLUTION:
        raise ValueError("Unsupported model artifact format or resolution")
    if file_sha256(weights_path) != config["checkpoint_sha256"]:
        raise ValueError("Model checkpoint checksum does not match its configuration")
    model = create_model(config["architecture"])
    weights = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(weights, dict) or not all(
        isinstance(v, torch.Tensor) for v in weights.values()
    ):
        raise ValueError("Checkpoint must contain tensor weights only")
    model.load_state_dict(weights, strict=True)
    model.eval()
    return Surrogate(model, Normalization(**config["normalization"]), config)
