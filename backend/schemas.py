"""Bounded numeric contracts; array layout is always [x_index][y_index]."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from ml.model import INPUT_MAX, INPUT_MIN, RESOLUTION

Coefficient = Annotated[float, Field(strict=True, ge=INPUT_MIN, le=INPUT_MAX, allow_inf_nan=False)]
Row = Annotated[list[Coefficient], Field(min_length=RESOLUTION, max_length=RESOLUTION)]
FieldArray = Annotated[list[Row], Field(min_length=RESOLUTION, max_length=RESOLUTION)]


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    coefficient: FieldArray
    example_id: Annotated[str | None, Field(pattern=r"^test-\d{2}$", max_length=16)] = None


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str | None
    device: str | None


class Metrics(BaseModel):
    mse: float
    mae: float
    relative_l2: float


class Summary(BaseModel):
    minimum: float
    maximum: float
    mean: float


class ExampleSummary(BaseModel):
    id: str
    name: str
    source_index: int
    split: Literal["test"]
    has_reference: bool = True


class Example(ExampleSummary):
    coefficient: list[list[float]]
    reference: list[list[float]]


class Prediction(BaseModel):
    prediction: list[list[float]]
    shape: list[int]
    array_layout: str
    inference_ms: float
    model_version: str
    checkpoint_sha256: str
    device: str
    summary: Summary
    warnings: list[str]
    reference: list[list[float]] | None = None
    absolute_error: list[list[float]] | None = None
    comparison: Metrics | None = None
    example_id: str | None = None


class Architecture(BaseModel):
    n_modes: list[int]
    hidden_channels: int
    n_layers: int
    in_channels: int
    out_channels: int
    positional_embedding: str
    projection_channel_ratio: int


class TrainingInfo(BaseModel):
    epochs: int
    best_epoch: int
    seed: int
    batch_size: int
    learning_rate: float
    training_seconds: float
    parameters: dict[str, int]
    split_sizes: dict[str, int]
    hardware: dict[str, str | int]
    versions: dict[str, str]
    trained_at_utc: str


class Provenance(BaseModel):
    source: str
    doi: str
    filename: str
    url: str
    license: str
    subset: str
    sample_count: int
    source_resolution: list[int]
    subsampling: str
    array_sha256: dict[str, str]
    full_source_checksum_verified: bool


class Latency(BaseModel):
    median_ms: float
    p95_ms: float
    warmup_calls: int
    timed_calls: int
    batch_size: int
    scope: str
    device: str
    threads: int
    machine: str


class HistoryRow(BaseModel):
    epoch: int
    train_normalized_mse: float
    validation_mse: float
    validation_relative_l2: float
    learning_rate: float
    seconds: float


class ModelInfo(BaseModel):
    model_version: str
    checkpoint_sha256: str
    pde: str
    equation: str
    domain: str
    boundary: str
    forcing: float
    resolution: list[int]
    input_shape: list[int]
    output_shape: list[int]
    supported_input_range: list[float]
    training_coefficient_values: list[float]
    array_layout: str
    solution_units: str
    boundary_enforced: bool
    architecture: Architecture
    normalization: dict[str, float]
    training: TrainingInfo
    evaluation: Metrics
    baseline: Metrics
    latency: Latency
    provenance: Provenance
    coordinates: dict[str, list[float]]
    limitations: list[str]
