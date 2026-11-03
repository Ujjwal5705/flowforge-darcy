"""Public API checks, including real trained-checkpoint integration tests."""

import copy
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import create_app


@pytest.fixture
def unavailable(tmp_path):
    with TestClient(create_app(tmp_path / "missing", tmp_path / "frontend")) as client:
        yield client


@pytest.fixture(scope="module")
def trained():
    directory = Path("artifacts/darcy-fno")
    assert (directory / "evaluation.json").is_file(), (
        "Integration tests require the real artifact. Run training and evaluation first."
    )
    with TestClient(create_app(directory)) as client:
        yield client


def valid_input():
    return {"coefficient": [[0.1] * 32 for _ in range(32)]}


def test_unavailable_health_is_honest_and_prediction_returns_503(unavailable):
    result = unavailable.get("/api/health")
    assert result.status_code == 200
    assert result.json()["model_loaded"] is False
    assert result.json()["status"] == "degraded"
    response = unavailable.post("/api/predict", json=valid_input())
    assert response.status_code == 503
    assert "Traceback" not in response.text
    assert "/Users/" not in response.text


@pytest.mark.parametrize(
    "mutation", ["rows", "columns", "negative", "too_high", "string", "bool", "extra"]
)
def test_invalid_input_is_rejected(unavailable, mutation):
    body = valid_input()
    if mutation == "rows":
        body["coefficient"].pop()
    elif mutation == "columns":
        body["coefficient"][0].pop()
    elif mutation == "extra":
        body["server_path"] = "/tmp/never-opened"
    else:
        body["coefficient"][0][0] = {"negative": -1, "too_high": 2, "string": "0.5", "bool": True}[
            mutation
        ]
    response = unavailable.post("/api/predict", json=body)
    assert response.status_code == 422
    assert "input" not in response.json().get("issues", [{}])[0]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_json_is_rejected_without_error_serialization_failure(unavailable, value):
    body = valid_input()
    body["coefficient"][1][2] = value
    response = unavailable.post(
        "/api/predict", content=json.dumps(body), headers={"Content-Type": "application/json"}
    )
    assert response.status_code == 422
    assert "NaN" not in response.text and "Infinity" not in response.text


def test_malformed_json_is_rejected(unavailable):
    assert (
        unavailable.post(
            "/api/predict", content=b'{"coefficient":', headers={"Content-Type": "application/json"}
        ).status_code
        == 422
    )


def test_large_body_is_rejected_before_parsing(unavailable):
    response = unavailable.post("/api/predict", content=b" " * (65536 + 1))
    assert response.status_code == 413


def test_chunked_body_limit_does_not_depend_on_content_length(unavailable):
    def chunks():
        yield b" " * 40000
        yield b" " * 40000

    assert unavailable.post("/api/predict", content=chunks()).status_code == 413


def test_real_health_metadata_and_examples(trained):
    health = trained.get("/api/health").json()
    assert health["model_loaded"] is True and health["device"] == "cpu"
    info = trained.get("/api/model-info")
    assert info.status_code == 200
    assert info.json()["training"]["split_sizes"] == {"train": 800, "validation": 112, "test": 112}
    assert len(info.json()["checkpoint_sha256"]) == 64
    examples = trained.get("/api/examples").json()
    assert len(examples) == 6
    assert all("coefficient" not in example for example in examples)
    assert trained.get("/api/examples/unknown").status_code == 404
    assert len(trained.get("/api/training-history").json()) == 80


def test_real_prediction_and_reference_metrics_match_independent_calculation(trained):
    sample = trained.get("/api/examples/test-01").json()
    before = id(trained.app.state.runtime.surrogate.model)
    response = trained.post(
        "/api/predict", json={"coefficient": sample["coefficient"], "example_id": sample["id"]}
    )
    assert response.status_code == 200
    result = response.json()
    prediction, reference = np.asarray(result["prediction"]), np.asarray(result["reference"])
    assert prediction.shape == (32, 32) and np.isfinite(prediction).all()
    assert result["inference_ms"] > 0
    expected = np.linalg.norm(prediction - reference) / np.linalg.norm(reference)
    assert result["comparison"]["relative_l2"] == pytest.approx(expected)
    np.testing.assert_allclose(
        result["absolute_error"], np.abs(prediction - reference), rtol=2e-5, atol=1e-8
    )
    assert id(trained.app.state.runtime.surrogate.model) == before


def test_reference_cannot_be_attached_to_an_edited_field(trained):
    sample = trained.get("/api/examples/test-01").json()
    field = copy.deepcopy(sample["coefficient"])
    field[0][0] = 0.5
    response = trained.post("/api/predict", json={"coefficient": field, "example_id": sample["id"]})
    assert response.status_code == 422
    result = trained.post("/api/predict", json={"coefficient": field}).json()
    assert result["reference"] is None and result["comparison"] is None
    assert len(result["warnings"]) >= 2
