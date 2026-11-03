"""The optional hosting adapter preserves the API, static UI and one model lifespan."""

from fastapi.testclient import TestClient

from deployment.gradio_app import create_space_app


def test_space_adapter_preserves_routes_and_shared_runtime():
    with TestClient(create_space_app()) as client:
        assert client.get("/api/health").json()["model_loaded"] is True
        identity = id(client.app.state.runtime.surrogate.model)
        assert client.get("/").status_code == 200
        assert client.get("/benchmark/").status_code == 200
        config = client.get("/benchmark/config").json()
        assert "benchmark_check" in [item.get("api_name") for item in config["dependencies"]]
        sample = client.get("/api/examples/test-01").json()
        response = client.post(
            "/api/predict", json={"coefficient": sample["coefficient"], "example_id": sample["id"]}
        )
        assert response.status_code == 200
        assert response.json()["comparison"]["relative_l2"] >= 0
        assert id(client.app.state.runtime.surrogate.model) == identity
        assert (
            client.post(
                "/benchmark/gradio_api/upload", files={"files": ("unused.pkl", b"not accepted")}
            ).status_code
            == 404
        )
        assert client.get("/benchmark/gradio_api/file=/etc/passwd").status_code == 404
