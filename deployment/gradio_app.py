"""Free Gradio Space adapter. The main application remains React + FastAPI.

The primary API uses CPU. The auxiliary benchmark genuinely runs the same weights
on an on-demand GPU on ZeroGPU, with a CPU fallback outside that provider.
"""

import json
import os
import time
from pathlib import Path

import spaces  # isort: skip

import gradio as gr
import numpy as np
import torch
from fastapi.staticfiles import StaticFiles

from backend.app import create_app
from ml.model import load_artifact
from ml.pipeline import metrics


def create_space_app():
    application = create_app(serve_frontend=False)
    directory = Path(os.getenv("FLOWFORGE_ARTIFACT_DIR", "artifacts/darcy-fno"))
    examples_path = directory / "examples.json"
    gpu_surrogate = None
    if os.getenv("SPACES_ZERO_GPU", "").lower() in {"true", "1", "t"}:
        # ZeroGPU emulates CUDA placement at startup, then supplies the real
        # device inside the decorated callback. Keep the primary CPU model separate.
        gpu_surrogate = load_artifact(directory)
        gpu_surrogate.model.to("cuda")
    choices = (
        [row["id"] for row in json.loads(examples_path.read_text())]
        if examples_path.is_file()
        else []
    )

    @spaces.GPU(duration=20)
    def benchmark(sample_id: str):
        runtime = application.state.runtime
        if runtime is None:
            raise gr.Error("The trained model is not ready. Please retry later.")
        sample = runtime.examples.get(sample_id)
        if sample is None:
            raise gr.Error("Select one of the bundled held-out examples.")
        if gpu_surrogate is None:
            result = runtime.predict(sample["coefficient"], sample_id)
            execution = "Local CPU fallback; primary workspace also uses CPU"
        else:
            field = np.asarray(sample["coefficient"], dtype=np.float32)
            torch.cuda.synchronize()
            started = time.perf_counter()
            prediction = (
                gpu_surrogate.predict(torch.from_numpy(field)[None, None].to("cuda"))[0, 0]
                .cpu()
                .numpy()
            )
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - started) * 1000
            reference = np.asarray(sample["reference"], dtype=np.float32)
            result = {
                "model_version": runtime.version,
                "checkpoint_sha256": gpu_surrogate.config["checkpoint_sha256"],
                "inference_ms": elapsed,
                "device": "cuda",
                "comparison": metrics(prediction[None], reference[None]),
                "prediction": prediction.tolist(),
                "array_layout": "field[x_index][y_index]",
            }
            execution = (
                f"On-demand {torch.cuda.get_device_name()}; timing includes input/output "
                "transfers, excludes queue/allocation and startup model placement"
            )
        return {
            "model_version": result["model_version"],
            "checkpoint_sha256": result["checkpoint_sha256"],
            "inference_ms": result["inference_ms"],
            "device": result["device"],
            "comparison": result["comparison"],
            "prediction": result["prediction"],
            "array_layout": result["array_layout"],
            "execution": execution,
        }

    with gr.Blocks(title="FLOWFORGE benchmark check", analytics_enabled=False) as demo:
        gr.Markdown(
            "# FLOWFORGE · benchmark check\n"
            "[Open the full interactive React workspace](/). This auxiliary interface "
            "runs the same FNO checkpoint and saved preprocessing on a held-out example. "
            "On this hosted Space, it requests a shared GPU and consumes free daily "
            "ZeroGPU quota; outside ZeroGPU it uses CPU. The main workspace/API use CPU."
        )
        sample_id = gr.Dropdown(
            choices=choices,
            value=choices[0] if choices else None,
            label="Held-out reference sample",
        )
        run = gr.Button("Run benchmark check")
        output = gr.JSON(label="Actual prediction and comparison")
        run.click(benchmark, inputs=sample_id, outputs=output, api_name="benchmark_check")
    application = gr.mount_gradio_app(
        application,
        demo,
        path="/benchmark",
        theme=gr.themes.Base(font=["Arial", "sans-serif"], font_mono=["Consolas", "monospace"]),
        show_error=False,
        ssr_mode=False,
        footer_links=[],
        run_history=False,
        max_file_size=65536,
    )
    frontend = Path(os.getenv("FLOWFORGE_FRONTEND_DIR", "frontend/dist"))
    application.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return application
