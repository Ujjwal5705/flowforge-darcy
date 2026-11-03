"""Free Gradio Space adapter. The main application remains React + FastAPI.

The provider's shared ZeroGPU tier hosts this Gradio-compatible app. The small FNO
uses CPU only: no GPU allocation, GPU quota, paid hardware or inference API is used.
"""

import json
import os
from pathlib import Path

import gradio as gr
from fastapi.staticfiles import StaticFiles

from backend.app import create_app


def create_space_app():
    application = create_app(serve_frontend=False)
    directory = Path(os.getenv("FLOWFORGE_ARTIFACT_DIR", "artifacts/darcy-fno"))
    examples_path = directory / "examples.json"
    choices = (
        [row["id"] for row in json.loads(examples_path.read_text())]
        if examples_path.is_file()
        else []
    )

    def benchmark(sample_id: str):
        runtime = application.state.runtime
        if runtime is None:
            raise gr.Error("The trained model is not ready. Please retry later.")
        sample = runtime.examples.get(sample_id)
        if sample is None:
            raise gr.Error("Select one of the bundled held-out examples.")
        result = runtime.predict(sample["coefficient"], sample_id)
        return {
            "model_version": result["model_version"],
            "inference_ms": result["inference_ms"],
            "device": result["device"],
            "comparison": result["comparison"],
            "prediction": result["prediction"],
            "array_layout": result["array_layout"],
        }

    with gr.Blocks(title="FLOWFORGE benchmark check", analytics_enabled=False) as demo:
        gr.Markdown(
            "# FLOWFORGE · benchmark check\n"
            "[Open the full interactive React workspace](/). This auxiliary interface "
            "runs the same CPU FNO and saved preprocessing on a held-out example. "
            "It does not call a remote model or allocate a GPU."
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
