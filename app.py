"""Hugging Face Gradio-SDK entrypoint; the primary UI/API remain React/FastAPI."""

import os

import uvicorn

from deployment.gradio_app import create_space_app

app = create_space_app()

if __name__ == "__main__":
    config = uvicorn.Config(
        app,
        host=os.getenv("FLOWFORGE_HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "7860")),
        workers=1,
        limit_concurrency=128,
    )
    server = uvicorn.Server(config)
    # Spaces normally reports startup from Gradio's launch() hook. Mounting
    # Gradio inside FastAPI requires the same pinned SDK hook explicitly.
    # Supply the actual server so GPU workers can close inherited listener FDs.
    app.state.benchmark_blocks.server = server
    if os.getenv("SPACES_ZERO_GPU", "").lower() in {"true", "1", "t"}:
        from spaces.zero import startup

        startup()
    server.run()
