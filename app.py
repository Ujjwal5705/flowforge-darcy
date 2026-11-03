"""Hugging Face Gradio-SDK entrypoint; the primary UI/API remain React/FastAPI."""

import os

import uvicorn

from deployment.gradio_app import create_space_app

app = create_space_app()

if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.getenv("FLOWFORGE_HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "7860")),
        workers=1,
        limit_concurrency=16,
    )
