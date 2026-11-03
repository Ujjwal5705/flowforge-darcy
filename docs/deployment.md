# Deployment

The portable runtime is one non-root Python service with the built React assets,
trained checkpoint and six held-out examples. It uses port 7860, one worker and
two CPU inference threads. Training data and credentials are not runtime inputs.

## Local container

```bash
docker compose up --build -d
curl -fsS http://127.0.0.1:7860/api/health
docker compose down
```

The image installs CPU-only PyTorch from its official CPU index before the pinned
runtime requirements. The official uv 0.10.8 binary and BuildKit cache mounts avoid
repeated serial dependency downloads. Frontend assets are built in a Node stage.
Only the six required model/metadata files enter the final application layer.
The healthcheck verifies `model_loaded`, not merely a listening port.

## Hugging Face hosting decision

The existing authenticated `ujjwal75` account is not Pro. On 3 November 2026,
creating an owner-private Docker Space on `cpu-basic` returned **HTTP 402**:
the provider requires a Pro subscription for that tier. No plan upgrade, paid
hardware or paid storage was requested.

The documented alternative is a free Gradio Space on shared ZeroGPU for eligible
personal accounts. The provider accepted an **owner-private** candidate at
`ujjwal75/flowforge-darcy-2026`, with SDK Gradio 6.31.0, Python 3.12 and requested
hardware `zero-a10g`. Creation alone is not deployment verification.

The optional `deployment/gradio_app.py` adapter mounts a functional Gradio benchmark
check at `/benchmark`, preserving the React app at `/` and the existing FastAPI
API at `/api`. The same CPU model is loaded once through the shared lifespan.
It does not allocate a GPU or consume a remote inference API. Gradio's current
Starlette requirement motivated the compatible FastAPI 0.162.0 / Starlette 1.10.0
serving stack; the trained FNO, preprocessing and checkpoint are unchanged.

Install `requirements-space.lock` and run the adapter with:

```bash
.venv/bin/python app.py
```

The prepared Space metadata is in `deployment/space-README.md`. Source and built
assets are prepared from this session; the model is included locally at runtime.
The Space remains private until publication is explicitly authorized. The final
verification record will identify the deployed revision, URL and measured smoke
test once that work has actually succeeded.

References: [Spaces policy](https://huggingface.co/docs/hub/en/spaces-overview),
[free ZeroGPU eligibility](https://huggingface.co/docs/hub/en/spaces-zerogpu),
[Docker setup](https://huggingface.co/docs/hub/en/spaces-sdks-docker).
