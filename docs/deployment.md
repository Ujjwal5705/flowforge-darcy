# Deployment

The portable runtime is one non-root Python service with the built React assets,
trained checkpoint and six held-out examples. It uses port 7860, one worker and
two CPU inference threads. Training data and credentials are not runtime inputs.
HTTP concurrency is capped at 128 so a browser can fetch the Gradio asset burst;
the independent model lock still serializes CPU inference with a five-second wait.
Text/JSON/static responses above 1 KiB use gzip when requested. The upstream
middleware excludes server-sent events, preserving the benchmark queue stream.

## Local container

```bash
docker compose up --build -d
curl -fsS http://127.0.0.1:7860/api/health
docker compose down
```

The image installs CPU-only PyTorch from its official CPU index before the pinned
runtime requirements. The official uv 0.10.8 binary and BuildKit cache mounts avoid
repeated serial dependency downloads. Frontend assets are built in a Node stage.
An optional `.build-wheels/` cache can supply official platform-compatible wheels
on slow connections; its wheel files are ignored and do not enter the final image.
The tracked empty directory permits ordinary builds directly from a fresh clone.
Only the six required model/metadata files enter the final application layer.
The healthcheck verifies `model_loaded`, not merely a listening port.

The ARM64 image was built and run on 3 November 2026 UTC: 286,636,421 image bytes,
about 235 MiB container memory after the browser workload, and model-ready health
3.57 seconds after the startup poll began (immediately after `docker run`). The
container runs as `flowforge`, health status is `healthy`, and five real browser
tests passed against its same-origin frontend/API on localhost:7861. A held-out
prediction returned finite 32×32 output and relative L2 0.04366723; independent
recalculation and the checkpoint SHA-256 matched. Startup/runtime logs contained
no application errors. These are local ARM64 observations, not hosted guarantees.

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
API at `/api`. The primary CPU model is loaded once through the shared lifespan.
The provider rejected startup without an `@spaces.GPU` function. The auxiliary
benchmark therefore genuinely runs the same checkpoint on an on-demand GPU using
the official Spaces 0.51.3 SDK. Its separate model is placed on emulated CUDA
during startup, following the provider's documented loading convention; allocation
occurs only when the benchmark is requested, with a 20-second duration limit.
It consumes free daily ZeroGPU quota and uses CPU outside ZeroGPU. The main API
and React workspace remain on CPU. No external model inference service is used.
The pinned SDK's `spaces.zero.startup()` hook is called explicitly by `app.py`:
its automatic report hooks `Blocks.launch()`, which a mounted FastAPI application
does not call. The mounted Blocks receive the actual Uvicorn server object so GPU
worker processes can close inherited listener sockets. This SDK integration is
version-sensitive and must be rechecked when upgrading Spaces.
Gradio's current Starlette requirement motivated the compatible FastAPI 0.162.0 / Starlette 1.10.0
serving stack; the trained FNO, preprocessing and checkpoint are unchanged.
The provider also installs Gradio's MCP and OAuth extras. Pydantic is pinned to
2.12.5 to satisfy their declared compatibility range; the first private build
identified the incompatible newer pin before any public release. ZeroGPU's
pre-build validator accepts the literal `torch==2.8.0` requirement but rejects
the compatible local-version suffix `+cpu`. The snapshot retains the accepted
public version and uses the provider-compatible CUDA build for the auxiliary
benchmark. The portable Docker runtime separately installs CPU-only PyTorch.
The snapshot also retains prebuilt `frontend/dist` in the SDK's generated Docker
context. The local Dockerfile builds that directory in its Node stage, whereas
the Gradio SDK copies it directly; their ignore rules therefore differ explicitly.

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
