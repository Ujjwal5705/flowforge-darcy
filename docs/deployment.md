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

The final ARM64 image was rebuilt and run on 4 November 2026: 286,636,476 image bytes,
and model-ready health 3.59 seconds after the startup poll began (immediately
after `docker run`). Observed memory was about 220 MiB after the final browser workload. The
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
Gradio's Starlette requirement motivated the compatible FastAPI 0.162.0 /
Starlette 1.10.0 hosting stack; the trained FNO, preprocessing and checkpoint are unchanged.
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

## Verified public release

Published with the owner's explicit approval on **4 November 2026** after private
CPU/CUDA verification. No paid upgrade, paid hardware or paid storage was requested.

- App: <https://ujjwal75-flowforge-darcy-2026.hf.space>
- Source: <https://huggingface.co/spaces/ujjwal75/flowforge-darcy-2026>
- Validated application source: `92a8c5cbca08d427f07ac75d2b336684540c1c52`.
- Validated Space revision: `6585fa77d21f8c60a00fc5f7d274e247921878c5`.
- Anonymous health/model-info/examples/prediction checks passed, with expected
  checkpoint SHA-256, finite 32×32 output, independently recalculated relative L2
  **0.04366721** for test-01, and matching local/remote CSS/JS hashes.
- **Six anonymous Chromium tests passed** in 1.3 minutes, including actual GPU
  callback execution, result visualization, downloads, editing, loading/error
  recovery, mobile layout and automated accessibility checks.
- Private CUDA verification used an NVIDIA RTX PRO 6000 Blackwell Server Edition
  MIG 2g.48gb. The sample relative L2 was **0.04366722**, and outputs agreed with
  CPU within `rtol=1e-4, atol=1e-5`. Its one observed call took 474.02 ms including
  input/output transfers, excluding allocation/queue/startup placement. This is
  a functional check, not a warmed GPU latency benchmark or speedup claim.
- Actual JavaScript response: **71,045 gzip bytes**, versus 228,009 identity bytes;
  decompression exactly matched the verified local build.

The sanitized machine-readable record is [deployment.json](results/deployment.json).
The revision identifiers describe the tested application; later documentation-only
updates retain the same runtime code and model. Startup/request logs were checked;
the initial concurrency rejection was fixed and the complete browser suite rerun.
Upstream Matplotlib/pyparsing and provider pip warnings were observed, without an
application exception in the successful runtime. Shared hosting availability and
GPU quota follow the provider's limits.

`scripts/prepare_space.py` creates a clean-commit snapshot and checks the model
hash. `scripts/upload_space.py` uploads it without changing visibility. The
prepared Space metadata is `deployment/space-README.md`. Raw training data,
credentials and the session log are excluded. Recheck the deployed app with:

```bash
.venv/bin/python scripts/check_deployment.py
# Optional: consumes a small amount of shared GPU quota
.venv/bin/python scripts/check_deployment.py --benchmark
FLOWFORGE_URL=https://ujjwal75-flowforge-darcy-2026.hf.space npm run test:e2e:space --prefix frontend
```

References: [Spaces policy](https://huggingface.co/docs/hub/en/spaces-overview),
[free ZeroGPU eligibility](https://huggingface.co/docs/hub/en/spaces-zerogpu),
[Docker setup](https://huggingface.co/docs/hub/en/spaces-sdks-docker).
