"""Run: uvicorn backend.app:app --host 0.0.0.0 --port 7860 --workers 1."""

import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.gzip import GZipMiddleware

from backend.runtime import ModelBusy, ReferenceMismatch, Runtime
from backend.schemas import (
    Example,
    ExampleSummary,
    Health,
    HistoryRow,
    ModelInfo,
    Prediction,
    PredictRequest,
)

logger = logging.getLogger("flowforge")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)
logger.propagate = False
MAX_BODY_BYTES = 64 * 1024


class BodyLimitMiddleware:
    def __init__(self, app, limit=MAX_BODY_BYTES):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"].startswith("/benchmark"):
            path = scope["path"]
            if any(part.startswith(("file=", "proxy=")) for part in path.split("/")) or (
                path.rstrip("/").split("/")[-1] in {"upload", "upload_progress"}
            ):
                return await JSONResponse(
                    {"detail": "File uploads and file-path requests are not supported"}, 404
                )(scope, receive, send)
        if scope["type"] != "http" or scope["method"] not in {"POST", "PUT", "PATCH"}:
            return await self.app(scope, receive, send)
        headers = dict(scope["headers"])
        if b"content-length" in headers:
            try:
                length = int(headers[b"content-length"])
            except ValueError:
                return await JSONResponse({"detail": "Invalid Content-Length"}, 400)(
                    scope, receive, send
                )
            if length < 0 or length > self.limit:
                return await JSONResponse({"detail": "Request exceeds the 64 KiB limit"}, 413)(
                    scope, receive, send
                )
        body = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return
            body.extend(message.get("body", b""))
            if len(body) > self.limit:
                return await JSONResponse({"detail": "Request exceeds the 64 KiB limit"}, 413)(
                    scope, receive, send
                )
            if not message.get("more_body", False):
                break
        delivered = False

        async def replay():
            nonlocal delivered
            if not delivered:
                delivered = True
                return {"type": "http.request", "body": bytes(body), "more_body": False}
            return {"type": "http.disconnect"}

        return await self.app(scope, replay, send)


def create_app(
    artifact_dir: Path | None = None,
    frontend_dir: Path | None = None,
    serve_frontend: bool = True,
) -> FastAPI:
    directory = artifact_dir or Path(os.getenv("FLOWFORGE_ARTIFACT_DIR", "artifacts/darcy-fno"))
    frontend = frontend_dir or Path(os.getenv("FLOWFORGE_FRONTEND_DIR", "frontend/dist"))

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.runtime = None
        try:
            application.state.runtime = Runtime(directory, int(os.getenv("FLOWFORGE_THREADS", "2")))
            logger.info(
                json.dumps(
                    {
                        "event": "model_loaded",
                        "model_version": application.state.runtime.version,
                        "device": "cpu",
                    }
                )
            )
        except Exception as error:
            logger.error(
                json.dumps({"event": "model_not_ready", "error_type": type(error).__name__})
            )
        yield
        application.state.runtime = None

    application = FastAPI(title="FLOWFORGE", version="0.1.0", lifespan=lifespan)
    application.add_middleware(BodyLimitMiddleware)
    application.add_middleware(GZipMiddleware, minimum_size=1024, compresslevel=6)

    @application.exception_handler(RequestValidationError)
    async def invalid_request(_request, error):
        issues = [
            {"location": list(item["loc"]), "message": item["msg"]} for item in error.errors()[:6]
        ]
        return JSONResponse(
            {
                "detail": "Provide a 32×32 numeric field with finite values from 0.1 to 1.0.",
                "issues": issues,
            },
            status_code=422,
        )

    @application.middleware("http")
    async def response_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        policy = (
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: blob:; connect-src 'self'; object-src 'none'; base-uri 'self'; "
            "frame-ancestors 'self' https://huggingface.co https://*.hf.space"
        )
        if request.url.path.startswith("/benchmark"):
            # Gradio's trusted bootstrap uses inline scripts. The React workspace
            # retains the stricter self-only script policy above.
            policy = policy.replace("script-src 'self'", "script-src 'self' 'unsafe-inline'")
        elif request.url.path in {"/docs", "/redoc"}:
            policy = policy.replace(
                "script-src 'self'", "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net"
            ).replace("style-src 'self'", "style-src 'self' https://cdn.jsdelivr.net")
        response.headers["Content-Security-Policy"] = policy
        return response

    def runtime() -> Runtime:
        model = application.state.runtime
        if model is None:
            raise HTTPException(503, "The trained model is not ready. Please try again later.")
        return model

    @application.get("/api/health", response_model=Health)
    def health():
        model = application.state.runtime
        return {
            "status": "ok" if model else "degraded",
            "model_loaded": model is not None,
            "model_version": model.version if model else None,
            "device": "cpu" if model else None,
        }

    @application.get("/api/model-info", response_model=ModelInfo)
    def model_info():
        return runtime().info()

    @application.get("/api/training-history", response_model=list[HistoryRow])
    def history():
        return runtime().history

    @application.get("/api/examples", response_model=list[ExampleSummary])
    def examples():
        return list(runtime().examples.values())

    @application.get("/api/examples/{example_id}", response_model=Example)
    def example(example_id: str):
        sample = runtime().examples.get(example_id)
        if sample is None:
            raise HTTPException(404, "Example not found")
        return sample

    @application.post("/api/predict", response_model=Prediction)
    def predict(request: PredictRequest):
        try:
            return runtime().predict(request.coefficient, request.example_id)
        except KeyError:
            raise HTTPException(404, "Example not found") from None
        except ReferenceMismatch as error:
            raise HTTPException(422, str(error)) from None
        except ModelBusy as error:
            raise HTTPException(503, str(error), headers={"Retry-After": "1"}) from None
        except HTTPException:
            raise
        except Exception as error:
            logger.error(
                json.dumps({"event": "inference_failed", "error_type": type(error).__name__})
            )
            raise HTTPException(500, "Inference could not be completed. Please retry.") from None

    if serve_frontend and frontend.is_dir():
        application.mount("/", StaticFiles(directory=frontend, html=True), name="frontend")
    return application


app = create_app()
