"""Verify a real Space over HTTP without logging credentials or signed URLs."""

import argparse
import hashlib
import json
import re
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, get_token


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="ujjwal75/flowforge-darcy-2026")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("artifacts/deployment-smoke.json"))
    args = parser.parse_args()
    token = get_token()
    space = HfApi(token=token).space_info(args.repo)
    host = space.host
    if not host:
        raise RuntimeError("The provider has not assigned an application host yet")
    base = host if host.startswith("https://") else "https://" + host
    origin = urllib.parse.urlparse(base)
    if origin.scheme != "https" or not origin.hostname.endswith(".hf.space"):
        raise RuntimeError("Refusing to send credentials to an unexpected application host")
    if args.private and not token:
        raise RuntimeError("Private verification needs the existing authenticated login")
    headers = {"Accept-Encoding": "identity"}
    if args.private:
        headers["Authorization"] = "Bearer " + token
    opener = urllib.request.build_opener(NoRedirect)

    def fetch(path: str, data=None):
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("Only same-origin application paths may be verified")
        request_headers = dict(headers)
        payload = None
        if data is not None:
            request_headers["Content-Type"] = "application/json"
            payload = json.dumps(data, allow_nan=False).encode()
        request = urllib.request.Request(
            base.rstrip("/") + path, data=payload, headers=request_headers
        )
        with opener.open(request, timeout=30) as response:
            if response.status != 200:
                raise RuntimeError("Unexpected application status")
            body = response.read(2 * 1024 * 1024 + 1)
        if len(body) > 2 * 1024 * 1024:
            raise RuntimeError("Unexpectedly large verification response")
        return body

    health = json.loads(fetch("/api/health"))
    info = json.loads(fetch("/api/model-info"))
    expected = json.loads(Path("artifacts/darcy-fno/config.json").read_text())["checkpoint_sha256"]
    assert health["model_loaded"] is True and health["device"] == "cpu"
    assert info["checkpoint_sha256"] == expected and info["resolution"] == [32, 32]
    examples = json.loads(fetch("/api/examples"))
    assert len(examples) == 6
    sample = json.loads(fetch("/api/examples/" + examples[0]["id"]))
    result = json.loads(
        fetch("/api/predict", {"coefficient": sample["coefficient"], "example_id": sample["id"]})
    )
    prediction, reference = np.asarray(result["prediction"]), np.asarray(result["reference"])
    assert prediction.shape == (32, 32) and np.isfinite(prediction).all()
    relative = np.linalg.norm(prediction - reference) / np.linalg.norm(reference)
    assert np.isclose(relative, result["comparison"]["relative_l2"], rtol=1e-6)
    assert result["checkpoint_sha256"] == expected
    html = fetch("/").decode()
    assert "FLOWFORGE" in html
    assets = re.findall(r'(?:src|href)="(/assets/[^"]+)"', html)
    assert len(assets) >= 2
    hashes = {}
    for asset in assets:
        remote_hash = hashlib.sha256(fetch(asset)).hexdigest()
        local = Path("frontend/dist") / asset.removeprefix("/")
        assert local.is_file() and hashlib.sha256(local.read_bytes()).hexdigest() == remote_hash
        hashes[asset] = remote_hash
    report = {
        "url": base,
        "private": space.private,
        "repo_id": args.repo,
        "hub_commit": space.sha,
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_version": health["model_version"],
        "checkpoint_sha256": expected,
        "prediction_shape": list(prediction.shape),
        "device": result["device"],
        "inference_ms": result["inference_ms"],
        "comparison": result["comparison"],
        "frontend_asset_sha256": hashes,
        "checks": [
            "health",
            "model-info",
            "examples",
            "prediction",
            "reference comparison",
            "frontend HTML",
            "CSS/JS equal the locally verified build",
        ],
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            json.dumps(
                {
                    "verification_failed": True,
                    "error_type": type(error).__name__,
                    "http_status": getattr(error, "code", None),
                }
            )
        )
        raise SystemExit(1) from None
