"""Create an immutable, credential-free Space snapshot from a clean source commit."""

import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    if status.strip():
        raise RuntimeError("Commit the reviewable source before preparing a deployment snapshot")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    target = ROOT / "deployment-staging" / commit[:12]
    if target.exists():
        raise FileExistsError("Snapshot already exists; retain and use its immutable contents")
    if not (ROOT / "frontend/dist/index.html").is_file():
        raise FileNotFoundError("Build the frontend before preparing the Space")
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    target.mkdir(parents=True)
    for name in filter(None, tracked):
        path = Path(name)
        if name == "docs/session-evidence.md" or path.suffix in {".jsonl", ".key", ".pem"}:
            continue
        if path.name.startswith(".env") and path.name != ".env.example":
            continue
        destination = target / ("TECHNICAL_README.md" if name == "README.md" else name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / path, destination)
    shutil.copytree(ROOT / "frontend/dist", target / "frontend/dist")
    # The local Dockerfile builds assets in Node; the Gradio SDK copies our
    # already-built assets instead. Its generated Docker build honors this file.
    dockerignore = (target / ".dockerignore").read_text().splitlines()
    (target / ".dockerignore").write_text(
        "\n".join(line for line in dockerignore if line.strip() != "frontend/dist") + "\n"
    )
    shutil.copy2(ROOT / "deployment/space-README.md", target / "README.md")
    shutil.copy2(ROOT / "requirements-space.lock", target / "requirements.txt")
    config = json.loads((target / "artifacts/darcy-fno/config.json").read_text())
    model_hash = sha256(target / "artifacts/darcy-fno/best.pt")
    if model_hash != config["checkpoint_sha256"]:
        raise ValueError("Checkpoint checksum mismatch")
    assets = {
        str(path.relative_to(target)): sha256(path)
        for path in sorted((target / "frontend/dist").rglob("*"))
        if path.is_file()
    }
    manifest = {
        "source_commit": commit,
        "checkpoint_sha256": model_hash,
        "frontend_assets": assets,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_training_data_included": False,
        "session_log_included": False,
        "hosting_sdk": "gradio",
        "inference_device": "cpu",
        "hosting_torch_requirement": "2.8.0; provider-compatible CUDA build",
        "auxiliary_benchmark_device": "On-demand ZeroGPU; local CPU fallback",
    }
    (target / "deployment-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "snapshot": str(target.relative_to(ROOT)),
                "source_commit": commit,
                "checkpoint_sha256": model_hash,
                "files": sum(p.is_file() for p in target.rglob("*")),
            }
        )
    )


if __name__ == "__main__":
    main()
