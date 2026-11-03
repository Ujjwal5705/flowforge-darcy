"""Upload a prepared snapshot to the existing private candidate; never change audience."""

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, get_token


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--repo", default="ujjwal75/flowforge-darcy-2026")
    parser.add_argument(
        "--allow-public-update",
        action="store_true",
        help="Use only after the owner has explicitly authorized public publication",
    )
    args = parser.parse_args()
    manifest = json.loads((args.snapshot / "deployment-manifest.json").read_text())
    token = get_token()
    if not token:
        raise RuntimeError(
            "An existing Hugging Face login is required; no token is accepted on argv"
        )
    api = HfApi(token=token)
    current = api.space_info(args.repo)
    if not current.private and not args.allow_public_update:
        raise RuntimeError("The target is public; explicit publication authorization is required")
    result = api.upload_folder(
        repo_id=args.repo,
        repo_type="space",
        folder_path=args.snapshot,
        commit_message="Deploy FLOWFORGE source " + manifest["source_commit"][:12],
        parent_commit=current.sha,
    )
    report = {
        "repo_id": args.repo,
        "hub_commit": result.oid,
        "source_commit": manifest["source_commit"],
        "private": current.private,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
    }
    Path("artifacts/space-upload.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # SDK exception strings can contain signed transport URLs. Keep output bounded and safe.
        print(
            json.dumps(
                {
                    "upload_failed": True,
                    "error_type": type(error).__name__,
                    "http_status": getattr(getattr(error, "response", None), "status_code", None),
                }
            )
        )
        raise SystemExit(1) from None
