"""Reproducible inspect/train/evaluate/predict commands: python -m ml.pipeline --help."""

import argparse
import json
import os
import platform
import shutil
import time
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ml.data import assert_no_duplicate_leakage, load_dataset, make_split
from ml.model import (
    Normalization,
    create_model,
    file_sha256,
    load_artifact,
    parameter_counts,
    save_weights,
)


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    prediction, reference = prediction.astype(np.float64), reference.astype(np.float64)
    error = prediction - reference
    norms = np.linalg.norm(error.reshape(len(error), -1), axis=1)
    denominators = np.maximum(np.linalg.norm(reference.reshape(len(reference), -1), axis=1), 1e-12)
    return {
        "mse": float(np.mean(error**2)),
        "mae": float(np.mean(np.abs(error))),
        "relative_l2": float(np.mean(norms / denominators)),
    }


def infer_batches(model, normalization, x, batch_size=32):
    model.eval()
    with torch.inference_mode():
        return torch.cat(
            [
                normalization.decode_y(model(normalization.encode_x(batch)))
                for batch in x.split(batch_size)
            ]
        )


def train(args) -> None:
    if (args.output / "best.pt").exists():
        raise ValueError("Output already has a checkpoint. Choose a new experiment directory.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(args.threads)
    x, y, provenance = load_dataset(args.data)
    split = make_split(len(x), args.train_size, args.val_size, args.seed)
    assert_no_duplicate_leakage(x, split)
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "split.json", split)
    write_json(args.output / "dataset.json", provenance)
    xt, yt = x[split["train"]], y[split["train"]]
    xv, yv = x[split["validation"]], y[split["validation"]]
    normalizer = Normalization.fit(xt, yt)
    architecture = {
        "n_modes": [args.modes, args.modes],
        "hidden_channels": args.hidden_channels,
        "n_layers": args.layers,
        "in_channels": 1,
        "out_channels": 1,
        "positional_embedding": "grid",
        "projection_channel_ratio": 2,
    }
    model = create_model(architecture)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loader = DataLoader(
        TensorDataset(xt, yt),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        num_workers=0,
    )
    history, best, best_epoch = [], float("inf"), 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        total = 0.0
        for batch_x, batch_y in loader:
            optimizer.zero_grad(set_to_none=True)
            output = model(normalizer.encode_x(batch_x))
            loss = (output - normalizer.encode_y(batch_y)).square().mean()
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite training loss; checkpoint is not publishable")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item() * len(batch_x)
        validation = infer_batches(model, normalizer, xv, args.batch_size)
        scores = metrics(validation.numpy(), yv.numpy())
        record = {
            "epoch": epoch,
            "train_normalized_mse": total / len(xt),
            "validation_mse": scores["mse"],
            "validation_relative_l2": scores["relative_l2"],
            "learning_rate": scheduler.get_last_lr()[0],
            "seconds": time.perf_counter() - epoch_start,
        }
        history.append(record)
        if scores["relative_l2"] < best:
            best, best_epoch = scores["relative_l2"], epoch
            save_weights(model, args.output / "best.pt")
        scheduler.step()
        write_json(args.output / "history.json", history)
        if epoch <= 3 or epoch % 5 == 0 or epoch == args.epochs:
            print(json.dumps(record), flush=True)
    elapsed = time.perf_counter() - started
    config = {
        "format_version": 1,
        "resolution": 32,
        "architecture": architecture,
        "normalization": asdict(normalizer),
        "seed": args.seed,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_validation_relative_l2": best,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "loss": "MSE in standardized solution units",
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "split_sizes": {k: len(v) for k, v in split.items()},
        "parameters": parameter_counts(model),
        "training_seconds": elapsed,
        "hardware": {
            "device": "cpu",
            "machine": platform.machine(),
            "system": platform.platform(),
            "logical_cpus": os.cpu_count(),
            "torch_threads": args.threads,
        },
        "versions": {k: version(k) for k in ["torch", "neuraloperator", "numpy"]},
        "checkpoint_sha256": file_sha256(args.output / "best.pt"),
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(args.output / "config.json", config)
    np.save(args.output / "mean_baseline.npy", yt.mean(dim=0).numpy(), allow_pickle=False)
    reloaded = load_artifact(args.output)
    fresh = reloaded.predict(xv[:1])
    if not torch.isfinite(fresh).all():
        raise RuntimeError("Reload smoke test failed")
    print(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "validation_relative_l2": best,
                "training_seconds": elapsed,
                "reload_shape": list(fresh.shape),
            }
        ),
        flush=True,
    )


def evaluate(args) -> None:
    torch.set_num_threads(args.threads)
    surrogate = load_artifact(args.artifact)
    x, y, provenance = load_dataset(args.data)
    recorded = json.loads((args.artifact / "dataset.json").read_text())
    if recorded["array_sha256"] != provenance["array_sha256"]:
        raise ValueError("Evaluation dataset differs from the training source")
    split = json.loads((args.artifact / "split.json").read_text())
    indices = split["test"]
    xt, yt = x[indices], y[indices]
    prediction = surrogate.predict(xt).numpy()
    baseline = np.load(args.artifact / "mean_baseline.npy", allow_pickle=False)
    baseline = np.broadcast_to(baseline, yt.shape)
    for _ in range(10):
        surrogate.predict(xt[:1])
    timings = []
    for _ in range(100):
        start = time.perf_counter()
        surrogate.predict(xt[:1])
        timings.append((time.perf_counter() - start) * 1000)
    report = {
        "model": metrics(prediction, yt.numpy()),
        "baseline": metrics(baseline, yt.numpy()),
        "test_size": len(indices),
        "resolution": [32, 32],
        "latency": {
            "median_ms": float(np.median(timings)),
            "p95_ms": float(np.percentile(timings, 95)),
            "warmup_calls": 10,
            "timed_calls": 100,
            "batch_size": 1,
            "scope": "normalize + FNO forward + decode; excludes network and JSON",
            "device": "cpu",
            "threads": args.threads,
            "machine": platform.machine(),
        },
        "checkpoint_sha256": surrogate.config["checkpoint_sha256"],
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metric_definition": "MSE/MAE over all pixels; mean per-sample relative L2 (epsilon 1e-12)",
    }
    write_json(args.artifact / "evaluation.json", report)
    examples = []
    for i in range(min(6, len(indices))):
        examples.append(
            {
                "id": f"test-{i + 1:02d}",
                "name": f"Held-out sample {i + 1:02d}",
                "source_index": indices[i],
                "split": "test",
                "coefficient": xt[i, 0].tolist(),
                "reference": yt[i, 0].tolist(),
            }
        )
    write_json(args.artifact / "examples.json", examples)
    np.savez_compressed(
        args.artifact / "test_predictions.npz",
        prediction=prediction,
        reference=yt.numpy(),
        source_indices=np.asarray(indices),
    )
    export_figures(args.artifact, xt[0, 0].numpy(), yt[0, 0].numpy(), prediction[0, 0])
    if args.publish_report:
        target = Path("docs/results")
        target.mkdir(parents=True, exist_ok=True)
        for name in [
            "evaluation.json",
            "history.json",
            "config.json",
            "comparison.png",
            "training.png",
        ]:
            shutil.copy2(args.artifact / name, target / name)
    print(json.dumps(report, indent=2))


def export_figures(directory, coefficient, reference, prediction) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    history = json.loads((directory / "history.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    epochs = [row["epoch"] for row in history]
    axes[0].plot(epochs, [row["train_normalized_mse"] for row in history], label="Train")
    normalizer = json.loads((directory / "config.json").read_text())["normalization"]
    axes[0].plot(
        epochs,
        [row["validation_mse"] / normalizer["y_std"] ** 2 for row in history],
        label="Validation",
    )
    axes[0].set(ylabel="Standardized MSE", yscale="log", xlabel="Epoch")
    axes[0].legend(frameon=False)
    axes[1].plot(epochs, [row["validation_relative_l2"] for row in history], color="#315adf")
    axes[1].set(ylabel="Validation relative L2", xlabel="Epoch", yscale="log")
    fig.savefig(directory / "training.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3), constrained_layout=True)
    low, high = min(reference.min(), prediction.min()), max(reference.max(), prediction.max())
    fields = [coefficient, reference, prediction, np.abs(prediction - reference)]
    for i, (ax, field, title) in enumerate(
        zip(
            axes,
            fields,
            ["Coefficient a", "Held-out reference u", "FNO prediction", "Absolute error"],
        )
    ):
        # Source is [x,y]; transpose is explicitly for Cartesian plotting only.
        limits = (
            {"vmin": 0.1, "vmax": 1}
            if i == 0
            else ({"vmin": low, "vmax": high} if i < 3 else {"vmin": 0})
        )
        im = ax.imshow(field.T, origin="lower", cmap="Blues" if i == 0 else "viridis", **limits)
        ax.set(title=title, xlabel="x index", ylabel="y index")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(directory / "comparison.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    inspect = commands.add_parser("inspect")
    inspect.add_argument("--data", type=Path, default=Path("data/pdebench-darcy-beta1.npz"))
    training = commands.add_parser("train")
    training.add_argument("--data", type=Path, default=Path("data/pdebench-darcy-beta1.npz"))
    training.add_argument("--output", type=Path, default=Path("artifacts/darcy-fno"))
    for name, default in [
        ("epochs", 80),
        ("train-size", 800),
        ("val-size", 112),
        ("seed", 42),
        ("threads", 2),
        ("batch-size", 32),
        ("hidden-channels", 32),
        ("modes", 12),
        ("layers", 4),
    ]:
        training.add_argument("--" + name, type=int, default=default)
    training.add_argument("--learning-rate", type=float, default=0.003)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--data", type=Path, default=Path("data/pdebench-darcy-beta1.npz"))
    evaluation.add_argument("--artifact", type=Path, default=Path("artifacts/darcy-fno"))
    evaluation.add_argument("--threads", type=int, default=2)
    evaluation.add_argument("--publish-report", action="store_true")
    inference = commands.add_parser("predict")
    inference.add_argument("--artifact", type=Path, default=Path("artifacts/darcy-fno"))
    inference.add_argument(
        "--input", type=Path, required=True, help="JSON object with coefficient array"
    )
    inference.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "inspect":
        x, y, provenance = load_dataset(args.data)
        print(
            json.dumps(
                {
                    "shape": list(x.shape),
                    "coefficient_range": [x.min().item(), x.max().item()],
                    "solution_range": [y.min().item(), y.max().item()],
                    "doi": provenance["doi"],
                    "array_sha256": provenance["array_sha256"],
                },
                indent=2,
            )
        )
    elif args.command == "train":
        if (
            min(
                args.epochs,
                args.threads,
                args.batch_size,
                args.hidden_channels,
                args.modes,
                args.layers,
            )
            < 1
        ):
            parser.error("Training hyperparameters must be positive")
        train(args)
    elif args.command == "evaluate":
        evaluate(args)
    else:
        torch.set_num_threads(2)
        # Reuse the exact public input contract, including finiteness/range checks.
        from backend.schemas import PredictRequest

        request = PredictRequest.model_validate_json(args.input.read_text())
        model = load_artifact(args.artifact)
        field = torch.tensor(request.coefficient, dtype=torch.float32)[None, None]
        prediction = model.predict(field)[0, 0].tolist()
        write_json(
            args.output,
            {
                "prediction": prediction,
                "shape": [32, 32],
                "checkpoint_sha256": model.config["checkpoint_sha256"],
            },
        )


if __name__ == "__main__":
    main()
