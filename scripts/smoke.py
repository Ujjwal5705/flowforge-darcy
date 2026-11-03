"""Verify the installed official loader and FNO API before training."""

import inspect
import json
from importlib.metadata import version
from pathlib import Path

import torch
from neuralop.data.datasets import load_darcy_flow_small
from neuralop.models import FNO


def main() -> None:
    torch.manual_seed(42)
    torch.set_num_threads(2)
    train, tests, _ = load_darcy_flow_small(
        n_train=4,
        n_tests=[2],
        batch_size=2,
        test_batch_sizes=[2],
        data_root=Path("data"),
        test_resolutions=[16],
        encode_input=False,
        encode_output=False,
    )
    x = torch.stack([train.dataset[i]["x"] for i in range(2)]).float()
    y = torch.stack([train.dataset[i]["y"] for i in range(2)]).float()
    model = FNO(n_modes=(4, 4), in_channels=1, out_channels=1, hidden_channels=8, n_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    losses = []
    for _ in range(4):
        optimizer.zero_grad()
        prediction = model(x)
        assert prediction.shape == y.shape
        assert torch.isfinite(prediction).all()
        loss = (prediction - y).square().mean()
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0], losses
    report = {
        "neuraloperator": version("neuraloperator"),
        "torch": torch.__version__,
        "device": "cpu",
        "loader_parameters": list(inspect.signature(load_darcy_flow_small).parameters),
        "input_shape": list(x.shape),
        "output_shape": list(prediction.shape),
        "loader_input_dtype": str(train.dataset[0]["x"].dtype),
        "explicit_input_cast": "float32",
        "losses": losses,
        "parameters_real_scalars": sum(
            p.numel() * (2 if p.is_complex() else 1) for p in model.parameters()
        ),
        "test_samples_loaded": len(tests[16].dataset),
    }
    Path("docs/smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
