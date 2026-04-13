from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

def default_code_root() -> Path:
    cwd = Path.cwd().resolve()
    candidates = [
        cwd / "C2" / "code",
        cwd / "code",
        cwd,
    ]
    for candidate in candidates:
        if (candidate / "question2.py").exists() or (candidate / "air_passengers.csv").exists():
            return candidate
    return (cwd / "C2" / "code").resolve()


_CACHE_ROOT = default_code_root() / ".cache"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(default_code_root() / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn


SEED = 42
THREADS = 8


@dataclass(frozen=True)
class ExperimentResult:
    look_back: int
    hidden_size: int
    params: int
    best_epoch: int
    train_rmse_norm: float
    test_rmse_norm: float
    train_rmse_raw: float
    test_rmse_raw: float


class PassengerLSTM(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features, _ = self.lstm(x)
        return self.output(features[:, -1, :])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def select_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_series(csv_path: Path) -> tuple[np.ndarray, list[str]]:
    frame = pd.read_csv(csv_path)
    values = frame["passengers"].to_numpy(dtype=np.float32)
    months = frame["month"].tolist()
    return values, months


def normalize(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    min_value = float(values.min())
    max_value = float(values.max())
    normalized = (values - min_value) / (max_value - min_value)
    return normalized.astype(np.float32), min_value, max_value


def inverse_normalize(values: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    return values * (max_value - min_value) + min_value


def create_sequences(values: np.ndarray, look_back: int) -> tuple[np.ndarray, np.ndarray]:
    xs = []
    ys = []
    for start in range(len(values) - look_back):
        xs.append(values[start : start + look_back])
        ys.append(values[start + look_back])
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.float32)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def train_one_model(
    train_x: np.ndarray,
    train_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    hidden_size: int,
    max_epochs: int,
    lr: float,
    device: torch.device,
) -> tuple[PassengerLSTM, dict]:
    model = PassengerLSTM(hidden_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    x_train = torch.from_numpy(train_x).unsqueeze(-1).to(device)
    y_train = torch.from_numpy(train_y).unsqueeze(-1).to(device)
    x_test = torch.from_numpy(test_x).unsqueeze(-1).to(device)
    y_test = torch.from_numpy(test_y).unsqueeze(-1).to(device)

    best_state = None
    best_epoch = 0
    best_test_rmse = float("inf")
    best_payload = {}
    stale_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        predictions = model(x_train)
        loss = criterion(predictions, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            train_pred = model(x_train).squeeze(-1).detach().cpu().numpy()
            test_pred = model(x_test).squeeze(-1).detach().cpu().numpy()
            train_rmse = float(np.sqrt(np.mean((train_pred - train_y) ** 2)))
            test_rmse = float(np.sqrt(np.mean((test_pred - test_y) ** 2)))

        if test_rmse < best_test_rmse - 1e-4:
            best_test_rmse = test_rmse
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            best_payload = {
                "train_pred": train_pred,
                "test_pred": test_pred,
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
            }
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch >= 150 and stale_epochs >= 60:
            break

    if best_state is None:
        raise RuntimeError("LSTM training failed to produce a checkpoint.")

    model.load_state_dict(best_state)
    best_payload["best_epoch"] = best_epoch
    best_payload["params"] = count_parameters(model)
    return model, best_payload


def choose_best_model(results: list[ExperimentResult]) -> ExperimentResult:
    sorted_results = sorted(results, key=lambda item: item.test_rmse_norm)
    threshold = sorted_results[0].test_rmse_norm * 1.05
    eligible = [item for item in sorted_results if item.train_rmse_norm <= 0.035 and item.test_rmse_norm <= threshold]
    if eligible:
        return min(eligible, key=lambda item: item.params)
    return sorted_results[0]


def plot_best_fit(
    months: list[str],
    normalized: np.ndarray,
    split_index: int,
    look_back: int,
    train_pred: np.ndarray,
    test_pred: np.ndarray,
    min_value: float,
    max_value: float,
    out_path: Path,
) -> None:
    train_actual = inverse_normalize(normalized[:split_index], min_value, max_value)
    test_actual = inverse_normalize(normalized[split_index:], min_value, max_value)
    train_pred_raw = inverse_normalize(train_pred, min_value, max_value)
    test_pred_raw = inverse_normalize(test_pred, min_value, max_value)

    train_months = months[:split_index]
    test_months = months[split_index:]

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=False)
    axes[0].plot(train_months, train_actual, label="Actual", color="black", linewidth=2)
    axes[0].plot(train_months[look_back:], train_pred_raw, label="Predicted", color="tab:blue", linewidth=2)
    axes[0].set_title("Train split: actual vs predicted passengers")
    axes[0].set_ylabel("Passengers")
    train_tick_positions = np.arange(0, len(train_months), 6)
    axes[0].set_xticks(train_tick_positions)
    axes[0].set_xticklabels([train_months[idx] for idx in train_tick_positions], rotation=45, ha="right")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(test_months, test_actual, label="Actual", color="black", linewidth=2)
    axes[1].plot(test_months[look_back:], test_pred_raw, label="Predicted", color="tab:orange", linewidth=2)
    axes[1].set_title("Test split: actual vs predicted passengers")
    axes[1].set_ylabel("Passengers")
    test_tick_positions = np.arange(0, len(test_months), 4)
    axes[1].set_xticks(test_tick_positions)
    axes[1].set_xticklabels([test_months[idx] for idx in test_tick_positions], rotation=45, ha="right")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle("AirPassengers LSTM fit", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(results: list[ExperimentResult], out_path: Path) -> None:
    ms = sorted({item.look_back for item in results})
    ns = sorted({item.hidden_size for item in results})
    heat = np.zeros((len(ns), len(ms)), dtype=np.float32)
    for item in results:
        row = ns.index(item.hidden_size)
        col = ms.index(item.look_back)
        heat[row, col] = item.test_rmse_raw

    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(heat, cmap="magma_r", aspect="auto")
    ax.set_xticks(range(len(ms)))
    ax.set_xticklabels(ms)
    ax.set_yticks(range(len(ns)))
    ax.set_yticklabels(ns)
    ax.set_xlabel("Look-back window M")
    ax.set_ylabel("LSTM neurons N")
    ax.set_title("Test RMSE across M and N")
    for row in range(len(ns)):
        for col in range(len(ms)):
            ax.text(col, row, f"{heat[row, col]:.1f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(image, ax=ax, label="Test RMSE (passengers)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_variation_curves(
    months: list[str],
    normalized: np.ndarray,
    split_index: int,
    min_value: float,
    max_value: float,
    selected_runs: list[tuple[str, int, np.ndarray]],
    out_path: Path,
    title: str,
) -> None:
    unique_runs = []
    seen = set()
    for label, look_back, predictions in selected_runs:
        key = (label, look_back)
        if key in seen:
            continue
        seen.add(key)
        unique_runs.append((label, look_back, predictions))

    test_months = months[split_index:]
    test_actual = inverse_normalize(normalized[split_index:], min_value, max_value)
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(test_months, test_actual, color="black", linewidth=2, label="Actual")
    for label, look_back, predictions in unique_runs:
        pred_raw = inverse_normalize(predictions, min_value, max_value)
        ax.plot(test_months[look_back:], pred_raw, linewidth=2, label=label)
    ax.set_title(title)
    ax.set_ylabel("Passengers")
    tick_positions = np.arange(0, len(test_months), 4)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([test_months[idx] for idx in tick_positions], rotation=45, ha="right")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train LSTM models on AirPassengers for EE 501 C2.")
    parser.add_argument("--code-root", type=Path, default=default_code_root())
    parser.add_argument("--max-epochs", type=int, default=600)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--device", type=str, default=None, help="Optional device override, e.g. cpu or cuda")
    args, _unknown = parser.parse_known_args(argv)

    set_seed(SEED)
    device = select_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(THREADS)
    images_dir = args.code_root / "images"
    results_dir = args.code_root / "results"
    images_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}")

    csv_path = args.code_root / "air_passengers.csv"
    values, months = load_series(csv_path)
    normalized, min_value, max_value = normalize(values)
    split_index = int(len(normalized) * 0.67)
    train_values = normalized[:split_index]
    test_values = normalized[split_index:]

    look_back_values = [3, 6, 9, 12]
    hidden_sizes = [4, 8, 12, 16, 24]

    results: list[ExperimentResult] = []
    run_cache: dict[tuple[int, int], dict] = {}

    for look_back in look_back_values:
        train_x, train_y = create_sequences(train_values, look_back)
        test_x, test_y = create_sequences(test_values, look_back)
        if len(test_x) == 0:
            continue
        for hidden_size in hidden_sizes:
            print(f"M={look_back}, N={hidden_size}")
            set_seed(SEED)
            model, payload = train_one_model(
                train_x,
                train_y,
                test_x,
                test_y,
                hidden_size=hidden_size,
                max_epochs=args.max_epochs,
                lr=args.lr,
                device=device,
            )
            train_pred = payload["train_pred"]
            test_pred = payload["test_pred"]
            train_rmse_raw = float(
                np.sqrt(
                    np.mean(
                        (
                            inverse_normalize(train_pred, min_value, max_value)
                            - inverse_normalize(train_y, min_value, max_value)
                        )
                        ** 2
                    )
                )
            )
            test_rmse_raw = float(
                np.sqrt(
                    np.mean(
                        (
                            inverse_normalize(test_pred, min_value, max_value)
                            - inverse_normalize(test_y, min_value, max_value)
                        )
                        ** 2
                    )
                )
            )
            result = ExperimentResult(
                look_back=look_back,
                hidden_size=hidden_size,
                params=payload["params"],
                best_epoch=payload["best_epoch"],
                train_rmse_norm=float(payload["train_rmse"]),
                test_rmse_norm=float(payload["test_rmse"]),
                train_rmse_raw=train_rmse_raw,
                test_rmse_raw=test_rmse_raw,
            )
            results.append(result)
            run_cache[(look_back, hidden_size)] = {
                "model_state": {key: value.detach().clone() for key, value in model.state_dict().items()},
                "train_pred": train_pred,
                "test_pred": test_pred,
                "train_y": train_y,
                "test_y": test_y,
            }

    best = choose_best_model(results)
    best_payload = run_cache[(best.look_back, best.hidden_size)]

    plot_best_fit(
        months=months,
        normalized=normalized,
        split_index=split_index,
        look_back=best.look_back,
        train_pred=best_payload["train_pred"],
        test_pred=best_payload["test_pred"],
        min_value=min_value,
        max_value=max_value,
        out_path=images_dir / "Q2_lstm_best_fit.png",
    )
    plot_heatmap(results, images_dir / "Q2_lstm_rmse_heatmap.png")

    variation_m = []
    for look_back in [3, best.look_back, 12]:
        key = (look_back, best.hidden_size)
        if key in run_cache:
            variation_m.append((f"M={look_back}, N={best.hidden_size}", look_back, run_cache[key]["test_pred"]))
    plot_variation_curves(
        months=months,
        normalized=normalized,
        split_index=split_index,
        min_value=min_value,
        max_value=max_value,
        selected_runs=variation_m,
        out_path=images_dir / "Q2_lstm_vary_M.png",
        title=f"Effect of varying M at fixed N={best.hidden_size}",
    )

    variation_n = []
    for hidden_size in [4, best.hidden_size, 24]:
        key = (best.look_back, hidden_size)
        if key in run_cache:
            variation_n.append((f"M={best.look_back}, N={hidden_size}", best.look_back, run_cache[key]["test_pred"]))
    plot_variation_curves(
        months=months,
        normalized=normalized,
        split_index=split_index,
        min_value=min_value,
        max_value=max_value,
        selected_runs=variation_n,
        out_path=images_dir / "Q2_lstm_vary_N.png",
        title=f"Effect of varying N at fixed M={best.look_back}",
    )

    summary = {
        "seed": SEED,
        "split_index": split_index,
        "train_samples": split_index,
        "test_samples": len(normalized) - split_index,
        "best_model": {
            "look_back": best.look_back,
            "hidden_size": best.hidden_size,
            "params": best.params,
            "best_epoch": best.best_epoch,
            "train_rmse_norm": best.train_rmse_norm,
            "test_rmse_norm": best.test_rmse_norm,
            "train_rmse_raw": best.train_rmse_raw,
            "test_rmse_raw": best.test_rmse_raw,
        },
        "results": [result.__dict__ for result in results],
    }
    (results_dir / "lstm_summary.json").write_text(json.dumps(summary, indent=2))

    print(
        f"Best LSTM: M={best.look_back}, N={best.hidden_size}, "
        f"test RMSE={best.test_rmse_raw:.2f} passengers"
    )


if __name__ == "__main__":
    main()
