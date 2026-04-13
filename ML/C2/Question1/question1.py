from __future__ import annotations

import argparse
import copy
import json
import random
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets


CLASS_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]

IMAGE_SIZE = 96
TRAIN_PER_CLASS = 1000
TEST_PER_CLASS = 200
SEED = 42
THREADS = 8


@dataclass(frozen=True)
class DesignConfig:
    name: str
    block_specs: Sequence[tuple[int, int]]
    use_batchnorm: bool
    dropout: float
    notes: str


DESIGNS: List[DesignConfig] = [
    DesignConfig(
        name="Design 1",
        block_specs=((32, 1), (64, 1)),
        use_batchnorm=False,
        dropout=0.0,
        notes="Baseline with two convolution stages and no regularization.",
    ),
    DesignConfig(
        name="Design 2",
        block_specs=((32, 1), (64, 1)),
        use_batchnorm=False,
        dropout=0.25,
        notes="Adds dropout after each max-pooling stage.",
    ),
    DesignConfig(
        name="Design 3",
        block_specs=((32, 1), (64, 1)),
        use_batchnorm=True,
        dropout=0.25,
        notes="Adds batch normalization after each convolution plus dropout.",
    ),
    DesignConfig(
        name="Design 4",
        block_specs=((32, 2), (64, 2)),
        use_batchnorm=True,
        dropout=0.25,
        notes="Uses two same-padded convolutions per block before pooling.",
    ),
    DesignConfig(
        name="Design 5",
        block_specs=((32, 2), (64, 2), (128, 2)),
        use_batchnorm=True,
        dropout=0.25,
        notes="Adds a third convolution block with 128 channels.",
    ),
    DesignConfig(
        name="Design 6",
        block_specs=((32, 2), (64, 2), (128, 2), (256, 2)),
        use_batchnorm=True,
        dropout=0.25,
        notes="Adds a fourth convolution block with 256 channels.",
    ),
]


def default_code_root() -> Path:
    cwd = Path.cwd().resolve()
    candidates = [
        cwd / "C2" / "code",
        cwd / "code",
        cwd,
    ]
    for candidate in candidates:
        if (candidate / "question1.py").exists():
            return candidate
    return (cwd / "C2" / "code").resolve()


def default_data_root() -> Path:
    code_root = default_code_root()
    candidates = [
        code_root.parent / "data",
        Path.cwd().resolve() / "C2" / "data",
        Path.cwd().resolve() / "data",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return (code_root.parent / "data").resolve()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def stratified_indices(labels: Sequence[int], per_class: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    picked: List[int] = []
    for class_id in range(len(CLASS_NAMES)):
        class_indices = np.flatnonzero(labels == class_id)
        chosen = rng.choice(class_indices, size=per_class, replace=False)
        picked.extend(chosen.tolist())
    rng.shuffle(picked)
    return np.asarray(picked, dtype=np.int64)


def resize_split(
    images: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    image_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    resized = np.empty((len(indices), 3, image_size, image_size), dtype=np.uint8)
    selected_labels = labels[indices].astype(np.int64)
    for out_idx, src_idx in enumerate(indices):
        if out_idx % 500 == 0:
            print(f"Resizing sample {out_idx + 1}/{len(indices)}", flush=True)
        upscaled = np.asarray(
            Image.fromarray(images[src_idx]).resize(
                (image_size, image_size),
                resample=Image.Resampling.BICUBIC,
            )
        )
        resized[out_idx] = np.transpose(upscaled, (2, 0, 1))
    return torch.from_numpy(resized), torch.from_numpy(selected_labels)


def save_sample_grid(images: torch.Tensor, labels: torch.Tensor, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for class_id, ax in enumerate(axes.flat):
        idx = int((labels == class_id).nonzero(as_tuple=False)[0].item())
        image = images[idx].permute(1, 2, 0).numpy()
        ax.imshow(image)
        ax.set_title(CLASS_NAMES[class_id], fontsize=11)
        ax.axis("off")
    fig.suptitle("Upscaled CIFAR-10 examples (32x32 to 96x96 via cubic interpolation)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def prepare_balanced_cifar_cache(
    data_root: Path,
    code_root: Path,
    images_dir: Path,
    force: bool = False,
) -> Dict[str, Path]:
    cache_dir = code_root / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "cifar_balanced_meta.json"
    train_path = cache_dir / "cifar_train_10000.pt"
    test_path = cache_dir / "cifar_test_2000.pt"
    if not force and meta_path.exists() and train_path.exists() and test_path.exists():
        return {"train": train_path, "test": test_path, "meta": meta_path}

    try:
        raw_train = datasets.CIFAR10(root=str(data_root), train=True, download=False)
        raw_test = datasets.CIFAR10(root=str(data_root), train=False, download=False)
    except RuntimeError:
        print("CIFAR-10 not found locally. Downloading it now...", flush=True)
        raw_train = datasets.CIFAR10(root=str(data_root), train=True, download=True)
        raw_test = datasets.CIFAR10(root=str(data_root), train=False, download=True)

    train_indices = stratified_indices(raw_train.targets, TRAIN_PER_CLASS, SEED)
    test_indices = stratified_indices(raw_test.targets, TEST_PER_CLASS, SEED)

    train_images, train_labels = resize_split(
        raw_train.data,
        np.asarray(raw_train.targets),
        train_indices,
        IMAGE_SIZE,
    )
    test_images, test_labels = resize_split(
        raw_test.data,
        np.asarray(raw_test.targets),
        test_indices,
        IMAGE_SIZE,
    )

    torch.save({"images": train_images, "labels": train_labels}, train_path)
    torch.save({"images": test_images, "labels": test_labels}, test_path)

    save_sample_grid(train_images, train_labels, images_dir / "Q1_cifar10_upscaled_examples.png")

    meta = {
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "train_per_class": TRAIN_PER_CLASS,
        "test_per_class": TEST_PER_CLASS,
        "classes": CLASS_NAMES,
        "train_indices": train_indices.tolist(),
        "test_indices": test_indices.tolist(),
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    return {"train": train_path, "test": test_path, "meta": meta_path}


class CIFARTensorDataset(TensorDataset):
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image, label = super().__getitem__(index)
        return image.float().div(255.0), label


class CNNClassifier(nn.Module):
    def __init__(self, config: DesignConfig, image_size: int = IMAGE_SIZE) -> None:
        super().__init__()
        self.config = config
        layers = OrderedDict()
        self.activation_labels: Dict[str, str] = {}
        in_channels = 3
        conv_counter = 0
        block_counter = 0

        for out_channels, repeats in config.block_specs:
            block_counter += 1
            for _ in range(repeats):
                conv_counter += 1
                conv_name = f"conv{conv_counter}"
                relu_name = f"relu{conv_counter}"
                layers[conv_name] = nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=4,
                    padding="same",
                )
                if config.use_batchnorm:
                    layers[f"bn{conv_counter}"] = nn.BatchNorm2d(out_channels)
                layers[relu_name] = nn.ReLU(inplace=True)
                self.activation_labels[relu_name] = f"Conv {conv_counter}"
                in_channels = out_channels
            layers[f"pool{block_counter}"] = nn.MaxPool2d(kernel_size=2)
            if config.dropout > 0:
                layers[f"dropout{block_counter}"] = nn.Dropout(p=config.dropout)

        self.features = nn.Sequential(layers)
        with torch.no_grad():
            dummy = torch.zeros(1, 3, image_size, image_size)
            flattened = self.features(dummy).flatten(1).shape[1]
        self.hidden = nn.Linear(flattened, 256)
        self.hidden_relu = nn.ReLU(inplace=True)
        self.output = nn.Linear(256, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.flatten(1)
        x = self.hidden(x)
        x = self.hidden_relu(x)
        return self.output(x)

    def forward_with_activations(self, x: torch.Tensor) -> tuple[torch.Tensor, OrderedDict[str, torch.Tensor]]:
        activations: OrderedDict[str, torch.Tensor] = OrderedDict()
        for name, module in self.features.named_children():
            x = module(x)
            if name in self.activation_labels:
                activations[self.activation_labels[name]] = x.detach().cpu()
        x = x.flatten(1)
        hidden = self.hidden_relu(self.hidden(x))
        activations["Dense hidden"] = hidden.detach().cpu()
        logits = self.output(hidden)
        activations["Dense output logits"] = logits.detach().cpu()
        activations["Dense output softmax"] = torch.softmax(logits, dim=1).detach().cpu()
        return logits, activations


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)
    return total_loss / total_samples, 100.0 * total_correct / total_samples


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    max_epochs: int,
    min_epochs: int,
    patience: int,
    device: torch.device,
) -> dict:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    history = []
    best_epoch = 0
    best_test_accuracy = -1.0
    best_snapshot = copy.deepcopy(model.state_dict())
    stale_epochs = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += labels.size(0)

        train_loss = total_loss / total_samples
        train_accuracy = 100.0 * total_correct / total_samples
        test_loss, test_accuracy = evaluate_model(model, test_loader, criterion, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "test_loss": test_loss,
                "test_accuracy": test_accuracy,
            }
        )

        print(
            f"Epoch {epoch:02d}/{max_epochs} | "
            f"Train loss {train_loss:.4f} | Train acc {train_accuracy:.2f}% | "
            f"Test loss {test_loss:.4f} | Test acc {test_accuracy:.2f}%",
            flush=True,
        )

        if test_accuracy > best_test_accuracy + 0.10:
            best_test_accuracy = test_accuracy
            best_epoch = epoch
            best_snapshot = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if epoch >= min_epochs and stale_epochs >= patience:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    model.load_state_dict(best_snapshot)
    final_train_loss, final_train_accuracy = evaluate_model(model, train_loader, criterion, device)
    final_test_loss, final_test_accuracy = evaluate_model(model, test_loader, criterion, device)
    return {
        "history": history,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "final_train_loss": final_train_loss,
        "final_train_accuracy": final_train_accuracy,
        "final_test_loss": final_test_loss,
        "final_test_accuracy": final_test_accuracy,
    }


def plot_accuracy_grid(results: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    for ax, config in zip(axes.flat, DESIGNS):
        result = results[config.name]
        epochs = [row["epoch"] for row in result["history"]]
        train_acc = [row["train_accuracy"] for row in result["history"]]
        test_acc = [row["test_accuracy"] for row in result["history"]]
        ax.plot(epochs, train_acc, marker="o", label="Train")
        ax.plot(epochs, test_acc, marker="s", label="Test")
        ax.set_title(config.name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Accuracy (%)")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle("CIFAR-10 CNN designs: train and test accuracy", fontsize=15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_final_accuracy_bar(results: dict, out_path: Path) -> None:
    design_names = [config.name for config in DESIGNS]
    train_scores = [results[name]["final_train_accuracy"] for name in design_names]
    test_scores = [results[name]["final_test_accuracy"] for name in design_names]
    x = np.arange(len(design_names))

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(x - 0.18, train_scores, width=0.36, label="Train")
    ax.bar(x + 0.18, test_scores, width=0.36, label="Test")
    ax.set_xticks(x)
    ax.set_xticklabels(design_names, rotation=15)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Final CNN accuracy comparison")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def feature_mosaic(feature_maps: np.ndarray, max_maps: int = 8) -> np.ndarray:
    count = min(feature_maps.shape[0], max_maps)
    maps = feature_maps[:count]
    tiles = []
    for fmap in maps:
        fmap = fmap.astype(np.float32)
        fmap = fmap - fmap.min()
        denom = fmap.max() if fmap.max() > 1e-8 else 1.0
        fmap = fmap / denom
        tiles.append(fmap)
    while len(tiles) < max_maps:
        tiles.append(np.zeros_like(tiles[0]))
    rows = []
    for start in range(0, max_maps, 4):
        rows.append(np.concatenate(tiles[start : start + 4], axis=1))
    return np.concatenate(rows, axis=0)


def find_visualization_samples(
    model: CNNClassifier,
    dataset: CIFARTensorDataset,
    device: torch.device,
) -> dict[str, int]:
    targets = {"horse": 7, "bird": 2, "frog": 6}
    found: dict[str, int] = {}
    model.eval()
    with torch.no_grad():
        for index in range(len(dataset)):
            image, label = dataset[index]
            if CLASS_NAMES[label.item()] not in targets:
                continue
            logits = model(image.unsqueeze(0).to(device))
            prediction = logits.argmax(dim=1).item()
            class_name = CLASS_NAMES[label.item()]
            if prediction == label.item() and class_name in targets and class_name not in found:
                found[class_name] = index
            if len(found) == len(targets):
                break
    return found


def plot_activation_figure(
    class_name: str,
    image_tensor: torch.Tensor,
    activations: OrderedDict[str, torch.Tensor],
    out_path: Path,
) -> None:
    conv_labels = [key for key in activations if key.startswith("Conv")]
    total_rows = 1 + len(conv_labels) + 2
    fig, axes = plt.subplots(total_rows, 1, figsize=(12, 2.8 * total_rows))
    if total_rows == 1:
        axes = [axes]

    image = image_tensor.permute(1, 2, 0).numpy()
    axes[0].imshow(image)
    axes[0].set_title(f"Input image: {class_name}", fontsize=13)
    axes[0].axis("off")

    row = 1
    for label in conv_labels:
        fmap = activations[label][0].numpy()
        axes[row].imshow(feature_mosaic(fmap), cmap="viridis")
        axes[row].set_title(f"{label}: first 8 feature maps", fontsize=11)
        axes[row].axis("off")
        row += 1

    hidden = activations["Dense hidden"][0].numpy()
    axes[row].plot(hidden, color="tab:blue", linewidth=1.0)
    axes[row].set_title("Dense hidden layer output (256 activations)", fontsize=11)
    axes[row].set_xlabel("Hidden neuron index")
    axes[row].set_ylabel("Activation")
    axes[row].grid(True, alpha=0.25)
    row += 1

    probs = activations["Dense output softmax"][0].numpy()
    axes[row].bar(CLASS_NAMES, probs, color="tab:green")
    axes[row].set_title("Dense output layer softmax probabilities", fontsize=11)
    axes[row].set_ylabel("Probability")
    axes[row].tick_params(axis="x", rotation=30)
    axes[row].grid(True, axis="y", alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def summarize_results(results: dict, out_path: Path, best_design: str) -> None:
    payload = {
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "train_per_class": TRAIN_PER_CLASS,
        "test_per_class": TEST_PER_CLASS,
        "best_design": best_design,
        "results": results,
    }
    out_path.write_text(json.dumps(payload, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train CIFAR-10 CNN designs for EE 501 C2.")
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--code-root", type=Path, default=default_code_root())
    parser.add_argument("--max-epochs", type=int, default=8)
    parser.add_argument("--min-epochs", type=int, default=5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--force-preprocess", action="store_true")
    parser.add_argument("--designs", nargs="*", default=None, help="Subset of design numbers to run, e.g. 1 3 6")
    parser.add_argument("--train-limit", type=int, default=None, help="Optional subset size for quick smoke tests")
    parser.add_argument("--test-limit", type=int, default=None, help="Optional subset size for quick smoke tests")
    parser.add_argument("--device", type=str, default=None, help="Optional device override, e.g. cpu or cuda")
    args, _unknown = parser.parse_known_args(argv)

    set_seed(SEED)
    device = select_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(THREADS)
    args.code_root.mkdir(parents=True, exist_ok=True)
    images_dir = args.code_root / "images"
    weights_dir = args.code_root / "weights"
    results_dir = args.code_root / "results"
    images_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using device: {device}", flush=True)

    cache = prepare_balanced_cifar_cache(
        args.data_root,
        args.code_root,
        images_dir,
        force=args.force_preprocess,
    )
    train_blob = torch.load(cache["train"], weights_only=False)
    test_blob = torch.load(cache["test"], weights_only=False)
    train_dataset: CIFARTensorDataset | Subset = CIFARTensorDataset(train_blob["images"], train_blob["labels"])
    test_dataset: CIFARTensorDataset | Subset = CIFARTensorDataset(test_blob["images"], test_blob["labels"])

    if args.train_limit is not None:
        train_dataset = Subset(train_dataset, list(range(min(args.train_limit, len(train_dataset)))))
    if args.test_limit is not None:
        test_dataset = Subset(test_dataset, list(range(min(args.test_limit, len(test_dataset)))))

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0)

    selected_designs = DESIGNS
    if args.designs:
        requested = {f"Design {int(token)}" for token in args.designs}
        selected_designs = [config for config in DESIGNS if config.name in requested]

    all_results: dict = {}
    best_design = ""
    best_test_accuracy = -1.0
    best_model: CNNClassifier | None = None

    for config in selected_designs:
        print(f"\n=== {config.name} ===", flush=True)
        set_seed(SEED)
        model = CNNClassifier(config).to(device)
        result = train_model(
            model,
            train_loader,
            test_loader,
            max_epochs=args.max_epochs,
            min_epochs=args.min_epochs,
            patience=args.patience,
            device=device,
        )
        result["notes"] = config.notes
        all_results[config.name] = result

        checkpoint_path = weights_dir / f"{config.name.lower().replace(' ', '_')}.pt"
        torch.save(
            {
                "config_name": config.name,
                "state_dict": model.state_dict(),
                "metrics": result,
            },
            checkpoint_path,
        )

        if result["final_test_accuracy"] > best_test_accuracy:
            best_test_accuracy = result["final_test_accuracy"]
            best_design = config.name
            best_model = model

    if len(selected_designs) == len(DESIGNS):
        plot_accuracy_grid(all_results, images_dir / "Q1_cnn_accuracy_grid.png")
        plot_final_accuracy_bar(all_results, images_dir / "Q1_cnn_final_accuracy_bar.png")

    if best_model is None:
        raise RuntimeError("No model trained successfully.")

    visual_samples = {}
    if isinstance(test_dataset, CIFARTensorDataset):
        visual_samples = find_visualization_samples(best_model, test_dataset, device)
    visual_summary = {}
    for class_name, index in visual_samples.items():
        image_tensor, label = test_dataset[index]
        logits, activations = best_model.forward_with_activations(image_tensor.unsqueeze(0).to(device))
        prediction = logits.argmax(dim=1).item()
        plot_activation_figure(
            class_name,
            image_tensor,
            activations,
            images_dir / f"Q1_best_model_{class_name}_activations.png",
        )
        visual_summary[class_name] = {
            "dataset_index": index,
            "label": CLASS_NAMES[label.item()],
            "prediction": CLASS_NAMES[prediction],
        }

    summary_path = results_dir / "cnn_summary.json"
    summarize_results(all_results, summary_path, best_design)
    full_summary = json.loads(summary_path.read_text())
    full_summary["visualization_samples"] = visual_summary
    summary_path.write_text(json.dumps(full_summary, indent=2))

    print(f"Best design: {best_design} with test accuracy {best_test_accuracy:.2f}%", flush=True)


if __name__ == "__main__":
    main()
