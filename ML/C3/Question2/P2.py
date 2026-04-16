from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parent
_CACHE_ROOT = _SCRIPT_ROOT / ".cache"
_MPLCONFIGDIR = _CACHE_ROOT / "mplconfig"
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


NODES = 8
KC = -0.5
KS_VALUES = np.array([0.0, 0.2, 0.3, 0.34, 0.35, 0.5, 0.75, 1.0, 2.0], dtype=float)
DEFAULT_DT = 0.02
DEFAULT_TOTAL_TIME = 25.0
DEFAULT_TRIALS = 40
DEFAULT_SEED = 42


@dataclass(frozen=True)
class PartitionRecord:
    labels: tuple[int, ...]
    cut_score: int
    lambda_max_at_ks0: float
    stabilization_threshold: float


def code_root() -> Path:
    return _SCRIPT_ROOT


def ensure_output_dirs() -> tuple[Path, Path]:
    images_dir = code_root() / "images"
    results_dir = code_root() / "results"
    images_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    return images_dir, results_dir


def mobius_ladder_adjacency() -> np.ndarray:
    matrix = np.zeros((NODES, NODES), dtype=float)
    for node in range(NODES):
        neighbor = (node + 1) % NODES
        matrix[node, neighbor] = 1.0
        matrix[neighbor, node] = 1.0
    for node in range(NODES // 2):
        opposite = (node + NODES // 2) % NODES
        matrix[node, opposite] = 1.0
        matrix[opposite, node] = 1.0
    return matrix


def mobius_positions() -> np.ndarray:
    angles = np.pi / 2.0 - (np.arange(NODES) + 0.5) * 2.0 * np.pi / NODES
    return np.column_stack((np.cos(angles), np.sin(angles)))


def canonicalize_partition(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=int).copy()
    if labels[0] == 1:
        labels = 1 - labels
    return labels


def partition_to_theta(labels: np.ndarray) -> np.ndarray:
    return np.where(labels == 0, 0.0, np.pi)


def partition_signature(labels: np.ndarray) -> tuple[int, ...]:
    return tuple(int(value) for value in canonicalize_partition(labels))


def format_partition(labels: np.ndarray | tuple[int, ...]) -> str:
    labels = np.asarray(labels, dtype=int)
    set_a = [idx + 1 for idx, value in enumerate(labels) if value == 0]
    set_b = [idx + 1 for idx, value in enumerate(labels) if value == 1]
    return f"A={set_a}, B={set_b}"


def cut_score(labels: np.ndarray, adjacency: np.ndarray) -> int:
    score = 0
    for i in range(NODES):
        for j in range(i + 1, NODES):
            if adjacency[i, j] > 0.0 and labels[i] != labels[j]:
                score += 1
    return score


def kuramoto_rhs(theta: np.ndarray, adjacency: np.ndarray, kc: float, ks: float) -> np.ndarray:
    diff = theta[:, None] - theta[None, :]
    coupling = -2.0 * kc * np.sum(adjacency * np.sin(diff), axis=1)
    shil = -2.0 * ks * np.sin(2.0 * theta)
    return coupling + shil


def kuramoto_energy(theta: np.ndarray, adjacency: np.ndarray, kc: float, ks: float) -> float:
    energy = 0.0
    for i in range(NODES):
        for j in range(i + 1, NODES):
            if adjacency[i, j] > 0.0:
                energy += -2.0 * kc * adjacency[i, j] * np.cos(theta[i] - theta[j])
    energy += -ks * float(np.sum(np.cos(2.0 * theta)))
    return float(energy)


def wrap_to_pi(theta: np.ndarray) -> np.ndarray:
    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def rk2_solve(
    theta0: np.ndarray,
    adjacency: np.ndarray,
    kc: float,
    ks: float,
    dt: float,
    total_time: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    steps = int(round(total_time / dt))
    times = np.linspace(0.0, steps * dt, steps + 1)
    phases = np.empty((steps + 1, NODES), dtype=float)
    energies = np.empty(steps + 1, dtype=float)
    theta = wrap_to_pi(theta0.astype(float))
    phases[0] = theta
    energies[0] = kuramoto_energy(theta, adjacency, kc, ks)
    for idx in range(steps):
        k1 = kuramoto_rhs(theta, adjacency, kc, ks)
        midpoint = wrap_to_pi(theta + 0.5 * dt * k1)
        k2 = kuramoto_rhs(midpoint, adjacency, kc, ks)
        theta = wrap_to_pi(theta + dt * k2)
        phases[idx + 1] = theta
        energies[idx + 1] = kuramoto_energy(theta, adjacency, kc, ks)
    return times, phases, energies


def binary_partition_from_phase(theta: np.ndarray) -> np.ndarray:
    labels = (np.cos(theta) < 0.0).astype(int)
    return canonicalize_partition(labels)


def jacobian_at_partition(labels: np.ndarray, adjacency: np.ndarray, kc: float, ks: float) -> np.ndarray:
    theta = partition_to_theta(labels)
    phase_diff = theta[:, None] - theta[None, :]
    cos_terms = np.cos(phase_diff)
    jacobian = 2.0 * kc * adjacency * cos_terms
    diagonal = -2.0 * kc * np.sum(adjacency * cos_terms, axis=1) - 4.0 * ks * np.cos(2.0 * theta)
    np.fill_diagonal(jacobian, diagonal)
    return jacobian


def enumerate_unique_partitions(adjacency: np.ndarray) -> list[PartitionRecord]:
    records: list[PartitionRecord] = []
    for bits in range(1 << (NODES - 1)):
        labels = np.zeros(NODES, dtype=int)
        for idx in range(1, NODES):
            labels[idx] = (bits >> (idx - 1)) & 1
        jacobian = jacobian_at_partition(labels, adjacency, KC, 0.0)
        eigenvalues = np.linalg.eigvalsh(jacobian)
        lambda_max = float(eigenvalues[-1])
        records.append(
            PartitionRecord(
                labels=partition_signature(labels),
                cut_score=cut_score(labels, adjacency),
                lambda_max_at_ks0=lambda_max,
                stabilization_threshold=lambda_max / 4.0,
            )
        )
    return records


def compute_lambda_max(record: PartitionRecord, ks: float) -> float:
    return record.lambda_max_at_ks0 - 4.0 * ks


def analyze_thresholds(records: list[PartitionRecord]) -> dict[str, object]:
    max_cut = max(record.cut_score for record in records)
    max_cut_records = [record for record in records if record.cut_score == max_cut]
    non_max_records = [record for record in records if record.cut_score != max_cut]
    max_cut_threshold = max(record.stabilization_threshold for record in max_cut_records)
    first_non_max_threshold = min(record.stabilization_threshold for record in non_max_records)
    return {
        "max_cut_score": max_cut,
        "max_cut_threshold": float(max_cut_threshold),
        "first_non_max_threshold": float(first_non_max_threshold),
        "exclusive_window_exists": bool(max_cut_threshold < first_non_max_threshold),
        "exclusive_window": [
            float(max_cut_threshold),
            float(first_non_max_threshold),
        ],
    }


def top_partition_counts(signatures: list[tuple[int, ...]], top_k: int = 5) -> list[dict[str, object]]:
    counts = Counter(signatures)
    top: list[dict[str, object]] = []
    for signature, count in counts.most_common(top_k):
        top.append(
            {
                "partition_bits": list(signature),
                "partition": format_partition(signature),
                "count": int(count),
            }
        )
    return top


def draw_graph(
    ax: plt.Axes,
    adjacency: np.ndarray,
    positions: np.ndarray,
    labels: np.ndarray | tuple[int, ...] | None = None,
    title: str = "",
) -> None:
    for i in range(NODES):
        for j in range(i + 1, NODES):
            if adjacency[i, j] > 0.0:
                ax.plot(
                    [positions[i, 0], positions[j, 0]],
                    [positions[i, 1], positions[j, 1]],
                    color="#9aa5b1",
                    lw=1.4,
                    zorder=1,
                )

    if labels is None:
        colors = ["#4c78a8"] * NODES
    else:
        labels = np.asarray(labels, dtype=int)
        colors = ["#4c78a8" if value == 0 else "#e45756" for value in labels]

    ax.scatter(positions[:, 0], positions[:, 1], s=220, c=colors, edgecolors="black", zorder=3)
    for idx, (x_pos, y_pos) in enumerate(positions):
        ax.text(x_pos, y_pos, str(idx + 1), ha="center", va="center", fontsize=10, color="white", zorder=4)
    ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_part_a_histogram(records: list[PartitionRecord], out_path: Path) -> None:
    scores = [record.cut_score for record in records]
    max_cut = max(scores)
    unique_scores = sorted(set(scores))
    counts = [scores.count(score) for score in unique_scores]
    colors = ["#e45756" if score == max_cut else "#4c78a8" for score in unique_scores]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(unique_scores, counts, width=0.8, color=colors, edgecolor="black")
    ax.set_xlabel("Cut score")
    ax.set_ylabel("Number of unique partitions")
    ax.set_title("Question 2(a): Histogram of Mobius-ladder cut scores")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xticks(unique_scores)
    ax.text(
        max_cut,
        max(counts) * 0.93,
        f"MaxCut = {max_cut}",
        ha="center",
        va="top",
        fontsize=10,
        color="#7f0000",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_maxcut_partitions(
    records: list[PartitionRecord],
    adjacency: np.ndarray,
    positions: np.ndarray,
    out_path: Path,
) -> None:
    max_cut = max(record.cut_score for record in records)
    max_cut_records = [record for record in records if record.cut_score == max_cut]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    for ax, record in zip(axes.ravel(), max_cut_records):
        draw_graph(
            ax,
            adjacency,
            positions,
            np.asarray(record.labels, dtype=int),
            title=f"Cut = {record.cut_score}\n{format_partition(record.labels)}",
        )
    fig.suptitle("Question 2(a): All unique MaxCut partitions", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_lambda_histograms(records: list[PartitionRecord], ks_values: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharex=False, sharey=False)
    axes = axes.ravel()
    for ax, ks in zip(axes, ks_values):
        lambda_values = [compute_lambda_max(record, ks) for record in records]
        ax.hist(lambda_values, bins=18, color="#4c78a8", edgecolor="black", alpha=0.85)
        ax.axvline(0.0, color="#d62728", lw=1.2, ls="--")
        ax.set_title(rf"$K_s={ks:g}$", fontsize=10)
        ax.set_xlabel(r"Maximum Jacobian eigenvalue $\lambda_{\max}$")
        ax.set_ylabel("Count")
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle("Question 2(b): Histogram of maximum Jacobian eigenvalues", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_lambda_vs_cut(records: list[PartitionRecord], ks_values: np.ndarray, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, ks in zip(axes, ks_values):
        stable_x = []
        stable_y = []
        unstable_x = []
        unstable_y = []
        for record in records:
            value = compute_lambda_max(record, ks)
            if value < 0.0:
                stable_x.append(record.cut_score)
                stable_y.append(value)
            else:
                unstable_x.append(record.cut_score)
                unstable_y.append(value)
        ax.scatter(unstable_x, unstable_y, color="#9aa5b1", s=26, alpha=0.85, label="unstable")
        ax.scatter(stable_x, stable_y, color="#2a9d8f", s=32, alpha=0.9, label="stable")
        ax.axhline(0.0, color="#d62728", lw=1.1, ls="--")
        ax.set_title(rf"$K_s={ks:g}$", fontsize=10)
        ax.set_xlabel("Cut score")
        ax.set_ylabel(r"$\lambda_{\max}$")
        ax.grid(True, alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=2)
    fig.suptitle("Question 2(b): Maximum eigenvalue vs cut score", fontsize=14, y=0.985)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def simulate_trials(
    adjacency: np.ndarray,
    ks_values: np.ndarray,
    trials: int,
    dt: float,
    total_time: float,
    seed: int,
) -> dict[float, dict[str, object]]:
    payload: dict[float, dict[str, object]] = {}
    for ks_index, ks in enumerate(ks_values):
        rng = np.random.default_rng(seed + ks_index)
        cut_scores: list[int] = []
        signatures: list[tuple[int, ...]] = []
        energies: list[np.ndarray] = []
        derivative_norms: list[float] = []
        for _ in range(trials):
            theta0 = rng.uniform(-np.pi, np.pi, size=NODES)
            times, phases, trial_energies = rk2_solve(theta0, adjacency, KC, ks, dt, total_time)
            final_theta = phases[-1]
            labels = binary_partition_from_phase(final_theta)
            cut_scores.append(cut_score(labels, adjacency))
            signatures.append(partition_signature(labels))
            energies.append(trial_energies)
            derivative_norms.append(float(np.linalg.norm(kuramoto_rhs(final_theta, adjacency, KC, ks))))
        payload[float(ks)] = {
            "times": times,
            "energy_trajectories": energies,
            "cut_scores": cut_scores,
            "signatures": signatures,
            "mean_cut_score": float(np.mean(cut_scores)),
            "std_cut_score": float(np.std(cut_scores)),
            "best_cut_score": int(np.max(cut_scores)),
            "mean_terminal_rhs_norm": float(np.mean(derivative_norms)),
            "top_partitions": top_partition_counts(signatures),
            "cut_score_histogram": {
                str(score): int(count)
                for score, count in sorted(Counter(cut_scores).items())
            },
        }
    return payload


def plot_energy_trajectories(simulation_payload: dict[float, dict[str, object]], out_path: Path) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(14, 11), sharex=True, sharey=False)
    axes = axes.ravel()
    for ax, ks in zip(axes, KS_VALUES):
        trial_payload = simulation_payload[float(ks)]
        times = trial_payload["times"]
        energies = trial_payload["energy_trajectories"]
        stacked = np.vstack(energies)
        sample_count = min(8, len(energies))
        for trace in stacked[:sample_count]:
            ax.plot(times, trace, color="#9aa5b1", lw=0.9, alpha=0.7)
        ax.plot(times, stacked.mean(axis=0), color="#264653", lw=2.0, label="mean")
        ax.set_title(rf"$K_s={ks:g}$", fontsize=10)
        ax.set_xlabel("Time")
        ax.set_ylabel("Energy")
        ax.grid(True, alpha=0.25)
    fig.suptitle("Question 2(c): Temporal energy trajectories from RK2", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_average_cut_vs_ks(
    simulation_payload: dict[float, dict[str, object]],
    threshold_info: dict[str, object],
    out_path: Path,
) -> None:
    mean_cuts = np.array([simulation_payload[float(ks)]["mean_cut_score"] for ks in KS_VALUES], dtype=float)
    std_cuts = np.array([simulation_payload[float(ks)]["std_cut_score"] for ks in KS_VALUES], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(KS_VALUES, mean_cuts, yerr=std_cuts, fmt="o-", color="#264653", capsize=4)
    if threshold_info["exclusive_window_exists"]:
        left, right = threshold_info["exclusive_window"]
        ax.axvspan(left, right, color="#2a9d8f", alpha=0.15, label="Only MaxCut binary states stable")
    ax.set_xlabel(r"$K_s$")
    ax.set_ylabel("Average cut score across trials")
    ax.set_title("Question 2(c): Average binarized cut score vs SHIL strength")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.0, max(mean_cuts + std_cuts) + 0.7)
    ax.legend(fontsize=9, loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_question2_summary(
    records: list[PartitionRecord],
    threshold_info: dict[str, object],
    simulation_payload: dict[float, dict[str, object]],
    results_dir: Path,
) -> None:
    max_cut = threshold_info["max_cut_score"]
    max_cut_records = [record for record in records if record.cut_score == max_cut]
    summary_payload = {
        "graph": {
            "name": "8-node Mobius ladder",
            "nodes": NODES,
            "edges": 12,
            "coupling_constant_kc": KC,
        },
        "part_a": {
            "unique_partitions_enumerated": len(records),
            "max_cut_score": max_cut,
            "max_cut_partitions": [
                {
                    "partition_bits": list(record.labels),
                    "partition": format_partition(record.labels),
                }
                for record in max_cut_records
            ],
        },
        "part_b": {
            "stability_thresholds": threshold_info,
            "records": [
                {
                    "partition_bits": list(record.labels),
                    "partition": format_partition(record.labels),
                    "cut_score": record.cut_score,
                    "lambda_max_at_ks0": record.lambda_max_at_ks0,
                    "stabilization_threshold": record.stabilization_threshold,
                }
                for record in records
            ],
        },
        "part_c": {
            str(ks): {
                "mean_cut_score": simulation_payload[float(ks)]["mean_cut_score"],
                "std_cut_score": simulation_payload[float(ks)]["std_cut_score"],
                "best_cut_score": simulation_payload[float(ks)]["best_cut_score"],
                "mean_terminal_rhs_norm": simulation_payload[float(ks)]["mean_terminal_rhs_norm"],
                "cut_score_histogram": simulation_payload[float(ks)]["cut_score_histogram"],
                "top_partitions": simulation_payload[float(ks)]["top_partitions"],
            }
            for ks in KS_VALUES
        },
    }
    (results_dir / "question2_summary.json").write_text(json.dumps(summary_payload, indent=2))

    if threshold_info["exclusive_window_exists"]:
        left, right = threshold_info["exclusive_window"]
        prediction_text = (
            f"Binary MaxCut partitions become stable for Ks > {left:.4f}, "
            f"while the first non-MaxCut binary partitions appear at Ks = {right:.4f}. "
            "This makes the narrow interval between these values the best binary-state window."
        )
    else:
        prediction_text = "No exclusive binary-stability window was found in the scanned range."

    lines = [
        "Question 2 summary",
        "",
        f"Unique partitions enumerated (after quotienting out global label swap): {len(records)}",
        f"Maximum cut score: {max_cut}",
        "MaxCut partitions:",
    ]
    for record in max_cut_records:
        lines.append(f"  {format_partition(record.labels)}")
    lines.extend(
        [
            "",
            "Part (b) prediction from the Jacobian spectra:",
            f"  {prediction_text}",
            "  Because J(Ks) = J(0) - 4*Ks*I at binary phase configurations, every Jacobian eigenvalue shifts downward linearly with Ks.",
            "  Small Ks keeps binary partitions weakly stabilized; very large Ks stabilizes many suboptimal binary cuts as well.",
            "",
            "Part (c) discussion scaffold:",
            "  Compare the average simulated cut scores against the predicted binary-stability window.",
            "  If small-Ks simulations still produce high cut scores, that comes from non-binary continuous-phase equilibria that part (b) does not classify.",
            "  If large-Ks simulations degrade, that reflects extra suboptimal binary attractors becoming stable.",
        ]
    )
    (results_dir / "question2_summary.txt").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Solve the Mobius-ladder Kuramoto / Max-Cut tasks from Assignment C3 Question 2."
    )
    parser.add_argument("--dt", type=float, default=DEFAULT_DT, help="Time step for the RK2 solver.")
    parser.add_argument(
        "--time",
        type=float,
        default=DEFAULT_TOTAL_TIME,
        help="Total integration time for each Kuramoto simulation trial.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=DEFAULT_TRIALS,
        help="Number of random-initial-phase trials per Ks value.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images_dir, results_dir = ensure_output_dirs()

    adjacency = mobius_ladder_adjacency()
    positions = mobius_positions()
    records = enumerate_unique_partitions(adjacency)
    threshold_info = analyze_thresholds(records)

    print("Running Question 2(a) partition enumeration and MaxCut analysis")
    plot_part_a_histogram(records, images_dir / "Q2_part_a_cut_histogram.png")
    plot_maxcut_partitions(records, adjacency, positions, images_dir / "Q2_part_a_maxcut_partitions.png")

    print("Running Question 2(b) Jacobian eigenvalue analysis")
    plot_lambda_histograms(records, KS_VALUES, images_dir / "Q2_part_b_lambda_histograms.png")
    plot_lambda_vs_cut(records, KS_VALUES, images_dir / "Q2_part_b_lambda_vs_cut.png")

    print("Running Question 2(c) RK2 simulations from random initial phases")
    simulation_payload = simulate_trials(
        adjacency=adjacency,
        ks_values=KS_VALUES,
        trials=args.trials,
        dt=args.dt,
        total_time=args.time,
        seed=args.seed,
    )
    plot_energy_trajectories(simulation_payload, images_dir / "Q2_part_c_energy_trajectories.png")
    plot_average_cut_vs_ks(
        simulation_payload,
        threshold_info,
        images_dir / "Q2_part_c_average_cut_vs_ks.png",
    )

    write_question2_summary(records, threshold_info, simulation_payload, results_dir)

    print(f"Saved figures to {images_dir}")
    print(f"Saved summaries to {results_dir}")


if __name__ == "__main__":
    main()
