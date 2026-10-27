from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


METRIC_KEYS = ("exact_match", "sequence_similarity", "length_control_rate", "clean_output_rate")

TASK_MODELS = {
    "completion": (
        ("uniform", "Uniform", None),
        ("scanpath", "Scanpath", "scanpath_vs_uniform"),
        ("eeg_workload", "EEG workload", "eeg_workload_vs_uniform"),
        ("scanpath_eeg", "Scanpath+EEG", "scanpath_eeg_vs_uniform"),
    ),
    "summarization": (
        ("uniform", "Uniform", None),
        ("scanpath", "Scanpath", "scanpath_vs_uniform"),
        ("eeg_workload", "EEG workload", "eeg_workload_vs_uniform"),
        ("scanpath_eeg", "Scanpath+EEG", "scanpath_eeg_vs_uniform"),
    ),
}

MODEL_COLORS = {
    "uniform": "#595959",
    "scanpath": "#8c8c8c",
    "eeg_workload": "#d94801",
    "scanpath_eeg": "#238b45",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute layer-group checkpoint metrics from task_weight_heatmaps cache "
            "and draw a reviewer-facing figure about EEG-guided adaptation."
        )
    )
    parser.add_argument(
        "--cache-json",
        default="outputs/figures/task_weight_heatmaps_uniform_baselines_cache.json",
        help="Input cache produced by plot_task_weight_heatmaps.py.",
    )
    parser.add_argument(
        "--output",
        default="outputs/figures/checkpoint_layer_group_metrics.png",
        help="Where to save the metrics figure.",
    )
    parser.add_argument(
        "--metrics-json",
        default="outputs/figures/checkpoint_layer_group_metrics.json",
        help="Where to save computed layer-group metrics.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of strongest layers used for top-k concentration.",
    )
    return parser.parse_args()


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def quality_score(metric_dict: dict[str, float]) -> float:
    return 100.0 * float(np.mean([metric_dict[key] for key in METRIC_KEYS]))


def gini(values: np.ndarray) -> float:
    flat = np.sort(values.astype(np.float64).ravel())
    total = float(flat.sum())
    if total <= 0.0:
        return 0.0
    n_items = flat.size
    weights = 2 * np.arange(1, n_items + 1) - n_items - 1
    return float(np.dot(weights, flat) / (n_items * total))


def normalized_entropy(values: np.ndarray) -> float:
    flat = values.astype(np.float64).ravel()
    total = float(flat.sum())
    if total <= 0.0:
        return 0.0
    probabilities = flat / total
    return float(-np.sum(probabilities * np.log(probabilities + 1e-12)) / np.log(flat.size))


def drift_metrics(matrix: np.ndarray, top_k: int) -> dict[str, float]:
    total = float(matrix.sum())
    if total <= 0.0:
        return {
            "mean_relative_drift": 0.0,
            "max_relative_drift": 0.0,
            "late_layer_share": 0.0,
            "topk_layer_share": 0.0,
            "attention_share": 0.0,
            "mlp_share": 0.0,
            "norm_share": 0.0,
            "gini": 0.0,
            "normalized_entropy": 0.0,
            "specialization": 0.0,
            "depth_center": 0.0,
        }

    layer_scores = matrix.sum(axis=1)
    top_layers = np.argsort(layer_scores)[-top_k:]
    layer_positions = np.arange(matrix.shape[0], dtype=np.float64)
    depth_center = float(np.dot(layer_positions, layer_scores) / ((matrix.shape[0] - 1) * total))
    entropy = normalized_entropy(matrix)

    return {
        "mean_relative_drift": float(matrix.mean()),
        "max_relative_drift": float(matrix.max()),
        "late_layer_share": float(matrix[matrix.shape[0] // 2 :, :].sum() / total),
        "topk_layer_share": float(matrix[top_layers, :].sum() / total),
        "attention_share": float(matrix[:, [0, 1]].sum() / total),
        "mlp_share": float(matrix[:, [2, 3, 4]].sum() / total),
        "norm_share": float(matrix[:, [5]].sum() / total),
        "gini": gini(matrix),
        "normalized_entropy": entropy,
        "specialization": float(1.0 - entropy),
        "depth_center": depth_center,
    }


def build_metrics(cache: dict[str, Any], top_k: int) -> dict[str, Any]:
    result: dict[str, Any] = {
        "research_question": (
            "Does EEG-guided training produce more behaviorally useful layer-group "
            "adaptation than scanpath-only or uniform training?"
        ),
        "metric_definitions": {
            "quality_score": "Mean of Exact, SeqSim, Len<=1.5x, and Clean, reported as percent.",
            "mean_relative_drift": "Mean relative Frobenius/L2 checkpoint drift from uniform over layer groups.",
            "drift_efficiency": "Quality-score gain over uniform divided by mean relative drift.",
            "late_layer_share": "Share of total checkpoint drift in the second half of transformer layers.",
            "topk_layer_share": "Share of total checkpoint drift in the top-k strongest layers.",
            "gini": "Concentration of drift across layer-group cells; larger means more localized adaptation.",
            "specialization": "One minus normalized entropy across layer-group cells; larger means less diffuse drift.",
            "depth_center": "Weighted average layer depth of checkpoint drift, normalized to [0, 1].",
        },
        "tasks": {},
    }

    for task, models in TASK_MODELS.items():
        task_metrics = cache["metrics"][task]
        uniform_quality = quality_score(task_metrics["uniform"])
        result["tasks"][task] = {}

        for model_name, label, comparison_name in models:
            row: dict[str, float | str | None] = {
                "model": model_name,
                "label": label,
                "comparison": comparison_name,
                "quality_score": quality_score(task_metrics[model_name]),
            }
            row["quality_gain_pp"] = float(row["quality_score"] - uniform_quality)

            if comparison_name is None:
                row.update(
                    {
                        "mean_relative_drift": 0.0,
                        "max_relative_drift": 0.0,
                        "drift_efficiency": None,
                        "late_layer_share": 0.0,
                        "topk_layer_share": 0.0,
                        "attention_share": 0.0,
                        "mlp_share": 0.0,
                        "norm_share": 0.0,
                        "gini": 0.0,
                        "normalized_entropy": 0.0,
                        "specialization": 0.0,
                        "depth_center": 0.0,
                    }
                )
            else:
                matrix = np.asarray(cache["drift_matrices"][task][comparison_name], dtype=np.float64)
                row.update(drift_metrics(matrix, top_k=top_k))
                mean_drift = float(row["mean_relative_drift"])
                row["drift_efficiency"] = float(row["quality_gain_pp"] / mean_drift) if mean_drift > 0.0 else None

            result["tasks"][task][model_name] = row

    return result


def annotate_bars(ax: plt.Axes, bars: Any, suffix: str = "") -> None:
    for bar in bars:
        height = float(bar.get_height())
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + max(0.8, 0.025 * height),
            f"{height:.1f}{suffix}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=90 if height > 15 else 0,
        )


def plot_quality_vs_drift(ax: plt.Axes, metrics: dict[str, Any], task: str) -> None:
    rows = list(metrics["tasks"][task].values())
    for row in rows:
        model_name = str(row["model"])
        ax.scatter(
            float(row["mean_relative_drift"]),
            float(row["quality_score"]),
            s=180 if model_name in {"eeg_workload", "scanpath_eeg"} else 120,
            color=MODEL_COLORS[model_name],
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        ax.text(
            float(row["mean_relative_drift"]) + 0.04,
            float(row["quality_score"]) + 1.1,
            str(row["label"]),
            fontsize=9,
        )

    ax.set_xlabel("Mean relative checkpoint drift from uniform")
    ax.set_ylabel("Composite quality score (%)")
    ax.set_title(f"{task.title()}: quality gained from comparable drift")
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.55)


def plot_efficiency(ax: plt.Axes, metrics: dict[str, Any], task: str) -> None:
    rows = [
        row
        for row in metrics["tasks"][task].values()
        if row["model"] != "uniform" and row["drift_efficiency"] is not None
    ]
    labels = [str(row["label"]) for row in rows]
    values = [float(row["drift_efficiency"]) for row in rows]
    colors = [MODEL_COLORS[str(row["model"])] for row in rows]
    bars = ax.bar(labels, values, color=colors, alpha=0.92)
    annotate_bars(ax, bars)
    ax.set_ylabel("Quality gain / mean drift")
    ax.set_title(f"{task.title()}: behavioral efficiency of checkpoint drift")
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.55)
    ax.tick_params(axis="x", rotation=20)


def plot_structure(ax: plt.Axes, metrics: dict[str, Any], task: str) -> None:
    rows = [
        metrics["tasks"][task][model_name]
        for model_name in ("scanpath", "eeg_workload", "scanpath_eeg")
        if model_name in metrics["tasks"][task]
    ]
    labels = [str(row["label"]) for row in rows]
    values = {
        "Late-layer share": [100.0 * float(row["late_layer_share"]) for row in rows],
        "Top-5 layer share": [100.0 * float(row["topk_layer_share"]) for row in rows],
        "MLP drift share": [100.0 * float(row["mlp_share"]) for row in rows],
    }
    x_positions = np.arange(len(labels))
    width = 0.24
    colors = ("#6baed6", "#fd8d3c", "#74c476")

    for offset_idx, (name, vals) in enumerate(values.items()):
        offset = (offset_idx - 1) * width
        bars = ax.bar(x_positions + offset, vals, width=width, label=name, color=colors[offset_idx], alpha=0.95)
        annotate_bars(ax, bars, suffix="%")

    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Share of total drift (%)")
    ax.set_title(f"{task.title()} layer-group drift allocation")
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.55)


def plot_metric_table(ax: plt.Axes, metrics: dict[str, Any]) -> None:
    rows = [
        ("Completion", "Scanpath", metrics["tasks"]["completion"]["scanpath"]),
        ("Completion", "EEG workload", metrics["tasks"]["completion"]["eeg_workload"]),
        ("Completion", "Scanpath+EEG", metrics["tasks"]["completion"]["scanpath_eeg"]),
        ("Summarization", "Scanpath", metrics["tasks"]["summarization"]["scanpath"]),
        ("Summarization", "EEG workload", metrics["tasks"]["summarization"]["eeg_workload"]),
        ("Summarization", "Scanpath+EEG", metrics["tasks"]["summarization"]["scanpath_eeg"]),
    ]
    cell_text = []
    for task, label, row in rows:
        cell_text.append(
            [
                task,
                label,
                f"{float(row['quality_score']):.1f}",
                f"{float(row['quality_gain_pp']):+.1f}",
                f"{float(row['mean_relative_drift']):.2f}",
                f"{float(row['drift_efficiency']):.2f}",
                f"{100.0 * float(row['late_layer_share']):.1f}",
                f"{100.0 * float(row['mlp_share']):.1f}",
            ]
        )

    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=("Task", "Model", "Quality", "Gain", "Drift", "Efficiency", "Late %", "MLP %"),
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.45)
    for (row_idx, col_idx), cell in table.get_celld().items():
        if row_idx == 0:
            cell.set_facecolor("#d9ead3")
            cell.set_text_props(weight="bold")
        elif row_idx in (2, 3, 5, 6):
            cell.set_facecolor("#fff2cc")
        else:
            cell.set_facecolor("#f7f7f7")
    ax.set_title("Layer-group checkpoint metrics tied to corpus behavior", pad=14)


def make_figure(metrics: dict[str, Any], output_path: Path) -> None:
    fig = plt.figure(figsize=(18, 15))
    grid = fig.add_gridspec(nrows=3, ncols=3, height_ratios=[1.0, 1.0, 0.88], hspace=0.42, wspace=0.32)

    plot_quality_vs_drift(fig.add_subplot(grid[0, 0]), metrics, "completion")
    plot_efficiency(fig.add_subplot(grid[0, 1]), metrics, "completion")
    plot_structure(fig.add_subplot(grid[0, 2]), metrics, "completion")
    plot_quality_vs_drift(fig.add_subplot(grid[1, 0]), metrics, "summarization")
    plot_efficiency(fig.add_subplot(grid[1, 1]), metrics, "summarization")
    plot_structure(fig.add_subplot(grid[1, 2]), metrics, "summarization")
    plot_metric_table(fig.add_subplot(grid[2, :]), metrics)

    fig.suptitle(
        "Research question: does EEG-guided training produce more useful layer-group adaptation?",
        fontsize=15,
        y=0.98,
    )
    fig.text(
        0.5,
        0.018,
        "Composite quality = mean of Exact, SeqSim, Len<=1.5x, and Clean. "
        "Efficiency = quality gain over uniform divided by mean relative checkpoint drift.",
        ha="center",
        fontsize=9,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cache = read_json(args.cache_json)
    metrics = build_metrics(cache, top_k=args.top_k)

    metrics_path = Path(args.metrics_json)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)

    output_path = Path(args.output)
    make_figure(metrics, output_path)
    print(
        json.dumps(
            {
                "figure_path": str(output_path),
                "metrics_json": str(metrics_path),
                "research_question": metrics["research_question"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
