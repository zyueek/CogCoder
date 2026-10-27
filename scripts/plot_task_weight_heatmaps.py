from __future__ import annotations

import argparse
import difflib
import json
import math
import os
from pathlib import Path
from statistics import mean
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import PowerNorm
from matplotlib.patches import Rectangle
from safetensors import safe_open


TASK_MODEL_PATHS = {
    "completion": {
        "uniform": "outputs/qwen35_base_uniform_completion_large/model.safetensors",
        "scanpath": "outputs/qwen35_base_scanpath_completion_large/model.safetensors",
        "scanpath_eeg": "outputs/qwen35_base_scanpath_eeg_completion_large/model.safetensors",
        "eeg_workload": "outputs/qwen35_base_scanpath_eeg_workload_completion_large/model.safetensors",
    },
    "summarization": {
        "uniform": "outputs/qwen35_base_uniform_summarization_large/model.safetensors",
        "scanpath": "outputs/qwen35_base_scanpath_summarization_large/model.safetensors",
        "scanpath_eeg": "outputs/qwen35_base_scanpath_eeg_summarization_large/model.safetensors",
        "eeg_workload": "outputs/qwen35_base_scanpath_eeg_workload_summarization_large/model.safetensors",
    },
}

TASK_EVAL_PATHS = {
    "completion": {
        "uniform": "outputs/qwen35_base_uniform_completion_eval.json",
        "scanpath": "outputs/qwen35_base_scanpath_completion_eval.json",
        "scanpath_eeg": "outputs/qwen35_base_scanpath_eeg_completion_eval.json",
        "eeg_workload": "outputs/qwen35_base_eeg_workload_scanpath_completion_eval.json",
    },
    "summarization": {
        "uniform": "outputs/qwen35_base_uniform_summarization_eval.json",
        "scanpath": "outputs/qwen35_base_scanpath_summarization_eval.json",
        "scanpath_eeg": "outputs/qwen35_base_scanpath_eeg_summarization_eval.json",
        "eeg_workload": "outputs/qwen35_base_scanpath_eeg_workload_summarization_eval2.json",
    },
}

DISPLAY_MODEL = {
    "uniform": "Uniform",
    "scanpath": "Scanpath",
    "scanpath_eeg": "Scanpath+EEG",
    "eeg_workload": "EEG workload",
}

MODEL_ORDER = {
    "completion": ("uniform", "scanpath", "eeg_workload", "scanpath_eeg"),
    "summarization": ("uniform", "scanpath", "eeg_workload", "scanpath_eeg"),
}

COMPARISON_DISPLAY = {
    "scanpath_vs_uniform": "Scanpath vs Uniform",
    "eeg_workload_vs_uniform": "EEG workload vs Uniform",
    "scanpath_eeg_vs_uniform": "Scanpath+EEG vs Uniform",
}

GROUP_SPECS: list[tuple[str, list[str]]] = [
    ("Attn In", ["self_attn.q_proj.weight", "linear_attn.in_proj_qkv.weight"]),
    ("Attn Out", ["self_attn.o_proj.weight", "linear_attn.out_proj.weight"]),
    ("MLP Gate", ["mlp.gate_proj.weight"]),
    ("MLP Up", ["mlp.up_proj.weight"]),
    ("MLP Down", ["mlp.down_proj.weight"]),
    ("Norms", ["input_layernorm.weight", "post_attention_layernorm.weight"]),
]

METRIC_SPECS: list[tuple[str, str]] = [
    ("exact_match", "Exact"),
    ("sequence_similarity", "SeqSim"),
    ("length_control_rate", "Len<=1.5x"),
    ("clean_output_rate", "Clean"),
]

CLEAN_OUTPUT_MARKERS = [
    "user",
    "assistant",
    "### instruction",
    "### java code",
    "### summary",
    "### completion",
    "<think>",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare checkpoint weight drift and corpus-level generation metrics "
            "for uniform, scanpath, EEG workload, and scanpath+EEG models."
        )
    )
    parser.add_argument(
        "--output",
        default="outputs/figures/task_weight_heatmaps_uniform_baselines.png",
        help="Where to save the final figure.",
    )
    parser.add_argument(
        "--cache-json",
        default="outputs/figures/task_weight_heatmaps_uniform_baselines_cache.json",
        help="Cache file for the expensive safetensors aggregation.",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=32,
        help="Number of transformer layers to aggregate.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Force recomputation instead of reusing the cached JSON.",
    )
    parser.add_argument(
        "--drift-scale",
        type=float,
        default=1000.0,
        help="Multiply relative weight drift by this factor for display.",
    )
    parser.add_argument(
        "--contrast-gamma",
        type=float,
        default=0.65,
        help=(
            "Power-law color normalization for heatmaps. Values below 1.0 expand "
            "mid-range drift so EEG-vs-uniform structure is easier to inspect."
        ),
    )
    return parser.parse_args()


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_text(text: str) -> str:
    return " ".join(str(text).split()).strip()


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_text(prediction) == normalize_text(reference))


def sequence_similarity(prediction: str, reference: str) -> float:
    return difflib.SequenceMatcher(None, normalize_text(prediction), normalize_text(reference)).ratio()


def clean_output_rate(prediction: str) -> float:
    lowered = normalize_text(prediction).lower()
    return float(not any(marker in lowered for marker in CLEAN_OUTPUT_MARKERS))


def length_control_rate(prediction: str, reference: str) -> float:
    pred_len = len(normalize_text(prediction))
    ref_len = max(1, len(normalize_text(reference)))
    return float(pred_len <= 1.5 * ref_len)


def compute_corpus_metrics(eval_rows: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "exact_match": mean(exact_match(row["generated_output"], row["ground_truth"]) for row in eval_rows),
        "sequence_similarity": mean(sequence_similarity(row["generated_output"], row["ground_truth"]) for row in eval_rows),
        "length_control_rate": mean(length_control_rate(row["generated_output"], row["ground_truth"]) for row in eval_rows),
        "clean_output_rate": mean(clean_output_rate(row["generated_output"]) for row in eval_rows),
    }


def relative_group_drift(
    ref_file: str,
    target_file: str,
    num_layers: int,
    drift_scale: float,
) -> np.ndarray:
    matrix = np.zeros((num_layers, len(GROUP_SPECS)), dtype=np.float32)

    with safe_open(ref_file, framework="pt", device="cpu") as ref_sf, safe_open(target_file, framework="pt", device="cpu") as target_sf:
        ref_keys = set(ref_sf.keys())
        target_keys = set(target_sf.keys())

        for layer_idx in range(num_layers):
            for group_idx, (_, suffixes) in enumerate(GROUP_SPECS):
                numerator = 0.0
                denominator = 0.0
                match_count = 0

                for suffix in suffixes:
                    key = f"model.language_model.layers.{layer_idx}.{suffix}"
                    if key not in ref_keys or key not in target_keys:
                        continue

                    ref_tensor = ref_sf.get_tensor(key).float()
                    target_tensor = target_sf.get_tensor(key).float()
                    diff_tensor = target_tensor - ref_tensor

                    numerator += float((diff_tensor * diff_tensor).sum().item())
                    denominator += float((ref_tensor * ref_tensor).sum().item())
                    match_count += 1

                if match_count > 0 and denominator > 0.0:
                    matrix[layer_idx, group_idx] = drift_scale * math.sqrt(numerator / denominator)

    return matrix


def comparison_pairs(model_paths: dict[str, str]) -> dict[str, tuple[str, str]]:
    pairs = {
        "scanpath_vs_uniform": (model_paths["uniform"], model_paths["scanpath"]),
    }
    if "eeg_workload" in model_paths:
        pairs["eeg_workload_vs_uniform"] = (model_paths["uniform"], model_paths["eeg_workload"])
    pairs["scanpath_eeg_vs_uniform"] = (model_paths["uniform"], model_paths["scanpath_eeg"])
    return pairs


def build_analysis_cache(args: argparse.Namespace) -> dict[str, Any]:
    cache: dict[str, Any] = {
        "num_layers": args.num_layers,
        "drift_scale": args.drift_scale,
        "contrast_gamma": args.contrast_gamma,
        "group_labels": [label for label, _ in GROUP_SPECS],
        "metrics": {},
        "drift_matrices": {},
        "top_layers": {},
    }

    for task, eval_paths in TASK_EVAL_PATHS.items():
        cache["metrics"][task] = {}
        for model_name, eval_path in eval_paths.items():
            cache["metrics"][task][model_name] = compute_corpus_metrics(read_json(eval_path))

    for task, model_paths in TASK_MODEL_PATHS.items():
        pairs = comparison_pairs(model_paths)
        cache["drift_matrices"][task] = {}
        cache["top_layers"][task] = {}

        for comparison_name, (ref_path, target_path) in pairs.items():
            matrix = relative_group_drift(
                ref_file=ref_path,
                target_file=target_path,
                num_layers=args.num_layers,
                drift_scale=args.drift_scale,
            )
            cache["drift_matrices"][task][comparison_name] = matrix.tolist()
            ranked_layers = sorted(
                range(args.num_layers),
                key=lambda layer_idx: float(matrix[layer_idx].mean()),
                reverse=True,
            )[:5]
            cache["top_layers"][task][comparison_name] = ranked_layers

    return cache


def cache_is_complete(cache: dict[str, Any], args: argparse.Namespace) -> bool:
    if int(cache.get("num_layers", -1)) != args.num_layers:
        return False
    if float(cache.get("drift_scale", -1.0)) != args.drift_scale:
        return False

    for task, eval_paths in TASK_EVAL_PATHS.items():
        if task not in cache.get("metrics", {}):
            return False
        for model_name in eval_paths:
            if model_name not in cache["metrics"][task]:
                return False

    for task, model_paths in TASK_MODEL_PATHS.items():
        if task not in cache.get("drift_matrices", {}) or task not in cache.get("top_layers", {}):
            return False
        for comparison_name in comparison_pairs(model_paths):
            if comparison_name not in cache["drift_matrices"][task]:
                return False
            if comparison_name not in cache["top_layers"][task]:
                return False

    return True


def maybe_load_cache(args: argparse.Namespace) -> dict[str, Any]:
    cache_path = Path(args.cache_json)
    if cache_path.exists() and not args.refresh_cache:
        cache = read_json(str(cache_path))
        if cache_is_complete(cache, args):
            cache["contrast_gamma"] = args.contrast_gamma
            return cache

    cache = build_analysis_cache(args)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2)
    return cache


def add_metric_highlights(ax: plt.Axes, metric_matrix: np.ndarray, row_groups: list[tuple[int, int]]) -> None:
    for task_start, task_end in row_groups:
        task_block = metric_matrix[task_start:task_end]
        for metric_idx in range(task_block.shape[1]):
            best_row = int(np.argmax(task_block[:, metric_idx]))
            rect = Rectangle(
                (metric_idx - 0.5, task_start + best_row - 0.5),
                1.0,
                1.0,
                fill=False,
                linewidth=2.0,
                edgecolor="white",
            )
            ax.add_patch(rect)


def plot_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    title: str,
    vmax: float,
    contrast_gamma: float,
    top_layers: list[int],
) -> Any:
    image = ax.imshow(
        matrix,
        aspect="auto",
        origin="lower",
        cmap="YlOrRd",
        norm=PowerNorm(gamma=contrast_gamma, vmin=0.0, vmax=vmax),
    )
    ax.set_xticks(np.arange(len(GROUP_SPECS)))
    ax.set_xticklabels([label for label, _ in GROUP_SPECS], rotation=30, ha="right")
    yticks = np.arange(0, matrix.shape[0], 4)
    ax.set_yticks(yticks)
    ax.set_yticklabels([str(int(tick)) for tick in yticks])
    ax.set_ylabel("Layer")
    ax.set_title(title, fontsize=11)
    for boundary in (8, 16, 24):
        ax.axhline(boundary - 0.5, color="white", linestyle="--", linewidth=0.8, alpha=0.8)
    for layer_idx in top_layers[:3]:
        ax.add_patch(
            Rectangle(
                (-0.5, layer_idx - 0.5),
                len(GROUP_SPECS),
                1.0,
                fill=False,
                linewidth=1.2,
                edgecolor="#184a90",
                alpha=0.9,
            )
        )
    return image


def drift_title(task_label: str, comparison_label: str, matrix: np.ndarray, top_layers: list[int]) -> str:
    return (
        f"{task_label}: {comparison_label}\n"
        f"mean={matrix.mean():.2f}, max={matrix.max():.2f}, top layers={top_layers[:3]}"
    )


def plot_metric_gain_panel(ax: plt.Axes, cache: dict[str, Any], task: str) -> None:
    task_metrics = cache["metrics"][task]
    baseline = task_metrics["uniform"]
    compared_models = [model_name for model_name in ("scanpath", "eeg_workload", "scanpath_eeg") if model_name in task_metrics]
    colors = {
        "scanpath": "#8c8c8c",
        "eeg_workload": "#d94801",
        "scanpath_eeg": "#238b45",
    }

    x_positions = np.arange(len(METRIC_SPECS))
    width = 0.24
    offsets = np.linspace(-width, width, len(compared_models))

    for offset, model_name in zip(offsets, compared_models):
        gains = [
            100.0 * (task_metrics[model_name][metric_key] - baseline[metric_key])
            for metric_key, _ in METRIC_SPECS
        ]
        ax.bar(
            x_positions + offset,
            gains,
            width=width,
            color=colors[model_name],
            label=DISPLAY_MODEL[model_name],
            alpha=0.92,
        )
        for x_pos, gain in zip(x_positions + offset, gains):
            ax.text(
                x_pos,
                gain + 1.2,
                f"+{gain:.1f}",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=90,
            )

    ax.axhline(0.0, color="black", linewidth=0.9)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([label for _, label in METRIC_SPECS], rotation=20, ha="right")
    ax.set_ylabel("Gain over uniform (percentage points)")
    ax.set_title(f"{task.title()} quality gains over uniform")
    max_gain = max(
        1.0,
        max(
            100.0 * (task_metrics[model_name][metric_key] - baseline[metric_key])
            for model_name in compared_models
            for metric_key, _ in METRIC_SPECS
        ),
    )
    ax.set_ylim(0.0, max(8.0, 1.18 * max_gain))
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax.grid(axis="y", linestyle=":", linewidth=0.7, alpha=0.55)


def make_figure(cache: dict[str, Any], output_path: Path) -> None:
    completion_scanpath = np.asarray(cache["drift_matrices"]["completion"]["scanpath_vs_uniform"], dtype=np.float32)
    completion_eeg_workload = np.asarray(cache["drift_matrices"]["completion"]["eeg_workload_vs_uniform"], dtype=np.float32)
    completion_eeg = np.asarray(cache["drift_matrices"]["completion"]["scanpath_eeg_vs_uniform"], dtype=np.float32)
    summarization_scanpath = np.asarray(cache["drift_matrices"]["summarization"]["scanpath_vs_uniform"], dtype=np.float32)
    summarization_eeg_workload = np.asarray(cache["drift_matrices"]["summarization"]["eeg_workload_vs_uniform"], dtype=np.float32)
    summarization_eeg = np.asarray(cache["drift_matrices"]["summarization"]["scanpath_eeg_vs_uniform"], dtype=np.float32)

    completion_vmax = float(max(completion_scanpath.max(), completion_eeg_workload.max(), completion_eeg.max()))
    summarization_vmax = float(max(summarization_scanpath.max(), summarization_eeg_workload.max(), summarization_eeg.max()))
    contrast_gamma = float(cache.get("contrast_gamma", 0.65))

    fig = plt.figure(figsize=(21, 18))
    grid = fig.add_gridspec(nrows=4, ncols=3, height_ratios=[1.0, 1.0, 0.72, 0.85], hspace=0.46, wspace=0.24)

    ax_completion_scan = fig.add_subplot(grid[0, 0])
    ax_completion_workload = fig.add_subplot(grid[0, 1])
    ax_completion_eeg = fig.add_subplot(grid[0, 2])
    ax_summarization_scan = fig.add_subplot(grid[1, 0])
    ax_summarization_workload = fig.add_subplot(grid[1, 1])
    ax_summarization_eeg = fig.add_subplot(grid[1, 2])
    ax_completion_gain = fig.add_subplot(grid[2, 0:2])
    ax_summarization_gain = fig.add_subplot(grid[2, 2])
    ax_metrics = fig.add_subplot(grid[3, :])

    completion_image = plot_heatmap(
        ax_completion_scan,
        completion_scanpath,
        title=drift_title(
            "Completion",
            COMPARISON_DISPLAY["scanpath_vs_uniform"],
            completion_scanpath,
            cache["top_layers"]["completion"]["scanpath_vs_uniform"],
        ),
        vmax=completion_vmax,
        contrast_gamma=contrast_gamma,
        top_layers=cache["top_layers"]["completion"]["scanpath_vs_uniform"],
    )
    plot_heatmap(
        ax_completion_workload,
        completion_eeg_workload,
        title=drift_title(
            "Completion",
            COMPARISON_DISPLAY["eeg_workload_vs_uniform"],
            completion_eeg_workload,
            cache["top_layers"]["completion"]["eeg_workload_vs_uniform"],
        ),
        vmax=completion_vmax,
        contrast_gamma=contrast_gamma,
        top_layers=cache["top_layers"]["completion"]["eeg_workload_vs_uniform"],
    )
    plot_heatmap(
        ax_completion_eeg,
        completion_eeg,
        title=drift_title(
            "Completion",
            COMPARISON_DISPLAY["scanpath_eeg_vs_uniform"],
            completion_eeg,
            cache["top_layers"]["completion"]["scanpath_eeg_vs_uniform"],
        ),
        vmax=completion_vmax,
        contrast_gamma=contrast_gamma,
        top_layers=cache["top_layers"]["completion"]["scanpath_eeg_vs_uniform"],
    )
    summarization_image = plot_heatmap(
        ax_summarization_scan,
        summarization_scanpath,
        title=drift_title(
            "Summarization",
            COMPARISON_DISPLAY["scanpath_vs_uniform"],
            summarization_scanpath,
            cache["top_layers"]["summarization"]["scanpath_vs_uniform"],
        ),
        vmax=summarization_vmax,
        contrast_gamma=contrast_gamma,
        top_layers=cache["top_layers"]["summarization"]["scanpath_vs_uniform"],
    )
    plot_heatmap(
        ax_summarization_workload,
        summarization_eeg_workload,
        title=drift_title(
            "Summarization",
            COMPARISON_DISPLAY["eeg_workload_vs_uniform"],
            summarization_eeg_workload,
            cache["top_layers"]["summarization"]["eeg_workload_vs_uniform"],
        ),
        vmax=summarization_vmax,
        contrast_gamma=contrast_gamma,
        top_layers=cache["top_layers"]["summarization"]["eeg_workload_vs_uniform"],
    )
    plot_heatmap(
        ax_summarization_eeg,
        summarization_eeg,
        title=drift_title(
            "Summarization",
            COMPARISON_DISPLAY["scanpath_eeg_vs_uniform"],
            summarization_eeg,
            cache["top_layers"]["summarization"]["scanpath_eeg_vs_uniform"],
        ),
        vmax=summarization_vmax,
        contrast_gamma=contrast_gamma,
        top_layers=cache["top_layers"]["summarization"]["scanpath_eeg_vs_uniform"],
    )
    plot_metric_gain_panel(ax_completion_gain, cache, "completion")
    plot_metric_gain_panel(ax_summarization_gain, cache, "summarization")

    completion_colorbar = fig.colorbar(
        completion_image,
        ax=[ax_completion_scan, ax_completion_workload, ax_completion_eeg],
        shrink=0.88,
    )
    completion_colorbar.set_label(
        f"Completion drift (x{int(cache['drift_scale'])}^-1 of uniform norm, gamma={contrast_gamma:g})"
    )
    summarization_colorbar = fig.colorbar(
        summarization_image,
        ax=[ax_summarization_scan, ax_summarization_workload, ax_summarization_eeg],
        shrink=0.88,
    )
    summarization_colorbar.set_label(
        f"Summarization drift (x{int(cache['drift_scale'])}^-1 of uniform norm, gamma={contrast_gamma:g})"
    )

    metric_rows: list[list[float]] = []
    metric_labels: list[str] = []
    metric_row_groups: list[tuple[int, int]] = []
    for task in ("completion", "summarization"):
        task_start = len(metric_rows)
        for model_name in MODEL_ORDER[task]:
            metric_labels.append(f"{task[:4]} | {DISPLAY_MODEL[model_name]}")
            metric_rows.append(
                [100.0 * cache["metrics"][task][model_name][metric_key] for metric_key, _ in METRIC_SPECS]
            )
        metric_row_groups.append((task_start, len(metric_rows)))

    metric_matrix = np.asarray(metric_rows, dtype=np.float32)
    metric_image = ax_metrics.imshow(metric_matrix, aspect="auto", cmap="YlGn", vmin=0.0, vmax=100.0)
    ax_metrics.set_xticks(np.arange(len(METRIC_SPECS)))
    ax_metrics.set_xticklabels([label for _, label in METRIC_SPECS])
    ax_metrics.set_yticks(np.arange(len(metric_labels)))
    ax_metrics.set_yticklabels(metric_labels)
    ax_metrics.set_title("Corpus-level metrics from saved eval outputs")
    for _, task_end in metric_row_groups[:-1]:
        ax_metrics.axhline(task_end - 0.5, color="black", linewidth=1.2)
    for row_idx in range(metric_matrix.shape[0]):
        for col_idx in range(metric_matrix.shape[1]):
            ax_metrics.text(
                col_idx,
                row_idx,
                f"{metric_matrix[row_idx, col_idx]:.1f}",
                ha="center",
                va="center",
                fontsize=9,
                color="black",
            )

    add_metric_highlights(ax_metrics, metric_matrix, metric_row_groups)
    metrics_colorbar = fig.colorbar(metric_image, ax=ax_metrics, shrink=0.85, pad=0.02)
    metrics_colorbar.set_label("Metric value (%)")

    fig.suptitle(
        "Checkpoint weight drift from uniform training, including EEG-workload training",
        fontsize=16,
        y=0.99,
    )
    fig.text(
        0.5,
        0.015,
        "Metrics: Exact = exact normalized match, SeqSim = SequenceMatcher similarity, "
        "Len<=1.5x = output-length control, Clean = no template/dialogue leakage markers.",
        ha="center",
        fontsize=9,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    cache = maybe_load_cache(args)
    output_path = Path(args.output)
    make_figure(cache, output_path)

    summary = {
        "figure_path": str(output_path),
        "cache_path": args.cache_json,
        "top_layers": cache["top_layers"],
        "metrics": cache["metrics"],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
