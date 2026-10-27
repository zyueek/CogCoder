from __future__ import annotations

import argparse
import os
import importlib.util
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from eeg_qwen.common import ensure_dir, mean, read_jsonl, write_json, write_jsonl

from build_eeg_scanpath_dataset import (
    build_eeg_sequences,
    compute_region_priors,
    count_processing_cost_sources,
    estimate_local_complexity,
    extract_java_ast_semantic_types,
    infer_line_region_types,
)


ACTR_SFT_ROOT = Path(os.environ.get("ACTR_SFT_ROOT", "data/actr_sft"))
if str(ACTR_SFT_ROOT) not in sys.path:
    sys.path.insert(0, str(ACTR_SFT_ROOT))

if importlib.util.find_spec("javalang") is not None:
    try:
        from generate_scanpath_training_file_fixed_complexity import generate_scanpath  # type: ignore  # noqa: E402

        ACTR_SIM_AVAILABLE = True
    except Exception:
        generate_scanpath = None  # type: ignore[assignment]
        ACTR_SIM_AVAILABLE = False
else:
    generate_scanpath = None  # type: ignore[assignment]
    ACTR_SIM_AVAILABLE = False


DEFAULT_NEURO_JSONL = Path("data/neuro_structural_dataset/train.jsonl")


def normalize_label(raw: Any) -> str:
    if isinstance(raw, bool):
        return "VULNERABLE" if raw else "SAFE"
    if isinstance(raw, (int, float)):
        return "VULNERABLE" if int(raw) != 0 else "SAFE"
    normalized = str(raw or "").strip().lower()
    if normalized in {"1", "true", "yes", "y", "vulnerable", "vuln", "buggy", "unsafe", "positive"}:
        return "VULNERABLE"
    if normalized in {"0", "false", "no", "n", "safe", "clean", "benign", "negative", "not_vulnerable"}:
        return "SAFE"
    raise ValueError(f"Unsupported vulnerability label: {raw!r}")


def normalize_fault_lines(raw: Any) -> list[int]:
    if raw is None:
        return []
    if isinstance(raw, int):
        return [raw] if raw > 0 else []
    if isinstance(raw, list):
        if raw and all(isinstance(item, (int, float)) for item in raw):
            return sorted({int(item) for item in raw if int(item) > 0})
        if raw and all(isinstance(item, bool) or isinstance(item, (int, float)) for item in raw):
            return sorted({index + 1 for index, value in enumerate(raw) if int(value) > 0})
    text = str(raw).strip()
    if not text:
        return []
    tokens = []
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            left, right = part.split("-", 1)
            try:
                start = int(left)
                end = int(right)
            except ValueError:
                continue
            if start > 0 and end >= start:
                tokens.extend(range(start, end + 1))
            continue
        try:
            tokens.append(int(part))
        except ValueError:
            continue
    return sorted({value for value in tokens if value > 0})


def infer_task_context(record: dict[str, Any], task: str) -> str:
    candidates = [
        record.get("task_context"),
        record.get("bug_report"),
        record.get("issue"),
        record.get("description"),
        record.get("commit_message"),
        record.get("message"),
    ]
    if task == "vulnerability_detection":
        cwe = record.get("cwe") or record.get("cwe_id")
        if cwe:
            candidates.append(f"CWE hint: {cwe}")
    for candidate in candidates:
        if candidate is None:
            continue
        text = str(candidate).strip()
        if text:
            return text
    return ""


def choose_content(record: dict[str, Any], task: str) -> tuple[str, dict[str, Any]]:
    metadata: dict[str, Any] = {}
    if task == "vulnerability_detection":
        label_source = None
        for key in ("label", "is_vulnerable", "vulnerable", "target"):
            if key in record:
                label_source = record[key]
                break
        if label_source is None:
            raise ValueError("Missing vulnerability label. Expected one of: label, is_vulnerable, vulnerable, target.")
        verdict = normalize_label(label_source)
        metadata["raw_label"] = label_source
        return verdict, metadata

    for key in ("fault_lines", "bug_lines", "ground_truth_lines", "suspicious_lines", "oracle_lines", "target_lines"):
        if key in record:
            lines = normalize_fault_lines(record[key])
            if lines:
                metadata["gold_lines"] = lines
                return ", ".join(str(line) for line in lines), metadata
    if "line_labels" in record:
        lines = normalize_fault_lines(record["line_labels"])
        if lines:
            metadata["gold_lines"] = lines
            return ", ".join(str(line) for line in lines), metadata
    raise ValueError("Missing fault lines. Expected one of: fault_lines, bug_lines, ground_truth_lines, suspicious_lines, oracle_lines, target_lines, line_labels.")


def maybe_use_existing_scanpath(record: dict[str, Any], line_count: int) -> tuple[list[int], list[int], list[float]] | None:
    mask = record.get("scanpath_mask")
    order = record.get("scanpath_order_sequence")
    complexity = record.get("scanpath_complexity_sequence")
    if mask is None or order is None:
        mask = record.get("mask")
        order = record.get("order_sequence")
        complexity = record.get("complexity_sequence")
    if mask is None or order is None:
        return None
    mask_list = [int(value) for value in list(mask)]
    order_list = [int(value) for value in list(order)]
    complexity_list = [float(value) for value in list(complexity or [0.0] * line_count)]
    if len(mask_list) != line_count or len(order_list) != line_count or len(complexity_list) != line_count:
        return None
    return mask_list, order_list, complexity_list


def normalize_complexities(complexity_values: list[float]) -> list[float]:
    if not complexity_values:
        return []
    max_value = max(complexity_values)
    if max_value <= 1.0:
        return [max(0.0, float(value)) for value in complexity_values]
    return [max(0.0, float(value)) / max_value for value in complexity_values]


def fallback_scanpath_features(code: str) -> tuple[list[int], list[int], list[float]]:
    lines = code.splitlines()
    line_count = len(lines)
    if line_count == 0:
        return [], [], []

    ast_types = extract_java_ast_semantic_types(code)
    line_scores: list[float] = []
    for index, line in enumerate(lines, start=1):
        if not line.strip():
            line_scores.append(0.0)
            continue
        region_types = infer_line_region_types(line, index, ast_types)
        local_complexity = estimate_local_complexity(line)
        structure_bonus = 0.0
        if "function_def" in region_types:
            structure_bonus += 0.35
        if "if_branch" in region_types or "loop_header" in region_types:
            structure_bonus += 0.25
        if "call" in region_types or "assignment" in region_types:
            structure_bonus += 0.10
        score = min(1.0, local_complexity + structure_bonus)
        line_scores.append(score)

    ranked = sorted(range(line_count), key=lambda idx: (line_scores[idx], -idx), reverse=True)
    selected_count = min(line_count, max(1, math.ceil(line_count * 0.35)))
    selected = ranked[:selected_count]
    if 0 not in selected:
        selected = [0] + selected[:-1] if selected_count == len(selected) else [0] + selected

    mask = [1 if idx in selected else 0 for idx in range(line_count)]
    order_sequence = [0] * line_count
    for order, idx in enumerate(selected, start=1):
        order_sequence[idx] = order
    return mask, order_sequence, line_scores


def simulate_scanpath_features(code: str, sim_id: int, seed: int) -> tuple[list[int], list[int], list[float]]:
    lines = code.splitlines()
    line_count = len(lines)
    if line_count == 0:
        return [], [], []

    if not ACTR_SIM_AVAILABLE or generate_scanpath is None:
        return fallback_scanpath_features(code)

    simulated_lines, complexity_map = generate_scanpath(code, sim_id=sim_id, seed=seed)
    ordered_unique: list[int] = []
    seen: set[int] = set()
    for line_number in simulated_lines:
        try:
            line_idx = int(line_number)
        except Exception:
            continue
        if 1 <= line_idx <= line_count and line_idx not in seen:
            ordered_unique.append(line_idx)
            seen.add(line_idx)

    if not ordered_unique:
        fallback_count = min(max(1, math.ceil(line_count * 0.35)), line_count)
        ordered_unique = list(range(1, fallback_count + 1))

    mask = [1 if (index + 1) in seen else 0 for index in range(line_count)]
    order_sequence = [0] * line_count
    for order, line_idx in enumerate(ordered_unique, start=1):
        order_sequence[line_idx - 1] = order

    raw_complexities = [float(complexity_map.get(index + 1, 0.0)) for index in range(line_count)]
    complexity_sequence = normalize_complexities(raw_complexities)
    return mask, order_sequence, complexity_sequence


def convert_record(
    record: dict[str, Any],
    *,
    task: str,
    sim_id: int,
    seed: int,
    priors: dict[str, dict[str, float]],
    global_cost_prior: float,
    signal_mode: str,
) -> dict[str, Any] | None:
    code = str(record.get("code") or "")
    code_tokens = code.splitlines()
    if not code_tokens:
        return None

    content, metadata = choose_content(record, task)
    task_context = infer_task_context(record, task)

    scanpath = maybe_use_existing_scanpath(record, len(code_tokens))
    if scanpath is None:
        scanpath_mask, scanpath_order_sequence, scanpath_complexity_sequence = simulate_scanpath_features(code, sim_id, seed)
    else:
        scanpath_mask, scanpath_order_sequence, scanpath_complexity_sequence = scanpath

    base_record = {
        "code": code,
        "content": content,
        "task_context": task_context,
        "flag": record.get("flag", ""),
        "code_content_hash": record.get("code_content_hash") or record.get("id") or f"{task}_{sim_id}",
        "code_tokens": code_tokens,
        "code_occurrence": [0] * len(code_tokens),
        "line_numbers": list(range(1, len(code_tokens) + 1)),
        "ngrams": [None] * len(code_tokens),
        "ngram_indices": [0] * len(code_tokens),
        "semantic_token_sequence": [1 if line.strip() else 0 for line in code_tokens],
        "scanpath_mask": scanpath_mask,
        "scanpath_order_sequence": scanpath_order_sequence,
        "scanpath_complexity_sequence": scanpath_complexity_sequence,
    }

    eeg_seed_record = dict(base_record)
    eeg_seed_record["mask"] = scanpath_mask
    eeg_seed_record["order_sequence"] = scanpath_order_sequence
    eeg_seed_record["complexity_sequence"] = scanpath_complexity_sequence
    eeg_fields = build_eeg_sequences(eeg_seed_record, priors, global_cost_prior, signal_mode=signal_mode)

    output = dict(base_record)
    output.update(eeg_fields)
    output["mask"] = list(eeg_fields["mask"])
    output["order_sequence"] = list(eeg_fields["order_sequence"])
    output["complexity_sequence"] = list(eeg_fields["complexity_sequence"])
    output["metadata"] = metadata
    return output


def convert_records(
    rows: list[dict[str, Any]],
    *,
    task: str,
    seed: int,
    priors: dict[str, dict[str, float]],
    global_cost_prior: float,
    signal_mode: str,
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        try:
            item = convert_record(
                row,
                task=task,
                sim_id=index,
                seed=seed + index,
                priors=priors,
                global_cost_prior=global_cost_prior,
                signal_mode=signal_mode,
            )
        except Exception:
            continue
        if item is not None:
            converted.append(item)
    return converted


def summarize_rows(rows: list[dict[str, Any]], task: str) -> dict[str, Any]:
    summary = {
        "count": len(rows),
        "mean_scanpath_coverage": mean(sum(record.get("scanpath_mask", [])) / max(1, len(record.get("scanpath_mask", []))) for record in rows),
        "mean_eeg_coverage": mean(sum(record.get("mask", [])) / max(1, len(record.get("mask", []))) for record in rows),
        "mean_eeg_snippet_complexity": mean(record.get("eeg_snippet_complexity", 0.0) for record in rows),
    }
    if task == "vulnerability_detection":
        labels = Counter(record.get("content", "") for record in rows)
        summary["label_distribution"] = dict(sorted(labels.items()))
    else:
        summary["mean_fault_line_count"] = mean(len((record.get("metadata") or {}).get("gold_lines", [])) for record in rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ACT-R-style scanpath+EEG datasets for vulnerability detection or fault localization.")
    parser.add_argument("--task", choices=["vulnerability_detection", "fault_localization"], required=True)
    parser.add_argument("--source-train-jsonl", required=True)
    parser.add_argument("--source-valid-jsonl", required=True)
    parser.add_argument("--source-test-jsonl")
    parser.add_argument("--neuro-jsonl", default=str(DEFAULT_NEURO_JSONL))
    parser.add_argument("--output-dir", default="data/actr_scanpath_eeg")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--signal-mode",
        choices=["salience", "workload_only"],
        default="salience",
        help="salience uses gaze attention + revisit + processing cost. workload_only uses only structural processing cost.",
    )
    parser.add_argument("--dataset-tag", default="eeg_scanpath")
    args = parser.parse_args()

    priors, global_cost_prior = compute_region_priors(args.neuro_jsonl)
    processing_cost_sources = count_processing_cost_sources(args.neuro_jsonl)
    output_dir = ensure_dir(args.output_dir)

    train_rows = convert_records(
        read_jsonl(args.source_train_jsonl),
        task=args.task,
        seed=args.seed,
        priors=priors,
        global_cost_prior=global_cost_prior,
        signal_mode=args.signal_mode,
    )
    valid_rows = convert_records(
        read_jsonl(args.source_valid_jsonl),
        task=args.task,
        seed=args.seed + 100_000,
        priors=priors,
        global_cost_prior=global_cost_prior,
        signal_mode=args.signal_mode,
    )
    test_rows: list[dict[str, Any]] = []
    if args.source_test_jsonl:
        test_rows = convert_records(
            read_jsonl(args.source_test_jsonl),
            task=args.task,
            seed=args.seed + 200_000,
            priors=priors,
            global_cost_prior=global_cost_prior,
            signal_mode=args.signal_mode,
        )

    train_out = output_dir / f"{args.task}_train_final_{args.dataset_tag}.jsonl"
    valid_out = output_dir / f"{args.task}_valid_final_{args.dataset_tag}.jsonl"
    test_out = output_dir / f"{args.task}_test_final_{args.dataset_tag}.jsonl"
    write_jsonl(train_out, train_rows)
    write_jsonl(valid_out, valid_rows)
    if args.source_test_jsonl:
        write_jsonl(test_out, test_rows)

    summary = {
        "task": args.task,
        "source_train_jsonl": str(args.source_train_jsonl),
        "source_valid_jsonl": str(args.source_valid_jsonl),
        "source_test_jsonl": str(args.source_test_jsonl) if args.source_test_jsonl else None,
        "neuro_jsonl": str(args.neuro_jsonl),
        "signal_mode": args.signal_mode,
        "dataset_tag": args.dataset_tag,
        "scanpath_source": "actr_simulator" if ACTR_SIM_AVAILABLE else "heuristic_fallback",
        "processing_cost_sources": processing_cost_sources,
        "train": summarize_rows(train_rows, args.task),
        "valid": summarize_rows(valid_rows, args.task),
        "test": summarize_rows(test_rows, args.task) if args.source_test_jsonl else None,
        "global_cost_prior": global_cost_prior,
    }
    write_json(output_dir / f"{args.task}_{args.dataset_tag}_summary.json", summary)
    print(f"Wrote {len(train_rows)} train rows to {train_out}")
    print(f"Wrote {len(valid_rows)} valid rows to {valid_out}")
    if args.source_test_jsonl:
        print(f"Wrote {len(test_rows)} test rows to {test_out}")


if __name__ == "__main__":
    main()
