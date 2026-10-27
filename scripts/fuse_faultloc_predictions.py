from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from eeg_qwen.common import mean, safe_div, write_json, write_jsonl


def read_jsonl(path: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def unique_keep_order(values: list[int], limit: int = 3) -> list[int]:
    output: list[int] = []
    for value in values:
        if value in output:
            continue
        output.append(value)
        if len(output) >= limit:
            break
    return output


def fuse_lines(scanpath_lines: list[int], eeg_lines: list[int], mode: str) -> list[int]:
    if mode == "scan_top1_eeg_singleton_second":
        if len(eeg_lines) == 1:
            return unique_keep_order(scanpath_lines[:1] + eeg_lines + scanpath_lines[1:])
        return unique_keep_order(scanpath_lines)
    if mode == "scan_top1_plus_eeg_singleton":
        if len(eeg_lines) == 1:
            return unique_keep_order(scanpath_lines[:1] + eeg_lines)
        return unique_keep_order(scanpath_lines[:1])
    if mode == "scan_first_append_eeg_if_singleton":
        if len(eeg_lines) == 1:
            return unique_keep_order(scanpath_lines + eeg_lines)
        return unique_keep_order(scanpath_lines)
    if mode == "scan_then_eeg":
        return unique_keep_order(scanpath_lines + eeg_lines)
    raise ValueError(f"Unknown fusion mode: {mode}")


def score_fault_localization(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        "task": "fault_localization",
        "num_examples": count,
        "exact_match_rate": safe_div(sum(int(row["exact_match"]) for row in rows), count),
        "hit_at_1": safe_div(sum(int(row["hit_at_1"]) for row in rows), count),
        "hit_at_3": safe_div(sum(int(row["hit_at_3"]) for row in rows), count),
        "mrr": mean(row["mrr"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fuse scanpath and EEG fault-localization predictions.")
    parser.add_argument("--scanpath-predictions", required=True, help="Path to scanpath predictions.jsonl")
    parser.add_argument("--eeg-predictions", required=True, help="Path to EEG predictions.jsonl")
    parser.add_argument("--output-dir", required=True, help="Directory for fused predictions and summary")
    parser.add_argument(
        "--mode",
        default="scan_top1_plus_eeg_singleton",
        choices=[
            "scan_top1_eeg_singleton_second",
            "scan_top1_plus_eeg_singleton",
            "scan_first_append_eeg_if_singleton",
            "scan_then_eeg",
        ],
        help="Fusion rule.",
    )
    args = parser.parse_args()

    scan_rows = read_jsonl(args.scanpath_predictions)
    eeg_rows = read_jsonl(args.eeg_predictions)
    if len(scan_rows) != len(eeg_rows):
        raise ValueError("Prediction files have different lengths.")

    fused_rows: list[dict[str, Any]] = []
    for scan_row, eeg_row in zip(scan_rows, eeg_rows):
        if scan_row["id"] != eeg_row["id"]:
            raise ValueError(f"Mismatched IDs: {scan_row['id']} vs {eeg_row['id']}")
        gold_lines = list(scan_row.get("gold_lines") or [])
        predicted_lines = fuse_lines(
            list(scan_row.get("predicted_lines") or []),
            list(eeg_row.get("predicted_lines") or []),
            args.mode,
        )
        hit_at_1 = bool(predicted_lines and predicted_lines[0] in gold_lines)
        hit_at_3 = any(line in gold_lines for line in predicted_lines[:3])
        reciprocal_rank = 0.0
        for rank, line in enumerate(predicted_lines, start=1):
            if line in gold_lines:
                reciprocal_rank = 1.0 / rank
                break
        fused = dict(scan_row)
        fused["fusion_mode"] = args.mode
        fused["scanpath_predicted_lines"] = list(scan_row.get("predicted_lines") or [])
        fused["eeg_predicted_lines"] = list(eeg_row.get("predicted_lines") or [])
        fused["predicted_lines"] = predicted_lines
        fused["generated_output"] = ", ".join(str(x) for x in predicted_lines)
        fused["exact_match"] = set(predicted_lines) == set(gold_lines)
        fused["hit_at_1"] = hit_at_1
        fused["hit_at_3"] = hit_at_3
        fused["mrr"] = reciprocal_rank
        fused_rows.append(fused)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"

    write_jsonl(str(predictions_path), fused_rows)
    summary = score_fault_localization(fused_rows)
    summary["fusion_mode"] = args.mode
    summary["sources"] = {
        "scanpath": args.scanpath_predictions,
        "eeg": args.eeg_predictions,
    }
    write_json(str(summary_path), summary)
    print(f"Wrote fused predictions to {predictions_path}")
    print(f"Wrote fused summary to {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
