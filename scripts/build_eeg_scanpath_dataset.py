from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from collections import defaultdict
from pathlib import Path
from typing import Any

from eeg_qwen.common import ensure_dir, mean, read_jsonl, write_json, write_jsonl

try:
    import javalang
except ImportError:  # pragma: no cover
    javalang = None


DEFAULT_ACTR_ROOT = Path("<USER_HOME>/actr/sft")
DEFAULT_NEURO_JSONL = Path("data/neuro_structural_dataset/train.jsonl")

REGION_TYPES = (
    "function_def",
    "parameter_list",
    "assignment",
    "call",
    "if_branch",
    "loop_header",
    "loop_body",
    "return",
    "comparison",
    "literal_or_constant",
    "container_access",
    "recursion_or_self_call",
)

AST_LABEL_TO_REGION_TYPES = {
    "ClassDeclaration": ("function_def",),
    "InterfaceDeclaration": ("function_def",),
    "EnumDeclaration": ("function_def",),
    "MethodDeclaration": ("function_def", "parameter_list"),
    "ConstructorDeclaration": ("function_def", "parameter_list"),
    "IfStatement": ("if_branch",),
    "ForStatement": ("loop_header",),
    "WhileStatement": ("loop_header",),
    "DoStatement": ("loop_header",),
    "EnhancedForControl": ("loop_header",),
    "ReturnStatement": ("return",),
    "Assignment": ("assignment",),
    "VariableDeclarator": ("assignment",),
    "MethodInvocation": ("call",),
    "BinaryOperation": ("comparison",),
    "ArraySelector": ("container_access",),
    "Literal": ("literal_or_constant",),
}


def default_source_file(task: str, split: str) -> Path:
    return DEFAULT_ACTR_ROOT / f"{task}_{split}_final_scanpath.jsonl"


def compute_region_priors(path: str | Path) -> tuple[dict[str, dict[str, float]], float]:
    rows = read_jsonl(path)
    by_region: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"attention": [], "revisit": [], "cost": []})
    global_costs: list[float] = []
    for row in rows:
        snippet_cost = float(((row.get("neuro_labels") or {}).get("snippet_targets") or {}).get("processing_cost", 0.0))
        global_costs.append(snippet_cost)
        for region in ((row.get("structural_labels") or {}).get("regions") or []):
            region_type = str(region.get("region_type") or "")
            if region_type not in REGION_TYPES:
                continue
            targets = region.get("targets") or {}
            by_region[region_type]["attention"].append(float(targets.get("attention_importance", 0.0)))
            by_region[region_type]["revisit"].append(float(targets.get("revisit_score", 0.0)))
            by_region[region_type]["cost"].append(snippet_cost)
    priors: dict[str, dict[str, float]] = {}
    for region_type in REGION_TYPES:
        buckets = by_region.get(region_type, {"attention": [], "revisit": [], "cost": []})
        priors[region_type] = {
            "attention": mean(buckets["attention"]) if buckets["attention"] else 0.0,
            "revisit": mean(buckets["revisit"]) if buckets["revisit"] else 0.0,
            "cost": mean(buckets["cost"]) if buckets["cost"] else 0.0,
        }
    return priors, mean(global_costs) if global_costs else 0.0


def count_processing_cost_sources(path: str | Path) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in read_jsonl(path):
        source = str((row.get("metadata") or {}).get("processing_cost_source") or "unknown")
        counts[source] += 1
    return dict(sorted(counts.items()))


def extract_java_ast_semantic_types(code: str) -> dict[int, str]:
    line_to_semantic: dict[int, str] = {}
    if javalang is None:
        return line_to_semantic
    try:
        tree = javalang.parse.parse(code)
    except Exception:
        return line_to_semantic

    line_to_type: dict[int, str] = {}
    preferred = {
        "IfStatement",
        "ForStatement",
        "WhileStatement",
        "DoStatement",
        "TryStatement",
        "CatchClause",
        "ReturnStatement",
        "Assignment",
        "MethodInvocation",
        "VariableDeclarator",
        "ClassDeclaration",
        "MethodDeclaration",
        "ConstructorDeclaration",
    }
    for _, node in tree:
        if hasattr(node, "position") and node.position:
            line = int(node.position[0])
            stmt_type = type(node).__name__
            current = line_to_type.get(line)
            if current is None or stmt_type in preferred:
                line_to_type[line] = stmt_type
    return line_to_type


def infer_line_region_types(line: str, line_number: int, ast_types: dict[int, str]) -> list[str]:
    stripped = line.strip()
    if not stripped:
        return []

    matched: list[str] = []
    ast_label = ast_types.get(line_number)
    if ast_label:
        matched.extend(AST_LABEL_TO_REGION_TYPES.get(ast_label, ()))

    if re.search(r"\b(public|private|protected)\b.*\([^;]*\)\s*\{?\s*$", stripped) and not re.search(r"\b(if|for|while|switch)\b", stripped):
        matched.extend(["function_def", "parameter_list"])
    if re.search(r"\bclass\b|\binterface\b|\benum\b", stripped):
        matched.append("function_def")
    if re.search(r"\bif\s*\(", stripped):
        matched.append("if_branch")
    if re.search(r"\b(for|while|do)\b", stripped):
        matched.append("loop_header")
    if re.search(r"\breturn\b", stripped):
        matched.append("return")
    if re.search(r"(?<![=!<>])=(?!=)", stripped):
        matched.append("assignment")
    if re.search(r"(==|!=|<=|>=|<|>)", stripped):
        matched.append("comparison")
    if "[" in stripped and "]" in stripped:
        matched.append("container_access")
    if re.search(r"\b[A-Za-z_][A-Za-z0-9_\.]*\s*\(", stripped) and not re.search(r"\b(if|for|while|switch|catch)\s*\(", stripped):
        matched.append("call")
    if re.search(r"\".*?\"|\'.*?\'|\b\d+\b", stripped):
        matched.append("literal_or_constant")

    deduped: list[str] = []
    for item in matched:
        if item not in deduped:
            deduped.append(item)
    return deduped or ["literal_or_constant"]


def estimate_local_complexity(line: str) -> float:
    stripped = line.strip()
    if not stripped:
        return 0.0
    control = len(re.findall(r"\b(if|else|for|while|switch|case|try|catch|return|throw)\b", stripped))
    calls = len(re.findall(r"\b[A-Za-z_][A-Za-z0-9_\.]*\s*\(", stripped))
    logic = len(re.findall(r"(&&|\|\||==|!=|<=|>=|<|>)", stripped))
    arith = len(re.findall(r"(?<!\+)\+(?!\+)|(?<!-)-(?!-)|\*|/|%", stripped))
    access = stripped.count("[") + stripped.count(".")
    raw = 1.2 * control + 0.8 * calls + 0.8 * logic + 0.5 * arith + 0.3 * access
    return min(1.0, raw / 6.0)


def difficulty_multiplier(label: str | None) -> float:
    normalized = str(label or "").strip().lower()
    if normalized == "easy":
        return 0.9
    if normalized == "medium":
        return 1.0
    if normalized == "hard":
        return 1.15
    return 1.0


def build_eeg_sequences(
    record: dict[str, Any],
    priors: dict[str, dict[str, float]],
    global_cost_prior: float,
    signal_mode: str,
) -> dict[str, Any]:
    code = str(record.get("code") or "")
    code_tokens = list(record.get("code_tokens") or code.splitlines())
    if not code_tokens:
        code_tokens = code.splitlines()
    ast_types = extract_java_ast_semantic_types(code)
    diff_mult = difficulty_multiplier(record.get("flag"))

    salience_scores: list[float] = []
    complexity_scores: list[float] = []
    region_types_by_line: list[list[str]] = []
    attention_scores: list[float] = []
    revisit_scores: list[float] = []
    rank_tie_breakers: list[float] = []

    for index, line in enumerate(code_tokens, start=1):
        region_types = infer_line_region_types(line, index, ast_types)
        region_types_by_line.append(region_types)
        region_attention = mean(priors.get(region_type, {}).get("attention", 0.0) for region_type in region_types)
        region_revisit = mean(priors.get(region_type, {}).get("revisit", 0.0) for region_type in region_types)
        region_cost = mean(priors.get(region_type, {}).get("cost", global_cost_prior) for region_type in region_types)
        local_complexity = estimate_local_complexity(line)

        if signal_mode == "workload_only":
            # Workload-only mode deliberately avoids gaze-derived attention and
            # revisit priors. The stored score is just structural processing
            # cost, optionally EEG-blended if the upstream neuro JSONL was built
            # with preprocess_novice_vs_expert_raw.py --enable-eeg.
            eeg_complexity = min(1.0, max(0.0, region_cost * diff_mult))
            salience = eeg_complexity
            region_attention = 0.0
            region_revisit = 0.0
            rank_tie_breakers.append(local_complexity)
        else:
            eeg_complexity = min(1.0, max(0.0, (0.65 * region_cost + 0.35 * local_complexity) * diff_mult))
            salience = min(1.0, max(0.0, 0.45 * region_attention + 0.15 * region_revisit + 0.40 * eeg_complexity))
            rank_tie_breakers.append(0.0)

        attention_scores.append(region_attention)
        revisit_scores.append(region_revisit)
        complexity_scores.append(eeg_complexity)
        salience_scores.append(salience)

    nonempty_indices = [idx for idx, token in enumerate(code_tokens) if str(token).strip()]
    if not nonempty_indices:
        mask = [0] * len(code_tokens)
        order_sequence = [0] * len(code_tokens)
    else:
        snippet_complexity = mean(complexity_scores[i] for i in nonempty_indices)
        if signal_mode == "workload_only":
            # EEG workload is a coarser snippet/structure signal than gaze
            # fixation, so select fewer lines and treat it as conservative
            # difficulty weighting rather than a precise attention trace.
            coverage_ratio = min(0.40, max(0.12, 0.12 + 0.28 * snippet_complexity))
        else:
            coverage_ratio = min(0.55, max(0.20, 0.20 + 0.35 * snippet_complexity))
        covered_count = max(1, int(math.ceil(len(nonempty_indices) * coverage_ratio)))
        ranked = sorted(
            nonempty_indices,
            key=lambda idx: (salience_scores[idx], rank_tie_breakers[idx], -idx),
            reverse=True,
        )
        selected = ranked[:covered_count]
        mask = [1 if idx in selected else 0 for idx in range(len(code_tokens))]
        order_sequence = [0] * len(code_tokens)
        for order, idx in enumerate(selected, start=1):
            order_sequence[idx] = order

    return {
        "mask": mask,
        "order_sequence": order_sequence,
        "complexity_sequence": [round(score, 6) for score in complexity_scores],
        "eeg_attention_sequence": [round(score, 6) for score in attention_scores],
        "eeg_revisit_sequence": [round(score, 6) for score in revisit_scores],
        "eeg_salience_sequence": [round(score, 6) for score in salience_scores],
        "eeg_region_types": region_types_by_line,
        "eeg_snippet_complexity": round(mean(complexity_scores), 6),
        "eeg_signal_mode": signal_mode,
    }


def convert_records(
    rows: list[dict[str, Any]],
    priors: dict[str, dict[str, float]],
    global_cost_prior: float,
    signal_mode: str,
) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for row in rows:
        code_tokens = list(row.get("code_tokens") or [])
        if not code_tokens:
            continue
        eeg_fields = build_eeg_sequences(row, priors, global_cost_prior, signal_mode=signal_mode)
        record = dict(row)
        record["scanpath_mask"] = list(row.get("mask") or [0] * len(code_tokens))
        record["scanpath_order_sequence"] = list(row.get("order_sequence") or [0] * len(code_tokens))
        record["scanpath_complexity_sequence"] = list(row.get("complexity_sequence") or [0.0] * len(code_tokens))
        record.update(eeg_fields)
        converted.append(record)
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ACT-R-style EEG-weighted scanpath datasets from the Java completion/summarization benchmark.")
    parser.add_argument("--task", choices=["completion", "summarization"], default="completion")
    parser.add_argument("--source-train-jsonl", help="Source ACT-R scanpath JSONL train file.")
    parser.add_argument("--source-valid-jsonl", help="Source ACT-R scanpath JSONL validation file.")
    parser.add_argument("--neuro-jsonl", default=str(DEFAULT_NEURO_JSONL))
    parser.add_argument("--output-dir", default="data/actr_scanpath_eeg")
    parser.add_argument(
        "--signal-mode",
        choices=["salience", "workload_only"],
        default="salience",
        help=(
            "salience uses gaze attention + revisit + processing cost. "
            "workload_only uses only structural processing_cost and avoids gaze attention/revisit."
        ),
    )
    parser.add_argument(
        "--dataset-tag",
        default="eeg_scanpath",
        help="Filename tag, e.g. eeg_scanpath or eeg_workload_scanpath.",
    )
    args = parser.parse_args()

    train_path = Path(args.source_train_jsonl) if args.source_train_jsonl else default_source_file(args.task, "train")
    valid_path = Path(args.source_valid_jsonl) if args.source_valid_jsonl else default_source_file(args.task, "valid")
    output_dir = ensure_dir(args.output_dir)

    priors, global_cost_prior = compute_region_priors(args.neuro_jsonl)
    processing_cost_sources = count_processing_cost_sources(args.neuro_jsonl)
    train_rows = convert_records(read_jsonl(train_path), priors, global_cost_prior, signal_mode=args.signal_mode)
    valid_rows = convert_records(read_jsonl(valid_path), priors, global_cost_prior, signal_mode=args.signal_mode)

    train_out = output_dir / f"{args.task}_train_final_{args.dataset_tag}.jsonl"
    valid_out = output_dir / f"{args.task}_valid_final_{args.dataset_tag}.jsonl"
    write_jsonl(train_out, train_rows)
    write_jsonl(valid_out, valid_rows)

    summary = {
        "task": args.task,
        "source_train_jsonl": str(train_path),
        "source_valid_jsonl": str(valid_path),
        "neuro_jsonl": str(args.neuro_jsonl),
        "signal_mode": args.signal_mode,
        "dataset_tag": args.dataset_tag,
        "processing_cost_sources": processing_cost_sources,
        "train_count": len(train_rows),
        "valid_count": len(valid_rows),
        "global_cost_prior": global_cost_prior,
        "mean_train_complexity": mean(row.get("eeg_snippet_complexity", 0.0) for row in train_rows),
        "mean_valid_complexity": mean(row.get("eeg_snippet_complexity", 0.0) for row in valid_rows),
    }
    write_json(output_dir / f"{args.task}_{args.dataset_tag}_summary.json", summary)
    print(f"Wrote {len(train_rows)} train rows to {train_out}")
    print(f"Wrote {len(valid_rows)} valid rows to {valid_out}")


if __name__ == "__main__":
    main()
