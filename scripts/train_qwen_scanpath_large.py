from __future__ import annotations

import argparse
import importlib.util
import inspect
import logging
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from eeg_qwen.common import mean, read_jsonl


TASK_PROMPTS = {
    "translation": {
        "instruction": "Translate the following Java code to C#.",
        "input_header": "### Java Code:",
        "output_header": "### C# Code:",
    },
    "summarization": {
        "instruction": "Summarize the following Java code.",
        "input_header": "### Java Code:",
        "output_header": "### Summary:",
    },
    "completion": {
        "instruction": "Complete the following Java code.",
        "input_header": "### Java Code:",
        "output_header": "### Completion:",
    },
    "vulnerability_detection": {
        "instruction": "Determine whether the following C/C++ code is vulnerable. Answer with exactly one label: VULNERABLE or SAFE.",
        "input_header": "### C/C++ Code:",
        "context_header": "### Context:",
        "output_header": "### Verdict:",
    },
    "fault_localization": {
        "instruction": "Identify the suspicious faulty line numbers in the following numbered Java code. Answer with a comma-separated list of the numbered 1-based lines only.",
        "input_header": "### Java Code:",
        "context_header": "### Context:",
        "output_header": "### Suspicious Lines:",
    },
}

DEFAULT_QWEN35_BASE = os.environ.get("QWEN35_BASE_PATH", "Qwen/Qwen3.5-4B")
DEFAULT_QWEN35_BASE_ID = "Qwen/Qwen3.5-4B"
DEFAULT_QWEN35_INSTRUCT_ID = "Qwen/Qwen3.5-4B-Instruct"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

SCANPATH_EEG_MODULATION = 0.30
SCANPATH_EEG_RESCUE_THRESHOLD = 0.65
SCANPATH_EEG_RESCUE_WEIGHT = 0.34
SCANPATH_EEG_SNIPPET_SCALE = 0.08
SCANPATH_EEG_OVERLAP_BOOST = 0.18
SCANPATH_WEAK_TOP_K = 1


def default_model_path(model_choice: str) -> str:
    if model_choice == "instruction":
        return DEFAULT_QWEN35_INSTRUCT_ID
    local_base = Path(DEFAULT_QWEN35_BASE)
    if local_base.exists():
        return str(local_base)
    return DEFAULT_QWEN35_BASE_ID


def _pick_optimizer_name() -> str:
    requested = os.environ.get("TRAIN_OPTIM")
    if requested:
        return requested
    try:
        import bitsandbytes  # noqa: F401

        return "adamw_bnb_8bit"
    except Exception:
        return "adafactor"


def _pick_report_to() -> str:
    requested = str(os.environ.get("TRAIN_REPORT_TO", "") or "").strip()
    if requested:
        return requested
    if importlib.util.find_spec("tensorboard") or importlib.util.find_spec("tensorboardX"):
        return "tensorboard"
    logging.warning("TensorBoard is not installed; falling back to report_to='none'.")
    return "none"


def _pick_attn_implementation() -> str:
    try:
        base = torch.__version__.split("+", 1)[0]
        parts = [int(p) for p in base.split(".")[:3]]
        torch_ver = tuple(parts + [0] * (3 - len(parts)))
    except Exception:
        torch_ver = (0, 0, 0)
    has_sdpa = hasattr(torch.nn.functional, "scaled_dot_product_attention")
    if has_sdpa and torch_ver >= (2, 1, 1):
        return "sdpa"
    return "eager"


def _is_bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    if not (hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()):
        return False
    try:
        x = torch.zeros((2, 2), device="cuda", dtype=torch.bfloat16)
        torch.triu(x, diagonal=1)
        return True
    except Exception:
        return False


def _set_reproducible_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _is_distributed_training() -> bool:
    try:
        return int(os.environ.get("WORLD_SIZE", "1")) > 1
    except ValueError:
        return False


def _log_cuda_memory(stage: str) -> None:
    if not torch.cuda.is_available():
        return
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    logging.info(f"[mem] stage={stage} CUDA_VISIBLE_DEVICES={visible if visible else '(unset)'}")
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        alloc = torch.cuda.memory_allocated(index)
        reserved = torch.cuda.memory_reserved(index)
        logging.info(
            f"[mem] gpu={index} free={free/1024**3:.2f}GiB total={total/1024**3:.2f}GiB "
            f"allocated={alloc/1024**3:.2f}GiB reserved={reserved/1024**3:.2f}GiB"
        )


def build_compatible_training_arguments(**kwargs) -> TrainingArguments:
    signature = inspect.signature(TrainingArguments.__init__)
    supported = set(signature.parameters)

    if "eval_strategy" not in supported and "evaluation_strategy" in supported and "eval_strategy" in kwargs:
        kwargs["evaluation_strategy"] = kwargs.pop("eval_strategy")

    if "overwrite_output_dir" not in supported:
        kwargs.pop("overwrite_output_dir", None)
    if "adafactor" not in supported:
        kwargs.pop("adafactor", None)

    filtered_kwargs = {key: value for key, value in kwargs.items() if key in supported}
    return TrainingArguments(**filtered_kwargs)


def format_plain_prompt(task: str, code: str, context: str | None = None) -> str:
    cfg = TASK_PROMPTS[task]
    if task == "fault_localization":
        numbered_lines = [f"{idx}: {line}" for idx, line in enumerate(str(code).splitlines(), start=1)]
        code = "\n".join(numbered_lines)
    prompt = (
        f"### Instruction:\n{cfg['instruction']}\n\n"
    )
    if context and str(context).strip():
        context_header = cfg.get("context_header", "### Context:")
        prompt += f"{context_header}\n{str(context).strip()}\n\n"
    prompt += (
        f"{cfg['input_header']}\n{code}\n\n"
        f"{cfg['output_header']}\n"
    )
    return prompt


def format_prompt(
    task: str,
    code: str,
    context: str | None = None,
    *,
    tokenizer: AutoTokenizer | None = None,
    prompt_format: str = "plain",
) -> str:
    prompt = format_plain_prompt(task, code, context=context)
    if prompt_format != "chat":
        return prompt
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt.strip()}],
        tokenize=False,
        add_generation_prompt=True,
    )


def _token_count(tokenizer: AutoTokenizer, text: str) -> int:
    return len(tokenizer(text, add_special_tokens=True)["input_ids"])


def _extract_focus_line_numbers(context: str | None) -> list[int]:
    if not context:
        return []
    numbers: list[int] = []
    for match in re.finditer(r"(?:inspect|lines?|line)[^.\n:]*?((?:\d+\s*,?\s*){1,12})", str(context), flags=re.IGNORECASE):
        numbers.extend(int(value) for value in re.findall(r"\d+", match.group(1)))
    deduped: list[int] = []
    for number in numbers:
        if number > 0 and number not in deduped:
            deduped.append(number)
    return deduped


def _merge_ranges(ranges: list[tuple[int, int]], line_count: int) -> list[tuple[int, int]]:
    normalized = [
        (max(0, start), min(line_count, end))
        for start, end in ranges
        if end > start
    ]
    if not normalized:
        return []
    normalized.sort()
    merged = [normalized[0]]
    for start, end in normalized[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def _compact_code_by_lines(
    code: str,
    *,
    focus_lines: list[int],
    head_keep: int,
    tail_keep: int,
    focus_window: int,
) -> str:
    lines = str(code).splitlines()
    if not lines:
        return str(code)
    ranges: list[tuple[int, int]] = []
    if head_keep > 0:
        ranges.append((0, head_keep))
    if tail_keep > 0:
        ranges.append((len(lines) - tail_keep, len(lines)))
    for line_number in focus_lines:
        idx = line_number - 1
        ranges.append((idx - focus_window, idx + focus_window + 1))
    merged = _merge_ranges(ranges, len(lines))
    if not merged:
        return "\n".join(lines[: max(1, min(4, len(lines)))])

    output: list[str] = []
    cursor = 0
    for start, end in merged:
        if start > cursor:
            output.append("/* ... omitted unchanged lines ... */")
        output.extend(lines[start:end])
        cursor = end
    if cursor < len(lines):
        output.append("/* ... omitted unchanged lines ... */")
    return "\n".join(output)


def format_prompt_for_budget(
    task: str,
    code: str,
    *,
    context: str | None = None,
    tokenizer: AutoTokenizer | None = None,
    max_length: int | None = None,
    target_text: str = "",
    prompt_format: str = "plain",
) -> str:
    """Truncate only the code body so the instruction, context, answer header, and target survive."""
    prompt = format_prompt(task, code, context=context, tokenizer=tokenizer, prompt_format=prompt_format)
    if tokenizer is None or not max_length or max_length <= 0:
        return prompt
    if _token_count(tokenizer, prompt + target_text) <= max_length:
        return prompt

    focus_lines = _extract_focus_line_numbers(context)
    configs = [
        (24, 24, 2),
        (16, 16, 1),
        (10, 10, 1),
        (6, 6, 0),
        (3, 3, 0),
        (1, 1, 0),
    ]
    for head_keep, tail_keep, focus_window in configs:
        compact_code = _compact_code_by_lines(
            code,
            focus_lines=focus_lines,
            head_keep=head_keep,
            tail_keep=tail_keep,
            focus_window=focus_window,
        )
        candidate = format_prompt(
            task,
            compact_code,
            context=context,
            tokenizer=tokenizer,
            prompt_format=prompt_format,
        )
        if _token_count(tokenizer, candidate + target_text) <= max_length:
            return candidate

    marker = "/* code omitted because the function exceeded the context budget */"
    candidate = format_prompt(task, marker, context=context, tokenizer=tokenizer, prompt_format=prompt_format)
    if _token_count(tokenizer, candidate + target_text) <= max_length:
        return candidate

    # Last-resort fallback: preserve the output header and target even if the context is very large.
    return format_prompt(task, "", context=None, tokenizer=tokenizer, prompt_format=prompt_format)


def _normalize_line_numbers(example: dict[str, Any], line_count: int) -> list[int]:
    line_numbers = list(example.get("line_numbers") or [])
    if len(line_numbers) != line_count:
        line_numbers = list(range(1, line_count + 1))
    normalized: list[int] = []
    for idx, value in enumerate(line_numbers, start=1):
        try:
            normalized.append(int(value))
        except Exception:
            normalized.append(idx)
    return normalized


def _top_scanpath_lines(example: dict[str, Any], line_count: int, top_k: int = 3) -> list[int]:
    mask = safe_list(example, "scanpath_mask", line_count, 0.0)
    order = safe_list(example, "scanpath_order_sequence", line_count, 0.0)
    line_numbers = _normalize_line_numbers(example, line_count)
    ranked = [
        (order[idx] if order[idx] > 0.0 else 10_000.0, idx)
        for idx in range(line_count)
        if mask[idx] > 0.0
    ]
    ranked.sort()
    return [line_numbers[idx] for _, idx in ranked[:top_k]]


def _top_scanpath_weak_lines(example: dict[str, Any], line_count: int, top_k: int = SCANPATH_WEAK_TOP_K) -> list[int]:
    # Deterministic degradation ablation:
    # keep only the strongest scanpath line and drop the broader ranked context.
    return _top_scanpath_lines(example, line_count, top_k=top_k)


def _top_eeg_lines(example: dict[str, Any], line_count: int, top_k: int = 3) -> list[int]:
    mask = safe_list(example, "mask", line_count, 0.0)
    line_numbers = _normalize_line_numbers(example, line_count)
    priority_meta = build_eeg_priority(example, line_count)
    priority = list(priority_meta["priority"])
    ranked = sorted(
        (
            (priority[idx], idx)
            for idx in range(line_count)
            if mask[idx] > 0.0 or priority[idx] > 0.0
        ),
        reverse=True,
    )
    return [line_numbers[idx] for _, idx in ranked[:top_k]]


def _top_scanpath_eeg_lines(example: dict[str, Any], line_count: int, top_k: int = 3) -> list[int]:
    scanpath_mask = safe_list(example, "scanpath_mask", line_count, 0.0)
    scanpath_order = safe_list(example, "scanpath_order_sequence", line_count, 0.0)
    eeg_mask = safe_list(example, "mask", line_count, 0.0)
    line_numbers = _normalize_line_numbers(example, line_count)
    priority_meta = build_eeg_priority(example, line_count)
    priority = list(priority_meta["priority"])

    def score(idx: int) -> float:
        overlap = 1.0 if scanpath_mask[idx] > 0.0 and eeg_mask[idx] > 0.0 else 0.0
        scan_bonus = 1.0 / max(1.0, scanpath_order[idx]) if scanpath_order[idx] > 0.0 else 0.0
        eeg_bonus = 0.35 if eeg_mask[idx] > 0.0 else 0.0
        return 1.25 * overlap + 0.65 * priority[idx] + 0.25 * scan_bonus + eeg_bonus

    ranked = sorted(((score(idx), idx) for idx in range(line_count)), reverse=True)
    return [line_numbers[idx] for _, idx in ranked[:top_k]]


def _focus_region_summary(example: dict[str, Any], lines: list[int], limit: int = 3) -> str:
    if not lines:
        return ""
    line_numbers = _normalize_line_numbers(example, len(list(example.get("code_tokens") or [])))
    index_by_line = {line_no: idx for idx, line_no in enumerate(line_numbers)}
    region_types = list(example.get("eeg_region_types") or [])
    counts: dict[str, int] = {}
    for line_no in lines:
        idx = index_by_line.get(line_no)
        if idx is None or idx >= len(region_types):
            continue
        for region in list(region_types[idx] or []):
            region_text = str(region).strip()
            if not region_text:
                continue
            counts[region_text] = counts.get(region_text, 0) + 1
    if not counts:
        return ""
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ", ".join(name for name, _ in ranked[:limit])


def _focus_line_snippets(example: dict[str, Any], lines: list[int], limit: int = 3, max_chars: int = 90) -> str:
    if not lines:
        return ""
    code_lines = list(example.get("code_tokens") or [])
    line_numbers = _normalize_line_numbers(example, len(code_lines))
    index_by_line = {line_no: idx for idx, line_no in enumerate(line_numbers)}
    snippets: list[str] = []
    for line_no in lines[:limit]:
        idx = index_by_line.get(line_no)
        if idx is None or idx >= len(code_lines):
            continue
        snippet = re.sub(r"\s+", " ", str(code_lines[idx]).strip())
        if not snippet:
            continue
        if len(snippet) > max_chars:
            snippet = snippet[: max_chars - 3].rstrip() + "..."
        snippets.append(f"{line_no}: {snippet}")
    return " | ".join(snippets)


def build_focus_hint(task: str, example: dict[str, Any], weight_mode: str) -> str:
    if task not in {"fault_localization", "vulnerability_detection"}:
        return ""
    line_count = len(list(example.get("code_tokens") or []))
    if line_count <= 0:
        return ""
    if weight_mode == "scanpath":
        lines = _top_scanpath_lines(example, line_count)
        source = "simulation"
    elif weight_mode == "scanpath_weak":
        lines = _top_scanpath_weak_lines(example, line_count)
        source = "simulation ablation"
    elif weight_mode in {"eeg", "eeg_workload"}:
        lines = _top_eeg_lines(example, line_count)
        source = "EEG workload prior" if weight_mode == "eeg_workload" else "EEG prior"
    elif weight_mode == "scanpath_eeg":
        lines = _top_scanpath_eeg_lines(example, line_count)
        source = "simulation+EEG consensus"
    else:
        return ""
    if not lines:
        return ""
    line_text = ", ".join(str(line) for line in lines)
    if task == "vulnerability_detection":
        region_text = _focus_region_summary(example, lines)
        snippet_text = _focus_line_snippets(example, lines)
        detail_parts = []
        if region_text:
            detail_parts.append(f"semantic hotspots: {region_text}")
        if snippet_text:
            detail_parts.append(f"hotspot snippets: {snippet_text}")
        if detail_parts:
            return f"Workload hint ({source}): inspect lines {line_text}; {'; '.join(detail_parts)}."
        return f"Workload hint ({source}): inspect lines {line_text}."
    if weight_mode in {"eeg", "scanpath_eeg"}:
        region_text = _focus_region_summary(example, lines)
        if region_text:
            return f"Focus hint ({source}): inspect lines {line_text} first. Structural hotspots: {region_text}."
    return f"Focus hint ({source}): inspect lines {line_text} first."


def calculate_scanpath_weight(mask: int | float, order: int | float) -> float:
    if float(mask) <= 0.0:
        return 1.0
    order_bonus = 1.0 / max(1.0, float(order))
    return 3.0 + order_bonus


def calculate_scanpath_weak_weight(mask: int | float, order: int | float) -> float:
    # Explicit weak ablation: much flatter weighting and only meaningful for the
    # very first few attended positions.
    if float(mask) <= 0.0:
        return 1.0
    order_value = float(order)
    if order_value <= 0.0:
        return 1.0
    if order_value > 2.0:
        return 1.0
    order_bonus = 1.0 / max(1.0, order_value)
    return 1.35 + 0.35 * order_bonus


def minmax_normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    low = min(values)
    high = max(values)
    if math.isclose(low, high):
        return [0.5 for _ in values]
    scale = high - low
    return [(value - low) / scale for value in values]


def safe_list(example: dict[str, Any], key: str, token_len: int, default_value: float) -> list[float]:
    values = list(example.get(key) or [])
    if len(values) != token_len:
        values = (values + [default_value] * token_len)[:token_len]
    return [float(v) for v in values]


def build_eeg_priority(example: dict[str, Any], token_len: int) -> dict[str, list[float] | float]:
    salience = safe_list(example, "eeg_salience_sequence", token_len, 0.0)
    attention = safe_list(example, "eeg_attention_sequence", token_len, 0.0)
    revisit = safe_list(example, "eeg_revisit_sequence", token_len, 0.0)
    complexity = safe_list(example, "complexity_sequence", token_len, 0.0)

    salience_norm = minmax_normalize(salience)
    attention_norm = minmax_normalize(attention)
    revisit_norm = minmax_normalize(revisit)
    complexity_norm = minmax_normalize(complexity)

    priority = [
        0.45 * salience_norm[idx]
        + 0.20 * attention_norm[idx]
        + 0.15 * revisit_norm[idx]
        + 0.20 * complexity_norm[idx]
        for idx in range(token_len)
    ]
    snippet_complexity = float(example.get("eeg_snippet_complexity", 0.0) or 0.0)
    return {
        "priority": priority,
        "snippet_complexity": min(1.0, max(0.0, snippet_complexity)),
    }


def normalize_weight_profile_to_scanpath_mean(weights: list[float], scanpath_weights: list[float]) -> list[float]:
    if not weights or not scanpath_weights:
        return weights
    base_mean = mean(scanpath_weights)
    hybrid_mean = mean(weights)
    if hybrid_mean <= 0.0:
        return weights
    scale = base_mean / hybrid_mean
    normalized = [max(1.0, weight * scale) for weight in weights]
    return normalized


def calculate_weight_profile(example: dict[str, Any], token_len: int, weight_mode: str) -> tuple[list[float], dict[str, float]]:
    if weight_mode == "uniform":
        return [1.0] * token_len, {
            "mean_weight": 1.0,
            "scanpath_active_ratio": 0.0,
            "eeg_active_ratio": 0.0,
            "overlap_ratio": 0.0,
            "rescue_ratio": 0.0,
        }

    scanpath_mask = safe_list(example, "scanpath_mask", token_len, 0.0)
    scanpath_order = safe_list(example, "scanpath_order_sequence", token_len, 0.0)
    eeg_mask = safe_list(example, "mask", token_len, 0.0)
    eeg_order = safe_list(example, "order_sequence", token_len, 0.0)
    eeg_priority_meta = build_eeg_priority(example, token_len)
    eeg_priority = list(eeg_priority_meta["priority"])
    snippet_complexity = float(eeg_priority_meta["snippet_complexity"])

    scanpath_weights = [
        calculate_scanpath_weight(scanpath_mask[idx], scanpath_order[idx])
        for idx in range(token_len)
    ]
    if weight_mode == "scanpath":
        active_scanpath = sum(1 for value in scanpath_mask if value > 0.0)
        active_eeg = sum(1 for value in eeg_mask if value > 0.0)
        overlap = sum(1 for s, e in zip(scanpath_mask, eeg_mask) if s > 0.0 and e > 0.0)
        return scanpath_weights, {
            "mean_weight": mean(scanpath_weights),
            "scanpath_active_ratio": active_scanpath / max(1, token_len),
            "eeg_active_ratio": active_eeg / max(1, token_len),
            "overlap_ratio": overlap / max(1, active_scanpath),
            "rescue_ratio": 0.0,
        }

    if weight_mode == "scanpath_weak":
        weak_scanpath_weights = [
            calculate_scanpath_weak_weight(scanpath_mask[idx], scanpath_order[idx])
            for idx in range(token_len)
        ]
        active_scanpath = sum(1 for idx, value in enumerate(scanpath_mask) if value > 0.0 and scanpath_order[idx] > 0.0 and scanpath_order[idx] <= 2.0)
        active_eeg = sum(1 for value in eeg_mask if value > 0.0)
        overlap = sum(
            1
            for idx, (s, e) in enumerate(zip(scanpath_mask, eeg_mask))
            if s > 0.0 and e > 0.0 and scanpath_order[idx] > 0.0 and scanpath_order[idx] <= 2.0
        )
        return weak_scanpath_weights, {
            "mean_weight": mean(weak_scanpath_weights),
            "scanpath_active_ratio": active_scanpath / max(1, token_len),
            "eeg_active_ratio": active_eeg / max(1, token_len),
            "overlap_ratio": overlap / max(1, active_scanpath),
            "rescue_ratio": 0.0,
        }

    if weight_mode in {"eeg", "eeg_workload"}:
        eeg_only_weights = []
        for idx in range(token_len):
            if eeg_mask[idx] <= 0.0:
                eeg_only_weights.append(1.0)
                continue
            order_bonus = 1.0 / max(1.0, eeg_order[idx])
            priority = eeg_priority[idx]
            eeg_only_weights.append(
                1.35
                + 1.50 * priority
                + 0.45 * order_bonus
                + 0.40 * snippet_complexity
            )
        eeg_only_weights = normalize_weight_profile_to_scanpath_mean(eeg_only_weights, scanpath_weights)
        active_scanpath = sum(1 for value in scanpath_mask if value > 0.0)
        active_eeg = sum(1 for value in eeg_mask if value > 0.0)
        overlap = sum(1 for s, e in zip(scanpath_mask, eeg_mask) if s > 0.0 and e > 0.0)
        return eeg_only_weights, {
            "mean_weight": mean(eeg_only_weights),
            "scanpath_active_ratio": active_scanpath / max(1, token_len),
            "eeg_active_ratio": active_eeg / max(1, token_len),
            "overlap_ratio": overlap / max(1, active_scanpath),
            "rescue_ratio": max(0, active_eeg - overlap) / max(1, token_len),
        }

    # Recommended mode: preserve scanpath weights and only let EEG gently modulate them.
    snippet_scale = 1.0 + SCANPATH_EEG_SNIPPET_SCALE * (snippet_complexity - 0.5)
    hybrid_weights: list[float] = []
    rescue_count = 0
    for idx in range(token_len):
        base_weight = scanpath_weights[idx]
        priority = eeg_priority[idx]
        if scanpath_mask[idx] > 0.0:
            centered = priority - 0.5
            overlap_boost = 1.0 + (SCANPATH_EEG_OVERLAP_BOOST if eeg_mask[idx] > 0.0 else 0.0)
            modulated = base_weight * overlap_boost * (1.0 + SCANPATH_EEG_MODULATION * centered) * snippet_scale
            hybrid_weights.append(max(1.0, modulated))
            continue

        eeg_rescue = eeg_mask[idx] > 0.0 and priority >= SCANPATH_EEG_RESCUE_THRESHOLD
        if eeg_rescue:
            order_bonus = 1.0 / max(1.0, eeg_order[idx])
            rescue_weight = 1.0 + SCANPATH_EEG_RESCUE_WEIGHT * priority + 0.15 * order_bonus
            hybrid_weights.append(rescue_weight)
            rescue_count += 1
        else:
            hybrid_weights.append(1.0)

    hybrid_weights = normalize_weight_profile_to_scanpath_mean(hybrid_weights, scanpath_weights)
    active_scanpath = sum(1 for value in scanpath_mask if value > 0.0)
    active_eeg = sum(1 for value in eeg_mask if value > 0.0)
    overlap = sum(1 for s, e in zip(scanpath_mask, eeg_mask) if s > 0.0 and e > 0.0)
    return hybrid_weights, {
        "mean_weight": mean(hybrid_weights),
        "scanpath_active_ratio": active_scanpath / max(1, token_len),
        "eeg_active_ratio": active_eeg / max(1, token_len),
        "overlap_ratio": overlap / max(1, active_scanpath),
        "rescue_ratio": rescue_count / max(1, token_len),
    }


def summarize_weight_profiles(rows: list[dict[str, Any]]) -> dict[str, float]:
    if not rows:
        return {}
    metrics = {
        "mean_weight": mean(float(row.get("weight_profile_mean", 1.0)) for row in rows),
        "mean_scanpath_active_ratio": mean(float(row.get("scanpath_active_ratio", 0.0)) for row in rows),
        "mean_eeg_active_ratio": mean(float(row.get("eeg_active_ratio", 0.0)) for row in rows),
        "mean_overlap_ratio": mean(float(row.get("overlap_ratio", 0.0)) for row in rows),
        "mean_rescue_ratio": mean(float(row.get("rescue_ratio", 0.0)) for row in rows),
    }
    return metrics


class SimpleListDataset(Dataset):
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.rows[index]


@dataclass
class CustomDataCollator:
    tokenizer: AutoTokenizer
    max_length: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch: dict[str, Any] = {}
        max_len = self.max_length
        for key in features[0].keys():
            if key == "weights":
                pad_value, dtype = 1.0, torch.float
            elif key == "labels":
                pad_value, dtype = -100, torch.long
            else:
                pad_value, dtype = self.tokenizer.pad_token_id, torch.long

            padded_list = []
            for feature in features:
                values = list(feature[key])[:max_len]
                values = values + [pad_value] * (max_len - len(values))
                padded_list.append(values)
            batch[key] = torch.tensor(padded_list, dtype=dtype)
        return batch


class WeightedLossTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):  # type: ignore[override]
        labels = inputs.pop("labels")
        weights = inputs.pop("weights", None)
        outputs = model(**inputs, use_cache=False)
        logits = outputs.logits

        if labels.device != logits.device:
            labels = labels.to(logits.device)
        if weights is not None and weights.device != logits.device:
            weights = weights.to(logits.device)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = CrossEntropyLoss(reduction="none")
        flat_logits = shift_logits.view(-1, shift_logits.shape[-1])
        flat_labels = shift_labels.view(-1)
        loss_per_token = loss_fct(flat_logits, flat_labels)

        active_loss_mask = flat_labels != -100
        active_losses = loss_per_token[active_loss_mask]
        if weights is not None:
            shift_weights = weights[..., 1:].contiguous().view(-1)
            active_weights = shift_weights[active_loss_mask]
            loss = (active_losses * active_weights).sum() / (active_weights.sum() + 1e-9)
        else:
            loss = active_losses.mean()

        return (loss, outputs) if return_outputs else loss


def load_and_validate_data(file_path: str) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = []
    required_keys = ["code_tokens", "mask", "ngram_indices", "semantic_token_sequence", "code", "content"]
    try:
        for line_index, record in enumerate(read_jsonl(file_path), start=1):
            if not all(key in record and record[key] is not None for key in required_keys):
                continue
            token_len = len(record["code_tokens"])
            keys_to_check = ["mask", "ngram_indices", "semantic_token_sequence"]
            if "order_sequence" in record:
                keys_to_check.append("order_sequence")
            if "complexity_sequence" in record:
                keys_to_check.append("complexity_sequence")
            if not all(len(record[key]) == token_len for key in keys_to_check):
                logging.warning(f"Skipping malformed row {line_index} in {file_path}: inconsistent sequence lengths")
                continue
            data.append(record)
    except FileNotFoundError:
        logging.error(f"Data file not found: {file_path}")
        return []
    return data


def create_and_process_data(
    tokenizer,
    task: str,
    train_file: str,
    valid_file: str,
    max_length: int,
    weight_mode: str,
    prompt_format: str = "plain",
    vulnerability_safe_weight: float = 1.0,
    vulnerability_vulnerable_weight: float = 1.0,
    vulnerability_eeg_hard_negative_scale: float = 0.0,
    normalize_vulnerability_label_tokens: bool = True,
    disable_focus_hint: bool = False,
) -> dict[str, Dataset]:
    logging.info(f"Loading training data from: {train_file}")
    logging.info(f"Loading validation data from: {valid_file}")
    train_data = load_and_validate_data(train_file)
    valid_data = load_and_validate_data(valid_file)
    if not train_data or not valid_data:
        logging.error("Training or validation data is empty. Check paths and content integrity.")
        sys.exit(1)

    def preprocess_and_weigh(example: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
        code_str, content_str = str(example["code"]), str(example["content"])
        base_context = str(example.get("task_context") or "").strip()
        focus_hint = "" if disable_focus_hint else build_focus_hint(task, example, weight_mode)
        if focus_hint:
            full_context = f"{base_context}\n{focus_hint}" if base_context else focus_hint
        else:
            full_context = base_context
        target_text = f"{content_str}{tokenizer.eos_token}"
        full_prompt = format_prompt_for_budget(
            task,
            code_str,
            context=full_context,
            tokenizer=tokenizer,
            max_length=max_length,
            target_text=target_text,
            prompt_format=prompt_format,
        )
        full_text = f"{full_prompt}{target_text}"

        tokenized_full = tokenizer(full_text, max_length=max_length, truncation=True)
        tokenized_prompt = tokenizer(full_prompt, max_length=max_length, truncation=True)

        input_ids = list(tokenized_full["input_ids"])
        attention_mask = [1] * len(input_ids)
        labels = list(input_ids)
        prompt_len = len(tokenized_prompt["input_ids"])
        labels[:prompt_len] = [-100] * prompt_len

        token_len = len(example["code_tokens"])
        weights_per_code_token, weight_stats = calculate_weight_profile(example, token_len, weight_mode)

        if task in {"fault_localization", "vulnerability_detection"}:
            weights = [1.0] * len(input_ids)
            if task == "vulnerability_detection":
                n_output = max(1, len(input_ids) - prompt_len)
                label_weight = (
                    vulnerability_vulnerable_weight
                    if content_str.strip().upper() == "VULNERABLE"
                    else vulnerability_safe_weight
                )
                if weight_mode in {"eeg", "eeg_workload", "scanpath_eeg"}:
                    token_len = len(example["code_tokens"])
                    priority = list(build_eeg_priority(example, token_len)["priority"])
                    max_priority = max(priority, default=0.0)
                    max_risk = max(safe_list(example, "c_security_risk_sequence", token_len, 0.0), default=0.0)
                    risk_signal = min(1.0, max(0.0, 0.55 * max_priority + 0.45 * max_risk))
                    if content_str.strip().upper() == "SAFE":
                        # Risky-looking safe functions are the hard negatives that control false positives.
                        label_weight *= 1.0 + vulnerability_eeg_hard_negative_scale * risk_signal
                    else:
                        label_weight *= 1.0 + 0.35 * vulnerability_eeg_hard_negative_scale * risk_signal
                if normalize_vulnerability_label_tokens:
                    label_weight = label_weight / n_output
                weights[prompt_len:] = [label_weight] * max(0, len(input_ids) - prompt_len)
        else:
            code_token_weights: list[float] = []
            for code_token, weight in zip(example["code_tokens"], weights_per_code_token):
                tokenized = tokenizer(str(code_token), add_special_tokens=False)
                code_token_weights.extend([weight] * len(tokenized["input_ids"]))

            n_output = len(input_ids) - prompt_len
            if len(code_token_weights) < n_output:
                code_token_weights.extend([1.0] * (n_output - len(code_token_weights)))
            elif len(code_token_weights) > n_output:
                code_token_weights = code_token_weights[:n_output]

            weights = [1.0] * prompt_len + code_token_weights
        processed = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "weights": weights,
        }
        profile = {
            "weight_profile_mean": weight_stats["mean_weight"],
            "scanpath_active_ratio": weight_stats["scanpath_active_ratio"],
            "eeg_active_ratio": weight_stats["eeg_active_ratio"],
            "overlap_ratio": weight_stats["overlap_ratio"],
            "rescue_ratio": weight_stats["rescue_ratio"],
        }
        return processed, profile

    processed_train: list[dict[str, Any]] = []
    processed_valid: list[dict[str, Any]] = []
    train_profiles: list[dict[str, Any]] = []
    valid_profiles: list[dict[str, Any]] = []
    for example in train_data:
        processed, profile = preprocess_and_weigh(example)
        processed_train.append(processed)
        train_profiles.append(profile)
    for example in valid_data:
        processed, profile = preprocess_and_weigh(example)
        processed_valid.append(processed)
        valid_profiles.append(profile)
    logging.info(f"Weight mode={weight_mode} train stats={summarize_weight_profiles(train_profiles)}")
    logging.info(f"Weight mode={weight_mode} valid stats={summarize_weight_profiles(valid_profiles)}")
    return {
        "train": SimpleListDataset(processed_train),
        "validation": SimpleListDataset(processed_valid),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Qwen 3.5 4B on ACT-R-style code tasks with uniform/scanpath/EEG weighting.")
    parser.add_argument("--task", choices=sorted(TASK_PROMPTS.keys()), default="completion")
    parser.add_argument("--model-choice", choices=["base", "instruction"], default="base")
    parser.add_argument("--model-id", help="Override the default Qwen model path/id.")
    parser.add_argument("--train-file", default="data/actr_scanpath_eeg/completion_train_final_eeg_scanpath.jsonl")
    parser.add_argument("--valid-file", default="data/actr_scanpath_eeg/completion_valid_final_eeg_scanpath.jsonl")
    parser.add_argument("--output-dir", default="outputs/qwen35_base_eeg_completion_large")
    parser.add_argument("--weight-mode", choices=["uniform", "scanpath", "scanpath_weak", "eeg", "eeg_workload", "scanpath_eeg"], default="scanpath_eeg")
    parser.add_argument("--max-length", type=int, default=int(os.environ.get("MAX_LENGTH", "1024")))
    parser.add_argument("--num-train-epochs", type=float, default=float(os.environ.get("NUM_TRAIN_EPOCHS", "3")))
    parser.add_argument("--per-device-train-batch-size", type=int, default=int(os.environ.get("TRAIN_BATCH_SIZE", "4")))
    parser.add_argument("--gradient-accumulation-steps", type=int, default=int(os.environ.get("GRAD_ACCUM_STEPS", "16")))
    parser.add_argument("--learning-rate", type=float, default=float(os.environ.get("LEARNING_RATE", "2e-5")))
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("MAX_STEPS", "-1")))
    parser.add_argument("--prompt-format", choices=["plain", "chat"], default=os.environ.get("PROMPT_FORMAT", "plain"))
    parser.add_argument("--use-lora", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=int(os.environ.get("LORA_RANK", "16")))
    parser.add_argument("--lora-alpha", type=int, default=int(os.environ.get("LORA_ALPHA", "32")))
    parser.add_argument("--lora-dropout", type=float, default=float(os.environ.get("LORA_DROPOUT", "0.05")))
    parser.add_argument("--lora-target-modules", default=os.environ.get("LORA_TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vulnerability-safe-weight", type=float, default=float(os.environ.get("VULN_SAFE_WEIGHT", "1.0")))
    parser.add_argument("--vulnerability-vulnerable-weight", type=float, default=float(os.environ.get("VULN_VULNERABLE_WEIGHT", "1.0")))
    parser.add_argument(
        "--vulnerability-eeg-hard-negative-scale",
        type=float,
        default=float(os.environ.get("VULN_EEG_HARD_NEGATIVE_SCALE", "0.0")),
    )
    parser.add_argument(
        "--no-normalize-vulnerability-label-tokens",
        action="store_true",
        help="Disable per-example label-token normalization for vulnerability labels.",
    )
    parser.add_argument("--disable-focus-hint", action="store_true", help="Disable scanpath/EEG focus hints in prompts while keeping loss weighting.")
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Disable gradient checkpointing. Useful for LoRA runs when memory is sufficient and speed matters.",
    )
    args = parser.parse_args()

    _set_reproducible_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")

    model_id = args.model_id or default_model_path(args.model_choice)
    output_dir = Path(args.output_dir)
    logging.info(f"Task={args.task} model_choice={args.model_choice} model={model_id}")
    logging.info(f"Weight mode={args.weight_mode}")
    logging.info(f"Prompt format={args.prompt_format}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    use_bf16 = _is_bf16_supported()
    load_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if torch.cuda.is_available() else torch.float32)
    distributed_training = _is_distributed_training()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        device_map=("auto" if torch.cuda.is_available() and not distributed_training else None),
        torch_dtype=load_dtype if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        attn_implementation=_pick_attn_implementation(),
    )
    model.config.use_cache = False
    if not args.no_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if args.use_lora:
        from peft import LoraConfig, get_peft_model

        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        target_modules = [item.strip() for item in args.lora_target_modules.split(",") if item.strip()]
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    _log_cuda_memory("after_model_load")

    datasets = create_and_process_data(
        tokenizer=tokenizer,
        task=args.task,
        train_file=args.train_file,
        valid_file=args.valid_file,
        max_length=args.max_length,
        weight_mode=args.weight_mode,
        prompt_format=args.prompt_format,
        vulnerability_safe_weight=args.vulnerability_safe_weight,
        vulnerability_vulnerable_weight=args.vulnerability_vulnerable_weight,
        vulnerability_eeg_hard_negative_scale=args.vulnerability_eeg_hard_negative_scale,
        normalize_vulnerability_label_tokens=not args.no_normalize_vulnerability_label_tokens,
        disable_focus_hint=args.disable_focus_hint,
    )

    optim_name = _pick_optimizer_name()
    report_to = _pick_report_to()
    logging.info(f"Using optimizer: {optim_name}")
    logging.info(f"Reporting backend: {report_to}")
    training_args = build_compatible_training_arguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        report_to=report_to,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        fp16=False,
        bf16=use_bf16,
        optim=optim_name,
        adafactor=(optim_name == "adafactor"),
        prediction_loss_only=True,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        max_grad_norm=0.0,
        dataloader_num_workers=0,
        seed=args.seed,
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
    )

    trainer = WeightedLossTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=CustomDataCollator(tokenizer=tokenizer, max_length=args.max_length),
    )
    _log_cuda_memory("after_trainer_init")

    logging.info(f"Starting Qwen ACT-R-style training for task '{args.task}' with weight_mode='{args.weight_mode}'")
    start_time = time.time()
    trainer.train(resume_from_checkpoint=False)
    end_time = time.time()
    logging.info("***** Training Complete *****")
    logging.info(f"Total training time: {end_time - start_time:.2f} seconds ({(end_time - start_time)/60:.2f} minutes)")

    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    logging.info(f"Final model saved to {output_dir}")


if __name__ == "__main__":
    main()
