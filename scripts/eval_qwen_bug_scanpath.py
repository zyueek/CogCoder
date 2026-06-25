from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from eeg_qwen.common import mean, read_jsonl, safe_div, write_json, write_jsonl
from train_qwen_scanpath_large import build_focus_hint, default_model_path, format_prompt_for_budget


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


TASKS = {"vulnerability_detection", "fault_localization"}


def _count_jsonl(path: str, max_examples: Optional[int] = None) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as handle:
        for _ in handle:
            count += 1
            if max_examples is not None and count >= max_examples:
                break
    return count


def _iter_jsonl(path: str, max_examples: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if max_examples is not None and index >= max_examples:
                break
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _batched(iterable: Iterable[Dict[str, Any]], batch_size: int) -> Iterator[List[Dict[str, Any]]]:
    batch: List[Dict[str, Any]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _pick_torch_dtype(precision: str) -> Optional[torch.dtype]:
    if precision == "fp32":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    if precision != "auto":
        raise ValueError(f"Unknown precision: {precision}")
    if not torch.cuda.is_available():
        return torch.float32
    if getattr(torch.cuda, "is_bf16_supported", None) and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _pick_attn_implementation(requested: str) -> Optional[str]:
    if requested in {"eager", "sdpa"}:
        return requested
    if requested != "auto":
        raise ValueError(f"Unknown attn implementation: {requested}")
    try:
        base = torch.__version__.split("+", 1)[0]
        parts = [int(part) for part in base.split(".")[:3]]
        torch_ver = tuple(parts + [0] * (3 - len(parts)))
    except Exception:
        torch_ver = (0, 0, 0)
    has_sdpa = hasattr(torch.nn.functional, "scaled_dot_product_attention")
    if has_sdpa and torch_ver >= (2, 1, 1):
        return "sdpa"
    return "eager"


def _parse_vulnerability_label(text: str) -> str:
    normalized = text.strip().upper()
    match = re.search(r"\b(VULNERABLE|SAFE|CLEAN|YES|NO)\b", normalized)
    if not match:
        return "UNKNOWN"
    value = match.group(1)
    if value == "VULNERABLE" or value == "YES":
        return "VULNERABLE"
    if value == "SAFE" or value == "CLEAN" or value == "NO":
        return "SAFE"
    return "UNKNOWN"


def _parse_line_numbers(text: str) -> list[int]:
    numbers = [int(match) for match in re.findall(r"\d+", text)]
    output: List[int] = []
    for number in numbers:
        if number <= 0 or number in output:
            continue
        output.append(number)
    return output


def _evaluate_vulnerability(record: dict[str, Any], prediction: str) -> dict[str, Any]:
    gold = str(record.get("content") or "").strip().upper()
    parsed = _parse_vulnerability_label(prediction)
    exact = parsed == gold
    return {
        "gold_label": gold,
        "predicted_label": parsed,
        "exact_match": exact,
        "is_positive_gold": gold == "VULNERABLE",
        "is_positive_pred": parsed == "VULNERABLE",
    }


def _evaluate_fault_localization(record: dict[str, Any], prediction: str) -> dict[str, Any]:
    gold_lines = list(((record.get("metadata") or {}).get("gold_lines") or []))
    if not gold_lines:
        gold_lines = _parse_line_numbers(str(record.get("content") or ""))
    predicted_lines = _parse_line_numbers(prediction)
    exact = predicted_lines == gold_lines
    hit1 = bool(predicted_lines and predicted_lines[0] in gold_lines)
    hit3 = any(line in gold_lines for line in predicted_lines[:3])
    reciprocal_rank = 0.0
    for rank, line in enumerate(predicted_lines, start=1):
        if line in gold_lines:
            reciprocal_rank = 1.0 / rank
            break
    return {
        "gold_lines": gold_lines,
        "predicted_lines": predicted_lines,
        "exact_match": exact,
        "hit_at_1": hit1,
        "hit_at_3": hit3,
        "mrr": reciprocal_rank,
    }


def summarize_results(task: str, results: list[dict[str, Any]]) -> dict[str, Any]:
    if task == "vulnerability_detection":
        exact = sum(int(result["exact_match"]) for result in results)
        tp = sum(1 for result in results if result["is_positive_gold"] and result["is_positive_pred"])
        fp = sum(1 for result in results if (not result["is_positive_gold"]) and result["is_positive_pred"])
        fn = sum(1 for result in results if result["is_positive_gold"] and (not result["is_positive_pred"]))
        tn = sum(1 for result in results if (not result["is_positive_gold"]) and (not result["is_positive_pred"]) and result["predicted_label"] != "UNKNOWN")
        unknown = sum(1 for result in results if result["predicted_label"] == "UNKNOWN")
        precision = safe_div(tp, tp + fp)
        recall = safe_div(tp, tp + fn)
        specificity = safe_div(tn, tn + fp)
        balanced_accuracy = 0.5 * (recall + specificity)
        f1 = safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
        denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = ((tp * tn) - (fp * fn)) / denominator if denominator > 0 else 0.0
        return {
            "task": task,
            "num_examples": len(results),
            "accuracy": safe_div(exact, len(results)),
            "precision": precision,
            "recall": recall,
            "specificity": specificity,
            "balanced_accuracy": balanced_accuracy,
            "f1": f1,
            "mcc": mcc,
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
            "unknown": unknown,
            "unknown_rate": safe_div(unknown, len(results)),
        }

    return {
        "task": task,
        "num_examples": len(results),
        "exact_match_rate": safe_div(sum(int(result["exact_match"]) for result in results), len(results)),
        "hit_at_1": safe_div(sum(int(result["hit_at_1"]) for result in results), len(results)),
        "hit_at_3": safe_div(sum(int(result["hit_at_3"]) for result in results), len(results)),
        "mrr": mean(result["mrr"] for result in results),
    }


def evaluate(
    *,
    task: str,
    model_path: str,
    data_file: str,
    output_dir: str,
    batch_size: int,
    max_new_tokens: int,
    truncate_inputs: bool,
    max_length: int,
    attn_implementation: str,
    precision: str,
    device: str,
    enable_tf32: bool,
    compile_model: bool,
    max_examples: Optional[int],
    prompt_weight_mode: str,
) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(enable_tf32)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high" if enable_tf32 else "highest")

    total = _count_jsonl(data_file, max_examples=max_examples)
    logging.info("Eval examples: %d (%s)", total, data_file)

    adapter_config = Path(model_path) / "adapter_config.json"
    is_lora_adapter = adapter_config.exists()
    tokenizer_source = model_path
    base_model_for_adapter = None
    if is_lora_adapter:
        try:
            payload = json.loads(adapter_config.read_text(encoding="utf-8"))
            base_model_for_adapter = payload.get("base_model_name_or_path") or default_model_path("base")
        except Exception:
            base_model_for_adapter = default_model_path("base")
        tokenizer_source = base_model_for_adapter

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"

    torch_dtype = _pick_torch_dtype(precision)
    attn_impl = _pick_attn_implementation(attn_implementation)
    use_device_map = device == "auto" and torch.cuda.is_available()
    load_path = base_model_for_adapter if is_lora_adapter else model_path
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        trust_remote_code=True,
        device_map="auto" if use_device_map else None,
        torch_dtype=(None if torch_dtype == torch.float32 else torch_dtype) if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl,
    )
    if is_lora_adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, model_path)
    if not use_device_map:
        model = model.to(device if torch.cuda.is_available() else "cpu")
    model.eval()
    model.config.use_cache = True
    eos_token_id = model.config.eos_token_id or tokenizer.eos_token_id

    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[assignment]
            logging.info("torch.compile enabled")
        except Exception as exc:
            logging.warning("torch.compile failed (%s: %s); continuing without compile.", type(exc).__name__, exc)

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    predictions_path = output_root / "predictions.jsonl"
    summary_path = output_root / "summary.json"
    model_device = next(model.parameters()).device

    results: list[dict[str, Any]] = []
    t0 = time.time()
    with predictions_path.open("w", encoding="utf-8") as handle:
        progress = tqdm(total=total, desc=f"{task}-eval", unit="ex")
        for batch in _batched(_iter_jsonl(data_file, max_examples=max_examples), batch_size):
            prompts: List[str] = []
            kept_rows: List[dict[str, Any]] = []
            for row in batch:
                code = row.get("code")
                if not code:
                    continue
                base_context = str(row.get("task_context") or "").strip()
                focus_hint = build_focus_hint(task, row, prompt_weight_mode)
                if focus_hint:
                    full_context = f"{base_context}\n{focus_hint}" if base_context else focus_hint
                else:
                    full_context = base_context
                prompts.append(
                    format_prompt_for_budget(
                        task,
                        str(code),
                        context=full_context,
                        tokenizer=tokenizer,
                        max_length=max_length if truncate_inputs else None,
                    )
                )
                kept_rows.append(row)
            if not prompts:
                progress.update(len(batch))
                continue

            tok_kwargs: Dict[str, Any] = {"return_tensors": "pt", "padding": True}
            if truncate_inputs:
                tok_kwargs.update({"truncation": True, "max_length": max_length})
            tok = tokenizer(prompts, **tok_kwargs)
            input_ids = tok["input_ids"].to(model_device)
            attention_mask = tok["attention_mask"].to(model_device)

            with torch.inference_mode():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    eos_token_id=eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )

            gen_only = outputs[:, input_ids.shape[1] :]
            predictions = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
            for row, prediction in zip(kept_rows, predictions):
                prediction = prediction.strip()
                if task == "vulnerability_detection":
                    metrics = _evaluate_vulnerability(row, prediction)
                else:
                    metrics = _evaluate_fault_localization(row, prediction)
                result = {
                    "id": row.get("code_content_hash") or row.get("id"),
                    "input": row.get("code"),
                    "context": row.get("task_context", ""),
                    "ground_truth": row.get("content", ""),
                    "generated_output": prediction,
                    **metrics,
                }
                results.append(result)
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            progress.update(len(batch))
        progress.close()

    summary = summarize_results(task, results)
    summary["elapsed_seconds"] = round(time.time() - t0, 3)
    write_json(summary_path, summary)
    logging.info("Wrote predictions to %s", predictions_path)
    logging.info("Wrote summary to %s", summary_path)
    logging.info("Summary: %s", summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen scanpath/EEG models on vulnerability detection or fault localization.")
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--model-choice", choices=["base", "instruction"], default="base")
    parser.add_argument("--model-path", help="Override the default model path.")
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--truncate-inputs", action="store_true")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--attn-implementation", choices=["auto", "eager", "sdpa"], default="auto")
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--enable-tf32", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--prompt-weight-mode", choices=["uniform", "scanpath", "scanpath_weak", "eeg", "eeg_workload", "scanpath_eeg"], default="uniform")
    args = parser.parse_args()

    model_path = args.model_path or default_model_path(args.model_choice)
    evaluate(
        task=args.task,
        model_path=model_path,
        data_file=args.data_file,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        truncate_inputs=args.truncate_inputs,
        max_length=args.max_length,
        attn_implementation=args.attn_implementation,
        precision=args.precision,
        device=args.device,
        enable_tf32=args.enable_tf32,
        compile_model=args.compile_model,
        max_examples=args.max_examples,
        prompt_weight_mode=args.prompt_weight_mode,
    )


if __name__ == "__main__":
    main()
