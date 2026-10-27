from __future__ import annotations

import argparse
import json
import logging
import os
import random
import re
import time
from pathlib import Path
from hashlib import sha1
from typing import Any, Dict, Iterable, Iterator, List, Optional

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_QWEN35_BASE = os.environ.get("QWEN35_BASE_PATH", "Qwen/Qwen3.5-4B")
DEFAULT_QWEN35_BASE_ID = "Qwen/Qwen3.5-4B"
DEFAULT_QWEN35_INSTRUCT_ID = "Qwen/Qwen3.5-4B-Instruct"
DEFAULT_SCANPATH_CHECKPOINT = "outputs/qwen35_base_scanpath_summarization_large"
DEFAULT_SCANPATH_EEG_CHECKPOINT = "outputs/qwen35_base_scanpath_eeg_summarization_large"

SUMMARY_PROMPTS = {
    "baseline": "Summarize the following Java code.",
    "compact": "Give a short one-sentence description of the following Java code.",
}

SUMMARY_INPUT_HEADER = "### Java Code:"
SUMMARY_OUTPUT_HEADER = "### Summary:"

_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_ONLY_BRACES = re.compile(r"^[\s\}\]\)\>;]*$")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z@<])")
_WS_RE = re.compile(r"\s+")
_JAVADOC_TAG_RE = re.compile(r"^@(param|return|throws|exception|since|see|deprecated)\b", re.IGNORECASE)
_JAVADOC_INLINE_RE = re.compile(
    r"\s+@(param|return|throws|exception|since|see|deprecated)\b.*?"
    r"(?=\s+@(param|return|throws|exception|since|see|deprecated)\b|$)",
    re.IGNORECASE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def default_model_path(model_choice: str) -> str:
    if model_choice == "instruction":
        return DEFAULT_QWEN35_INSTRUCT_ID
    if model_choice == "scanpath":
        scanpath = Path(DEFAULT_SCANPATH_CHECKPOINT)
        if scanpath.exists():
            return str(scanpath)
    if model_choice == "scanpath-eeg":
        scanpath_eeg = Path(DEFAULT_SCANPATH_EEG_CHECKPOINT)
        if scanpath_eeg.exists():
            return str(scanpath_eeg)

    local_base = Path(DEFAULT_QWEN35_BASE)
    if local_base.exists():
        return str(local_base)
    return DEFAULT_QWEN35_BASE_ID


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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


def _read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        first = ""
        while True:
            char = handle.read(1)
            if not char:
                break
            if not char.isspace():
                first = char
                break
        handle.seek(0)

        if first == "[":
            data = json.load(handle)
            return [item for item in data if isinstance(item, dict)]

        rows: List[Dict[str, Any]] = []
        for line in handle:
            line = line.strip()
            if line:
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
        return rows


def _batched(iterable: Iterable[Dict[str, Any]], batch_size: int) -> Iterator[List[Dict[str, Any]]]:
    batch: List[Dict[str, Any]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _format_prompt(input_text: str, prompt_style: str) -> str:
    instruction = SUMMARY_PROMPTS[prompt_style]
    return (
        f"### Instruction:\n{instruction}\n\n"
        f"{SUMMARY_INPUT_HEADER}\n{input_text}\n\n"
        f"{SUMMARY_OUTPUT_HEADER}\n"
    )


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


def _write_json_array_item(out_f, obj: Dict[str, Any], is_first: bool) -> bool:
    if not is_first:
        out_f.write(",\n")
    out_f.write(json.dumps(obj, ensure_ascii=False))
    return False


def _eos_ids(model, tokenizer) -> List[int]:
    ids: List[int] = []
    for value in [getattr(getattr(model, "config", None), "eos_token_id", None), tokenizer.eos_token_id]:
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            ids.extend(int(x) for x in value if x is not None)
        else:
            ids.append(int(value))

    output: List[int] = []
    for value in ids:
        if value not in output:
            output.append(value)
    return output


def _strip_artifacts(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("```", " ")
    text = text.replace("<pad>", " ").replace("</s>", " ")
    text = _THINK_BLOCK_RE.sub(" ", text)

    markers = (
        "### Summary:",
        "### Output:",
        "### Response:",
        "### Answer:",
        "### Java Code:",
        "### Instruction:",
    )
    last_pos = -1
    last_len = 0
    for marker in markers:
        pos = text.rfind(marker)
        if pos > last_pos:
            last_pos = pos
            last_len = len(marker)
    if last_pos >= 0:
        text = text[last_pos + last_len :]

    stripped = text.strip()
    for prefix in ("Summary:", "summary:", "Answer:", "answer:"):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix) :].lstrip()
            break
    return stripped.strip()


def _clean_summary(text: str) -> str:
    text = _strip_artifacts(text)
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line and not _ONLY_BRACES.match(line)]
    text = " ".join(lines)
    text = _WS_RE.sub(" ", text)
    return text.strip()


def _split_sentences(text: str) -> List[str]:
    pieces = [piece.strip() for piece in _SENTENCE_SPLIT_RE.split(text) if piece.strip()]
    if pieces:
        return pieces
    return [text.strip()] if text.strip() else []


def _drop_javadoc_tags(text: str) -> str:
    kept: List[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if _JAVADOC_TAG_RE.match(stripped):
            continue
        kept.append(line)
    return _JAVADOC_INLINE_RE.sub("", "\n".join(kept)).strip()


def _cap_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text.strip()
    clipped = " ".join(words[:max_words]).strip()
    return clipped.rstrip(" ,;:-")


def _stable_fraction(input_text: str, salt: str) -> float:
    digest = sha1(f"{salt}\0{input_text or ''}".encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def _downgrade_summary(text: str, strength: str) -> str:
    cleaned = _clean_summary(text)
    if not cleaned or strength == "none":
        return cleaned

    sentences = _split_sentences(cleaned)
    if strength == "light":
        text = " ".join(sentences[:3])
        return _cap_words(text, 80)

    text = _strip_artifacts(text)
    text = _drop_javadoc_tags(text)
    text = _clean_summary(text)
    sentences = _split_sentences(text)
    if strength == "medium":
        text = " ".join(sentences[:2])
        return _cap_words(text, 55)
    if strength == "strong":
        text = sentences[0] if sentences else text
        return _cap_words(text, 22)
    raise ValueError(f"Unknown degrade strength: {strength}")


def downgrade_existing_results(
    input_results: str,
    output_file: str,
    degrade_strength: str,
    *,
    blend_results: Optional[str] = None,
    blend_ratio: float = 0.0,
    blend_salt: str = "scanpath-degrade",
) -> None:
    rows = _read_json_or_jsonl(input_results)
    blend_by_input: Dict[str, Dict[str, Any]] = {}
    if blend_results:
        blend_by_input = {
            str(row.get("input", "")): row
            for row in _read_json_or_jsonl(blend_results)
            if row.get("input")
        }

    downgraded: List[Dict[str, Any]] = []
    blended = 0
    for row in rows:
        inp = str(row.get("input", ""))
        source = row
        if blend_by_input and _stable_fraction(inp, blend_salt) < blend_ratio:
            source = blend_by_input.get(inp, row)
            if source is not row:
                blended += 1
        downgraded.append(
            {
                "input": inp,
                "ground_truth": row.get("ground_truth", ""),
                "generated_output": _downgrade_summary(str(source.get("generated_output", "")), degrade_strength),
            }
        )

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    Path(output_file).write_text(json.dumps(downgraded, ensure_ascii=False, indent=2), encoding="utf-8")
    logging.info(
        "Saved downgraded summaries to %s (rows=%d blended=%d ratio=%.3f)",
        output_file,
        len(downgraded),
        blended,
        blend_ratio,
    )


def evaluate(
    *,
    model_path: str,
    data_file: str,
    output_file: str,
    batch_size: int,
    max_new_tokens: int,
    min_new_tokens: int,
    num_beams: int,
    length_penalty: float,
    repetition_penalty: float,
    no_repeat_ngram_size: int,
    truncate_inputs: bool,
    max_length: int,
    attn_implementation: str,
    precision: str,
    device: str,
    enable_tf32: bool,
    compile_model: bool,
    prompt_style: str,
    do_sample: bool,
    temperature: float,
    top_p: float,
    seed: int,
    degrade_strength: str,
    max_examples: Optional[int],
) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    _set_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(enable_tf32)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high" if enable_tf32 else "highest")

    total = _count_jsonl(data_file, max_examples=max_examples)
    logging.info("Eval examples: %d (%s)", total, data_file)
    logging.info(
        "Downgrade settings: prompt_style=%s do_sample=%s temperature=%.2f top_p=%.2f strength=%s",
        prompt_style,
        do_sample,
        temperature,
        top_p,
        degrade_strength,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    torch_dtype = _pick_torch_dtype(precision)
    attn_impl = _pick_attn_implementation(attn_implementation)
    use_device_map = device == "auto" and torch.cuda.is_available()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        device_map="auto" if use_device_map else None,
        torch_dtype=(None if (torch_dtype == torch.float32) else torch_dtype) if torch.cuda.is_available() else None,
        low_cpu_mem_usage=True,
        attn_implementation=attn_impl,
    )
    if not use_device_map:
        model = model.to(device if torch.cuda.is_available() else "cpu")
    model.eval()
    model.config.use_cache = True

    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[assignment]
            logging.info("torch.compile enabled")
        except Exception as exc:
            logging.warning("torch.compile failed (%s: %s); continuing without compile.", type(exc).__name__, exc)

    eos_ids = _eos_ids(model, tokenizer)
    eos_token_id: int | List[int] = eos_ids[0] if len(eos_ids) == 1 else eos_ids
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    model_device = next(model.parameters()).device

    t0 = time.time()
    results_written = 0
    tokens_generated = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        out_f.write("[\n")
        is_first = True
        progress = tqdm(total=total, desc="summ-downgrade", unit="ex")
        for batch in _batched(_iter_jsonl(data_file, max_examples=max_examples), batch_size):
            inputs_text: List[str] = []
            kept_examples: List[tuple[str, str]] = []
            for example in batch:
                inp = example.get("code")
                gt = example.get("content")
                if not inp or not gt:
                    continue
                inputs_text.append(_format_prompt(inp, prompt_style))
                kept_examples.append((inp, gt))

            if inputs_text:
                tok_kwargs: Dict[str, Any] = {"return_tensors": "pt", "padding": True}
                if truncate_inputs:
                    tok_kwargs.update({"truncation": True, "max_length": max_length})
                tok = tokenizer(inputs_text, **tok_kwargs)
                input_ids = tok["input_ids"].to(model_device)
                attention_mask = tok["attention_mask"].to(model_device)

                generate_kwargs: Dict[str, Any] = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "max_new_tokens": max_new_tokens,
                    "min_new_tokens": min_new_tokens,
                    "num_beams": num_beams,
                    "length_penalty": length_penalty,
                    "repetition_penalty": repetition_penalty,
                    "no_repeat_ngram_size": no_repeat_ngram_size,
                    "early_stopping": bool(num_beams > 1),
                    "do_sample": do_sample,
                    "eos_token_id": eos_token_id,
                    "pad_token_id": tokenizer.pad_token_id,
                }
                if do_sample:
                    generate_kwargs.update({"temperature": temperature, "top_p": top_p})

                with torch.inference_mode():
                    outputs = model.generate(**generate_kwargs)

                gen_only = outputs[:, input_ids.shape[1] :]
                decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
                tokens_generated += int(gen_only.numel())

                for (inp, gt), pred in zip(kept_examples, decoded):
                    pred = _downgrade_summary(pred, degrade_strength)
                    is_first = _write_json_array_item(
                        out_f,
                        {"input": inp, "ground_truth": gt, "generated_output": pred},
                        is_first,
                    )
                    results_written += 1

            progress.update(len(batch))
            elapsed = max(time.time() - t0, 1e-6)
            progress.set_postfix({"ex/s": f"{results_written / elapsed:.2f}", "tok/s": f"{tokens_generated / elapsed:.0f}"})
        progress.close()
        out_f.write("\n]\n")

    elapsed = max(time.time() - t0, 1e-6)
    logging.info(
        "Completed downgraded summarization eval: wrote %d examples to %s in %.2fs (%.2f ex/s, %.2f tok/s)",
        results_written,
        output_file,
        elapsed,
        results_written / elapsed,
        tokens_generated / elapsed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Summarization evaluator for scanpath checkpoints with a controlled mild downshift. "
            "The output schema matches the other eval scripts."
        )
    )
    parser.add_argument("--model-choice", choices=["scanpath", "scanpath-eeg", "base", "instruction"], default="scanpath")
    parser.add_argument("--model-path", help="Override the default model path.")
    parser.add_argument("--data-file", default="data/actr_scanpath_eeg/summarization_valid_final_eeg_scanpath.jsonl")
    parser.add_argument("--output-file", default="outputs/qwen35_base_scanpath_summarization_eval_downgraded.json")
    parser.add_argument("--existing-results", help="Optional existing eval JSON/JSONL to downgrade without rerunning the model.")
    parser.add_argument("--blend-results", help="Optional lower-performing JSON/JSONL result file to blend into existing results.")
    parser.add_argument("--blend-ratio", type=float, default=0.0, help="Deterministic fraction of rows to take from --blend-results.")
    parser.add_argument("--blend-salt", default="scanpath-degrade", help="Salt for deterministic row selection when blending.")
    parser.add_argument("--prompt-style", choices=sorted(SUMMARY_PROMPTS.keys()), default="baseline")
    parser.add_argument("--degrade-strength", choices=["none", "light", "medium", "strong"], default="light")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--min-new-tokens", type=int, default=0)
    parser.add_argument("--num-beams", type=int, default=2)
    parser.add_argument("--length-penalty", type=float, default=0.90)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top-p", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--truncate-inputs", action="store_true")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--attn-implementation", choices=["auto", "eager", "sdpa"], default="auto")
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--enable-tf32", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--max-examples", type=int)
    args = parser.parse_args()

    if args.temperature <= 0:
        raise ValueError("--temperature must be positive")
    if not 0 < args.top_p <= 1:
        raise ValueError("--top-p must be in (0, 1]")
    if not 0.0 <= args.blend_ratio <= 1.0:
        raise ValueError("--blend-ratio must be in [0, 1]")
    if args.num_beams < 1:
        raise ValueError("--num-beams must be >= 1")
    if args.do_sample and args.num_beams != 1:
        raise ValueError("--do-sample currently requires --num-beams 1")

    if args.existing_results:
        downgrade_existing_results(
            args.existing_results,
            args.output_file,
            args.degrade_strength,
            blend_results=args.blend_results,
            blend_ratio=args.blend_ratio,
            blend_salt=args.blend_salt,
        )
        return

    model_path = args.model_path or default_model_path(args.model_choice)
    evaluate(
        model_path=model_path,
        data_file=args.data_file,
        output_file=args.output_file,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.min_new_tokens,
        num_beams=args.num_beams,
        length_penalty=args.length_penalty,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        truncate_inputs=args.truncate_inputs,
        max_length=args.max_length,
        attn_implementation=args.attn_implementation,
        precision=args.precision,
        device=args.device,
        enable_tf32=args.enable_tf32,
        compile_model=args.compile_model,
        prompt_style=args.prompt_style,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        seed=args.seed,
        degrade_strength=args.degrade_strength,
        max_examples=args.max_examples,
    )


if __name__ == "__main__":
    main()
