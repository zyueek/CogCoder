from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_QWEN35_BASE = os.environ.get("QWEN35_BASE_PATH", "Qwen/Qwen3.5-4B")
DEFAULT_QWEN35_BASE_ID = "Qwen/Qwen3.5-4B"
DEFAULT_QWEN35_INSTRUCT_ID = "Qwen/Qwen3.5-4B-Instruct"

SUMMARY_PROMPTS = {
    "baseline": "Summarize the following Java code.",
    "advanced": (
        "Write a concise documentation-style summary of the following Java code. "
        "Use 1 to 3 short sentences. Focus on the main behavior, key side effect, or return value. "
        "Do not include reasoning, analysis, bullet points, code, or extra sections."
    ),
}

SUMMARY_INPUT_HEADER = "### Java Code:"
SUMMARY_OUTPUT_HEADER = "### Summary:"

_WORD_RE = re.compile(r"[A-Za-z0-9_@]+")
_ONLY_BRACES = re.compile(r"^[\s\}\]\)\>;]*$")
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z@<])")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def default_model_path(model_choice: str) -> str:
    if model_choice == "instruction":
        return DEFAULT_QWEN35_INSTRUCT_ID
    local_base = Path(DEFAULT_QWEN35_BASE)
    if local_base.exists():
        return str(local_base)
    return DEFAULT_QWEN35_BASE_ID


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


def _format_plain_prompt(input_text: str, prompt_style: str) -> str:
    instruction = SUMMARY_PROMPTS[prompt_style]
    return (
        f"### Instruction:\n{instruction}\n\n"
        f"{SUMMARY_INPUT_HEADER}\n{input_text}\n\n"
        f"{SUMMARY_OUTPUT_HEADER}\n"
    )


def _format_prompt(input_text: str, prompt_style: str, *, tokenizer=None, prompt_format: str = "plain") -> str:
    prompt = _format_plain_prompt(input_text, prompt_style)
    if prompt_format != "chat":
        return prompt
    if tokenizer is None or not hasattr(tokenizer, "apply_chat_template"):
        return prompt
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt.strip()}],
        tokenize=False,
        add_generation_prompt=True,
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


def _normalize_sentence(text: str) -> str:
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9@ ]+", "", text)
    return text.strip()


def _dedupe_sentences(sentences: List[str]) -> List[str]:
    seen: set[str] = set()
    output: List[str] = []
    for sentence in sentences:
        normalized = _normalize_sentence(sentence)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        output.append(sentence.strip())
    return output


def _trim_repetition(text: str, *, min_run: int = 6) -> str:
    if not text:
        return text
    for marker in ("\n### ", "### Instruction:", "### Java Code:", "### Summary:", "```"):
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx]
    words = [match.group(0).lower() for match in _WORD_RE.finditer(text)]
    if not words:
        return text.strip()
    run_word = None
    run_len = 0
    for word in words:
        if word == run_word:
            run_len += 1
        else:
            run_word = word
            run_len = 1
        if run_len >= min_run:
            idx = text.lower().find(word)
            if idx > 0:
                text = text[:idx]
            break
    return text.strip()


def _strip_artifacts(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("```", " ")
    text = text.replace("<pad>", " ").replace("</s>", " ")
    text = _THINK_BLOCK_RE.sub(" ", text)
    for prefix in ("Summary:", "summary:", "Answer:", "answer:"):
        if text.strip().startswith(prefix):
            text = text.strip()[len(prefix) :].lstrip()

    lower = text.lower()
    for marker in (
        "assistant",
        "the user is asking me",
        "key points:",
        "the method signature should be",
        "@jls",
        "@debug",
    ):
        idx = lower.find(marker)
        if idx > 0:
            text = text[:idx]
            lower = text.lower()

    for marker in ("### Instruction:", "### Java Code:", "### Summary:", "### Response:", "### Answer:"):
        idx = text.find(marker)
        if idx > 0:
            text = text[:idx]

    return text.strip()


def _extract_summary_core(text: str, *, max_sentences: int = 3, max_words: int = 80, max_tag_lines: int = 2) -> str:
    text = _strip_artifacts(text)
    text = _trim_repetition(text)

    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line and not _ONLY_BRACES.match(line)]
    if not lines:
        return ""

    description_lines: List[str] = []
    tag_lines: List[str] = []
    for line in lines:
        if line.startswith(("@param", "@return", "@throws")):
            tag_lines.append(line)
        else:
            description_lines.append(line)

    description = " ".join(description_lines).strip()
    description = re.sub(r"\s+", " ", description)
    description = re.sub(r"^(assistant|summary)\s*", "", description, flags=re.IGNORECASE).strip()

    sentences = _dedupe_sentences([part.strip() for part in _SENTENCE_SPLIT_RE.split(description) if part.strip()])
    selected_sentences: List[str] = []
    word_budget = 0
    for sentence in sentences:
        sentence_words = len(sentence.split())
        if selected_sentences and len(selected_sentences) >= max_sentences:
            break
        if selected_sentences and word_budget + sentence_words > max_words:
            break
        if not selected_sentences and sentence_words > max_words:
            clipped = " ".join(sentence.split()[:max_words]).strip()
            if clipped:
                selected_sentences.append(clipped)
            break
        selected_sentences.append(sentence)
        word_budget += sentence_words

    if not selected_sentences and description:
        selected_sentences.append(" ".join(description.split()[:max_words]).strip())

    cleaned_tag_lines: List[str] = []
    seen_tags: set[str] = set()
    for line in tag_lines:
        normalized = _normalize_sentence(line)
        if not normalized or normalized in seen_tags:
            continue
        seen_tags.add(normalized)
        if len(line.split()) > 24:
            line = " ".join(line.split()[:24]).strip()
        cleaned_tag_lines.append(line)
        if len(cleaned_tag_lines) >= max_tag_lines:
            break

    parts = [" ".join(selected_sentences).strip()]
    if cleaned_tag_lines and len(parts[0].split()) < 55:
        parts.extend(cleaned_tag_lines)

    output = "\n".join(part for part in parts if part).strip()
    output = re.sub(r"\s+\n", "\n", output)
    output = re.sub(r"\n{3,}", "\n\n", output)
    return output.strip()


def _candidate_quality_score(text: str) -> float:
    if not text:
        return -100.0
    words = text.split()
    word_count = len(words)
    score = 0.0
    if 10 <= word_count <= 70:
        score += 3.0
    elif word_count < 6:
        score -= 4.0
    elif word_count > 90:
        score -= 2.0

    if any(marker in text.lower() for marker in ("assistant", "<think>", "the user is asking me", "```")):
        score -= 5.0
    if text.count("\n\n") >= 1:
        score -= 0.5
    if any(token in text for token in ("{", "}", ";")):
        score -= 0.6
    if any(word in text.lower() for word in ("public static", "class ", "void ", "return new")):
        score -= 0.6
    if text.endswith((".", "!", "?", ")")):
        score += 0.3
    unique_ratio = len(set(word.lower() for word in words)) / max(1, word_count)
    score += unique_ratio
    return score


def _group_candidates(texts: List[str], num_examples: int, num_return_sequences: int) -> List[List[str]]:
    grouped: List[List[str]] = []
    cursor = 0
    for _ in range(num_examples):
        grouped.append(texts[cursor : cursor + num_return_sequences])
        cursor += num_return_sequences
    return grouped


def _choose_best_summary(candidates: List[str]) -> str:
    best_text = ""
    best_score = float("-inf")
    for candidate in candidates:
        cleaned = _extract_summary_core(candidate)
        score = _candidate_quality_score(cleaned)
        if score > best_score:
            best_score = score
            best_text = cleaned
    return best_text


def repair_existing_results(input_results: str, output_file: str) -> None:
    rows = json.loads(Path(input_results).read_text())
    repaired: List[Dict[str, Any]] = []
    for row in rows:
        repaired.append(
            {
                "input": row.get("input", ""),
                "ground_truth": row.get("ground_truth", ""),
                "generated_output": _extract_summary_core(str(row.get("generated_output", ""))),
            }
        )
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    Path(output_file).write_text(json.dumps(repaired, ensure_ascii=False, indent=2))
    logging.info("Saved repaired summaries to %s", output_file)


def evaluate(
    *,
    model_path: str,
    data_file: str,
    output_file: str,
    batch_size: int,
    max_new_tokens: int,
    min_new_tokens: int,
    num_beams: int,
    num_return_sequences: int,
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
    max_examples: Optional[int],
    prompt_format: str,
) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(enable_tf32)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high" if enable_tf32 else "highest")

    total = _count_jsonl(data_file, max_examples=max_examples)
    logging.info("Eval examples: %d (%s)", total, data_file)

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
    logging.info("Using eos_token_id=%s", eos_token_id)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    model_device = next(model.parameters()).device

    t0 = time.time()
    results_written = 0
    tokens_generated = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        out_f.write("[\n")
        is_first = True
        progress = tqdm(total=total, desc="summ-eval", unit="ex")
        for batch in _batched(_iter_jsonl(data_file, max_examples=max_examples), batch_size):
            inputs_text: List[str] = []
            kept_examples: List[tuple[str, str]] = []
            for example in batch:
                inp = example.get("code")
                gt = example.get("content")
                if not inp or not gt:
                    continue
                inputs_text.append(_format_prompt(inp, prompt_style, tokenizer=tokenizer, prompt_format=prompt_format))
                kept_examples.append((inp, gt))

            if inputs_text:
                tok_kwargs: Dict[str, Any] = {"return_tensors": "pt", "padding": True}
                if truncate_inputs:
                    tok_kwargs.update({"truncation": True, "max_length": max_length})
                tok = tokenizer(inputs_text, **tok_kwargs)
                input_ids = tok["input_ids"].to(model_device)
                attention_mask = tok["attention_mask"].to(model_device)

                with torch.inference_mode():
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        min_new_tokens=min_new_tokens,
                        num_beams=num_beams,
                        num_return_sequences=num_return_sequences,
                        early_stopping=True,
                        length_penalty=length_penalty,
                        repetition_penalty=repetition_penalty,
                        no_repeat_ngram_size=no_repeat_ngram_size,
                        do_sample=False,
                        eos_token_id=eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                expanded_input_len = input_ids.shape[1]
                gen_only = outputs[:, expanded_input_len:]
                decoded = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
                tokens_generated += int(gen_only.numel())

                grouped = _group_candidates(decoded, len(kept_examples), num_return_sequences)
                for (inp, gt), candidates in zip(kept_examples, grouped):
                    pred = _choose_best_summary(candidates)
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
        "Completed summarization eval: wrote %d examples to %s in %.2fs (%.2f ex/s, %.2f tok/s)",
        results_written,
        output_file,
        elapsed,
        results_written / elapsed,
        tokens_generated / elapsed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarization-specific evaluator for advanced ACT-R EEG runs.")
    parser.add_argument("--model-choice", choices=["base", "instruction"], default="base")
    parser.add_argument("--model-path", help="Override the default model path.")
    parser.add_argument("--data-file", default="data/actr_scanpath_eeg/summarization_valid_final_eeg_scanpath.jsonl")
    parser.add_argument("--output-file", default="outputs/qwen35_base_scanpath_eeg_summarization_eval_advanced.json")
    parser.add_argument("--existing-results", help="Optional existing eval JSON array to clean without rerunning the model.")
    parser.add_argument("--prompt-style", choices=["baseline", "advanced"], default="advanced")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--min-new-tokens", type=int, default=8)
    parser.add_argument("--num-beams", type=int, default=4)
    parser.add_argument("--num-return-sequences", type=int, default=2)
    parser.add_argument("--length-penalty", type=float, default=0.85)
    parser.add_argument("--repetition-penalty", type=float, default=1.08)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=4)
    parser.add_argument("--truncate-inputs", action="store_true")
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--attn-implementation", choices=["auto", "eager", "sdpa"], default="auto")
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--enable-tf32", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--prompt-format", choices=["plain", "chat"], default=os.environ.get("PROMPT_FORMAT", "plain"))
    args = parser.parse_args()

    if args.num_return_sequences > args.num_beams:
        raise ValueError("--num-return-sequences must be <= --num-beams")

    if args.existing_results:
        repair_existing_results(args.existing_results, args.output_file)
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
        num_return_sequences=args.num_return_sequences,
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
        max_examples=args.max_examples,
        prompt_format=args.prompt_format,
    )


if __name__ == "__main__":
    main()
