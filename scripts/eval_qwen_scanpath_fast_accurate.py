from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


TASK_PROMPTS = {
    "translation": {
        "instruction": "Translate the following Java code to C#.",
        "input_header": "### Java Code:",
        "output_header": "### C# Code:",
        "input_key": "code",
    },
    "summarization": {
        "instruction": "Summarize the following Java code.",
        "input_header": "### Java Code:",
        "output_header": "### Summary:",
        "input_key": "code",
    },
    "completion": {
        "instruction": "Complete the following Java code.",
        "input_header": "### Java Code:",
        "output_header": "### Completion:",
        "input_key": "code",
    },
}

DEFAULT_QWEN35_BASE = os.environ.get("QWEN35_BASE_PATH", "Qwen/Qwen3.5-4B")
DEFAULT_QWEN35_BASE_ID = "Qwen/Qwen3.5-4B"
DEFAULT_QWEN35_INSTRUCT_ID = "Qwen/Qwen3.5-4B-Instruct"

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


def _iter_jsonl(path: str, max_examples: Optional[int] = None) -> Iterator[Dict]:
    with open(path, "r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if max_examples is not None and index >= max_examples:
                break
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _batched(iterable: Iterable[Dict], batch_size: int) -> Iterator[List[Dict]]:
    batch: List[Dict] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def _format_plain_prompt(task: str, input_text: str) -> str:
    cfg = TASK_PROMPTS[task]
    return (
        f"### Instruction:\n{cfg['instruction']}\n\n"
        f"{cfg['input_header']}\n{input_text}\n\n"
        f"{cfg['output_header']}\n"
    )


def _format_prompt(task: str, input_text: str, *, tokenizer=None, prompt_format: str = "plain") -> str:
    prompt = _format_plain_prompt(task, input_text)
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


def _write_json_array_item(out_f, obj: Dict, is_first: bool) -> bool:
    if not is_first:
        out_f.write(",\n")
    out_f.write(json.dumps(obj, ensure_ascii=False))
    return False


def evaluate(
    *,
    model_path: str,
    task: str,
    data_file: str,
    output_file: str,
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
    prompt_format: str,
) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(enable_tf32)
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high" if enable_tf32 else "highest")

    total = _count_jsonl(data_file, max_examples=max_examples)
    logging.info(f"Eval examples: {total} ({data_file})")

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
    eos_token_id = model.config.eos_token_id or tokenizer.eos_token_id

    if compile_model and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[assignment]
            logging.info("torch.compile enabled")
        except Exception as exc:
            logging.warning(f"torch.compile failed ({type(exc).__name__}: {exc}); continuing without compile.")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    model_device = next(model.parameters()).device
    cfg = TASK_PROMPTS[task]

    t0 = time.time()
    results_written = 0
    tokens_generated = 0

    with open(output_file, "w", encoding="utf-8") as out_f:
        out_f.write("[\n")
        is_first = True
        progress = tqdm(total=total, desc="eval", unit="ex")
        for batch in _batched(_iter_jsonl(data_file, max_examples=max_examples), batch_size):
            inputs_text: List[str] = []
            kept_examples: List[tuple[str, str]] = []
            for ex in batch:
                inp = ex.get(cfg["input_key"])
                gt = ex.get("content")
                if not inp or not gt:
                    continue
                inputs_text.append(_format_prompt(task, inp, tokenizer=tokenizer, prompt_format=prompt_format))
                kept_examples.append((inp, gt))

            if inputs_text:
                tok_kwargs: dict[str, Any] = {"return_tensors": "pt", "padding": True}
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
                        do_sample=False,
                        eos_token_id=eos_token_id,
                        pad_token_id=tokenizer.pad_token_id,
                    )

                gen_only = outputs[:, input_ids.shape[1] :]
                gen_texts = tokenizer.batch_decode(gen_only, skip_special_tokens=True)
                tokens_generated += int(gen_only.numel())
                for (inp, gt), pred in zip(kept_examples, gen_texts):
                    pred = pred.split(cfg["output_header"])[-1].strip()
                    is_first = _write_json_array_item(
                        out_f,
                        {"input": inp, "ground_truth": gt, "generated_output": pred},
                        is_first,
                    )
                    results_written += 1

            progress.update(len(batch))
        progress.close()
        out_f.write("\n]\n")

    elapsed = max(time.time() - t0, 1e-6)
    logging.info(
        "Completed eval: wrote %d examples to %s in %.2fs (%.2f ex/s, %.2f tok/s)",
        results_written,
        output_file,
        elapsed,
        results_written / elapsed,
        tokens_generated / elapsed,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen 3.5 4B on the ACT-R-style Java benchmark.")
    parser.add_argument("--task", choices=sorted(TASK_PROMPTS.keys()), default="completion")
    parser.add_argument("--model-choice", choices=["base", "instruction"], default="base")
    parser.add_argument("--model-path", help="Override the default model path.")
    parser.add_argument("--data-file", default="data/actr_scanpath_eeg/completion_valid_final_eeg_scanpath.jsonl")
    parser.add_argument("--output-file", default="outputs/qwen35_base_eeg_completion_eval.json")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--truncate-inputs", action="store_true")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--attn-implementation", choices=["auto", "eager", "sdpa"], default="auto")
    parser.add_argument("--precision", choices=["auto", "bf16", "fp16", "fp32"], default="auto")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--enable-tf32", action="store_true")
    parser.add_argument("--compile-model", action="store_true")
    parser.add_argument("--max-examples", type=int)
    parser.add_argument("--prompt-format", choices=["plain", "chat"], default=os.environ.get("PROMPT_FORMAT", "plain"))
    args = parser.parse_args()

    model_path = args.model_path or default_model_path(args.model_choice)
    evaluate(
        model_path=model_path,
        task=args.task,
        data_file=args.data_file,
        output_file=args.output_file,
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
        prompt_format=args.prompt_format,
    )


if __name__ == "__main__":
    main()
