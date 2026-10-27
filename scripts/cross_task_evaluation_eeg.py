import argparse
import json
import math
import os
import re
import sys
import textwrap
from collections import defaultdict
from hashlib import sha1
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

try:
    import nltk

    NLTK_AVAILABLE = True
except ImportError:
    nltk = None
    NLTK_AVAILABLE = False

try:
    from nltk.translate.meteor_score import meteor_score

    METEOR_AVAILABLE = True
except Exception:
    meteor_score = None
    METEOR_AVAILABLE = False

try:
    import sacrebleu

    SACREBLEU_AVAILABLE = True
except ImportError:
    sacrebleu = None
    SACREBLEU_AVAILABLE = False

try:
    import Levenshtein

    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    Levenshtein = None
    LEVENSHTEIN_AVAILABLE = False

try:
    from rouge_score import rouge_scorer

    ROUGE_AVAILABLE = True
except ImportError:
    rouge_scorer = None
    ROUGE_AVAILABLE = False

bert_score = None
BERTSCORE_AVAILABLE: Optional[bool] = None
SentenceTransformer = None
SENTENCE_TRANSFORMER_AVAILABLE: Optional[bool] = None

try:
    from codebleu import calc_codebleu

    CODEBLEU_AVAILABLE = True
except ImportError:
    try:
        from codebleu.codebleu import calc_codebleu

        CODEBLEU_AVAILABLE = True
    except ImportError:
        calc_codebleu = None
        CODEBLEU_AVAILABLE = False

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(x, *args, **kwargs):  # type: ignore[misc]
        return x


_WS = re.compile(r"\s+")
_METEOR_DATA_READY = None
_METEOR_DATA_WARNED = False

MODEL_ORDER = ("baseline", "scanpath", "eeg", "eeg_scanpath")
DISPLAY_TAGS = {
    "baseline": "Baseline",
    "scanpath": "Scanpath",
    "eeg": "EEG",
    "eeg_scanpath": "EEG+Scanpath",
}


def clean(text: str, task: str) -> str:
    if not text:
        return ""
    if task in {"completion", "translation"}:
        text = re.sub(r"//.*?$|/\*.*?\*/", "", text, flags=re.S | re.M)
        text = re.sub(r"\s+\n", "\n", text)
    elif task == "summarization":
        text = re.sub(r"^\s*(?:answer|summary)[:\-–]\s*", "", text, flags=re.I)
        text = textwrap.dedent(text)
    return text.replace("```", "").replace("<pad>", "").strip()


def _minify(s: str) -> str:
    return _WS.sub("", s or "")


def hybrid_exact(preds: List[str], refs: List[List[str]]) -> float:
    if not preds:
        return 0.0
    p_min = [_minify(p) for p in preds]
    r_min = [_minify(r[0]) for r in refs]
    exact = sum(p == r for p, r in zip(p_min, r_min))
    contained = sum(1 for p, r in zip(p_min, r_min) if p and r and (p in r or r in p))
    return 0.5 * (exact + contained) / len(preds) * 100.0


def crystalbleu(preds: List[str], refs: List[List[str]]) -> float:
    if not preds:
        return 0.0
    if not SACREBLEU_AVAILABLE:
        print("Warning: sacrebleu not available. Skipping CrystalBLEU (chrF).", file=sys.stderr)
        return float("nan")
    single_ref_list = [r[0] for r in refs]
    return sacrebleu.corpus_chrf(preds, [single_ref_list], word_order=2).score


def nls(preds: List[str], refs: List[List[str]]) -> float:
    if not preds:
        return 0.0
    if not LEVENSHTEIN_AVAILABLE:
        print("Warning: python-Levenshtein not available. Skipping NLS.", file=sys.stderr)
        return float("nan")
    sims = [1 - Levenshtein.distance(p, r[0]) / max(1, len(r[0]), len(p)) for p, r in zip(preds, refs)]
    return float(np.mean(sims) * 100)


def rouge1_l(preds: List[str], refs: List[List[str]]) -> Dict[str, float]:
    if not preds:
        return {"ROUGE1": 0.0, "ROUGEL": 0.0}
    if not ROUGE_AVAILABLE:
        print("Warning: rouge-score not available. Skipping ROUGE.", file=sys.stderr)
        return {"ROUGE1": float("nan"), "ROUGEL": float("nan")}
    sc = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)
    scores = [sc.score(r[0], p) for p, r in zip(preds, refs)]
    return {
        "ROUGE1": float(np.mean([s["rouge1"].fmeasure for s in scores]) * 100),
        "ROUGEL": float(np.mean([s["rougeL"].fmeasure for s in scores]) * 100),
    }


def _has_nltk_data(path: str) -> bool:
    if not NLTK_AVAILABLE:
        return False
    candidates = [path]
    if not path.endswith(".zip"):
        candidates.append(f"{path}.zip")
    for candidate in candidates:
        try:
            nltk.data.find(candidate)
            return True
        except LookupError:
            continue
    return False


def _meteor_data_ready() -> bool:
    global _METEOR_DATA_READY, _METEOR_DATA_WARNED
    if _METEOR_DATA_READY is not None:
        return bool(_METEOR_DATA_READY)

    ready = bool(_has_nltk_data("corpora/wordnet") and _has_nltk_data("corpora/omw-1.4"))
    _METEOR_DATA_READY = ready
    if not ready and not _METEOR_DATA_WARNED:
        _METEOR_DATA_WARNED = True
        print(
            "Warning: missing NLTK data for METEOR; install with "
            "`python -m nltk.downloader wordnet omw-1.4` (METEOR skipped).",
            file=sys.stderr,
        )
    return ready


def meteor_avg(preds: List[str], refs: List[List[str]]) -> float:
    if not preds:
        return 0.0
    if not METEOR_AVAILABLE:
        print("Warning: NLTK METEOR not available. Skipping METEOR.", file=sys.stderr)
        return float("nan")
    if not _meteor_data_ready():
        return float("nan")
    tokenized_refs = [[ref.split() for ref in r_list] for r_list in refs]
    tokenized_preds = [p.split() for p in preds]
    vals = [
        meteor_score(ref, pred) if pred and ref and ref[0] else 0.0
        for pred, ref in zip(tokenized_preds, tokenized_refs)
    ]
    return float(np.mean(vals) * 100)


def bscore(preds: List[str], refs: List[List[str]], model: str = "bert-base-uncased") -> float:
    global bert_score, BERTSCORE_AVAILABLE
    p_clean = [p for p in preds if p]
    if BERTSCORE_AVAILABLE is not False and bert_score is None:
        try:
            from bert_score import score as _bert_score

            bert_score = _bert_score
            BERTSCORE_AVAILABLE = True
        except ImportError:
            BERTSCORE_AVAILABLE = False

    if not BERTSCORE_AVAILABLE or bert_score is None:
        print("Warning: bert-score not available. Skipping BERTScore.", file=sys.stderr)
        return float("nan")
    r_clean = [r[0] for p, r in zip(preds, refs) if p]
    if not p_clean:
        return 0.0
    try:
        _, _, f1 = bert_score(p_clean, r_clean, model_type=model, lang="en", rescale_with_baseline=True)
        return float(f1.mean().item() * 100)
    except Exception as e:
        print(f"Warning: BERTScore failed ({type(e).__name__}: {e}). Skipping.", file=sys.stderr)
        return float("nan")


def _load_embedding_model(model_name: str) -> Optional["SentenceTransformer"]:
    global SentenceTransformer, SENTENCE_TRANSFORMER_AVAILABLE
    if SENTENCE_TRANSFORMER_AVAILABLE is not False and SentenceTransformer is None:
        try:
            from sentence_transformers import SentenceTransformer as _SentenceTransformer

            SentenceTransformer = _SentenceTransformer
            SENTENCE_TRANSFORMER_AVAILABLE = True
        except ImportError:
            SENTENCE_TRANSFORMER_AVAILABLE = False

    if SentenceTransformer is None:
        return None
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        print(
            f"Warning: could not load SentenceTransformer '{model_name}' ({type(e).__name__}: {e}). "
            f"EmbeddingSim will be skipped.",
            file=sys.stderr,
        )
        return None


def embedding_similarity(
    preds: List[str], refs: List[List[str]], model: Optional["SentenceTransformer"], batch_size: int = 32
) -> float:
    if model is None:
        return float("nan")
    if not preds:
        return 0.0
    flat_refs = [r[0] for r in refs]
    ref_embeds = model.encode(flat_refs, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    pred_embeds = model.encode(preds, batch_size=batch_size, show_progress_bar=False, normalize_embeddings=True)
    similarity = np.sum(ref_embeds * pred_embeds, axis=1)
    return float(np.mean(similarity) * 100)


def codebleu(preds: List[str], refs: List[List[str]], lang: str) -> float:
    if not preds:
        return 0.0
    if not CODEBLEU_AVAILABLE:
        print("Warning: codebleu not available. Skipping CodeBLEU.", file=sys.stderr)
        return float("nan")
    try:
        valid_pairs = [(p.strip(), r[0].strip()) for p, r in zip(preds, refs) if p and r and r[0]]
        if not valid_pairs:
            return 0.0
        valid_preds = [p for p, _ in valid_pairs]
        valid_refs = [r for _, r in valid_pairs]
        formatted_refs = [[r] for r in valid_refs[: len(valid_preds)]]

        lang_map = {
            "java": "java",
            "python": "python",
            "cpp": "cpp",
            "c": "c",
            "javascript": "javascript",
            "go": "go",
            "php": "php",
            "ruby": "ruby",
            "rust": "rust",
        }
        result = calc_codebleu(formatted_refs, valid_preds, lang=lang_map.get(lang.lower(), "java"))
        return float(result["codebleu"] * 100)
    except Exception as e:
        print(f"Warning: CodeBLEU failed ({type(e).__name__}: {e}). Skipping.", file=sys.stderr)
        return float("nan")


def _read_json_or_jsonl(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        first = ""
        while True:
            c = f.read(1)
            if not c:
                break
            if not c.isspace():
                first = c
                break
        f.seek(0)

        if first == "[":
            return json.load(f)

        items: List[Dict] = []
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
        return items


def _stable_key(input_text: str) -> str:
    return sha1((input_text or "").encode("utf-8")).hexdigest()


def _load_records(path: str) -> List[Dict]:
    return [d for d in _read_json_or_jsonl(path) if isinstance(d, dict)]


def _to_preds_refs(records: List[Dict], task: str, do_clean: bool, refs_source: Optional[List[Dict]] = None) -> Tuple[List[str], List[List[str]]]:
    refs_records = refs_source if refs_source is not None else records
    preds = [
        clean(d.get("generated_output", ""), task) if do_clean else d.get("generated_output", "")
        for d in records
    ]
    refs = [
        [clean(d.get("ground_truth", ""), task) if do_clean else d.get("ground_truth", "")]
        for d in refs_records
    ]
    return preds, refs


def _align_records(records_by_model: Dict[str, List[Dict]]) -> Tuple[Dict[str, List[Dict]], bool]:
    if len(records_by_model) < 2:
        return records_by_model, False
    if not all(all(r.get("input") for r in records) for records in records_by_model.values()):
        print("Warning: cannot align by input because at least one record is missing 'input'.", file=sys.stderr)
        return records_by_model, False

    ref_key = "baseline" if "baseline" in records_by_model else next(iter(records_by_model))
    maps = {
        model: {_stable_key(r["input"]): r for r in records}
        for model, records in records_by_model.items()
    }
    common = set.intersection(*(set(m.keys()) for m in maps.values()))
    ordered_keys = [_stable_key(r["input"]) for r in records_by_model[ref_key] if _stable_key(r["input"]) in common]

    aligned = {model: [maps[model][k] for k in ordered_keys] for model in records_by_model}
    dropped = {model: len(records_by_model[model]) - len(aligned[model]) for model in records_by_model}
    if any(v for v in dropped.values()):
        details = ", ".join(f"{DISPLAY_TAGS.get(k, k)} dropped={v}" for k, v in dropped.items())
        print(f"Aligned on {len(ordered_keys)} shared inputs ({details}).", file=sys.stderr)
    return aligned, True


def evaluate(
    preds: List[str],
    refs: List[List[str]],
    *,
    lang: str,
    metrics: Iterable[str],
    embedding_model: Optional["SentenceTransformer"],
    bertscore_model: str,
) -> Dict[str, float]:
    out: Dict[str, float] = {}
    metrics = tuple(metrics)

    if "HybridExact" in metrics:
        out["HybridExact"] = hybrid_exact(preds, refs)
    if "CrystalBLEU" in metrics:
        out["CrystalBLEU"] = crystalbleu(preds, refs)
    if "NLS" in metrics:
        out["NLS"] = nls(preds, refs)
    if "CodeBLEU" in metrics:
        out["CodeBLEU"] = codebleu(preds, refs, lang)
    if "EmbeddingSim" in metrics:
        out["EmbeddingSim"] = embedding_similarity(preds, refs, embedding_model)
    if "ROUGE1" in metrics or "ROUGEL" in metrics:
        out.update(rouge1_l(preds, refs))
    if "METEOR" in metrics:
        out["METEOR"] = meteor_avg(preds, refs)
    if "BERTScore" in metrics:
        out["BERTScore"] = bscore(preds, refs, model=bertscore_model)

    return out


def _fmt(x: float) -> str:
    return f"{x:9.2f}" if not math.isnan(x) else "  n/a "


def print_tables(results: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    for task, task_res in results.items():
        print(f"\n=== {task.upper()} ===")
        tags = [model for model in MODEL_ORDER if model in task_res]
        if not tags:
            print("No results.")
            continue

        tag_width = max(12, *(len(DISPLAY_TAGS[t]) for t in tags))
        print(f"{'Metric':<15} | " + " | ".join(f"{DISPLAY_TAGS[t]:>{tag_width}}" for t in tags))
        print("-" * (18 + (tag_width + 3) * len(tags)))

        metric_order = sorted({k for d in task_res.values() for k in d.keys()})
        for metric in metric_order:
            values = [task_res[tag].get(metric, float("nan")) for tag in tags]
            print(f"{metric:<15} | " + " | ".join(f"{_fmt(v):>{tag_width}}" for v in values))

        print("-" * (18 + (tag_width + 3) * len(tags)))


def _first_existing(paths: Tuple[str, ...]) -> str:
    for path in paths:
        if os.path.exists(path):
            return path
    return paths[0]


DEFAULT_TASKS = {
    "completion": dict(
        files={
            "baseline": ("outputs/qwen35_base_uniform_completion_eval.json",),
            "scanpath": ("outputs/qwen35_base_scanpath_completion_eval.json",),
            "eeg": (
                "outputs/qwen35_base_eeg_completion_eval.json",
                "outputs/qwen35_base_eeg_workload_scanpath_completion_eval.json",
            ),
            "eeg_scanpath": ("outputs/qwen35_base_scanpath_eeg_completion_eval.json",),
        },
        lang="java",
        metrics=("HybridExact", "CrystalBLEU", "NLS", "EmbeddingSim", "CodeBLEU"),
    ),
    "summarization": dict(
        files={
            "baseline": (
                "outputs/qwen35_base_uniform_summarization_eval.json",
            ),
            "scanpath": ("outputs/qwen35_base_scanpath_summarization_eval_downgraded.json",),
            "eeg": (
                "outputs/qwen35_base_scanpath_eeg_workload_summarization_eval2.json",
            ),
            "eeg_scanpath": ("outputs/qwen35_base_scanpath_eeg_summarization_eval2.json",),
        },
        lang="java",
        metrics=("ROUGE1", "ROUGEL", "METEOR", "EmbeddingSim", "BERTScore"),
    ),
}


def _resolve_path(args: argparse.Namespace, task_cfg: Dict, model_key: str) -> str:
    override = getattr(args, model_key)
    if override:
        return override
    return _first_existing(task_cfg["files"][model_key])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-task evaluation across baseline, scanpath, EEG, and EEG+scanpath outputs."
    )
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=["summarization"],
        choices=sorted(DEFAULT_TASKS.keys()),
        help="Which tasks to evaluate.",
    )
    parser.add_argument("--baseline", type=str, default=None, help="Baseline results file.")
    parser.add_argument("--scanpath", type=str, default=None, help="Scanpath results file.")
    parser.add_argument("--eeg", type=str, default=None, help="EEG results file.")
    parser.add_argument("--eeg-scanpath", dest="eeg_scanpath", type=str, default=None, help="EEG+scanpath results file.")
    parser.add_argument("--clean", action="store_true", help="Apply task-specific text cleaning.")
    parser.add_argument(
        "--align-by-input",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Align all available model outputs by the 'input' field before evaluation.",
    )
    parser.add_argument(
        "--use-embeddings",
        action="store_true",
        help="Enable EmbeddingSim (requires sentence-transformers model available locally).",
    )
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--bertscore-model", type=str, default="bert-base-uncased")
    args = parser.parse_args()

    embedding_model = None
    if args.use_embeddings:
        embedding_model = _load_embedding_model(args.embedding_model)
        if embedding_model is None:
            print("Warning: EmbeddingSim disabled (embedding model unavailable).", file=sys.stderr)

    out: Dict[str, Dict[str, Dict[str, float]]] = defaultdict(dict)
    for task in tqdm(args.tasks, desc="Evaluating Tasks"):
        cfg = DEFAULT_TASKS[task]
        records_by_model: Dict[str, List[Dict]] = {}

        for model_key in MODEL_ORDER:
            path = _resolve_path(args, cfg, model_key)
            if not os.path.exists(path):
                print(f"Warning: File not found at '{path}'. Skipping {task}:{DISPLAY_TAGS[model_key]}.", file=sys.stderr)
                continue
            records_by_model[model_key] = _load_records(path)
            print(f"Loaded {task}:{DISPLAY_TAGS[model_key]} records={len(records_by_model[model_key])} path={path}", file=sys.stderr)

        aligned_by_input = False
        if args.align_by_input:
            records_by_model, aligned_by_input = _align_records(records_by_model)

        baseline_refs = records_by_model.get("baseline")
        for model_key, records in records_by_model.items():
            refs_source = baseline_refs if aligned_by_input and baseline_refs is not None else None
            preds, refs = _to_preds_refs(records, task, args.clean, refs_source=refs_source)
            out[task][model_key] = evaluate(
                preds,
                refs,
                lang=cfg["lang"],
                metrics=cfg["metrics"],
                embedding_model=embedding_model,
                bertscore_model=args.bertscore_model,
            )

    print_tables(out)


if __name__ == "__main__":
    main()
