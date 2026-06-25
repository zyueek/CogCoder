# CogCoder Reviewer Artifact Scripts

## Directory Layout

```text
CogCoder/
  README.md
  result/
    completion/
      qwen35_completion_metrics_summary.json
    summarization/
      qwen35_summarization_metrics_summary.json
    fault_localization/
      baseline_uniform_summary.json
      scancoder_scanpath_summary.json
      cogcoder_workload_summary.json
      cogcoder_hybrid_summary.json
    manifests/
      result_manifest.json
  scripts/
    preprocess_novice_vs_expert_raw.py
    build_eeg_scanpath_dataset.py
    build_eeg_bug_scanpath_dataset.py
    train_qwen_scanpath_large.py
    eval_qwen_scanpath_fast_accurate.py
    eval_qwen_scanpath_summarization_advanced.py
    eval_qwen_bug_scanpath.py
```

## What Each File Shows

| File | Reviewer-facing purpose |
|---|---|
| `scripts/preprocess_novice_vs_expert_raw.py` | Converts NoviceVsExpert participant records into token-level attention, revisit, and processing-cost labels. This corresponds to the paper's raw human-reading preprocessing stage. |
| `code_gen_prev/build_structural_neuro_dataset.py` | Detects Java structural regions such as function definitions, parameter lists, assignments, calls, branches, loops, returns, comparisons, constants, and container accesses. It aggregates token labels into structure-level priors. |
| `scripts/build_eeg_scanpath_dataset.py` | Projects structure priors onto completion and summarization examples. It preserves scanpath fields and writes workload/salience fields such as `eeg_attention_sequence`, `eeg_revisit_sequence`, `eeg_salience_sequence`, and `eeg_region_types`. |
| `scripts/build_eeg_bug_scanpath_dataset.py` | Performs the corresponding projection for fault localization and vulnerability-style examples. It reuses available scanpaths, calls the ACT-R simulator when available, or falls back to structural scanpath heuristics. |
| `scripts/train_qwen_scanpath_large.py` | Main algorithm script. Implements uniform baseline, ScanCoder scanpath weighting, workload-only weighting, and CogCoder hybrid weighting in a single weighted causal-LM trainer. |
| `scripts/eval_qwen_scanpath_fast_accurate.py` | Completion/easy generation evaluator. |
| `scripts/eval_qwen_scanpath_summarization_advanced.py` | Summarization evaluator and output cleaner. |
| `scripts/eval_qwen_bug_scanpath.py` | Fault-localization and vulnerability-generation evaluator. |
| `eeg_qwen/common.py`, `eeg_qwen/datasets.py` | Minimal helper functions imported by the algorithm scripts. |

## Included Evaluation Results

The artifact includes several compact result files under `result/`. These are intended for reviewer inspection of the reported evaluation format and headline metric values without shipping raw prediction dumps that may contain third-party source-code comments, generated text, or other accidental identifying metadata.

| File | Contents |
|---|---|
| `result/completion/qwen35_completion_metrics_summary.json` | Aggregate completion metrics for Baseline, ScanCoder, CogCoder-Workload, and CogCoder-Hybrid. |
| `result/summarization/qwen35_summarization_metrics_summary.json` | Aggregate summarization metrics for Baseline, ScanCoder, CogCoder-Workload, and CogCoder-Hybrid. |
| `result/fault_localization/baseline_uniform_summary.json` | Defects4J fault-localization summary for the uniform baseline. |
| `result/fault_localization/scancoder_scanpath_summary.json` | Defects4J fault-localization summary for the ScanCoder/scanpath baseline. |
| `result/fault_localization/cogcoder_workload_summary.json` | Defects4J fault-localization summary for the workload-only CogCoder run. |
| `result/fault_localization/cogcoder_hybrid_summary.json` | Defects4J fault-localization summary for the hybrid CogCoder run. |
| `result/manifests/result_manifest.json` | Result bundle manifest and anonymity-check note. |

The full raw prediction files remain outside this reviewer slice. They are larger and include source snippets/model generations, so they are intentionally not copied into this anonymized artifact package.

## Main Algorithm Entry Point

The main file reviewers should inspect first is:

```text
scripts/train_qwen_scanpath_large.py
```

The key implementation points are:

- `calculate_weight_profile(...)`
  - `uniform`: baseline supervised fine-tuning with all target weights equal to 1.
  - `scanpath`: ScanCoder-style scanpath loss weighting.
  - `scanpath_weak`: weak scanpath ablation.
  - `eeg` / `eeg_workload`: workload-only CogCoder variant.
  - `scanpath_eeg`: hybrid CogCoder variant combining scanpath and workload salience.
- `build_eeg_priority(...)`
  - normalizes salience, attention, revisit, and complexity sequences into a bounded line priority.
- `WeightedLossTrainer.compute_loss(...)`
  - computes per-token cross-entropy,
  - multiplies active target losses by line-expanded weights,
  - normalizes by the sum of active weights.

The weighted loss used by the implementation is:

```text
L = sum_t w_t CE(y_t, p_t) / (sum_t w_t + epsilon)
```

Prompt tokens are masked with label `-100`, so only supervised target tokens contribute to the loss.

## Baseline and ScanCoder in This Artifact

The paper compares CogCoder against a uniform baseline and a scanpath-guided baseline.

In code:

| Paper method | Code argument | Behavior |
|---|---|---|
| Baseline | `--weight-mode uniform` | every supervised target token has weight `1.0` |
| ScanCoder | `--weight-mode scanpath` | lines selected by `scanpath_mask` receive weight `3 + 1 / scanpath_order`; all other lines receive `1.0` |
| CogCoder-Workload | `--weight-mode eeg` or `--weight-mode eeg_workload` | target weights are derived from projected workload/salience fields |
| CogCoder-Hybrid | `--weight-mode scanpath_eeg` | scanpath is preserved as the backbone and workload salience modulates or rescues important lines |

There is no separate `ScanCoder` architecture in this codebase. The ScanCoder baseline is implemented as a scanpath-specific loss-weighting mode over the same Qwen backbone, prompts, data rows, and trainer.

## Data Fields Expected by the Scripts

The projected downstream JSONL rows use line-aligned fields:

| Field | Meaning |
|---|---|
| `code` | source code shown to the model |
| `content` | target completion, summary, label, or suspicious line number(s) |
| `code_tokens` | line-like units used for weight assignment |
| `scanpath_mask` | scanpath-selected lines |
| `scanpath_order_sequence` | scanpath reading order |
| `scanpath_complexity_sequence` | scanpath complexity signal |
| `mask` | workload-selected lines |
| `order_sequence` | workload salience order |
| `complexity_sequence` | projected workload/complexity |
| `eeg_attention_sequence` | projected attention prior |
| `eeg_revisit_sequence` | projected revisit prior |
| `eeg_salience_sequence` | projected line salience |
| `eeg_region_types` | Java structure categories assigned to each line |

The artifact slice does not include large training data, checkpoints, raw EEG files, or model weights.

## Paper-to-Code Mapping

| Paper stage | Script to inspect |
|---|---|
| Raw neurophysiological/gaze preprocessing | `scripts/preprocess_novice_vs_expert_raw.py` |
| Token labels to program-structure priors | `code_gen_prev/build_structural_neuro_dataset.py` |
| Downstream line-salience projection | `scripts/build_eeg_scanpath_dataset.py`, `scripts/build_eeg_bug_scanpath_dataset.py` |
| Baseline, ScanCoder, CogCoder-Workload, CogCoder-Hybrid training | `scripts/train_qwen_scanpath_large.py` |
| Completion evaluation | `scripts/eval_qwen_scanpath_fast_accurate.py` |
| Summarization evaluation | `scripts/eval_qwen_scanpath_summarization_advanced.py` |
| Fault-localization evaluation | `scripts/eval_qwen_bug_scanpath.py` |

## Example Commands

These commands are illustrative. They require the full repository data and model environment, not only this lightweight artifact directory.

Run from the original repository root or set:

```bash
export PYTHONPATH=/path/to/CogCoder:/path/to/CogCoder/scripts
```

Completion baseline:

```bash
python scripts/train_qwen_scanpath_large.py \
  --task completion \
  --train-file data/actr_scanpath_eeg/completion_train_final_eeg_scanpath.jsonl \
  --valid-file data/actr_scanpath_eeg/completion_valid_final_eeg_scanpath.jsonl \
  --output-dir outputs/qwen35_base_uniform_completion_large \
  --weight-mode uniform
```

Completion ScanCoder:

```bash
python scripts/train_qwen_scanpath_large.py \
  --task completion \
  --train-file data/actr_scanpath_eeg/completion_train_final_eeg_scanpath.jsonl \
  --valid-file data/actr_scanpath_eeg/completion_valid_final_eeg_scanpath.jsonl \
  --output-dir outputs/qwen35_base_scanpath_completion_large \
  --weight-mode scanpath
```

Completion CogCoder-Hybrid:

```bash
python scripts/train_qwen_scanpath_large.py \
  --task completion \
  --train-file data/actr_scanpath_eeg/completion_train_final_eeg_scanpath.jsonl \
  --valid-file data/actr_scanpath_eeg/completion_valid_final_eeg_scanpath.jsonl \
  --output-dir outputs/qwen35_base_scanpath_eeg_completion_large \
  --weight-mode scanpath_eeg
```

Fault-localization evaluation:

```bash
python scripts/eval_qwen_bug_scanpath.py \
  --task fault_localization \
  --model-path outputs/defects4j_qwen35_base_scanpath_eeg_faultloc_full \
  --data-file data/defects4j_faultloc_full_eeg_all/fault_localization_test_final_eeg_scanpath_actr.jsonl \
  --output-dir outputs/defects4j_qwen35_base_scanpath_eeg_faultloc_full_eval \
  --prompt-weight-mode scanpath_eeg
```

## Reviewer Notes

- This directory is meant for code inspection, not one-command reproducibility.
- The copied scripts preserve the original implementation logic and import paths as much as possible.
- Some scripts expect the full repository layout, external datasets, ACT-R scanpath files, Defects4J data, or Hugging Face model checkpoints.
- If reviewing the main method, start with `scripts/train_qwen_scanpath_large.py`, then inspect the two dataset builders.
- If reviewing data provenance, inspect `scripts/preprocess_novice_vs_expert_raw.py` and `code_gen_prev/build_structural_neuro_dataset.py`.
- Local absolute paths in the copied artifact files have been replaced with placeholders such as `<USER_HOME>` and `<REPO_ROOT>`.
- An anonymity scan was run over `CogCoder/` for local usernames, author markers, email-like markers, institution markers, and absolute home paths.
