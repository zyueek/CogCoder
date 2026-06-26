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

