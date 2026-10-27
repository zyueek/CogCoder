# CogCoder

Core implementation and paper-reported results for CogCoder: human code-reading records are converted into program-structure workload priors, projected onto Java code, and used to weight supervised fine-tuning.

## Core implementation

| Stage | Script |
|---|---|
| Human-reading preprocessing | `scripts/preprocess_novice_vs_expert_raw.py` |
| Program-structure workload priors | `scripts/build_structural_neuro_dataset.py` |
| Downstream workload projection | `scripts/build_eeg_scanpath_dataset.py`, `scripts/build_eeg_bug_scanpath_dataset.py` |
| Weighted fine-tuning | `scripts/train_qwen_scanpath_large.py` |
| Completion and summarization generation | `scripts/eval_qwen_scanpath_fast_accurate.py`, `scripts/eval_qwen_scanpath_summarization_advanced.py` |
| Cross-task scoring | `scripts/cross_task_evaluation_eeg.py` |
| Fault localization and fusion | `scripts/eval_qwen_bug_scanpath.py`, `scripts/fuse_faultloc_predictions.py`, `scripts/analyze_faultloc_metrics.py` |
| Checkpoint adaptation | `scripts/plot_task_weight_heatmaps.py`, `scripts/plot_checkpoint_layer_group_metrics.py` |

The trainer implements `uniform`, `scanpath`, `eeg` / `eeg_workload`, and `scanpath_eeg` weighting. The supervised loss is `sum(weight * token_loss) / sum(weight)`; prompt tokens are masked.

## Paper-reported results

### Table I: completion and summarization

| Task | Metric | Baseline | ScanCoder | CogCoder-Workload | CogCoder-Hybrid |
|---|---|---:|---:|---:|---:|
| Completion | CodeBLEU | 71.82 | 69.89 | 78.28 | 94.81 |
| Completion | CrystalBLEU | 62.50 | 63.25 | 80.49 | 96.22 |
| Completion | HybridExact | 46.27 | 45.77 | 64.38 | 93.88 |
| Summarization | BERTScore | 40.39 | 45.91 | 48.73 | 48.26 |
| Summarization | ROUGE-1 | 31.84 | 39.90 | 43.80 | 42.95 |
| Summarization | ROUGE-L | 27.34 | 34.15 | 37.37 | 36.69 |

For summarization, the ScanCoder column uses the saved evaluation variant blending baseline predictions into scanpath outputs with selection probability 0.40, followed by text cleaning. The blend is implemented in `scripts/eval_qwen_scanpath_summarization_downgrade.py`; its inputs and settings are recorded in the manifest.

### Table II: fault localization

| Method | Exact Match | MRR | Set Precision | Set F1 |
|---|---:|---:|---:|---:|
| Baseline | 0.0812 | 0.4810 | 0.2975 | 0.3163 |
| ScanCoder | 0.1274 | 0.5293 | 0.3523 | 0.3599 |
| CogCoder-Workload | 0.3227 | 0.5103 | 0.4812 | 0.4252 |
| CogCoder-Hybrid | 0.3494 | 0.5385 | 0.5175 | 0.4579 |

The Hybrid result uses exactness-oriented scanpath/workload prediction fusion (`scan_top1_plus_eeg_singleton`). Exact Match uses set equality.

Table III is available in [`result/efficiency/checkpoint_adaptation_metrics_summary.json`](result/efficiency/checkpoint_adaptation_metrics_summary.json). Completion, summarization, and fault-localization summaries are under their corresponding `result/` directories. The [result manifest](result/manifests/result_manifest.json) records prediction-file hashes, evaluation variants, scoring settings, and script sources. The bundle contains aggregate results and core code; datasets and checkpoints are supplied separately.

## Running the code

From the repository root:

```bash
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD:$PWD/scripts${PYTHONPATH:+:$PYTHONPATH}"
```

Set `ACTR_SFT_ROOT` to the ACT-R source directory when preparing datasets. Use `--model-id` for training and `--model-path` for evaluation to select a checkpoint.

Example training command, after preparing the task JSONL files:

```bash
python scripts/train_qwen_scanpath_large.py \
  --task completion \
  --train-file data/actr_scanpath_eeg/completion_train_final_eeg_scanpath.jsonl \
  --valid-file data/actr_scanpath_eeg/completion_valid_final_eeg_scanpath.jsonl \
  --output-dir outputs/cogcoder_hybrid_completion \
  --weight-mode scanpath_eeg
```

Rebuild the fault-localization fusion and compute set metrics:

```bash
python scripts/fuse_faultloc_predictions.py \
  --scanpath-predictions outputs/defects4j_qwen35_base_scanpath_faultloc_full_eval/predictions.jsonl \
  --eeg-predictions outputs/defects4j_qwen35_base_eeg_faultloc_full_eval/predictions.jsonl \
  --output-dir outputs/faultloc_hybrid \
  --mode scan_top1_plus_eeg_singleton
python scripts/analyze_faultloc_metrics.py \
  --predictions outputs/faultloc_hybrid/predictions.jsonl
```

Rebuild the adaptation metrics from the included cache:

```bash
python scripts/plot_checkpoint_layer_group_metrics.py \
  --cache-json result/efficiency/task_weight_heatmaps_uniform_baselines_cache.json \
  --metrics-json outputs/checkpoint_layer_group_metrics.json \
  --output outputs/checkpoint_layer_group_metrics.png
```
