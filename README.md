# CogCoder

Paper-reported results and selected core components of CogCoder.

## Core code

| Script | Component |
|---|---|
| [`weighted_loss.py`](scripts/weighted_loss.py) | Token-weighted supervised loss |
| [`scanpath_weights.py`](scripts/scanpath_weights.py) | Scanpath baseline weight functions |
| [`data_collator.py`](scripts/data_collator.py) | Padding and collation of prepared examples |
| [`faultloc_fusion.py`](scripts/faultloc_fusion.py) | Fault-localization prediction fusion |
| [`faultloc_metrics.py`](scripts/faultloc_metrics.py) | Fault-localization set and ranking metrics |

These focused components are extracted from the experiment code for inspection.

## Results

| Paper table | Results |
|---|---|
| Table I: code completion | [Completion summary](result/completion/qwen35_completion_metrics_summary.json) |
| Table I: code summarization | [Summarization summary](result/summarization/qwen35_summarization_metrics_summary.json) |
| Table II: fault localization | [Fault-localization summaries](result/fault_localization) |
| Table III: adaptation efficiency | [Adaptation summary](result/efficiency/checkpoint_adaptation_metrics_summary.json) |

The summary files identify the evaluation variants used for the reported results.
