# CogCoder

Paper-reported results and the core weighted-loss component of CogCoder.

## Core code

[`scripts/weighted_loss.py`](scripts/weighted_loss.py) contains the token-weighted supervised-loss implementation extracted from the training code. It consumes prepared token weights and masks prompt labels with `-100`.

This repository provides the core implementation excerpt for inspection.

## Results

| Paper table | Results |
|---|---|
| Table I: code completion | [Completion summary](result/completion/qwen35_completion_metrics_summary.json) |
| Table I: code summarization | [Summarization summary](result/summarization/qwen35_summarization_metrics_summary.json) |
| Table II: fault localization | [Fault-localization summaries](result/fault_localization) |
| Table III: adaptation efficiency | [Adaptation summary](result/efficiency/checkpoint_adaptation_metrics_summary.json) |

The summary files identify the evaluation variants used for the reported results.
