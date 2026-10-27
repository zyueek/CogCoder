"""Set and ranking metrics for fault-localization predictions."""
from __future__ import annotations

from statistics import mean
from typing import Any


def safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    exact_values: list[float] = []
    hit1_values: list[float] = []
    hit3_values: list[float] = []
    mrr_values: list[float] = []
    precision_at_1_values: list[float] = []
    precision_at_3_values: list[float] = []
    recall_at_1_values: list[float] = []
    recall_at_3_values: list[float] = []
    average_precision_values: list[float] = []
    exam_values: list[float] = []
    wasted_effort_values: list[float] = []
    set_precision_values: list[float] = []
    set_recall_values: list[float] = []
    set_f1_values: list[float] = []
    jaccard_values: list[float] = []
    singleton_exact_values: list[float] = []
    singleton_hit1_values: list[float] = []
    multi_hit3_values: list[float] = []
    pred_size_values: list[float] = []

    for row in rows:
        predicted_lines = list(row.get("predicted_lines") or [])
        gold_lines = list(row.get("gold_lines") or [])
        predicted_set = set(predicted_lines)
        gold_set = set(gold_lines)
        intersection = len(predicted_set & gold_set)
        gold_count = max(1, len(gold_set))

        exact_values.append(float(predicted_set == gold_set))
        hit1_values.append(float(bool(predicted_lines) and predicted_lines[0] in gold_set))
        hit3_values.append(float(any(line in gold_set for line in predicted_lines[:3])))
        pred_size_values.append(float(len(predicted_lines)))

        top1 = predicted_lines[:1]
        top3 = predicted_lines[:3]
        top1_hits = sum(1 for line in top1 if line in gold_set)
        top3_hits = sum(1 for line in top3 if line in gold_set)
        precision_at_1_values.append(top1_hits / 1.0)
        precision_at_3_values.append(top3_hits / 3.0)
        recall_at_1_values.append(top1_hits / gold_count)
        recall_at_3_values.append(top3_hits / gold_count)

        reciprocal_rank = 0.0
        first_hit_rank: int | None = None
        average_precision = 0.0
        relevant_seen = 0
        for rank, line in enumerate(predicted_lines, start=1):
            if line in gold_set:
                relevant_seen += 1
                average_precision += relevant_seen / rank
                if first_hit_rank is None:
                    first_hit_rank = rank
                    reciprocal_rank = 1.0 / rank
        average_precision_values.append(average_precision / gold_count)
        mrr_values.append(reciprocal_rank)

        snippet_line_count = max(1, len((row.get("input") or "").splitlines()))
        if first_hit_rank is None:
            exam_values.append(1.0)
            wasted_effort_values.append(float(snippet_line_count))
        else:
            exam_values.append(first_hit_rank / snippet_line_count)
            wasted_effort_values.append(float(first_hit_rank - 1))

        set_precision = intersection / len(predicted_set) if predicted_set else 0.0
        set_recall = intersection / len(gold_set) if gold_set else 0.0
        set_f1 = (2.0 * set_precision * set_recall / (set_precision + set_recall)) if (set_precision + set_recall) else 0.0
        jaccard = intersection / len(predicted_set | gold_set) if (predicted_set or gold_set) else 1.0
        set_precision_values.append(set_precision)
        set_recall_values.append(set_recall)
        set_f1_values.append(set_f1)
        jaccard_values.append(jaccard)

        if len(gold_set) == 1:
            singleton_exact_values.append(float(predicted_set == gold_set))
            singleton_hit1_values.append(float(bool(predicted_lines) and predicted_lines[0] in gold_set))
        else:
            multi_hit3_values.append(float(any(line in gold_set for line in predicted_lines[:3])))

    return {
        "num_examples": float(len(rows)),
        "exact_match_rate": safe_mean(exact_values),
        "hit_at_1": safe_mean(hit1_values),
        "hit_at_3": safe_mean(hit3_values),
        "mrr": safe_mean(mrr_values),
        "precision_at_1": safe_mean(precision_at_1_values),
        "precision_at_3": safe_mean(precision_at_3_values),
        "recall_at_1": safe_mean(recall_at_1_values),
        "recall_at_3": safe_mean(recall_at_3_values),
        "map": safe_mean(average_precision_values),
        "exam": safe_mean(exam_values),
        "wasted_effort": safe_mean(wasted_effort_values),
        "set_precision": safe_mean(set_precision_values),
        "set_recall": safe_mean(set_recall_values),
        "set_f1": safe_mean(set_f1_values),
        "jaccard": safe_mean(jaccard_values),
        "singleton_exact_match_rate": safe_mean(singleton_exact_values),
        "singleton_hit_at_1": safe_mean(singleton_hit1_values),
        "multi_hit_at_3": safe_mean(multi_hit3_values),
        "mean_predicted_lines": safe_mean(pred_size_values),
    }
