from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from eeg_qwen.common import read_jsonl
from prompt_templates import render_neuro_prompt, render_training_prompt


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def simple_code_tokenize(text: str) -> list[str]:
    return [match.group(0) for match in TOKEN_PATTERN.finditer(text)]


def sequential_token_spans(text: str, tokens: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        match_start = text.find(token, cursor)
        if match_start < 0:
            match_start = text.find(token)
            if match_start < 0:
                continue
        match_end = match_start + len(token)
        spans.append((match_start, match_end))
        cursor = match_end
    return spans


def overlap(span_a: tuple[int, int], span_b: tuple[int, int]) -> bool:
    return max(span_a[0], span_b[0]) < min(span_a[1], span_b[1])


def truncate_prompt_and_target(
    prompt_ids: list[int],
    target_ids: list[int],
    max_length: int,
) -> tuple[list[int], list[int]]:
    if len(prompt_ids) + len(target_ids) <= max_length:
        return prompt_ids, target_ids
    target_budget = min(len(target_ids), max(1, min(max_length - 1, max_length // 3 or 1)))
    target_ids = target_ids[:target_budget]
    prompt_budget = max(1, max_length - len(target_ids))
    return prompt_ids[-prompt_budget:], target_ids


def normalize_distribution(values: list[float]) -> list[float]:
    total = sum(max(value, 0.0) for value in values)
    if total <= 0:
        return values
    return [max(value, 0.0) / total for value in values]


@dataclass
class ExampleFeatures:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    processing_cost: torch.Tensor
    has_processing_cost: torch.Tensor
    attention_distribution: torch.Tensor
    has_attention: torch.Tensor
    revisit_scores: torch.Tensor
    has_revisit: torch.Tensor
    neuro_mask: torch.Tensor
    efficiency_group: torch.Tensor
    has_efficiency_group: torch.Tensor


def _empty_targets(seq_len: int) -> dict[str, torch.Tensor]:
    zeros = torch.zeros(seq_len, dtype=torch.float32)
    return {
        "processing_cost": torch.tensor(0.0, dtype=torch.float32),
        "has_processing_cost": torch.tensor(0, dtype=torch.long),
        "attention_distribution": zeros.clone(),
        "has_attention": torch.tensor(0, dtype=torch.long),
        "revisit_scores": zeros.clone(),
        "has_revisit": torch.tensor(0, dtype=torch.long),
        "neuro_mask": torch.zeros(seq_len, dtype=torch.long),
        "efficiency_group": torch.tensor(-100, dtype=torch.long),
        "has_efficiency_group": torch.tensor(0, dtype=torch.long),
    }


def _build_target_vector(
    offset_mapping: list[tuple[int, int]],
    token_spans: list[tuple[int, int]],
    token_values: list[float],
    prompt_offset: int,
) -> tuple[list[float], list[int]]:
    seq_values = [0.0] * len(offset_mapping)
    seq_mask = [0] * len(offset_mapping)
    for span, value in zip(token_spans, token_values):
        global_span = (span[0] + prompt_offset, span[1] + prompt_offset)
        matched_positions = [
            index
            for index, offsets in enumerate(offset_mapping)
            if offsets[1] > offsets[0] and overlap(global_span, offsets)
        ]
        if not matched_positions:
            continue
        split_value = float(value) / len(matched_positions)
        for position in matched_positions:
            seq_values[position] += split_value
            seq_mask[position] = 1
    return seq_values, seq_mask


def build_sequence_features(prompt: str, target: str, tokenizer, max_length: int) -> ExampleFeatures:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"] if target else []
    eos_id = tokenizer.eos_token_id
    if eos_id is not None and target_ids:
        target_ids = target_ids + [eos_id]
    prompt_ids, target_ids = truncate_prompt_and_target(prompt_ids, target_ids, max_length)

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    attention_mask = [1] * len(input_ids)
    defaults = _empty_targets(len(input_ids))

    return ExampleFeatures(
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        attention_mask=torch.tensor(attention_mask, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
        processing_cost=defaults["processing_cost"],
        has_processing_cost=defaults["has_processing_cost"],
        attention_distribution=defaults["attention_distribution"],
        has_attention=defaults["has_attention"],
        revisit_scores=defaults["revisit_scores"],
        has_revisit=defaults["has_revisit"],
        neuro_mask=defaults["neuro_mask"],
        efficiency_group=defaults["efficiency_group"],
        has_efficiency_group=defaults["has_efficiency_group"],
    )


def apply_neuro_labels(
    features: ExampleFeatures,
    tokenizer,
    prompt: str,
    target: str,
    source_text: str,
    neuro_labels: dict[str, Any],
) -> ExampleFeatures:
    if not neuro_labels or not source_text:
        return features

    prompt_offset = prompt.index(source_text)
    full_text = prompt + target
    tokenized = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    offset_mapping = tokenized["offset_mapping"][: len(features.input_ids)]

    code_tokens = neuro_labels.get("tokens") or simple_code_tokenize(source_text)
    token_spans = sequential_token_spans(source_text, code_tokens)
    attention_values = neuro_labels.get("token_targets", {}).get("attention_importance") or [0.0] * len(token_spans)
    revisit_values = neuro_labels.get("token_targets", {}).get("revisit_score") or [0.0] * len(token_spans)

    attention_distribution, attention_mask = _build_target_vector(
        offset_mapping=offset_mapping,
        token_spans=token_spans,
        token_values=attention_values,
        prompt_offset=prompt_offset,
    )
    revisit_scores, revisit_mask = _build_target_vector(
        offset_mapping=offset_mapping,
        token_spans=token_spans,
        token_values=revisit_values,
        prompt_offset=prompt_offset,
    )

    attention_distribution = normalize_distribution(attention_distribution)
    neuro_mask = [max(mask_a, mask_b) for mask_a, mask_b in zip(attention_mask, revisit_mask)]

    snippet_targets = neuro_labels.get("snippet_targets", {})
    efficiency_group = str(snippet_targets.get("efficiency_group", "")).lower()
    efficiency_lookup = {"low": 0, "medium": 1, "high": 2}

    return ExampleFeatures(
        input_ids=features.input_ids,
        attention_mask=features.attention_mask,
        labels=features.labels,
        processing_cost=torch.tensor(float(snippet_targets.get("processing_cost", 0.0)), dtype=torch.float32),
        has_processing_cost=torch.tensor(int("processing_cost" in snippet_targets), dtype=torch.long),
        attention_distribution=torch.tensor(attention_distribution, dtype=torch.float32),
        has_attention=torch.tensor(int(any(attention_distribution)), dtype=torch.long),
        revisit_scores=torch.tensor(revisit_scores, dtype=torch.float32),
        has_revisit=torch.tensor(int(any(neuro_mask)), dtype=torch.long),
        neuro_mask=torch.tensor(neuro_mask, dtype=torch.long),
        efficiency_group=torch.tensor(efficiency_lookup.get(efficiency_group, -100), dtype=torch.long),
        has_efficiency_group=torch.tensor(int(efficiency_group in efficiency_lookup), dtype=torch.long),
    )


def build_supervised_features(record: dict[str, Any], tokenizer, max_length: int) -> ExampleFeatures:
    prompt = render_training_prompt(record)
    target = (record["target_file_contents"] or "").rstrip() + "\n"
    return build_sequence_features(prompt, target, tokenizer, max_length)


def build_multitask_features(record: dict[str, Any], tokenizer, max_length: int) -> ExampleFeatures:
    prompt = render_training_prompt(record)
    target = (record["target_file_contents"] or "").rstrip() + "\n"
    features = build_sequence_features(prompt, target, tokenizer, max_length)
    return apply_neuro_labels(
        features=features,
        tokenizer=tokenizer,
        prompt=prompt,
        target=target,
        source_text=record.get("file_content", ""),
        neuro_labels=record.get("neuro_labels") or {},
    )


def build_neuro_only_features(record: dict[str, Any], tokenizer, max_length: int) -> ExampleFeatures:
    prompt = render_neuro_prompt(record)
    target = ""
    features = build_sequence_features(prompt, target, tokenizer, max_length)
    source_text = record.get("file_content") or record.get("code") or ""
    return apply_neuro_labels(
        features=features,
        tokenizer=tokenizer,
        prompt=prompt,
        target=target,
        source_text=source_text,
        neuro_labels=record.get("neuro_labels") or {},
    )


class JsonlFeatureDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer,
        max_length: int,
        multitask: bool = False,
        mode: str | None = None,
    ) -> None:
        self.records = read_jsonl(path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        if mode is None:
            self.mode = "multitask" if multitask else "edit"
        else:
            self.mode = mode

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record = self.records[index]
        if self.mode == "neuro_only":
            builder = build_neuro_only_features
        elif self.mode == "multitask":
            builder = build_multitask_features
        else:
            builder = build_supervised_features
        features = builder(record, self.tokenizer, self.max_length)
        return {
            "input_ids": features.input_ids,
            "attention_mask": features.attention_mask,
            "labels": features.labels,
            "processing_cost": features.processing_cost,
            "has_processing_cost": features.has_processing_cost,
            "attention_distribution": features.attention_distribution,
            "has_attention": features.has_attention,
            "revisit_scores": features.revisit_scores,
            "has_revisit": features.has_revisit,
            "neuro_mask": features.neuro_mask,
            "efficiency_group": features.efficiency_group,
            "has_efficiency_group": features.has_efficiency_group,
        }


class FeatureCollator:
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        max_length = max(feature["input_ids"].shape[0] for feature in features)

        batch: dict[str, list[torch.Tensor]] = {}
        for feature in features:
            pad_width = max_length - feature["input_ids"].shape[0]
            for key, value in feature.items():
                if value.ndim == 0:
                    batch.setdefault(key, []).append(value)
                    continue
                if key == "input_ids":
                    padded = torch.nn.functional.pad(value, (0, pad_width), value=pad_id)
                elif key == "labels":
                    padded = torch.nn.functional.pad(value, (0, pad_width), value=-100)
                else:
                    padded = torch.nn.functional.pad(value, (0, pad_width), value=0)
                batch.setdefault(key, []).append(padded)

        collated: dict[str, torch.Tensor] = {}
        for key, values in batch.items():
            collated[key] = torch.stack(values, dim=0)
        return collated
