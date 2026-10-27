"""Batch collation for prepared tokenized examples."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoTokenizer


@dataclass
class CustomDataCollator:
    tokenizer: AutoTokenizer
    max_length: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch: dict[str, Any] = {}
        max_len = self.max_length
        for key in features[0].keys():
            if key == "weights":
                pad_value, dtype = 1.0, torch.float
            elif key == "labels":
                pad_value, dtype = -100, torch.long
            else:
                pad_value, dtype = self.tokenizer.pad_token_id, torch.long

            padded_list = []
            for feature in features:
                values = list(feature[key])[:max_len]
                values = values + [pad_value] * (max_len - len(values))
                padded_list.append(values)
            batch[key] = torch.tensor(padded_list, dtype=dtype)
        return batch
