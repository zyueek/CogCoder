"""Scanpath baseline weighting helpers."""
from __future__ import annotations


def calculate_scanpath_weight(mask: int | float, order: int | float) -> float:
    if float(mask) <= 0.0:
        return 1.0
    order_bonus = 1.0 / max(1.0, float(order))
    return 3.0 + order_bonus


def calculate_scanpath_weak_weight(mask: int | float, order: int | float) -> float:
    # Explicit weak ablation: much flatter weighting and only meaningful for the
    # very first few attended positions.
    if float(mask) <= 0.0:
        return 1.0
    order_value = float(order)
    if order_value <= 0.0:
        return 1.0
    if order_value > 2.0:
        return 1.0
    order_bonus = 1.0 / max(1.0, order_value)
    return 1.35 + 0.35 * order_bonus
