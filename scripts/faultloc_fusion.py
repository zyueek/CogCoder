"""Prediction-list fusion helpers for fault localization."""
from __future__ import annotations


def unique_keep_order(values: list[int], limit: int = 3) -> list[int]:
    output: list[int] = []
    for value in values:
        if value in output:
            continue
        output.append(value)
        if len(output) >= limit:
            break
    return output


def fuse_lines(scanpath_lines: list[int], eeg_lines: list[int], mode: str) -> list[int]:
    if mode == "scan_top1_eeg_singleton_second":
        if len(eeg_lines) == 1:
            return unique_keep_order(scanpath_lines[:1] + eeg_lines + scanpath_lines[1:])
        return unique_keep_order(scanpath_lines)
    if mode == "scan_top1_plus_eeg_singleton":
        if len(eeg_lines) == 1:
            return unique_keep_order(scanpath_lines[:1] + eeg_lines)
        return unique_keep_order(scanpath_lines[:1])
    if mode == "scan_first_append_eeg_if_singleton":
        if len(eeg_lines) == 1:
            return unique_keep_order(scanpath_lines + eeg_lines)
        return unique_keep_order(scanpath_lines)
    if mode == "scan_then_eeg":
        return unique_keep_order(scanpath_lines + eeg_lines)
    raise ValueError(f"Unknown fusion mode: {mode}")
