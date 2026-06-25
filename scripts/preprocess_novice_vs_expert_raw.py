from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from eeg_qwen.common import ensure_dir, write_json, write_jsonl
from eeg_qwen.datasets import normalize_distribution


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)
FLOAT_PATTERN = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
CHOICE_SUFFIXES = ("code", "input", "right", "wrong1", "wrong2", "dontknow", "next")


@dataclass
class SnippetLayout:
    snippet_id: str
    file_path: str
    file_content: str
    tokens: list[str]
    token_boxes: list[tuple[float, float, float, float]]
    image_width: float
    image_height: float
    char_width: float


@dataclass
class TrialWindow:
    subject_id: str
    snippet_id: str
    start_ms: float
    stop_ms: float
    correct: bool | None
    response_time_s: float | None
    display_width: float | None
    display_height: float | None
    gaze_processing_cost_raw: float = 0.0
    eeg_processing_cost_raw: float | None = None


class TrialAccumulator:
    def __init__(
        self,
        trial: TrialWindow,
        layout: SnippetLayout,
        screen_width: int,
        screen_height: int,
        min_valid_samples: int,
        max_sample_gap_ms: float,
    ) -> None:
        self.trial = trial
        self.layout = layout
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.min_valid_samples = min_valid_samples
        self.max_sample_gap_ms = max_sample_gap_ms

        display_width = float(trial.display_width or layout.image_width)
        display_height = float(trial.display_height or layout.image_height)
        x_scale = display_width / max(layout.image_width, 1.0)
        y_scale = display_height / max(layout.image_height, 1.0)
        left = (screen_width - display_width) / 2.0
        top = (screen_height - display_height) / 2.0

        self.image_box = (left, top, left + display_width, top + display_height)
        self.scaled_char_width = layout.char_width * x_scale
        self.scaled_boxes = [
            (
                left + x0 * x_scale,
                top + y0 * y_scale,
                left + x1 * x_scale,
                top + y1 * y_scale,
            )
            for (x0, y0, x1, y1) in layout.token_boxes
        ]
        self.total_samples = 0
        self.valid_samples = 0
        self.image_samples = 0
        self.token_sample_hits = 0
        self.attention_mass = [0.0] * len(layout.tokens)
        self.revisit_count = [0.0] * len(layout.tokens)
        self.seen_tokens: set[int] = set()
        self.unique_tokens: set[int] = set()
        self.previous_timestamp_ms: float | None = None
        self.previous_token_idx: int | None = None

    def contains(self, timestamp_ms: float) -> bool:
        return self.trial.start_ms <= timestamp_ms <= self.trial.stop_ms

    def _find_token(self, x_px: float, y_px: float) -> int | None:
        line_candidates: list[tuple[float, int]] = []
        for index, (x0, y0, x1, y1) in enumerate(self.scaled_boxes):
            if x0 <= x_px <= x1 and y0 <= y_px <= y1:
                return index
            if y0 <= y_px <= y1:
                horizontal_gap = 0.0
                if x_px < x0:
                    horizontal_gap = x0 - x_px
                elif x_px > x1:
                    horizontal_gap = x_px - x1
                line_candidates.append((horizontal_gap, index))
        if not line_candidates:
            return None
        line_candidates.sort(key=lambda item: item[0])
        closest_gap, closest_index = line_candidates[0]
        if closest_gap <= max(6.0, self.scaled_char_width * 1.5):
            return closest_index
        return None

    def observe(self, timestamp_ms: float, x_norm: float, y_norm: float) -> None:
        self.total_samples += 1
        if not (0.0 <= x_norm <= 1.0 and 0.0 <= y_norm <= 1.0):
            return

        self.valid_samples += 1
        x_px = x_norm * self.screen_width
        y_px = y_norm * self.screen_height

        previous_token_idx = self.previous_token_idx
        if self.previous_timestamp_ms is not None and previous_token_idx is not None:
            delta_ms = min(max(timestamp_ms - self.previous_timestamp_ms, 0.0), self.max_sample_gap_ms)
            self.attention_mass[previous_token_idx] += delta_ms

        self.previous_timestamp_ms = timestamp_ms
        self.previous_token_idx = None

        left, top, right, bottom = self.image_box
        if not (left <= x_px <= right and top <= y_px <= bottom):
            return

        self.image_samples += 1
        token_idx = self._find_token(x_px, y_px)
        if token_idx is None:
            return

        self.token_sample_hits += 1
        if token_idx in self.seen_tokens and token_idx != previous_token_idx:
            self.revisit_count[token_idx] += 1.0
        self.seen_tokens.add(token_idx)
        self.unique_tokens.add(token_idx)
        self.previous_token_idx = token_idx

    def finalize(self) -> dict[str, Any] | None:
        if self.valid_samples < self.min_valid_samples:
            return None
        token_count = max(1, len(self.layout.tokens))
        coverage = len(self.unique_tokens) / token_count
        view_duration_ms = max(0.0, self.trial.stop_ms - self.trial.start_ms)
        mean_token_dwell = sum(self.attention_mass) / max(1, self.token_sample_hits)
        revisit_density = sum(self.revisit_count) / max(1, self.token_sample_hits)
        image_ratio = self.image_samples / max(1, self.valid_samples)
        duration_ratio = min(1.0, view_duration_ms / 60000.0)
        raw_cost = (
            0.30 * duration_ratio
            + 0.25 * min(1.0, mean_token_dwell / 80.0)
            + 0.25 * min(1.0, revisit_density / 0.20)
            + 0.20 * (1.0 - coverage)
        )
        raw_cost = min(1.0, max(0.0, raw_cost + 0.10 * (1.0 - image_ratio)))

        self.trial.gaze_processing_cost_raw = raw_cost
        return {
            "coverage": coverage,
            "image_ratio": image_ratio,
            "mean_token_dwell_ms": mean_token_dwell,
            "revisit_density": revisit_density,
            "attention_distribution": normalize_distribution(self.attention_mass),
            "revisit_score": min_max_scale(self.revisit_count),
        }


def normalize_key(name: str) -> str:
    return "".join(ch.lower() for ch in name if ch.isalnum())


def read_csv_rows(path: str | Path) -> list[dict[str, str]]:
    return list(iter_csv_rows(path))


RAW_GAZE_COLUMNS = [
    "left_gaze_point_in_user_coordinate_system",
    "left_gaze_point_validity",
    "right_gaze_point_validity",
    "right_gaze_point_in_user_coordinate_system",
    "left_gaze_origin_in_user_coordinate_system",
    "right_gaze_origin_in_user_coordinate_system",
    "left_gaze_point_on_display_area",
    "right_gaze_point_on_display_area",
    "system_time_stamp",
    "left_pupil_diameter",
    "right_pupil_diameter",
]


def detect_delimiter(path: str | Path) -> str:
    with Path(path).open("r", encoding="utf-8-sig", errors="ignore") as handle:
        sample = handle.read(8192)
    delimiter = ";"
    if sample.count(",") > sample.count(";"):
        delimiter = ","
    elif sample.count("\t") > sample.count(";"):
        delimiter = "\t"
    return delimiter


def iter_csv_rows(path: str | Path):
    path = Path(path)
    if path.name == "experiment_data.csv":
        yield from iter_raw_gaze_rows(path)
        return

    delimiter = detect_delimiter(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=delimiter)
        for row in reader:
            yield row


def iter_raw_gaze_rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter=";")
        for row in reader:
            if not row:
                continue
            if len(row) < len(RAW_GAZE_COLUMNS):
                row = row + [""] * (len(RAW_GAZE_COLUMNS) - len(row))
            elif len(row) > len(RAW_GAZE_COLUMNS):
                row = row[: len(RAW_GAZE_COLUMNS)]
            yield dict(zip(RAW_GAZE_COLUMNS, row))


def get_first(record: dict[str, Any], candidates: list[str], default: Any = None) -> Any:
    normalized = {normalize_key(key): value for key, value in record.items()}
    for candidate in candidates:
        value = normalized.get(normalize_key(candidate))
        if value not in {None, ""}:
            return value
    return default


def parse_float(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    match = FLOAT_PATTERN.search(str(value))
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def parse_pair(value: Any) -> tuple[float, float] | None:
    if value in {None, ""}:
        return None
    matches = FLOAT_PATTERN.findall(str(value))
    if len(matches) < 2:
        return None
    try:
        return float(matches[0]), float(matches[1])
    except ValueError:
        return None


def min_max_scale(values: list[float]) -> list[float]:
    if not values:
        return values
    minimum = min(values)
    maximum = max(values)
    if maximum <= minimum:
        return [0.0 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


def deterministic_split(subject_id: str, train_ratio: float, val_ratio: float) -> str:
    digest = hashlib.sha256(subject_id.encode("utf-8")).hexdigest()
    score = int(digest[:8], 16) / 0xFFFFFFFF
    if score < train_ratio:
        return "train"
    if score < train_ratio + val_ratio:
        return "val"
    return "test"


def format_progress_bar(current: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return "[" + "-" * width + "]"
    current = max(0, min(current, total))
    filled = int(round((current / total) * width))
    filled = max(0, min(filled, width))
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def compute_efficiency_groups(skill_by_subject: dict[str, float]) -> dict[str, str]:
    ordered = sorted(skill_by_subject.values())
    if not ordered:
        return {}
    lower = ordered[int(0.33 * (len(ordered) - 1))]
    upper = ordered[int(0.66 * (len(ordered) - 1))]
    groups = {}
    for subject_id, score in skill_by_subject.items():
        if score < lower:
            groups[subject_id] = "low"
        elif score > upper:
            groups[subject_id] = "high"
        else:
            groups[subject_id] = "medium"
    return groups


def discover_participant_dirs(raw_root: Path, requested: list[str] | None) -> list[Path]:
    if not raw_root.exists():
        raise FileNotFoundError(
            "Raw NoviceVsExpert data directory not found: "
            f"{raw_root}\n"
            "Expected participant folders such as "
            "'data/neuro/raw/NoviceVsExpert/rawData/Participant04'.\n"
            "You need to download the OSF raw data first, or point --raw-root at the directory that already contains the ParticipantXX folders."
        )
    participant_dirs = [path for path in raw_root.iterdir() if path.is_dir() and path.name.lower().startswith("participant")]
    participant_dirs.sort()
    if not requested:
        return participant_dirs
    requested_set = {item.lower() for item in requested}
    return [path for path in participant_dirs if path.name.lower() in requested_set]


def detect_behavioral_csv(participant_dir: Path) -> Path | None:
    candidates = []
    for path in participant_dir.glob("*.csv"):
        if path.name == "experiment_data.csv":
            continue
        candidates.append(path)
    if not candidates:
        return None
    candidates.sort(key=lambda path: (-path.stat().st_size, path.name))
    return candidates[0]


def detect_fif(participant_dir: Path) -> Path | None:
    fif_files = sorted(participant_dir.glob("*.fif"))
    if not fif_files:
        return None
    fif_files.sort(key=lambda path: (-path.stat().st_size, path.name))
    return fif_files[0]


def subject_id_from_dir(participant_dir: Path) -> str:
    digits = "".join(ch for ch in participant_dir.name if ch.isdigit())
    return digits or participant_dir.name


def strip_choice_suffix(stem: str) -> str:
    lowered = stem.lower()
    for suffix in CHOICE_SUFFIXES:
        if lowered.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def resolve_snippet_id(raw_value: str | None, known_ids: set[str]) -> str | None:
    if not raw_value:
        return None
    stem = Path(str(raw_value).replace("\\", "/")).stem
    normalized_known = {normalize_key(item): item for item in known_ids}

    direct = normalized_known.get(normalize_key(stem))
    if direct:
        return direct

    stripped = strip_choice_suffix(stem)
    direct = normalized_known.get(normalize_key(stripped))
    if direct:
        return direct

    candidates = []
    normalized_stem = normalize_key(stem)
    for snippet_id in known_ids:
        normalized_id = normalize_key(snippet_id)
        if normalized_id and normalized_id in normalized_stem:
            candidates.append((len(normalized_id), snippet_id))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def load_snippet_layouts(source_dir: Path, generator_dir: Path | None) -> dict[str, SnippetLayout]:
    layouts: dict[str, SnippetLayout] = {}
    for source_path in sorted(source_dir.glob("*.java")):
        snippet_id = source_path.stem
        generator_path = generator_dir / f"{snippet_id}.json" if generator_dir else None
        source_text = source_path.read_text(encoding="utf-8")
        lines = source_text.rstrip("\n").splitlines() or [source_text]
        font_size = 19.0
        width_margin = 5.0
        height_margin = 5.0
        spacing = 1.0

        if generator_path and generator_path.exists():
            payload = json.loads(generator_path.read_text(encoding="utf-8"))
            lines = payload.get("source-code") or lines
            font_size = float(payload.get("font-size", font_size))
            width_margin = float(payload.get("width-margin", width_margin))
            height_margin = float(payload.get("height-margin", height_margin))
            spacing = float(payload.get("spacing", spacing))
            source_text = "\n".join(lines).rstrip() + "\n"

        char_width = font_size * 0.62
        line_height = font_size * 1.35 + spacing
        tokens: list[str] = []
        token_boxes: list[tuple[float, float, float, float]] = []
        for line_index, line in enumerate(lines):
            for match in TOKEN_PATTERN.finditer(line):
                tokens.append(match.group(0))
                x0 = width_margin + match.start() * char_width
                x1 = width_margin + match.end() * char_width
                y0 = height_margin + line_index * line_height
                y1 = y0 + line_height
                token_boxes.append((x0, y0, x1, y1))

        image_width = max((len(line) for line in lines), default=1) * char_width + 2.0 * width_margin
        image_height = len(lines) * line_height + 2.0 * height_margin

        layouts[snippet_id] = SnippetLayout(
            snippet_id=snippet_id,
            file_path=str(source_path),
            file_content=source_text,
            tokens=tokens,
            token_boxes=token_boxes,
            image_width=image_width,
            image_height=image_height,
            char_width=char_width,
        )
    return layouts


def merge_lookback_rows(rows: list[dict[str, str]], index: int, lookback: int) -> dict[str, str]:
    merged: dict[str, str] = {}
    start_index = max(0, index - lookback)
    for row in rows[start_index : index + 1]:
        for key, value in row.items():
            if value not in {None, ""}:
                merged[key] = value
    return merged


def parse_behavioral_trials(path: Path, subject_id: str, layouts: dict[str, SnippetLayout]) -> list[TrialWindow]:
    rows = read_csv_rows(path)
    trials: list[TrialWindow] = []
    dedupe: set[tuple[str, int, int]] = set()
    known_ids = set(layouts)

    for index, row in enumerate(rows):
        context = merge_lookback_rows(rows, index, lookback=6)
        start_s = parse_float(get_first(context, ["Image.started"]))
        if start_s is None:
            continue

        stop_s = parse_float(get_first(context, ["Image.stopped"]))
        if stop_s is None:
            stop_s = parse_float(get_first(context, ["image.started"]))
        if stop_s is None:
            stop_s = parse_float(get_first(context, ["key_resp_3.started"]))
        if stop_s is None or stop_s <= start_s:
            stop_s = start_s + 180.0

        image_path = str(get_first(context, ["ImagePath", "ImagePathInputs"], default=""))
        snippet_id = resolve_snippet_id(image_path, known_ids)
        if snippet_id is None:
            continue

        chosen_answer = get_first(context, ["ChoosenAnwer", "ChosenAnswer", "ChoosenAnswer"])
        right_answer = get_first(context, ["RightPath"])
        correct = None
        if chosen_answer not in {None, ""} and right_answer not in {None, ""}:
            chosen_stem = Path(str(chosen_answer).replace("\\", "/")).stem.lower()
            right_stem = Path(str(right_answer).replace("\\", "/")).stem.lower()
            correct = chosen_stem == right_stem
        elif chosen_answer not in {None, ""}:
            chosen_stem = Path(str(chosen_answer).replace("\\", "/")).stem.lower()
            if chosen_stem.endswith("right"):
                correct = True
            elif chosen_stem in {"none", "skipped"}:
                correct = False

        response_time_s = parse_float(get_first(context, ["key_resp_3.rt"]))
        display_width = parse_float(get_first(context, ["ImagePathWidth"]))
        display_height = parse_float(get_first(context, ["ImagePathHeight"]))

        key = (snippet_id, int(start_s * 1000), int(stop_s * 1000))
        if key in dedupe:
            continue
        dedupe.add(key)
        trials.append(
            TrialWindow(
                subject_id=subject_id,
                snippet_id=snippet_id,
                start_ms=max(0.0, start_s * 1000.0),
                stop_ms=max(0.0, stop_s * 1000.0),
                correct=correct,
                response_time_s=response_time_s,
                display_width=display_width,
                display_height=display_height,
            )
        )

    trials.sort(key=lambda trial: trial.start_ms)
    return trials


def infer_timestamp_scale_to_ms(path: Path, sample_limit: int = 2000) -> tuple[float, float]:
    timestamps = []
    for row in iter_csv_rows(path):
        timestamp = parse_float(get_first(row, ["system_time_stamp", "systemtimestamp", "timestamp"]))
        if timestamp is None:
            continue
        timestamps.append(timestamp)
        if len(timestamps) >= sample_limit:
            break
    if not timestamps:
        return 1.0, 0.0
    deltas = [
        max(0.0, right - left)
        for left, right in zip(timestamps, timestamps[1:])
        if right > left
    ]
    median_delta = statistics.median(deltas) if deltas else 0.0
    if median_delta >= 1000.0:
        return 1.0 / 1000.0, timestamps[0]
    if 0.0 < median_delta < 1.0:
        return 1000.0, timestamps[0]
    return 1.0, timestamps[0]


def combine_display_point(row: dict[str, str]) -> tuple[float, float] | None:
    left = parse_pair(get_first(row, ["left_gaze_point_on_display_area"]))
    right = parse_pair(get_first(row, ["right_gaze_point_on_display_area"]))
    points = [point for point in (left, right) if point is not None]
    if not points:
        return None
    x = sum(point[0] for point in points) / len(points)
    y = sum(point[1] for point in points) / len(points)
    return x, y


def stream_gaze_into_trials(
    gaze_csv: Path,
    trials: list[TrialWindow],
    layouts: dict[str, SnippetLayout],
    screen_width: int,
    screen_height: int,
    min_valid_samples: int,
    max_sample_gap_ms: float,
) -> tuple[list[tuple[TrialWindow, dict[str, Any]]], int]:
    scale_to_ms, base_timestamp = infer_timestamp_scale_to_ms(gaze_csv)
    accumulators = [
        TrialAccumulator(
            trial=trial,
            layout=layouts[trial.snippet_id],
            screen_width=screen_width,
            screen_height=screen_height,
            min_valid_samples=min_valid_samples,
            max_sample_gap_ms=max_sample_gap_ms,
        )
        for trial in trials
    ]

    session_index = 0
    processed_samples = 0
    for row in iter_csv_rows(gaze_csv):
        timestamp = parse_float(get_first(row, ["system_time_stamp", "systemtimestamp", "timestamp"]))
        if timestamp is None:
            continue
        point = combine_display_point(row)
        if point is None:
            continue

        timestamp_ms = (timestamp - base_timestamp) * scale_to_ms
        while session_index < len(accumulators) and timestamp_ms > accumulators[session_index].trial.stop_ms:
            session_index += 1
        if session_index >= len(accumulators):
            break
        current = accumulators[session_index]
        if current.contains(timestamp_ms):
            current.observe(timestamp_ms, point[0], point[1])
            processed_samples += 1

    finalized = []
    for accumulator in accumulators:
        metrics = accumulator.finalize()
        if metrics is None:
            continue
        finalized.append((accumulator.trial, metrics))
    return finalized, processed_samples


def maybe_read_raw_fif(path: Path | None):
    if path is None or not path.exists():
        return None
    try:
        import mne
    except ImportError:
        return None
    return mne.io.read_raw_fif(str(path), preload=False, verbose="ERROR")


def extract_eeg_cost(raw_fif, start_ms: float, stop_ms: float) -> float | None:
    if raw_fif is None:
        return None
    try:
        import mne
    except ImportError:
        return None

    sfreq = float(raw_fif.info["sfreq"])
    start_index = max(0, int((start_ms / 1000.0) * sfreq))
    stop_index = min(raw_fif.n_times, int((stop_ms / 1000.0) * sfreq))
    if stop_index - start_index < int(max(1.0, sfreq * 2.0)):
        return None

    picks = mne.pick_types(raw_fif.info, eeg=True, meg=False, eog=False, stim=False, exclude="bads")
    if len(picks) == 0:
        return None
    data = raw_fif.get_data(picks=picks, start=start_index, stop=stop_index)
    if data.size == 0:
        return None
    data = data - data.mean(axis=1, keepdims=True)
    power = np.abs(np.fft.rfft(data, axis=1)) ** 2
    freqs = np.fft.rfftfreq(data.shape[1], d=1.0 / sfreq)

    def band_power(low: float, high: float) -> float:
        mask = (freqs >= low) & (freqs < high)
        if not np.any(mask):
            return 0.0
        return float(power[:, mask].mean())

    theta = band_power(4.0, 8.0)
    alpha = band_power(8.0, 13.0)
    beta = band_power(13.0, 30.0)
    return theta / (alpha + beta + 1e-6)


def build_subject_efficiency_scores(records: list[dict[str, Any]]) -> dict[str, float]:
    by_subject: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_subject[record["subject_id"]].append(record)

    scores = {}
    for subject_id, subject_records in by_subject.items():
        known_correct = [
            1.0 if item["metadata"].get("behavior_correct") else 0.0
            for item in subject_records
            if item["metadata"].get("behavior_correct") is not None
        ]
        duration_minutes = sum(item["metadata"].get("view_duration_ms", 0.0) for item in subject_records) / 60000.0
        mean_cost = statistics.mean(
            float(item["neuro_labels"]["snippet_targets"]["processing_cost"])
            for item in subject_records
        )
        if known_correct:
            accuracy = sum(known_correct) / len(known_correct)
            scores[subject_id] = accuracy / max(duration_minutes, 1e-3)
        else:
            scores[subject_id] = max(0.0, 1.0 - mean_cost)
    return scores


def build_records(
    finalized_trials: list[tuple[TrialWindow, dict[str, Any]]],
    layouts: dict[str, SnippetLayout],
    use_eeg: bool,
) -> list[dict[str, Any]]:
    blended_processing_values: list[float] = []
    for trial, _metrics in finalized_trials:
        raw_value = trial.gaze_processing_cost_raw
        if trial.eeg_processing_cost_raw is None:
            blended_processing_values.append(raw_value)
        else:
            blended_processing_values.append(0.5 * raw_value + 0.5 * trial.eeg_processing_cost_raw)

    scaled_blended = min_max_scale(blended_processing_values)
    records = []
    for index, (trial, metrics) in enumerate(finalized_trials):
        layout = layouts[trial.snippet_id]
        record = {
            "snippet_id": trial.snippet_id,
            "subject_id": trial.subject_id,
            "language": "java",
            "file_path": layout.file_path,
            "file_content": layout.file_content,
            "metadata": {
                "source_dataset": "NoviceVsExpert",
                "preprocessing_mode": "raw_fallback",
                "processing_cost_source": "gaze_plus_eeg" if use_eeg and trial.eeg_processing_cost_raw is not None else "gaze_only",
                "behavior_correct": trial.correct,
                "response_time_s": trial.response_time_s,
                "view_duration_ms": max(0.0, trial.stop_ms - trial.start_ms),
                "coverage": metrics["coverage"],
                "image_ratio": metrics["image_ratio"],
                "mean_token_dwell_ms": metrics["mean_token_dwell_ms"],
                "revisit_density": metrics["revisit_density"],
                "raw_gaze_processing_cost": trial.gaze_processing_cost_raw,
                "raw_eeg_processing_cost": trial.eeg_processing_cost_raw,
            },
            "neuro_labels": {
                "snippet_id": trial.snippet_id,
                "subject_id": trial.subject_id,
                "tokens": layout.tokens,
                "snippet_targets": {
                    "processing_cost": scaled_blended[index],
                    "efficiency_group": "medium",
                },
                "token_targets": {
                    "attention_importance": metrics["attention_distribution"],
                    "revisit_score": metrics["revisit_score"],
                },
            },
        }
        records.append(record)
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Approximate a Stage 2 neuro-only dataset directly from raw NoviceVsExpert participant folders. "
            "This is a pure-Python fallback for cases where the published MATLAB/EEGLAB preprocessing path is unavailable."
        )
    )
    parser.add_argument("--raw-root", required=True, help="Directory containing ParticipantXX raw-data folders.")
    parser.add_argument("--snippet-source-dir", required=True, help="Directory containing CodeSnippets/Source/*.java.")
    parser.add_argument("--generator-dir", help="Optional CodeSnippets/Generators_Raw directory for code-image layout reconstruction.")
    parser.add_argument("--output-dir", required=True, help="Directory where train/val/test JSONL files will be written.")
    parser.add_argument("--participants", nargs="*", help="Optional subset of participant directory names, e.g. Participant04 Participant05.")
    parser.add_argument("--screen-width", type=int, default=1920)
    parser.add_argument("--screen-height", type=int, default=1080)
    parser.add_argument("--min-valid-samples", type=int, default=20)
    parser.add_argument("--max-sample-gap-ms", type=float, default=100.0)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument(
        "--progress",
        action="store_true",
        help="Print a per-participant progress bar while preprocessing raw folders.",
    )
    parser.add_argument(
        "--enable-eeg",
        action="store_true",
        help="If mne is installed and .fif files are available, blend a simple EEG band-power proxy into processing_cost.",
    )
    args = parser.parse_args()

    raw_root = Path(args.raw_root)
    source_dir = Path(args.snippet_source_dir)
    generator_dir = Path(args.generator_dir) if args.generator_dir else None
    output_dir = ensure_dir(args.output_dir)

    layouts = load_snippet_layouts(source_dir, generator_dir)
    participant_dirs = discover_participant_dirs(raw_root, args.participants)
    if not participant_dirs:
        raise FileNotFoundError(f"No participant directories found under {raw_root}")

    finalized_trials: list[tuple[TrialWindow, dict[str, Any]]] = []
    summary = {
        "participants_requested": args.participants or [],
        "participants_found": [path.name for path in participant_dirs],
        "participants_processed": [],
        "participants_skipped": [],
        "records_written": {"train": 0, "val": 0, "test": 0},
        "notes": [
            "This fallback path reconstructs token regions from generator metadata and approximates processing_cost from gaze, optionally blended with a rough EEG ratio.",
            "The published notebook path remains the preferred route for final experiments.",
        ],
    }

    total_participants = len(participant_dirs)
    if args.progress:
        print(
            f"{format_progress_bar(0, total_participants)} 0/{total_participants} Starting raw fallback preprocessing",
            flush=True,
        )

    for index, participant_dir in enumerate(participant_dirs, start=1):
        subject_id = subject_id_from_dir(participant_dir)
        gaze_csv = participant_dir / "experiment_data.csv"
        behavioral_csv = detect_behavioral_csv(participant_dir)
        fif_path = detect_fif(participant_dir)
        status_detail = "ok"

        if not gaze_csv.exists() or behavioral_csv is None:
            status_detail = "skipped: missing gaze or behavioral csv"
            summary["participants_skipped"].append(
                {
                    "participant": participant_dir.name,
                    "reason": "missing gaze or behavioral csv",
                }
            )
            if args.progress:
                print(
                    f"{format_progress_bar(index, total_participants)} {index}/{total_participants} "
                    f"{participant_dir.name} {status_detail}",
                    flush=True,
                )
            continue

        trials = parse_behavioral_trials(behavioral_csv, subject_id=subject_id, layouts=layouts)
        if not trials:
            status_detail = "skipped: no code-viewing trial windows detected"
            summary["participants_skipped"].append(
                {
                    "participant": participant_dir.name,
                    "reason": "no code-viewing trial windows detected",
                }
            )
            if args.progress:
                print(
                    f"{format_progress_bar(index, total_participants)} {index}/{total_participants} "
                    f"{participant_dir.name} {status_detail}",
                    flush=True,
                )
            continue

        trial_metrics, processed_samples = stream_gaze_into_trials(
            gaze_csv=gaze_csv,
            trials=trials,
            layouts=layouts,
            screen_width=args.screen_width,
            screen_height=args.screen_height,
            min_valid_samples=args.min_valid_samples,
            max_sample_gap_ms=args.max_sample_gap_ms,
        )
        if not trial_metrics:
            status_detail = "skipped: no trials survived gaze filtering"
            summary["participants_skipped"].append(
                {
                    "participant": participant_dir.name,
                    "reason": "no trials survived gaze filtering",
                }
            )
            if args.progress:
                print(
                    f"{format_progress_bar(index, total_participants)} {index}/{total_participants} "
                    f"{participant_dir.name} {status_detail}",
                    flush=True,
                )
            continue

        raw_fif = maybe_read_raw_fif(fif_path) if args.enable_eeg else None
        for trial, _metrics in trial_metrics:
            trial.eeg_processing_cost_raw = extract_eeg_cost(raw_fif, trial.start_ms, trial.stop_ms)

        finalized_trials.extend(trial_metrics)
        summary["participants_processed"].append(
            {
                "participant": participant_dir.name,
                "subject_id": subject_id,
                "behavioral_csv": str(behavioral_csv),
                "gaze_csv": str(gaze_csv),
                "fif_file": str(fif_path) if fif_path else None,
                "candidate_trials": len(trials),
                "usable_trials": len(trial_metrics),
                "processed_gaze_samples": processed_samples,
            }
        )
        status_detail = (
            f"usable_trials={len(trial_metrics)} candidate_trials={len(trials)} "
            f"gaze_samples={processed_samples}"
        )
        if args.progress:
            print(
                f"{format_progress_bar(index, total_participants)} {index}/{total_participants} "
                f"{participant_dir.name} {status_detail}",
                flush=True,
            )

    if not finalized_trials:
        raise RuntimeError("No usable trial records were produced from the raw participant folders.")

    records = build_records(finalized_trials, layouts=layouts, use_eeg=args.enable_eeg)
    efficiency_scores = build_subject_efficiency_scores(records)
    efficiency_groups = compute_efficiency_groups(efficiency_scores)

    by_split = {"train": [], "val": [], "test": []}
    for record in records:
        subject_id = record["subject_id"]
        record["metadata"]["subject_efficiency_score"] = efficiency_scores.get(subject_id, 0.0)
        record["neuro_labels"]["snippet_targets"]["efficiency_group"] = efficiency_groups.get(subject_id, "medium")
        split = deterministic_split(subject_id, args.train_ratio, args.val_ratio)
        by_split[split].append(record)

    for split, split_records in by_split.items():
        write_jsonl(output_dir / f"{split}.jsonl", split_records)
        summary["records_written"][split] = len(split_records)
        print(f"Wrote {len(split_records)} records to {output_dir / f'{split}.jsonl'}")

    write_json(output_dir / "preprocess_summary.json", summary)
    print(f"Wrote summary to {output_dir / 'preprocess_summary.json'}")


if __name__ == "__main__":
    main()
