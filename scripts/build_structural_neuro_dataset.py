from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from eeg_qwen.common import ensure_dir, read_jsonl, write_json, write_jsonl
from eeg_qwen.datasets import normalize_distribution


IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
NUMBER_PATTERN = re.compile(r"^\d+(?:\.\d+)?(?:[eE][-+]?\d+)?$")

CONTROL_KEYWORDS = {
    "if",
    "else",
    "for",
    "while",
    "do",
    "switch",
    "case",
    "catch",
    "return",
    "throw",
    "new",
    "try",
    "finally",
    "synchronized",
}

MODIFIER_KEYWORDS = {
    "public",
    "private",
    "protected",
    "static",
    "final",
    "abstract",
    "synchronized",
    "native",
    "strictfp",
}

SHARED_REGION_TYPES = [
    "function_def",
    "parameter_list",
    "assignment",
    "call",
    "if_branch",
    "loop_header",
    "loop_body",
    "return",
    "comparison",
    "literal_or_constant",
    "container_access",
    "recursion_or_self_call",
]


@dataclass
class Region:
    region_type: str
    start_token: int
    end_token: int
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_region_id: str | None = None
    depth: int = 0
    region_id: str | None = None


def min_max_scale(values: list[float]) -> list[float]:
    if not values:
        return []
    minimum = min(values)
    maximum = max(values)
    if maximum <= minimum:
        return [0.0 for _ in values]
    return [(value - minimum) / (maximum - minimum) for value in values]


def is_identifier(token: str) -> bool:
    return bool(IDENTIFIER_PATTERN.match(token))


def is_literal_or_constant(token: str) -> bool:
    lowered = token.lower()
    return lowered in {"true", "false", "null"} or bool(NUMBER_PATTERN.match(token)) or token in {'"', "'"}


def build_match_map(tokens: list[str], opening: str, closing: str) -> dict[int, int]:
    stack: list[int] = []
    matches: dict[int, int] = {}
    for index, token in enumerate(tokens):
        if token == opening:
            stack.append(index)
        elif token == closing and stack:
            start = stack.pop()
            matches[start] = index
    return matches


def statement_start(tokens: list[str], index: int) -> int:
    cursor = index
    while cursor > 0 and tokens[cursor - 1] not in {";", "{", "}"}:
        cursor -= 1
    return cursor


def statement_end(tokens: list[str], index: int) -> int:
    cursor = index
    while cursor + 1 < len(tokens) and tokens[cursor] not in {";", "{", "}"}:
        cursor += 1
        if tokens[cursor] == ";":
            return cursor
    return min(cursor, len(tokens) - 1)


def infer_body_span(tokens: list[str], start_index: int, brace_matches: dict[int, int]) -> tuple[int, int] | None:
    if start_index >= len(tokens):
        return None
    if tokens[start_index] == "{":
        close = brace_matches.get(start_index)
        if close is None or close <= start_index:
            return None
        return start_index + 1, close - 1
    end = statement_end(tokens, start_index)
    return start_index, end


def is_assignment_operator(tokens: list[str], index: int) -> bool:
    if tokens[index] != "=":
        return False
    if index > 0 and tokens[index - 1] in {"=", "!", "<", ">", "+", "-", "*", "/", "%", "&", "|", "^"}:
        return False
    if index + 1 < len(tokens) and tokens[index + 1] == "=":
        return False
    return True


def collect_function_regions(tokens: list[str], paren_matches: dict[int, int], brace_matches: dict[int, int]) -> tuple[list[Region], set[int], str | None]:
    regions: list[Region] = []
    function_name_indices: set[int] = set()
    primary_function_name: str | None = None

    for index in range(len(tokens) - 1):
        if not is_identifier(tokens[index]) or tokens[index] in CONTROL_KEYWORDS:
            continue
        if tokens[index + 1] != "(":
            continue
        close_paren = paren_matches.get(index + 1)
        if close_paren is None or close_paren + 1 >= len(tokens) or tokens[close_paren + 1] != "{":
            continue

        start = index
        while start > 0 and tokens[start - 1] not in {";", "{", "}"}:
            start -= 1
        if start == index and index > 0 and tokens[index - 1] == ".":
            continue

        regions.append(
            Region(
                region_type="function_def",
                start_token=start,
                end_token=close_paren,
                metadata={"function_name": tokens[index]},
            )
        )
        if index + 2 <= close_paren - 1:
            regions.append(
                Region(
                    region_type="parameter_list",
                    start_token=index + 2,
                    end_token=close_paren - 1,
                    metadata={"function_name": tokens[index]},
                )
            )
        function_name_indices.add(index)
        if primary_function_name is None:
            primary_function_name = tokens[index]

    return regions, function_name_indices, primary_function_name


def collect_branch_and_loop_regions(tokens: list[str], paren_matches: dict[int, int], brace_matches: dict[int, int]) -> list[Region]:
    regions: list[Region] = []
    for index, token in enumerate(tokens):
        if token == "if":
            header_end = index
            if index + 1 < len(tokens) and tokens[index + 1] == "(":
                header_end = paren_matches.get(index + 1, index)
            regions.append(Region(region_type="if_branch", start_token=index, end_token=header_end))
        elif token in {"for", "while"}:
            header_end = index
            if index + 1 < len(tokens) and tokens[index + 1] == "(":
                header_end = paren_matches.get(index + 1, index)
            regions.append(Region(region_type="loop_header", start_token=index, end_token=header_end))
            body_span = infer_body_span(tokens, header_end + 1, brace_matches)
            if body_span is not None and body_span[0] <= body_span[1]:
                regions.append(
                    Region(
                        region_type="loop_body",
                        start_token=body_span[0],
                        end_token=body_span[1],
                        metadata={"loop_keyword": token},
                    )
                )
    return regions


def collect_return_regions(tokens: list[str]) -> list[Region]:
    regions = []
    for index, token in enumerate(tokens):
        if token == "return":
            end = statement_end(tokens, index)
            regions.append(Region(region_type="return", start_token=index, end_token=end))
    return regions


def collect_assignment_regions(tokens: list[str]) -> list[Region]:
    regions = []
    for index in range(len(tokens)):
        if is_assignment_operator(tokens, index):
            start = statement_start(tokens, index)
            end = statement_end(tokens, index)
            regions.append(Region(region_type="assignment", start_token=start, end_token=end))
    return regions


def collect_call_regions(
    tokens: list[str],
    paren_matches: dict[int, int],
    function_name_indices: set[int],
    primary_function_name: str | None,
) -> list[Region]:
    regions = []
    for index in range(len(tokens) - 1):
        token = tokens[index]
        if not is_identifier(token) or token in CONTROL_KEYWORDS:
            continue
        if tokens[index + 1] != "(":
            continue
        if index in function_name_indices:
            continue
        if index > 0 and tokens[index - 1] == "." and token in {"if", "for", "while"}:
            continue
        close_paren = paren_matches.get(index + 1)
        if close_paren is None:
            continue
        regions.append(Region(region_type="call", start_token=index, end_token=close_paren, metadata={"callee": token}))
        if primary_function_name is not None and token == primary_function_name:
            regions.append(
                Region(
                    region_type="recursion_or_self_call",
                    start_token=index,
                    end_token=close_paren,
                    metadata={"callee": token},
                )
            )
    return regions


def collect_comparison_regions(tokens: list[str]) -> list[Region]:
    regions = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        span_end = index
        matched = False
        if token in {"<", ">"}:
            if index + 1 < len(tokens) and tokens[index + 1] == "=":
                span_end = index + 1
            matched = True
        elif token in {"!", "="} and index + 1 < len(tokens) and tokens[index + 1] == "=":
            span_end = index + 1
            matched = True

        if matched:
            start = max(0, index - 1)
            end = min(len(tokens) - 1, span_end + 1)
            regions.append(Region(region_type="comparison", start_token=start, end_token=end))
            index = span_end + 1
        index += 1
    return regions


def collect_literal_regions(tokens: list[str]) -> list[Region]:
    regions = []
    for index, token in enumerate(tokens):
        if is_literal_or_constant(token):
            regions.append(Region(region_type="literal_or_constant", start_token=index, end_token=index))
    return regions


def collect_container_regions(tokens: list[str], bracket_matches: dict[int, int]) -> list[Region]:
    regions = []
    for index in range(len(tokens) - 1):
        if is_identifier(tokens[index]) and tokens[index + 1] == "[":
            close = bracket_matches.get(index + 1)
            if close is not None:
                regions.append(Region(region_type="container_access", start_token=index, end_token=close))
    return regions


def dedupe_regions(regions: list[Region]) -> list[Region]:
    deduped: list[Region] = []
    seen: set[tuple[str, int, int, tuple[tuple[str, Any], ...]]] = set()
    for region in regions:
        if region.start_token > region.end_token:
            continue
        key = (
            region.region_type,
            region.start_token,
            region.end_token,
            tuple(sorted(region.metadata.items())),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(region)
    deduped.sort(key=lambda item: (item.start_token, item.end_token, item.region_type))
    return deduped


def assign_region_hierarchy(regions: list[Region]) -> list[dict[str, Any]]:
    for index, region in enumerate(regions):
        region.region_id = f"r{index:04d}"

    for region in regions:
        parent: Region | None = None
        for candidate in regions:
            if candidate.region_id == region.region_id:
                continue
            if candidate.start_token <= region.start_token and candidate.end_token >= region.end_token:
                if candidate.start_token == region.start_token and candidate.end_token == region.end_token:
                    continue
                if parent is None:
                    parent = candidate
                else:
                    candidate_span = candidate.end_token - candidate.start_token
                    parent_span = parent.end_token - parent.start_token
                    if candidate_span < parent_span:
                        parent = candidate
        if parent is not None:
            region.parent_region_id = parent.region_id

    region_by_id = {region.region_id: region for region in regions if region.region_id is not None}
    for region in regions:
        depth = 0
        parent_id = region.parent_region_id
        while parent_id is not None and parent_id in region_by_id:
            depth += 1
            parent_id = region_by_id[parent_id].parent_region_id
        region.depth = depth

    return [serialize_region(region) for region in regions]


def serialize_region(region: Region) -> dict[str, Any]:
    return {
        "region_id": region.region_id,
        "region_type": region.region_type,
        "start_token": region.start_token,
        "end_token": region.end_token,
        "token_count": region.end_token - region.start_token + 1,
        "depth": region.depth,
        "parent_region_id": region.parent_region_id,
        "metadata": region.metadata,
    }


def build_control_edges(regions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    ordered = sorted(regions, key=lambda item: (item["start_token"], item["end_token"], item["region_type"]))
    for left, right in zip(ordered, ordered[1:]):
        edges.append({"source": left["region_id"], "target": right["region_id"], "type": "next"})
    for region in ordered:
        parent = region.get("parent_region_id")
        if parent:
            edges.append({"source": parent, "target": region["region_id"], "type": "contains"})
    return edges


def aggregate_region_targets(
    regions: list[dict[str, Any]],
    attention_values: list[float],
    revisit_values: list[float],
    processing_cost: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not regions:
        return [], {
            "attention_importance": [],
            "revisit_score": [],
            "processing_cost": processing_cost,
        }

    attention_raw = []
    revisit_raw = []
    for region in regions:
        start = region["start_token"]
        end = region["end_token"] + 1
        span_attention = attention_values[start:end]
        span_revisit = revisit_values[start:end]
        attention_raw.append(sum(span_attention))
        revisit_raw.append(max(span_revisit) if span_revisit else 0.0)

    attention_norm = normalize_distribution(attention_raw)
    revisit_norm = min_max_scale(revisit_raw)

    enriched = []
    for region, raw_attention, raw_revisit, norm_attention, norm_revisit in zip(
        regions,
        attention_raw,
        revisit_raw,
        attention_norm,
        revisit_norm,
    ):
        enriched_region = dict(region)
        enriched_region["targets"] = {
            "attention_mass_raw": raw_attention,
            "revisit_peak_raw": raw_revisit,
            "attention_importance": norm_attention,
            "revisit_score": norm_revisit,
        }
        enriched.append(enriched_region)

    return enriched, {
        "attention_importance": attention_norm,
        "revisit_score": revisit_norm,
        "processing_cost": processing_cost,
    }


def lift_record(record: dict[str, Any]) -> dict[str, Any]:
    tokens = list(record["neuro_labels"]["tokens"])
    attention_values = [float(value) for value in record["neuro_labels"]["token_targets"]["attention_importance"]]
    revisit_values = [float(value) for value in record["neuro_labels"]["token_targets"]["revisit_score"]]
    processing_cost = float(record["neuro_labels"]["snippet_targets"]["processing_cost"])

    paren_matches = build_match_map(tokens, "(", ")")
    brace_matches = build_match_map(tokens, "{", "}")
    bracket_matches = build_match_map(tokens, "[", "]")

    function_regions, function_name_indices, primary_function_name = collect_function_regions(tokens, paren_matches, brace_matches)
    regions = []
    regions.extend(function_regions)
    regions.extend(collect_branch_and_loop_regions(tokens, paren_matches, brace_matches))
    regions.extend(collect_return_regions(tokens))
    regions.extend(collect_assignment_regions(tokens))
    regions.extend(collect_call_regions(tokens, paren_matches, function_name_indices, primary_function_name))
    regions.extend(collect_comparison_regions(tokens))
    regions.extend(collect_literal_regions(tokens))
    regions.extend(collect_container_regions(tokens, bracket_matches))
    deduped_regions = dedupe_regions(regions)
    serialized_regions = assign_region_hierarchy(deduped_regions)
    enriched_regions, region_targets = aggregate_region_targets(
        serialized_regions,
        attention_values=attention_values,
        revisit_values=revisit_values,
        processing_cost=processing_cost,
    )

    output = dict(record)
    output["structural_labels"] = {
        "ontology_version": "shared_struct_v1",
        "builder": "heuristic_java_control_skeleton",
        "shared_region_types": SHARED_REGION_TYPES,
        "regions": enriched_regions,
        "region_targets": region_targets,
        "control_edges": build_control_edges(enriched_regions),
    }
    return output


def process_split(path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    records = read_jsonl(path)
    lifted_records = []
    region_type_counts: Counter[str] = Counter()
    region_counts = []
    for record in records:
        lifted = lift_record(record)
        lifted_records.append(lifted)
        regions = lifted["structural_labels"]["regions"]
        region_counts.append(len(regions))
        for region in regions:
            region_type_counts[region["region_type"]] += 1

    summary = {
        "records": len(lifted_records),
        "avg_regions_per_record": sum(region_counts) / len(region_counts) if region_counts else 0.0,
        "max_regions_per_record": max(region_counts) if region_counts else 0,
        "region_type_counts": dict(sorted(region_type_counts.items())),
    }
    return lifted_records, summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Lift existing token-level NoviceVsExpert neuro targets into a shared structural "
            "ontology using a lightweight Java control-skeleton heuristic."
        )
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing train/val/test JSONL neuro splits.")
    parser.add_argument("--output-dir", required=True, help="Directory where structural JSONL splits will be written.")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = ensure_dir(args.output_dir)

    overall_summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "ontology_version": "shared_struct_v1",
        "builder": "heuristic_java_control_skeleton",
        "shared_region_types": SHARED_REGION_TYPES,
        "splits": {},
    }

    for split in ("train", "val", "test"):
        input_path = input_dir / f"{split}.jsonl"
        if not input_path.exists():
            raise FileNotFoundError(f"Missing expected input split: {input_path}")
        records, summary = process_split(input_path)
        write_jsonl(output_dir / f"{split}.jsonl", records)
        overall_summary["splits"][split] = summary
        print(f"Wrote {len(records)} structural records to {output_dir / f'{split}.jsonl'}")

    write_json(output_dir / "structural_summary.json", overall_summary)
    print(f"Wrote summary to {output_dir / 'structural_summary.json'}")


if __name__ == "__main__":
    main()
