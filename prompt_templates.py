from __future__ import annotations

import json


SYSTEM_PROMPT = (
    "You are editing an existing code file.\n"
    "Return the full updated file contents only.\n"
)

NEURO_PROMPT = (
    "You are analyzing source code.\n"
    "Read the file carefully so your hidden states can model human processing difficulty, attention, and revisit patterns.\n"
)

LIVECODEBENCH_SYSTEM_PROMPT = (
    "You are a strong Python competitive programming model.\n"
    "Solve the problem and return only runnable Python code.\n"
    "Do not include markdown fences or explanation.\n"
)


def _parse_metadata(record: dict) -> dict:
    raw = record.get("metadata") or {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


def _infer_livecodebench_task_format(record: dict) -> str:
    task_format = str(record.get("task_format") or "").strip().lower()
    if task_format in {"functional", "stdin"}:
        return task_format
    starter_code = (record.get("starter_code") or "").strip()
    metadata = _parse_metadata(record)
    if metadata.get("func_name") or "class Solution" in starter_code:
        return "functional"
    return "stdin"


def _format_context_files(record: dict) -> str:
    context_files = record.get("context_files") or []
    if not context_files:
        return "None\n"

    chunks: list[str] = []
    for item in context_files:
        path = item.get("path", "unknown")
        language = item.get("language") or record.get("language") or "text"
        content = item.get("content", "")
        chunks.append(
            f"Context file: {path}\n```{language}\n{content}\n```\n"
        )
    return "\n".join(chunks)


def render_training_prompt(record: dict) -> str:
    language = record.get("language") or "text"
    file_path = record.get("file_path") or "main.txt"
    instruction = record.get("instruction", "").strip()
    file_content = record.get("file_content", "")
    test_feedback = (record.get("optional_test_feedback") or "").strip()

    sections = [
        SYSTEM_PROMPT,
        f"Instruction:\n{instruction}\n",
        f"File: {file_path}\n```{language}\n{file_content}\n```\n",
        "Additional context:\n" + _format_context_files(record),
    ]
    if test_feedback:
        sections.append(f"Optional test feedback:\n{test_feedback}\n")
    sections.append("Updated file:\n")
    return "\n".join(sections)


def render_generation_prompt(record: dict) -> str:
    return render_training_prompt(record)


def render_neuro_prompt(record: dict) -> str:
    language = record.get("language") or "text"
    file_path = record.get("file_path") or record.get("snippet_id") or "snippet.txt"
    file_content = record.get("file_content") or record.get("code") or ""

    sections = [
        NEURO_PROMPT,
        "Task:\nAnalyze the code for human processing signals.\n",
        f"File: {file_path}\n```{language}\n{file_content}\n```\n",
        "Analysis:\n",
    ]
    return "\n".join(sections)


def render_livecodebench_prompt(record: dict) -> str:
    question_content = (record.get("question_content") or "").strip()
    starter_code = (record.get("starter_code") or "").strip()
    task_format = _infer_livecodebench_task_format(record)
    use_task_format_hint = bool(record.get("use_task_format_hint", False))
    use_focus_hint = bool(record.get("use_cognitive_focus_hint", False))
    focus_hint = (record.get("cognitive_focus_hint") or "").strip() if use_focus_hint else ""
    sections = [
        LIVECODEBENCH_SYSTEM_PROMPT,
        f"Problem:\n{question_content}\n",
    ]
    if starter_code:
        sections.append(f"Starter code:\n```python\n{starter_code}\n```\n")
    if use_task_format_hint:
        if task_format == "functional":
            sections.append(
                "Task format:\n"
                "Complete the provided class and method implementation.\n"
                "Preserve the class name, method name, and signature.\n"
                "Do not write stdin parsing code or explanation.\n"
            )
        else:
            sections.append(
                "Task format:\n"
                "Write a complete Python program that reads from stdin and writes to stdout.\n"
                "Do not wrap the answer in class Solution unless the starter code requires it.\n"
            )
    if focus_hint:
        sections.append(f"Human focus cue:\n{focus_hint}\n")
    sections.append("Return only the final Python program.\n")
    return "\n".join(sections)
