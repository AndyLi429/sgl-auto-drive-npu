#!/usr/bin/env python3
"""Compare two plain-text SGLang benchmark outputs."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


NUMBER = r"[-+]?(?:\d[\d,]*\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
METRIC_RE = re.compile(rf"^\s*(?P<name>[^:]+?)\s*:\s*(?P<value>{NUMBER})\b(?P<unit>.*)$")


def parse_metrics(text: str) -> dict[str, tuple[float, str]]:
    metrics: dict[str, tuple[float, str]] = {}
    for line in text.splitlines():
        match = METRIC_RE.match(line)
        if not match or set(line.strip()) <= {"=", "-", "_"}:
            continue
        name = re.sub(r"\s+", " ", match.group("name").strip())
        value = float(match.group("value").replace(",", ""))
        unit = match.group("unit").strip()
        metrics[name] = (value, unit)
    return metrics


def split_combined(text: str) -> tuple[str, str]:
    labels = list(re.finditer(r"^.*\b(before|after)\b.*$", text, re.I | re.M))
    before = next((m for m in labels if m.group(1).lower() == "before"), None)
    after = next((m for m in labels if m.group(1).lower() == "after"), None)
    if not before or not after:
        raise ValueError("combined input must contain labels with 'before' and 'after'")
    if before.start() < after.start():
        return text[before.end():after.start()], text[after.end():]
    return text[after.end():before.start()], text[before.end():]


def direction(name: str) -> str:
    lower = name.lower()
    if any(word in lower for word in ("throughput", "successful", "cache hit")):
        return "higher"
    if any(word in lower for word in ("ttft", "tpot", "itl", "latency", "duration", "queue", "failed", "error")):
        return "lower"
    return "unknown"


def format_value(value: float, unit: str) -> str:
    return f"{value:g}{(' ' + unit) if unit else ''}"


def render(before: dict[str, tuple[float, str]], after: dict[str, tuple[float, str]], title: str) -> str:
    names = list(dict.fromkeys([*before.keys(), *after.keys()]))
    rows = []
    unmatched = []
    for name in names:
        if name not in before or name not in after:
            unmatched.append(name)
            continue
        old, old_unit = before[name]
        new, new_unit = after[name]
        change = "n/a" if old == 0 else f"{(new - old) / abs(old) * 100:+.2f}% {'↑' if new >= old else '↓'}"
        kind = direction(name)
        if kind == "higher":
            assessment = "improvement" if new > old else "regression" if new < old else "unchanged"
        elif kind == "lower":
            assessment = "improvement" if new < old else "regression" if new > old else "unchanged"
        else:
            assessment = "direction depends on objective"
        unit = new_unit or old_unit
        category = "Latency" if kind == "lower" else "Throughput" if kind == "higher" else "Other"
        rows.append((category, name, format_value(old, unit), format_value(new, unit), change, assessment))

    lines = [f"# {title}", "", "## Performance Metrics", "", "| Category | Metric | Before | After | Change | Assessment |", "|---|---|---:|---:|---:|---|"]
    lines.extend(f"| {a} | {b} | {c} | {d} | {e} | {f} |" for a, b, c, d, e, f in rows)
    lines.extend(["", "## AI Summary", "", "<!-- Add a 4–6 bullet evidence-based summary here. -->", "", "## Unmatched Metrics", ""])
    lines.extend(f"- `{name}`" for name in unmatched) if unmatched else lines.append("- None")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path)
    source.add_argument("--before", type=Path)
    parser.add_argument("--after", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("./outputs"))
    parser.add_argument("--title", default="SGLang Benchmark Comparison")
    args = parser.parse_args()
    if args.before and not args.after:
        parser.error("--after is required with --before")
    if args.input:
        before_text, after_text = split_combined(args.input.read_text(encoding="utf-8"))
    else:
        before_text = args.before.read_text(encoding="utf-8")
        after_text = args.after.read_text(encoding="utf-8")
    report = render(parse_metrics(before_text), parse_metrics(after_text), args.title)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "sglang_benchmark_comparison.md"
    output.write_text(report, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
