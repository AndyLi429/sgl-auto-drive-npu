---
name: sglang-benchmark-result-analysis
description: Compare SGLang serving benchmark results before and after a code or configuration change. Parse plain-text SGLang benchmark output, calculate per-metric percentage changes, distinguish improvements from regressions using metric direction, and produce a concise Markdown performance report. Use whenever the user provides SGLang benchmark output, asks to compare two runs, analyze performance changes, or evaluate a PR's serving performance impact.
---

# SGLang Benchmark Result Analysis

Use this skill to turn two SGLang serving benchmark runs into an evidence-based comparison. Separate measured changes from interpretation: a percentage delta describes what changed, while the direction rules determine whether that change is better or worse.

## Workflow

1. Identify the input. Accept either one text file containing `before` and `after` sections or two separate files.
2. Preserve the original benchmark output. Do not manually retype metrics or silently discard metrics that the parser cannot recognize.
3. Run `scripts/compare_benchmarks.py` to parse both runs and generate the metrics table.
4. Read the generated report and add an `## AI Summary` section with the key findings.
5. Show the complete report and state any missing, incomparable, or ambiguous metrics.

## Running the comparison

For a combined before/after file:

```bash
python3 skills/sglang-benchmark-result-analysis/scripts/compare_benchmarks.py \
  --input benchmark.txt \
  --output-dir ./outputs \
  --title "SGLang Benchmark Comparison"
```

For two separate files:

```bash
python3 skills/sglang-benchmark-result-analysis/scripts/compare_benchmarks.py \
  --before before.txt --after after.txt \
  --output-dir ./outputs \
  --title "SGLang Benchmark Comparison"
```

The script uses only the Python standard library. Use `python` instead of `python3` on Windows when needed.

## Accepted input

The parser recognizes labels containing `before` and `after` (case-insensitive), followed by SGLang-style metric lines such as:

```text
Before change
============ Serving Benchmark Result ============
Request throughput (req/s): 12.50
Input token throughput (tok/s): 1536.00
Output token throughput (tok/s): 768.00
Mean TTFT (ms): 42.10
P99 TTFT (ms): 85.20
Mean TPOT (ms): 18.40
P99 TPOT (ms): 31.70
Failed requests: 0

After change
============ Serving Benchmark Result ============
Request throughput (req/s): 14.10
...
```

Metric lines must contain a name, a colon, and a numeric value. Commas in numbers, units after the value, and separator lines are supported. Metrics present in only one run are listed as unmatched instead of being compared.

## Interpretation rules

Use these defaults unless the benchmark or user specifies a different objective:

- Higher is better: request throughput, input token throughput, output token throughput, total token throughput, successful requests, and cache hit rate.
- Lower is better: TTFT, TPOT, ITL, request latency, benchmark duration, queue time, and failed requests.
- For percentile metrics, compare the same percentile only: P50 with P50, P99 with P99.
- Never infer end-to-end improvement from one metric. Check throughput together with TTFT/TPOT/ITL, failures, and workload comparability.
- Treat small changes as noise when they are within the benchmark's stated variance. If no repeated-run variance is available, explicitly say that significance is unknown.

The comparison column uses signed percentage change with an arrow, for example `+12.80% ↑` or `-6.40% ↓`. The arrow shows numerical direction only; the report must separately say whether that direction is an improvement.

## AI Summary requirements

Write 4–6 concise bullets covering:

- The most important improvement, with before → after values and percentage change.
- Request and token throughput trends.
- TTFT, TPOT, and ITL changes, including percentile tail latency when available.
- Failures, errors, or anomalous metrics.
- Whether the overall result is clearly better, mixed, unchanged, or inconclusive.
- Any workload, hardware, model, batch/concurrency, precision, or sampling differences that weaken the comparison.

Use bold for the most important values, such as `**12.50 → 14.10 req/s (+12.80%)**`. Do not use ✅/❌ markers. Do not claim causality from benchmark output alone; phrase causes as hypotheses and suggest the next measurement needed to validate them.

## Report format

```markdown
# SGLang Benchmark Comparison

## Test Context
<!-- model, hardware, backend, dtype, workload, concurrency, input/output lengths, commit, command -->

## Performance Metrics
| Category | Metric | Before | After | Change | Assessment |
|---|---|---:|---:|---:|---|

## AI Summary
- ...

## Comparability and Limitations
- ...

## Unmatched or Unparsed Metrics
- ...
```

Keep raw numbers and units in the table. Group latency metrics separately from throughput metrics when the result contains enough metrics to make that useful.

## Notes

- Prefer the benchmark's own metadata and command line when determining comparability.
- Keep the generated report under the user-specified output directory; default to `./outputs`.
- If only one run is provided, summarize it without inventing a comparison and ask for the missing baseline when appropriate.
