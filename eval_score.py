"""Score saved traces. Reads traces.json, writes a report, spends no agent calls.

    python eval_score.py             # all metrics, judge included
    python eval_score.py --no-judge  # deterministic metrics only, zero quota
"""

import json
import sys
from pathlib import Path

# Must happen before deepeval prints anything: it emits symbols the default
# Windows codepage cannot encode, which otherwise kills the run at the last line.
for stream in (sys.stdout, sys.stderr):
    stream.reconfigure(encoding="utf-8", errors="replace")

from deepeval import evaluate  # noqa: E402
from deepeval.evaluate.configs import AsyncConfig, CacheConfig, DisplayConfig, ErrorConfig
from deepeval.test_case import LLMTestCase

from eval_cases import CASES
from eval_metrics import (
    TaskExecutionMetric,
    ToolEfficiencyMetric,
    TaskGroundingMetric,
    ToolSelectionMetric,
    summary_faithfulness_metric,
)

TRACE_FILE = Path("traces.json")
REPORT_FILE = Path("eval_report.json")


def build_test_cases(traces: list[dict]) -> list[LLMTestCase]:
    by_name = {case.name: case for case in CASES}
    test_cases = []

    for trace in traces:
        case = by_name[trace["case"]]
        final = trace.get("final") or {}

        # Context is the evidence the judge is allowed to reason from: what the
        # tools actually returned, not what the agent says they returned.
        context = [
            f"{call['tool']} -> {json.dumps(call.get('output') or call.get('error'))}"
            for call in trace.get("tool_calls", [])
        ] or ["no tools were called"]

        test_cases.append(
            LLMTestCase(
                name=case.name,
                input=case.instruction,
                actual_output=final.get("summary") or trace.get("error") or "(no summary produced)",
                context=context,
                metadata={"trace": trace, "case": case.model_dump()},
            )
        )
    return test_cases


def main() -> None:
    use_judge = "--no-judge" not in sys.argv

    if not TRACE_FILE.exists():
        raise SystemExit(f"{TRACE_FILE} not found. Run: python eval_run.py")

    payload = json.loads(TRACE_FILE.read_text(encoding="utf-8"))
    test_cases = build_test_cases(payload["results"])

    metrics = [ToolSelectionMetric(), ToolEfficiencyMetric(), TaskExecutionMetric(), TaskGroundingMetric()]
    if use_judge:
        metrics.append(summary_faithfulness_metric())

    print(f"Scoring {len(test_cases)} traces from {payload['model']} recorded {payload['recorded_at']}")
    print(f"Metrics: {', '.join(m.__name__ for m in metrics)}\n")

    results = evaluate(
        test_cases=test_cases,
        metrics=metrics,
        # Serial on purpose: the judge shares the agent's per-minute quota.
        async_config=AsyncConfig(run_async=False),
        display_config=DisplayConfig(print_results=False, show_indicator=False, inspect_after_run=False),
        cache_config=CacheConfig(write_cache=False),
        error_config=ErrorConfig(ignore_errors=True),
    )

    report, totals = [], {}
    for result in results.test_results:
        row = {"case": result.name, "metrics": {}}
        for metric in result.metrics_data or []:
            row["metrics"][metric.name] = {
                "score": round(metric.score or 0.0, 2),
                "passed": bool(metric.success),
                "reason": metric.reason,
            }
            totals.setdefault(metric.name, []).append(metric.score or 0.0)
        report.append(row)

    for row in report:
        failures = [name for name, m in row["metrics"].items() if not m["passed"]]
        print(f"{'PASS' if not failures else 'FAIL'}  {row['case']}")
        for name, m in row["metrics"].items():
            mark = "ok  " if m["passed"] else "FAIL"
            print(f"      {mark} {name:<22} {m['score']:.2f}  {m['reason']}")
        print()

    print("AGGREGATE")
    for name, scores in totals.items():
        print(f"  {name:<22} {sum(scores) / len(scores):.2f}")

    REPORT_FILE.write_text(
        json.dumps({"model": payload["model"], "recorded_at": payload["recorded_at"], "cases": report}, indent=2),
        encoding="utf-8",
    )
    print(f"\nWrote {REPORT_FILE}")


if __name__ == "__main__":
    main()
