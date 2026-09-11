"""Fail when a score drops, instead of hoping somebody notices.

    python eval_gate.py            # compare runs/report.json against the baseline
    python eval_gate.py --accept   # bless the current report as the new baseline

A suite of numbers that nobody diffs is a suite of numbers nobody reads. This
compares the latest report against a blessed one, per case and per metric, and
exits non-zero on any drop. That exit code is the whole point: it turns "the
evals look fine" into something a script can assert.

Deliberately not part of CI. Producing a report costs model quota, so this runs
locally after `eval_run.py`, and the exit code is there for whoever wants to
wire it into a release step.

A case in the baseline but missing from the report counts as a regression.
Coverage going backwards is the cheapest way to make scores improve.
"""

import json
import sys
from pathlib import Path

from pydantic import BaseModel, Field

RUNS_DIR = Path("runs")
REPORT_FILE = RUNS_DIR / "report.json"
BASELINE_FILE = RUNS_DIR / "baseline.json"

# Scores are rounded to two places when written, so anything smaller than this is
# noise rather than a change.
TOLERANCE = 0.005

# Token counts move run to run even with the prompt unchanged, because the model
# is not deterministic. This is set wide enough to ignore that and narrow enough
# to catch a change that actually made the agent more expensive: an extra turn on
# a short case is roughly a third more tokens.
TOKEN_TOLERANCE = 0.25


class Change(BaseModel):
    case: str
    metric: str
    baseline: float
    current: float

    @property
    def delta(self) -> float:
        return self.current - self.baseline

    def line(self) -> str:
        return f"{self.case} / {self.metric}: {self.baseline:.2f} -> {self.current:.2f} ({self.delta:+.2f})"


class TokenChange(BaseModel):
    case: str
    baseline: int
    current: int

    @property
    def ratio(self) -> float:
        return self.current / self.baseline if self.baseline else 1.0

    def line(self) -> str:
        return (
            f"{self.case}: {self.baseline} -> {self.current} tokens "
            f"({(self.ratio - 1) * 100:+.0f}%)"
        )


class GateResult(BaseModel):
    regressions: list[Change] = Field(default_factory=list)
    improvements: list[Change] = Field(default_factory=list)
    missing_cases: list[str] = Field(default_factory=list, description="In the baseline, absent now.")
    new_cases: list[str] = Field(default_factory=list, description="Not in the baseline yet.")
    missing_metrics: list[str] = Field(default_factory=list, description="Scored before, not now.")
    costlier: list[TokenChange] = Field(
        default_factory=list, description="Cases that now spend materially more tokens."
    )
    cheaper: list[TokenChange] = Field(default_factory=list)
    config_changes: dict[str, tuple[object, object]] = Field(default_factory=dict)

    @property
    def failed(self) -> bool:
        return bool(
            self.regressions or self.missing_cases or self.missing_metrics or self.costlier
        )


def scores_of(report: dict) -> dict[str, dict[str, float]]:
    """{case name: {metric name: score}} from a report file."""
    return {
        row["case"]: {name: metric["score"] for name, metric in row["metrics"].items()}
        for row in report.get("cases", [])
    }


def tokens_of(report: dict) -> dict[str, int]:
    """{case name: total tokens}, skipping cases recorded before spend was tracked."""
    totals = {}
    for row in report.get("cases", []):
        usage = row.get("usage") or {}
        if usage:
            totals[row["case"]] = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
    return totals


def compare(baseline: dict, current: dict) -> GateResult:
    old, new = scores_of(baseline), scores_of(current)
    result = GateResult()

    for key, was in (baseline.get("config") or {}).items():
        now = (current.get("config") or {}).get(key)
        if now != was:
            result.config_changes[key] = (was, now)

    result.missing_cases = sorted(set(old) - set(new))
    result.new_cases = sorted(set(new) - set(old))

    for case in sorted(set(old) & set(new)):
        for metric, was in old[case].items():
            if metric not in new[case]:
                # Scoring with --no-judge drops a metric. Silently comparing the
                # rest would let a judged regression hide behind a cheaper run.
                result.missing_metrics.append(f"{case} / {metric}")
                continue
            now = new[case][metric]
            change = Change(case=case, metric=metric, baseline=was, current=now)
            if now < was - TOLERANCE:
                result.regressions.append(change)
            elif now > was + TOLERANCE:
                result.improvements.append(change)

    old_tokens, new_tokens = tokens_of(baseline), tokens_of(current)
    for case in sorted(set(old_tokens) & set(new_tokens)):
        was, now = old_tokens[case], new_tokens[case]
        change = TokenChange(case=case, baseline=was, current=now)
        if now > was * (1 + TOKEN_TOLERANCE):
            result.costlier.append(change)
        elif now < was * (1 - TOKEN_TOLERANCE):
            result.cheaper.append(change)

    result.missing_metrics.sort()
    return result


def render(result: GateResult) -> str:
    lines: list[str] = []

    if result.config_changes:
        lines.append("CONFIG CHANGED SINCE THE BASELINE")
        for key, (was, now) in sorted(result.config_changes.items()):
            lines.append(f"  {key}: {was} -> {now}")
        lines.append("  Scores are not strictly comparable across this change.")
        lines.append("")

    if result.regressions:
        lines.append(f"REGRESSIONS ({len(result.regressions)})")
        lines += [f"  {change.line()}" for change in result.regressions]
        lines.append("")

    if result.missing_cases:
        lines.append("CASES MISSING FROM THIS RUN")
        lines += [f"  {case}" for case in result.missing_cases]
        lines.append("")

    if result.missing_metrics:
        lines.append("METRICS NOT SCORED THIS RUN")
        lines += [f"  {entry}" for entry in result.missing_metrics]
        lines.append("  Re-run without --no-judge, or accept a new baseline on purpose.")
        lines.append("")

    if result.costlier:
        lines.append(f"MORE EXPENSIVE ({len(result.costlier)})")
        lines += [f"  {change.line()}" for change in result.costlier]
        lines.append("  A right answer for more tokens is still worse.")
        lines.append("")

    if result.cheaper:
        lines.append(f"CHEAPER ({len(result.cheaper)})")
        lines += [f"  {change.line()}" for change in result.cheaper]
        lines.append("")

    if result.improvements:
        lines.append(f"IMPROVEMENTS ({len(result.improvements)})")
        lines += [f"  {change.line()}" for change in result.improvements]
        lines.append("")

    if result.new_cases:
        lines.append("NEW CASES, NOT IN THE BASELINE")
        lines += [f"  {case}" for case in result.new_cases]
        lines.append("")

    lines.append(
        "FAIL: something went backwards" if result.failed else "PASS: nothing went backwards"
    )
    return "\n".join(lines)


def main() -> None:
    if not REPORT_FILE.exists():
        raise SystemExit(f"{REPORT_FILE} not found. Run: python eval_score.py")

    current = json.loads(REPORT_FILE.read_text(encoding="utf-8"))

    if "--accept" in sys.argv:
        BASELINE_FILE.write_text(json.dumps(current, indent=2), encoding="utf-8")
        cases = len(current.get("cases", []))
        print(f"Baseline set from {REPORT_FILE}: {cases} cases, config {current.get('config')}")
        return

    if not BASELINE_FILE.exists():
        raise SystemExit(
            f"No baseline at {BASELINE_FILE}. Set one with: python eval_gate.py --accept"
        )

    result = compare(json.loads(BASELINE_FILE.read_text(encoding="utf-8")), current)
    print(render(result))
    if result.failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
