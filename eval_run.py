"""Run every eval case and save the traces. Scoring happens separately.

Splitting run from score is the whole point: model calls are slow, rate limited
and non-deterministic, so you pay for them once and then score the saved traces
as many times as you like while tuning metrics.
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from agent import MODEL, config_fingerprint, run
from calendar_backend import MockCalendar, use_backend
from eval_cases import CASES
from mock_data import CALENDAR, reset_calendar
from task_store import disable_persistence, reset_tasks

# The suite is pinned to the mock, whatever CALENDAR_BACKEND says. Scoring a run
# against a real calendar would create real events, defeat reset_calendar(), and
# make every score depend on that week's meetings.
use_backend(MockCalendar())

# A booking ticks its task off, and the suite books plenty. Keep that in memory:
# tasks.json belongs to whoever is using the UI.
disable_persistence()

RUNS_DIR = Path("runs")
TRACE_FILE = RUNS_DIR / "traces.json"


def run_case(case) -> dict:
    # Every case starts from the same calendar and the same task list, or results
    # depend on run order and on whatever the UI was last used to edit.
    reset_calendar()
    reset_tasks()
    started = time.perf_counter()

    # Scripted answers, consumed one per pause. Cases that say nothing get the
    # old behaviour: the gate is still asked, the harness still says yes.
    answers = list(case.approvals)

    def decide(request: dict) -> bool | str:
        return answers.pop(0) if answers else True

    try:
        state = run(case.instruction, decide=decide)
        final = state.final.model_dump() if state.final else None
        error = None
    except Exception as exc:  # a crashed case is a result, not a reason to abandon the suite
        state, final, error = None, None, f"{type(exc).__name__}: {exc}"

    return {
        "case": case.name,
        "instruction": case.instruction,
        "error": error,
        "seconds": round(time.perf_counter() - started, 1),
        "turns": state.turns if state else 0,
        "usage": state.usage.model_dump() if state else {},
        "path": state.steps if state else [],
        "tool_calls": [entry.model_dump() for entry in state.tool_log] if state else [],
        "final": final,
        # Ground truth the agent cannot fake: what the calendar actually holds now.
        "calendar_after": [dict(event) for event in CALENDAR],
    }


def main() -> None:
    only = sys.argv[1] if len(sys.argv) > 1 else None
    cases = [c for c in CASES if not only or c.name == only]
    if not cases:
        raise SystemExit(f"No case named {only!r}. Options: {', '.join(c.name for c in CASES)}")

    results = []
    for index, case in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case.name}: {case.instruction}", flush=True)
        result = run_case(case)
        status = result["error"] or f"{len(result['tool_calls'])} tool calls in {result['seconds']}s"
        print(f"      -> {status}\n", flush=True)
        results.append(result)

    # Re-running one case must not wipe the traces for the others.
    if only and TRACE_FILE.exists():
        previous = json.loads(TRACE_FILE.read_text(encoding="utf-8"))["results"]
        fresh = {result["case"] for result in results}
        results = [old for old in previous if old["case"] not in fresh] + results
        order = [case.name for case in CASES]
        results.sort(key=lambda result: order.index(result["case"]))

    payload = {
        "model": MODEL,
        "config": config_fingerprint(),
        "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "results": results,
    }
    RUNS_DIR.mkdir(exist_ok=True)
    TRACE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {len(results)} traces to {TRACE_FILE}")


if __name__ == "__main__":
    main()
