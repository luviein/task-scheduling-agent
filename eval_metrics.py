"""Metrics.

Four are plain assertions over the trace. One uses an LLM judge.

That ratio is deliberate. "Did it call the right tool" and "does this title
exist" are set comparisons; running a model to answer them would be slower,
costlier and less reliable than the code below. The judge is reserved for the
one question that is genuinely fuzzy: is the summary honest about what happened?
"""

from collections import Counter

from deepeval.metrics import BaseMetric, GEval
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from google.genai import types

from agent import MODEL, call_model
from mock_data import TASKS as SEED_TASKS

SEEDED_EVENT_TITLES = {"Standup", "Design review"}
# The seed, deliberately: a run is scored against the list it ran against, not
# against whatever the UI has since been used to add or rename.
REAL_TASK_TITLES = {task["title"] for task in SEED_TASKS}


# --- Judge model ------------------------------------------------------------
class GeminiJudge(DeepEvalBaseLLM):
    """DeepEval defaults to an OpenAI judge. This points it at Gemini instead."""

    def __init__(self, model: str = MODEL):
        self.model_name = model

    def load_model(self):
        return self

    def get_model_name(self) -> str:
        return f"Gemini {self.model_name}"

    def generate(self, prompt: str, schema=None):
        config = types.GenerateContentConfig()
        if schema is not None:
            config.response_mime_type = "application/json"
            config.response_schema = schema
        # Reuses the agent's rate-limit retry; the judge shares the same quota.
        response = call_model(model=self.model_name, contents=prompt, config=config)
        return response.parsed if schema is not None else response.text

    async def a_generate(self, prompt: str, schema=None):
        return self.generate(prompt, schema=schema)


# --- Deterministic metrics --------------------------------------------------
class _RuleMetric(BaseMetric):
    """Shared plumbing. Subclasses only implement `evaluate`."""

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self.async_mode = False
        self.evaluation_model = "rule-based, no LLM"
        self.evaluation_cost = 0.0

    def evaluate(self, trace: dict, case: dict) -> tuple[float, str]:
        raise NotImplementedError

    def measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        meta = test_case.metadata or {}
        self.score, self.reason = self.evaluate(meta.get("trace", {}), meta.get("case", {}))
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args, **kwargs) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)


class ToolSelectionMetric(_RuleMetric):
    """Did it reach for the right tools, and keep its hands off the wrong ones?"""

    __name__ = "Tool Selection"

    def evaluate(self, trace, case):
        called = {call["tool"] for call in trace.get("tool_calls", [])}
        expected = set(case.get("expected_tools", []))
        forbidden = set(case.get("forbidden_tools", []))

        missing = expected - called
        trespassed = forbidden & called

        hit_rate = len(expected & called) / len(expected) if expected else 1.0
        score = 0.0 if trespassed else hit_rate

        notes = []
        if missing:
            notes.append(f"never called {', '.join(sorted(missing))}")
        if trespassed:
            notes.append(f"called forbidden {', '.join(sorted(trespassed))}")
        return score, "; ".join(notes) or f"called exactly what was expected: {', '.join(sorted(called)) or 'nothing'}"


class TaskExecutionMetric(_RuleMetric):
    """Did the work actually land?

    Scored against the calendar, not the agent's own summary. An agent that
    claims success while booking nothing must not be able to pass this.
    """

    __name__ = "Task Execution"

    def evaluate(self, trace, case):
        booked = {event["title"] for event in trace.get("calendar_after", [])} - SEEDED_EVENT_TITLES
        expected = set(case.get("expect_events", []))

        if expected:
            booking_score = len(expected & booked) / len(expected)
            detail = f"booked {sorted(booked) or 'nothing'}, expected {sorted(expected)}"
        else:
            booking_score = 0.0 if booked else 1.0
            detail = f"expected no bookings, got {sorted(booked) or 'none'}"

        final = trace.get("final") or {}
        flagged = bool(final.get("unresolved"))
        honest = flagged == case.get("expect_unresolved", False)
        if not honest:
            detail += f"; unresolved={flagged} but expected {case.get('expect_unresolved')}"

        return (booking_score + (1.0 if honest else 0.0)) / 2, detail


class TaskGroundingMetric(_RuleMetric):
    """Every task title the agent reports must exist in the source data.

    This is the hallucination check, and it needs no judge: the set of real
    task titles is known, so a made-up name is a lookup miss.
    """

    __name__ = "Task Grounding"

    def evaluate(self, trace, case):
        final = trace.get("final") or {}
        claimed = [*final.get("tasks_found", []), *final.get("events_created", [])]
        if not claimed:
            return 1.0, "reported no task titles, nothing to invent"

        invented = [title for title in claimed if title not in REAL_TASK_TITLES]
        score = 1 - len(invented) / len(claimed)
        if invented:
            return score, f"invented {len(invented)} of {len(claimed)} titles: {invented}"
        return score, f"all {len(claimed)} reported titles exist in the task list"


class ToolEfficiencyMetric(_RuleMetric):
    """Did it get there without wasting calls?

    Correctness metrics are blind to an agent that reaches the right answer the
    long way round. On a metered API that costs real money, and on a free tier
    with a daily cap it costs you the ability to run the suite twice. A ceiling
    per case turns "it worked" into "it worked without thrashing".
    """

    __name__ = "Tool Efficiency"

    def evaluate(self, trace, case):
        used = len(trace.get("tool_calls", []))
        ceiling = case.get("max_tool_calls", 0)
        if not ceiling:
            return 1.0, "no ceiling set for this case"
        if used <= ceiling:
            return 1.0, f"{used} calls, ceiling {ceiling}"

        repeats = Counter(call["tool"] for call in trace["tool_calls"])
        worst = max(repeats.items(), key=lambda item: item[1])
        # Graded, not binary: one call over is not the same failure as five.
        return ceiling / used, f"{used} calls, ceiling {ceiling}; {worst[0]} called {worst[1]} times"


# --- LLM-judged metric ------------------------------------------------------
def summary_faithfulness_metric() -> GEval:
    """The one thing worth a judge: does the prose match what the tools returned?

    A summary can be fully grounded in real titles and still misdescribe the
    outcome, for example claiming a booking that the tool refused.
    """
    return GEval(
        name="Summary Faithfulness",
        model=GeminiJudge(),
        threshold=0.7,
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.CONTEXT],
        evaluation_steps=[
            "The context lists every tool call the agent made and the exact result of each.",
            "Check whether the summary in the output describes only outcomes the tool results support.",
            "Penalise claiming a booking that the tool results show was refused or never attempted.",
            "Penalise stating a date or time that contradicts the booked event in the tool results.",
            "Do not penalise brevity, and do not penalise omitting detail that the user did not ask for.",
        ],
    )
