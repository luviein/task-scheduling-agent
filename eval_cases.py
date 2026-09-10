"""The eval set. Expectations are written down before the agent runs.

Each case says what a correct run looks like. If you find yourself editing an
expectation to make a case pass, that is the moment to stop and ask whether the
agent is wrong or the expectation was.
"""

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    name: str
    instruction: str
    expected_tools: list[str] = Field(description="Tools that must be called at least once.")
    forbidden_tools: list[str] = Field(default_factory=list, description="Tools that must never be called.")
    expect_tasks: list[str] = Field(default_factory=list, description="Task titles that must appear in tasks_found.")
    expect_events: list[str] = Field(default_factory=list, description="Event titles that must be booked.")
    expect_unresolved: bool = Field(default=False, description="Should the agent report something it could not do?")
    max_tool_calls: int = Field(description="Ceiling on tool calls: the minimum this case needs, plus one retry.")
    notes: str = ""


CASES: list[EvalCase] = [
    EvalCase(
        name="search_only",
        max_tool_calls=2,  # One search is enough. A second means it did not trust the first.
        instruction="What QA work do I have open? Don't book anything.",
        expected_tools=["search_tasks"],
        forbidden_tools=["create_calendar_event"],
        expect_tasks=["Refactor Playwright login suite"],
        notes="Reads only. Booking here is an over-reach, not a bonus.",
    ),
    EvalCase(
        name="book_high_priority",
        max_tool_calls=4,  # Search, check availability, book. One spare.
        instruction="Find my open AI tasks and book an hour tomorrow for the highest priority one.",
        expected_tools=["search_tasks", "find_free_slots", "create_calendar_event"],
        expect_tasks=["Write LangGraph agent draft"],
        expect_events=["Write LangGraph agent draft"],
        notes="Happy path: search, pick, book.",
    ),
    EvalCase(
        name="conflict_retry",
        max_tool_calls=4,  # Search, check availability, book. One spare.
        instruction="Book an hour at 9am tomorrow to work on the Playwright login suite refactor.",
        expected_tools=["search_tasks", "find_free_slots", "create_calendar_event"],
        expect_events=["Refactor Playwright login suite"],
        notes="09:00 collides with Standup, so availability offers 09:30. Tests that stated times yield to real availability.",
    ),
    EvalCase(
        name="no_such_task",
        max_tool_calls=3,  # Two searches is legitimate here; broadening is the point. No more.
        instruction="Schedule two hours tomorrow to do my tax return.",
        expected_tools=["search_tasks"],
        expect_events=[],
        expect_unresolved=True,
        notes="No matching task exists. The agent must say so, not invent one.",
    ),
    EvalCase(
        name="multi_step",
        max_tool_calls=6,  # Search, check availability, two bookings. Two spare.
        instruction="Book separate one-hour slots tomorrow afternoon for both of my open high priority AI tasks.",
        expected_tools=["search_tasks", "find_free_slots", "create_calendar_event"],
        expect_events=["Write LangGraph agent draft", "Publish DeepEval results"],
        notes="Two bookings, and 14:00 is taken by the design review.",
    ),
    EvalCase(
        name="ambiguous_target",
        max_tool_calls=2,  # One search to see the options. Nothing else is warranted.
        instruction="Book two hours on Friday.",
        expected_tools=["search_tasks"],
        forbidden_tools=["create_calendar_event"],
        expect_events=[],
        expect_unresolved=True,
        notes="No task is named. The agent must ask which one, not quietly pick for you.",
    ),
]
