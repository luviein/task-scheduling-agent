"""The parts of the graph that decide things without calling a model.

Routing, the conflict warning shown at the approval gate, and the schema adapter
between the two providers. All pure, all cheap, and all places where a quiet bug
would be expensive: a routing mistake ends a run early, and a wrong conflict
warning tells a human the opposite of the truth.
"""

import pytest

from agent import (
    MAX_TURNS,
    AgentState,
    FinalAnswer,
    _strip_for_gemini,
    nudge,
    overlaps_slot,
    route_after_reason,
    system_prompt,
)

BOOKED = {"title": "Design review", "from": "14:00", "to": "15:00"}


def state(**kwargs) -> AgentState:
    return AgentState(user_input="anything", **kwargs)


# --- the conflict warning at the approval gate ------------------------------
@pytest.mark.parametrize(
    "start, minutes, expected",
    [
        ("14:00", 25, True),  # starts exactly when the meeting does
        ("13:45", 25, True),  # runs into the front of it
        ("14:30", 15, True),  # sits wholly inside it
        ("13:00", 60, False),  # ends exactly as it begins
        ("15:00", 30, False),  # starts exactly as it ends
        ("09:00", 30, False),  # nowhere near
    ],
)
def test_overlaps_slot(start, minutes, expected):
    assert overlaps_slot("2026-09-12", start, minutes, BOOKED) is expected


@pytest.mark.parametrize(
    "day, start, booked",
    [
        ("", "14:00", BOOKED),  # no date
        ("2026-09-12", None, BOOKED),  # the model omitted a start time
        ("not-a-date", "14:00", BOOKED),  # unparseable
        ("2026-09-12", "14:00", {"title": "Broken"}),  # agenda entry missing its times
    ],
)
def test_overlaps_slot_stays_quiet_on_bad_input(day, start, booked):
    """A malformed proposal is the dispatcher's problem. The warning must not crash."""
    assert overlaps_slot(day, start, 30, booked) is False


# --- routing ----------------------------------------------------------------
def test_wanting_tools_goes_to_the_approval_gate():
    assert route_after_reason(state(wants_tools=True, turns=1)) == "confirm"


def test_finishing_normally_goes_to_summarize():
    assert route_after_reason(state(wants_tools=False, turns=1)) == "summarize"


def test_cannot_summarize_while_a_revision_is_outstanding():
    """Asked for a different time, the agent does not get to stop and write it up."""
    assert route_after_reason(state(wants_tools=False, turns=1, awaiting_revision=True)) == "nudge"


def test_the_turn_ceiling_still_wins_over_the_nudge():
    """The guard must not be able to loop forever."""
    assert (
        route_after_reason(state(wants_tools=False, turns=MAX_TURNS, awaiting_revision=True))
        == "summarize"
    )


def test_the_turn_ceiling_stops_tool_calls_too():
    assert route_after_reason(state(wants_tools=True, turns=MAX_TURNS)) == "summarize"


def test_nudge_appends_one_message_and_records_the_step():
    result = nudge(state(contents=[]))
    assert result["steps"] == ["nudge"]
    assert len(result["contents"]) == 1


# --- the system prompt ------------------------------------------------------
def test_system_prompt_carries_the_active_calendars_date():
    """The mock clock is frozen, so this is the date the evals depend on."""
    prompt = system_prompt()
    assert "2026-09-10" in prompt
    assert "09:00 to 18:00" in prompt


# --- the schema adapter -----------------------------------------------------
def test_strip_removes_keywords_gemini_rejects():
    cleaned = _strip_for_gemini(
        {
            "type": "object",
            "title": "SomeModel",
            "additionalProperties": False,
            "strict": True,
            "properties": {"when": {"type": "string", "format": "date"}},
        }
    )
    assert "title" not in cleaned
    assert "additionalProperties" not in cleaned
    assert "strict" not in cleaned


def test_strip_folds_format_into_the_description():
    """Dropping `format` silently would lose the model its only hint about shape."""
    cleaned = _strip_for_gemini({"type": "string", "format": "date", "description": "The day."})
    assert "format" not in cleaned
    assert "date" in cleaned["description"]


def test_strip_does_not_mistake_a_field_named_title_for_the_title_keyword():
    """One of our own tool inputs is literally called `title`."""
    cleaned = _strip_for_gemini(
        {"type": "object", "properties": {"title": {"type": "string", "title": "Title"}}}
    )
    assert "title" in cleaned["properties"]
    assert "title" not in cleaned["properties"]["title"]


def test_strip_recurses_into_array_items():
    cleaned = _strip_for_gemini(
        {"type": "array", "items": {"type": "string", "title": "Tag", "format": "uuid"}}
    )
    assert "title" not in cleaned["items"]
    assert "uuid" in cleaned["items"]["description"]


# --- the agent's output contract --------------------------------------------
def test_final_answer_requires_every_field():
    """Evals assert against these, so a missing one has to fail loudly."""
    with pytest.raises(ValueError):
        FinalAnswer(summary="done")


# --- streaming labels -------------------------------------------------------
def test_every_graph_node_has_a_progress_label():
    """Add a node without a label and the UI shows a raw function name."""
    from session import STEP_LABELS
    from agent import build_agent

    nodes = {name for name in build_agent().nodes if not name.startswith("__")}
    assert nodes <= set(STEP_LABELS), f"no progress label for {nodes - set(STEP_LABELS)}"
