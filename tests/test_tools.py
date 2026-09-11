"""The tool layer: validation, refusal, and the schemas the model is handed.

`dispatch` is the boundary between whatever the model produced and code that
runs. Its contract is that it never raises, and these tests hold it to that.
"""

import pytest

from tools import (
    REGISTRY,
    FindFreeSlotsInput,
    SearchTasksInput,
    anthropic_tool_defs,
    day_agenda,
    dispatch,
    find_free_slots,
    search_tasks,
)

DAY = "2026-09-11"


# --- search_tasks -----------------------------------------------------------
def test_search_matches_on_title():
    result = search_tasks(SearchTasksInput(query="playwright", status="open"))
    assert [task.title for task in result.matches] == ["Refactor Playwright login suite"]
    assert result.count == 1


def test_search_matches_on_tag():
    result = search_tasks(SearchTasksInput(query="evals", status="open"))
    assert [task.title for task in result.matches] == ["Publish DeepEval results"]


def test_search_is_case_insensitive():
    upper = search_tasks(SearchTasksInput(query="PLAYWRIGHT", status="open"))
    lower = search_tasks(SearchTasksInput(query="playwright", status="open"))
    assert upper.matches == lower.matches


def test_empty_query_matches_every_open_task():
    result = search_tasks(SearchTasksInput(query="", status="open"))
    assert result.count == 4  # five seeded, one of them already done
    assert all(task.status == "open" for task in result.matches)


def test_status_all_includes_done_work():
    titles = [t.title for t in search_tasks(SearchTasksInput(query="", status="all")).matches]
    assert "Renew domain name" in titles


def test_no_match_is_an_empty_result_not_an_error():
    result = search_tasks(SearchTasksInput(query="tax return", status="open"))
    assert result.matches == []
    assert result.count == 0


# --- find_free_slots --------------------------------------------------------
def test_free_slots_count_matches_the_list():
    result = find_free_slots(FindFreeSlotsInput(date=DAY, duration_minutes=60))
    assert result.count == len(result.slots)


# --- create_calendar_event --------------------------------------------------
def test_booking_a_free_slot_succeeds():
    result = dispatch(
        "create_calendar_event",
        {"title": "Deep work", "date": DAY, "start_time": "10:00", "duration_minutes": 60},
    )
    assert result.ok
    assert result.output["created"] is True
    assert result.output["event"]["title"] == "Deep work"


def test_booking_over_an_existing_event_is_refused_by_name():
    """Refusal, not an exception: the agent has to be able to read it and retry."""
    result = dispatch(
        "create_calendar_event",
        {"title": "Clash", "date": DAY, "start_time": "14:30", "duration_minutes": 30},
    )
    assert result.ok  # the call worked; the booking did not
    assert result.output["created"] is False
    assert result.output["conflict_with"] == "Design review"
    assert result.output["event"] is None


def test_booking_that_merely_touches_an_event_is_allowed():
    """Design review runs 14:00-15:00, so 13:00-14:00 is free."""
    result = dispatch(
        "create_calendar_event",
        {"title": "Right before", "date": DAY, "start_time": "13:00", "duration_minutes": 60},
    )
    assert result.output["created"] is True


def test_a_refused_booking_does_not_change_the_calendar():
    before = day_agenda(DAY)
    dispatch(
        "create_calendar_event",
        {"title": "Clash", "date": DAY, "start_time": "14:30", "duration_minutes": 30},
    )
    assert day_agenda(DAY) == before


# --- dispatch contract ------------------------------------------------------
def test_unknown_tool_is_reported_not_raised():
    result = dispatch("delete_everything", {})
    assert result.ok is False
    assert "Unknown tool" in result.error


def test_malformed_input_is_reported_not_raised():
    result = dispatch(
        "create_calendar_event",
        {"title": "Bad", "date": "not-a-date", "start_time": "10:00", "duration_minutes": 60},
    )
    assert result.ok is False
    assert "date" in result.error


def test_out_of_range_duration_is_rejected():
    result = dispatch(
        "create_calendar_event",
        {"title": "Marathon", "date": DAY, "start_time": "09:00", "duration_minutes": 5000},
    )
    assert result.ok is False


def test_extra_fields_are_rejected():
    """extra=forbid, so a model that invents an argument is caught here."""
    result = dispatch("search_tasks", {"query": "ai", "status": "open", "limit": 3})
    assert result.ok is False


@pytest.mark.parametrize("name", sorted(REGISTRY))
def test_every_registered_tool_dispatches_without_raising(name):
    """Garbage in, a ToolCallResult out. Never an exception, for any tool."""
    result = dispatch(name, {"nonsense": True})
    assert result.ok is False
    assert result.error


# --- schemas handed to the model -------------------------------------------
def test_tool_defs_cover_the_registry():
    assert {spec["name"] for spec in anthropic_tool_defs()} == set(REGISTRY)


def test_tool_defs_are_strict_and_closed():
    for spec in anthropic_tool_defs():
        schema = spec["input_schema"]
        assert spec["strict"] is True
        assert schema["additionalProperties"] is False
        # Every property is required, so the model cannot omit one and leave the
        # tool guessing at a default.
        assert set(schema["required"]) == set(schema["properties"])
        assert spec["description"]


# --- day_agenda -------------------------------------------------------------
def test_day_agenda_is_sorted_and_shows_end_times():
    agenda = day_agenda(DAY)
    assert [item["title"] for item in agenda] == ["Standup", "Design review"]
    assert agenda[0] == {"title": "Standup", "from": "09:00", "to": "09:15"}


def test_day_agenda_of_an_empty_day():
    assert day_agenda("2026-09-20") == []
