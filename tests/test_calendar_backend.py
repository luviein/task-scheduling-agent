"""The overlap rule and the free-slot grid.

These are the two pieces of arithmetic the whole agent rests on. If `overlaps`
is wrong the approval card warns about the wrong thing and the tool refuses the
wrong writes; if the grid is wrong the agent books over your meetings.
"""

from datetime import date, datetime, time

import pytest

from calendar_backend import (
    BUSINESS_END,
    BUSINESS_START,
    CalendarEvent,
    MockCalendar,
    get_backend,
    overlaps,
    use_backend,
)

DAY = date(2026, 9, 11)


def span(start: str, end: str) -> tuple[datetime, datetime]:
    return datetime.fromisoformat(f"2026-09-11T{start}"), datetime.fromisoformat(f"2026-09-11T{end}")


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (("10:00", "11:00"), ("10:30", "11:30"), True),  # partial, b starts inside a
        (("10:30", "11:30"), ("10:00", "11:00"), True),  # partial, the other way
        (("10:00", "11:00"), ("10:15", "10:45"), True),  # b wholly inside a
        (("10:15", "10:45"), ("10:00", "11:00"), True),  # a wholly inside b
        (("10:00", "11:00"), ("11:00", "12:00"), False),  # b starts exactly as a ends
        (("11:00", "12:00"), ("10:00", "11:00"), False),  # a starts exactly as b ends
        (("10:00", "11:00"), ("13:00", "14:00"), False),  # nowhere near
    ],
)
def test_overlaps(a, b, expected):
    """Touching at the boundary is not an overlap: a 14:00 meeting does not block 13:00-14:00."""
    assert overlaps(span(*a), span(*b)) is expected


def test_event_span_uses_duration():
    event = CalendarEvent(
        id="E-1", title="Standup", date="2026-09-11", start_time="09:00", duration_minutes=15
    )
    begin, end = event.span()
    assert begin == datetime(2026, 9, 11, 9, 0)
    assert end == datetime(2026, 9, 11, 9, 15)


def test_list_events_only_returns_that_day_sorted():
    events = MockCalendar().list_events(DAY)
    assert [event.title for event in events] == ["Standup", "Design review"]
    assert all(event.date == "2026-09-11" for event in events)


def test_list_events_empty_for_a_free_day():
    assert MockCalendar().list_events(date(2026, 9, 20)) == []


def test_free_slots_exclude_anything_that_overlaps():
    slots = MockCalendar().find_free_slots(DAY, 60)
    # Design review is 14:00-15:00, so no 60-minute block may start at 13:30 or 14:30.
    assert "13:30" not in slots
    assert "14:00" not in slots
    assert "14:30" not in slots
    assert "15:00" in slots


def test_free_slots_stay_inside_business_hours():
    slots = MockCalendar().find_free_slots(DAY, 60)
    assert slots[0] >= BUSINESS_START.strftime("%H:%M")
    # The last block must end by closing time, so it cannot start at 17:30.
    assert max(slots) == "17:00"
    assert BUSINESS_END == time(18, 0)


def test_a_block_too_long_for_the_day_has_nowhere_to_go():
    assert MockCalendar().find_free_slots(DAY, 480) == []


def test_create_event_appends_and_is_visible_immediately():
    cal = MockCalendar()
    before = len(cal.list_events(DAY))
    made = cal.create_event("Deep work", DAY, time(10, 0), 90)

    assert made.title == "Deep work"
    assert made.start_time == "10:00"
    assert len(cal.list_events(DAY)) == before + 1
    # The backend does not police conflicts; that is the tool layer's job.
    assert "10:00" not in cal.find_free_slots(DAY, 90)


def test_list_range_spans_days_and_sorts():
    cal = MockCalendar()
    cal.create_event("Later", date(2026, 9, 13), time(11, 0), 30)
    events = cal.list_range(date(2026, 9, 11), date(2026, 9, 13))

    assert [event.title for event in events] == ["Standup", "Design review", "Later"]


def test_list_range_of_one_day_matches_list_events():
    cal = MockCalendar()
    assert cal.list_range(DAY, DAY) == cal.list_events(DAY)


def test_mock_clock_is_frozen():
    """Evals depend on this: a moving clock would move every expectation."""
    assert MockCalendar().today() == date(2026, 9, 10)


def test_use_backend_overrides_selection():
    mine = MockCalendar()
    use_backend(mine)
    assert get_backend() is mine
