"""The task store, including the reconcile that Refresh runs.

The reconcile is the part worth testing hardest: it is the only code that can
untick something a human ticked, and it decides on evidence read from a calendar
that may be unreadable.
"""

from datetime import date, time

import pytest

import mock_data
import task_store
from calendar_backend import MockCalendar, use_backend
from task_store import (
    TaskDraft,
    TaskPatch,
    create_task,
    delete_task,
    get_task,
    list_tasks,
    mark_booked,
    resync_bookings,
    update_task,
)

DAY = "2026-09-11"


# --- basics -----------------------------------------------------------------
def test_seed_is_five_tasks_one_done():
    tasks = list_tasks()
    assert len(tasks) == 5
    assert sum(task.status == "done" for task in tasks) == 1


def test_create_assigns_the_next_free_id():
    made = create_task(TaskDraft(title="Something new"))
    assert made.id == "T-6"
    assert get_task("T-6").title == "Something new"


def test_ids_keep_climbing_after_a_delete():
    """Reusing T-6 would attach a new task to a deleted one's history."""
    create_task(TaskDraft(title="First"))
    delete_task("T-6")
    assert create_task(TaskDraft(title="Second")).id == "T-6"


def test_defaults_are_open_and_medium():
    made = create_task(TaskDraft(title="Bare"))
    assert (made.status, made.priority, made.tags, made.booked) == ("open", "medium", [], None)


def test_patch_changes_only_what_it_names():
    before = get_task("T-1")
    update_task("T-1", TaskPatch(status="done"))
    after = get_task("T-1")
    assert after.status == "done"
    assert (after.title, after.priority, after.tags) == (before.title, before.priority, before.tags)


def test_patch_can_clear_tags_to_empty():
    """exclude_unset, not exclude_none: an explicit [] is a real edit."""
    update_task("T-1", TaskPatch(tags=[]))
    assert get_task("T-1").tags == []


def test_patch_of_a_missing_task_returns_none():
    assert update_task("T-99", TaskPatch(status="done")) is None


def test_delete_reports_whether_it_did_anything():
    assert delete_task("T-1") is True
    assert delete_task("T-1") is False
    assert get_task("T-1") is None


# --- mark_booked ------------------------------------------------------------
def test_mark_booked_records_the_time_and_ticks_off():
    task = mark_booked("Review React hook PR", DAY, "11:00", 45, "E-9")
    assert task.status == "done"
    assert task.booked.model_dump() == {
        "date": DAY,
        "start_time": "11:00",
        "duration_minutes": 45,
        "event_id": "E-9",
    }


def test_mark_booked_ignores_a_title_that_is_not_a_task():
    assert mark_booked("Not a task", DAY, "11:00", 45, "E-9") is None


# --- resync -----------------------------------------------------------------
def book_on_calendar(title: str, day: date, start: time, minutes: int) -> str:
    return MockCalendar().create_event(title, day, start, minutes).id


def test_resync_adopts_an_event_that_matches_a_task():
    """The case that matters on a calendar this app did not write."""
    event_id = book_on_calendar("Publish DeepEval results", date(2026, 9, 11), time(11, 0), 30)

    result = resync_bookings()

    assert result.adopted == ["T-3"]
    assert result.cleared == []
    task = get_task("T-3")
    assert task.status == "done"
    assert task.booked.event_id == event_id
    assert task.booked.start_time == "11:00"


def test_resync_clears_a_booking_whose_event_is_gone():
    mark_booked("Publish DeepEval results", DAY, "11:00", 30, "E-does-not-exist")

    result = resync_bookings()

    assert result.cleared == ["T-3"]
    task = get_task("T-3")
    assert task.status == "open"
    assert task.booked is None


def test_resync_keeps_a_booking_whose_event_is_still_there():
    event_id = book_on_calendar("Publish DeepEval results", date(2026, 9, 11), time(11, 0), 30)
    mark_booked("Publish DeepEval results", DAY, "11:00", 30, event_id)

    result = resync_bookings()

    assert (result.cleared, result.adopted) == ([], [])
    assert get_task("T-3").status == "done"


def test_resync_matches_by_title_and_time_when_there_is_no_event_id():
    """Bookings written before ids were stored still have to reconcile."""
    book_on_calendar("Publish DeepEval results", date(2026, 9, 11), time(11, 0), 30)
    mark_booked("Publish DeepEval results", DAY, "11:00", 30, None)

    assert resync_bookings().cleared == []


def test_resync_without_an_id_does_not_match_a_different_time():
    """Same title, different hour, is a different commitment."""
    book_on_calendar("Publish DeepEval results", date(2026, 9, 11), time(16, 0), 30)
    mark_booked("Publish DeepEval results", DAY, "11:00", 30, None)

    assert resync_bookings().cleared == ["T-3"]


def test_resync_ignores_events_beyond_the_lookahead():
    far = date(2026, 9, 10) + __import__("datetime").timedelta(days=task_store.LOOKAHEAD_DAYS + 5)
    book_on_calendar("Publish DeepEval results", far, time(11, 0), 30)

    assert resync_bookings().adopted == []


def test_resync_changes_nothing_when_the_calendar_cannot_be_read():
    """A failed lookup is not evidence that your meetings were cancelled."""

    class Broken(MockCalendar):
        def list_range(self, start, end):
            raise OSError("network down")

    mark_booked("Publish DeepEval results", DAY, "11:00", 30, "E-9")
    use_backend(Broken())

    result = resync_bookings()

    assert (result.cleared, result.adopted) == ([], [])
    assert get_task("T-3").status == "done"


def test_resync_is_idempotent():
    book_on_calendar("Publish DeepEval results", date(2026, 9, 11), time(11, 0), 30)
    first = resync_bookings()
    second = resync_bookings()

    assert first.adopted == ["T-3"]
    assert (second.adopted, second.cleared) == ([], [])


def test_resync_never_touches_the_calendar():
    """It reconciles in one direction only, so a bad match costs a tick, not a meeting."""
    before = [dict(event) for event in mock_data.CALENDAR]
    mark_booked("Publish DeepEval results", DAY, "11:00", 30, "E-gone")

    resync_bookings()

    assert mock_data.CALENDAR == before


# --- persistence ------------------------------------------------------------
def test_the_suite_cannot_write_the_store_file():
    """conftest disables persistence process-wide; this is the assertion of that."""
    assert task_store._persist is False
    create_task(TaskDraft(title="Never on disk"))
    task_store._save()  # a no-op while persistence is off


def test_reset_restores_the_seed():
    create_task(TaskDraft(title="Temporary"))
    delete_task("T-1")
    task_store.reset_tasks()
    assert [task.id for task in list_tasks()] == ["T-1", "T-2", "T-3", "T-4", "T-5"]
