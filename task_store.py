"""The task list, editable at runtime and persisted to disk.

`mock_data.TASKS` is the seed and stays frozen: the eval suite grounds its
hallucination check against exactly those titles, so a suite that read your
edited list would score differently every time you added a task.

So the same split the calendar has. Edits made through the UI go to `tasks.json`
and are loaded from there on start. The eval suite calls `reset_tasks()` and runs
against the seed, in memory, without touching the file.
"""

import json
from datetime import date as DateType
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mock_data import TASKS as SEED_TASKS

STORE_FILE = Path("tasks.json")

Status = Literal["open", "done"]
Priority = Literal["low", "medium", "high"]


class Booking(BaseModel):
    """Where a task ended up on the calendar. Written by the agent, not by hand."""

    model_config = ConfigDict(extra="forbid")

    date: str
    start_time: str
    duration_minutes: int
    event_id: str | None = Field(
        default=None, description="The calendar's own id, so a resync can match exactly."
    )


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str = Field(min_length=1)
    status: Status = "open"
    priority: Priority = "medium"
    tags: list[str] = Field(default_factory=list)
    booked: Booking | None = Field(
        default=None, description="Set when a booking for this task actually succeeded."
    )


class TaskDraft(BaseModel):
    """What a caller may send when creating. The id is ours to assign."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1)
    status: Status = "open"
    priority: Priority = "medium"
    tags: list[str] = Field(default_factory=list)


class TaskPatch(BaseModel):
    """Every field optional: a patch changes only what it names."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1)
    status: Status | None = None
    priority: Priority | None = None
    tags: list[str] | None = None


_tasks: list[Task] | None = None

# Evals mutate the list: a successful booking ticks its task off. They must not
# write that to the file a human edits, or a suite run would silently replace
# your task list with the seed. Same shape as pinning the evals to the mock
# calendar: the harness says so once, at import.
_persist = True


def disable_persistence() -> None:
    """Keep every later change in memory only. Called by the eval harness."""
    global _persist
    _persist = False


def _seed() -> list[Task]:
    return [Task(**task) for task in SEED_TASKS]


def _load() -> list[Task]:
    global _tasks
    if _tasks is None:
        if STORE_FILE.exists():
            try:
                raw = json.loads(STORE_FILE.read_text(encoding="utf-8"))
                _tasks = [Task(**task) for task in raw]
            except (json.JSONDecodeError, ValueError):
                # A corrupt store should not take the app down. Fall back to the
                # seed and leave the bad file alone for the human to look at.
                _tasks = _seed()
        else:
            _tasks = _seed()
    return _tasks


def _save() -> None:
    if not _persist:
        return
    STORE_FILE.write_text(
        json.dumps([task.model_dump() for task in _load()], indent=2), encoding="utf-8"
    )


def reset_tasks() -> None:
    """Back to the seed, in memory only. The eval suite calls this per case."""
    global _tasks
    _tasks = _seed()


def list_tasks() -> list[Task]:
    return list(_load())


def as_dicts() -> list[dict]:
    """What `search_tasks` reads. Plain dicts, matching the old fixture shape."""
    return [task.model_dump() for task in _load()]


def get_task(task_id: str) -> Task | None:
    return next((task for task in _load() if task.id == task_id), None)


def _next_id() -> str:
    used = []
    for task in _load():
        head, _, tail = task.id.partition("-")
        if head == "T" and tail.isdigit():
            used.append(int(tail))
    return f"T-{max(used, default=0) + 1}"


def create_task(draft: TaskDraft) -> Task:
    task = Task(id=_next_id(), **draft.model_dump())
    _load().append(task)
    _save()
    return task


def update_task(task_id: str, patch: TaskPatch) -> Task | None:
    task = get_task(task_id)
    if task is None:
        return None
    # exclude_unset, not exclude_none: clearing tags to [] is a real edit, while
    # a field the caller never mentioned must keep its current value.
    for field, value in patch.model_dump(exclude_unset=True).items():
        setattr(task, field, value)
    _save()
    return task


def mark_booked(
    title: str, date: str, start_time: str, duration_minutes: int, event_id: str | None = None
) -> Task | None:
    """Record that a task got time on the calendar, and tick it off.

    Called only after `create_calendar_event` actually returned created=true, so
    the tick means an event exists, not that the agent intended one. Booking is
    not the same as finishing the work, which is why the time is stored rather
    than just flipping a flag: the list can say when, not merely that.
    """
    task = next((task for task in _load() if task.title == title), None)
    if task is None:
        return None
    task.booked = Booking(
        date=date, start_time=start_time, duration_minutes=duration_minutes, event_id=event_id
    )
    task.status = "done"
    _save()
    return task


def resync_bookings() -> list[str]:
    """Drop bookings whose calendar event is gone, and untick those tasks.

    The calendar is the source of truth, and it can change behind this app's
    back: you delete the event in Google Calendar and the task is left claiming
    time it no longer holds. This reconciles in that direction only. It never
    creates or moves an event, so the worst it can do is tell the truth.

    Matches on the calendar's own event id where there is one, and falls back to
    title and start time for bookings recorded before ids were stored.
    """
    from calendar_backend import get_backend

    booked = [task for task in _load() if task.booked]
    if not booked:
        return []

    backend = get_backend()
    # One listing per distinct day, not one per task.
    days = {task.booked.date for task in booked}
    events_by_day = {}
    for day in days:
        try:
            events_by_day[day] = backend.list_events(DateType.fromisoformat(day))
        except (ValueError, OSError):
            # A day we cannot read is a day we cannot judge. Leave those tasks be
            # rather than unticking on the strength of a failed lookup.
            events_by_day[day] = None

    cleared = []
    for task in booked:
        events = events_by_day.get(task.booked.date)
        if events is None:
            continue
        if task.booked.event_id:
            still_there = any(event.id == task.booked.event_id for event in events)
        else:
            still_there = any(
                event.title == task.title and event.start_time == task.booked.start_time
                for event in events
            )
        if not still_there:
            task.booked = None
            task.status = "open"
            cleared.append(task.id)

    if cleared:
        _save()
    return cleared


def delete_task(task_id: str) -> bool:
    tasks = _load()
    remaining = [task for task in tasks if task.id != task_id]
    if len(remaining) == len(tasks):
        return False
    tasks[:] = remaining
    _save()
    return True


if __name__ == "__main__":
    reset_tasks()
    print("seeded:", [task.id for task in list_tasks()])
    made = create_task(TaskDraft(title="Try the new UI", priority="high", tags=["ui"]))
    print("created:", made.model_dump())
    print("updated:", update_task(made.id, TaskPatch(status="done")).model_dump())
    print("deleted:", delete_task(made.id), "->", [task.id for task in list_tasks()])
