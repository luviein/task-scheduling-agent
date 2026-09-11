"""Shared fixtures.

Two rules hold for every test in this suite:

  1. No test may touch a real calendar. The mock backend is forced.
  2. No test may touch `tasks.json`. Persistence is switched off process-wide
     and the list is reseeded before each test.

Neither is politeness. The task store and the calendar are module-level state,
so a test that forgot either would quietly rewrite the file a human edits, or
create events on a real Google Calendar.
"""

import pytest

import mock_data
import task_store
from calendar_backend import MockCalendar, use_backend

# Process-wide, once: nothing in this suite is allowed to write the store file.
task_store.disable_persistence()


@pytest.fixture(autouse=True)
def clean_state():
    """Every test starts from the same calendar and the same task list."""
    use_backend(MockCalendar())
    mock_data.reset_calendar()
    task_store.reset_tasks()
    yield
    mock_data.reset_calendar()
    task_store.reset_tasks()


@pytest.fixture
def seeded_day() -> str:
    """The day the seed calendar has events on: 09:00 standup, 14:00 design review."""
    return "2026-09-11"
