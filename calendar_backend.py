"""Calendar backends: one interface, a mock implementation, and a real one later.

The tool layer never touches a calendar directly. It asks whatever backend is
active, so pointing the agent at Google Calendar is a config change rather than
a rewrite of `tools.py`.

Three operations, as agreed:
  list_events(day)               -> what is already booked
  find_free_slots(day, minutes)  -> where a block of that length fits
  create_event(...)              -> book one

`find_free_slots` has a default implementation derived from `list_events`, so a
backend only has to supply the two primitives. Google can override it later with
the freebusy endpoint if that turns out to be cheaper than listing.
"""

from abc import ABC, abstractmethod
from datetime import date as DateType, datetime, time as TimeType, timedelta
from os import getenv

from pydantic import BaseModel

import mock_data


# Business hours are a property of how this user schedules, not of any one
# backend, so they live here and every backend honours them.
BUSINESS_START = TimeType(9, 0)
BUSINESS_END = TimeType(18, 0)
SLOT_GRID_MINUTES = 30


class CalendarEvent(BaseModel):
    id: str
    title: str
    date: str
    start_time: str
    duration_minutes: int

    def span(self) -> tuple[datetime, datetime]:
        begin = datetime.fromisoformat(f"{self.date}T{self.start_time}")
        return begin, begin + timedelta(minutes=self.duration_minutes)


def overlaps(a: tuple[datetime, datetime], b: tuple[datetime, datetime]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


class CalendarBackend(ABC):
    """What the tool layer is allowed to assume about a calendar."""

    name: str

    @abstractmethod
    def list_events(self, day: DateType) -> list[CalendarEvent]:
        """Every event on that day, earliest first."""

    @abstractmethod
    def create_event(
        self, title: str, day: DateType, start: TimeType, duration_minutes: int
    ) -> CalendarEvent:
        """Book the event. Conflict checking happens in the tool layer, above this."""

    def find_free_slots(self, day: DateType, duration_minutes: int) -> list[str]:
        """Start times, on the half-hour grid, where the block fits inside business hours."""
        busy = [event.span() for event in self.list_events(day)]
        opens = datetime.combine(day, BUSINESS_START)
        closes = datetime.combine(day, BUSINESS_END)

        free: list[str] = []
        start = opens
        while start + timedelta(minutes=duration_minutes) <= closes:
            block = (start, start + timedelta(minutes=duration_minutes))
            if not any(overlaps(block, taken) for taken in busy):
                free.append(start.strftime("%H:%M"))
            start += timedelta(minutes=SLOT_GRID_MINUTES)
        return free


class MockCalendar(CalendarBackend):
    """Backed by the in-memory list in `mock_data`, so `reset_calendar()` still works.

    The eval suite is pinned to this one permanently: running it against a real
    calendar would create real events and make scores depend on that week's
    meetings.
    """

    name = "mock"

    def list_events(self, day: DateType) -> list[CalendarEvent]:
        stamp = day.isoformat()
        events = [CalendarEvent(**row) for row in mock_data.CALENDAR if row["date"] == stamp]
        return sorted(events, key=lambda event: event.start_time)

    def create_event(
        self, title: str, day: DateType, start: TimeType, duration_minutes: int
    ) -> CalendarEvent:
        event = CalendarEvent(
            id=f"E-{len(mock_data.CALENDAR) + 1}",
            title=title,
            date=day.isoformat(),
            start_time=start.strftime("%H:%M"),
            duration_minutes=duration_minutes,
        )
        mock_data.CALENDAR.append(event.model_dump())
        return event


_BACKENDS: dict[str, type[CalendarBackend]] = {"mock": MockCalendar}
_active: CalendarBackend | None = None


def get_backend() -> CalendarBackend:
    """The backend named by CALENDAR_BACKEND, built once and reused."""
    global _active
    if _active is None:
        choice = getenv("CALENDAR_BACKEND", "mock").strip().lower()
        if choice not in _BACKENDS:
            known = ", ".join(sorted(_BACKENDS))
            raise ValueError(f"Unknown CALENDAR_BACKEND {choice!r}. Known backends: {known}")
        _active = _BACKENDS[choice]()
    return _active


def use_backend(backend: CalendarBackend) -> None:
    """Override the active backend. For tests and for the eval suite's mock pin."""
    global _active
    _active = backend


if __name__ == "__main__":
    from datetime import date

    cal = get_backend()
    day = date(2026, 9, 11)
    print(f"backend: {cal.name}")
    print("events :", [e.model_dump() for e in cal.list_events(day)])
    print("free60 :", cal.find_free_slots(day, 60))
