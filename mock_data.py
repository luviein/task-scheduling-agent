"""Zero-config fake backend. Swap for a real API later; the tool layer won't change."""

from datetime import date

# Frozen "today" keeps evals deterministic. Real clock comes later.
TODAY = date(2026, 9, 10)

TASKS: list[dict] = [
    {"id": "T-1", "title": "Write LangGraph agent draft", "status": "open", "priority": "high", "tags": ["ai", "portfolio"]},
    {"id": "T-2", "title": "Refactor Playwright login suite", "status": "open", "priority": "medium", "tags": ["qa", "testing"]},
    {"id": "T-3", "title": "Publish DeepEval results", "status": "open", "priority": "high", "tags": ["ai", "evals"]},
    {"id": "T-4", "title": "Renew domain name", "status": "done", "priority": "low", "tags": ["admin"]},
    {"id": "T-5", "title": "Review React hook PR", "status": "open", "priority": "low", "tags": ["frontend"]},
]

# Mutated by create_calendar_event. Reset with reset_calendar() between eval cases.
CALENDAR: list[dict] = [
    {"id": "E-1", "title": "Standup", "date": "2026-09-11", "start_time": "09:00", "duration_minutes": 15},
    {"id": "E-2", "title": "Design review", "date": "2026-09-11", "start_time": "14:00", "duration_minutes": 60},
]

_SEED = [dict(event) for event in CALENDAR]


def reset_calendar() -> None:
    CALENDAR[:] = [dict(event) for event in _SEED]
