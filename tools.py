"""Tool layer: Pydantic in, Pydantic out, JSON Schema for the model.

Every tool has three pieces:
  1. An input model  -> becomes the JSON Schema Claude sees.
  2. An output model -> what the node writes back into agent state.
  3. A plain function that takes the input model and returns the output model.
"""

# Aliased: a field named `date` would otherwise shadow the `date` type.
from datetime import date as DateType, datetime, time as TimeType, timedelta
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from calendar_backend import CalendarEvent, get_backend, overlaps
from mock_data import TASKS


# --- Tool 1: search_tasks ---------------------------------------------------
class SearchTasksInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(description="Keyword matched against task title and tags. Empty string matches all.")
    status: Literal["open", "done", "all"] = Field(description="Filter by task status.")


class Task(BaseModel):
    id: str
    title: str
    status: str
    priority: str
    tags: list[str]


class SearchTasksOutput(BaseModel):
    matches: list[Task]
    count: int


def search_tasks(args: SearchTasksInput) -> SearchTasksOutput:
    needle = args.query.strip().lower()
    hits = [
        Task(**t)
        for t in TASKS
        if (args.status == "all" or t["status"] == args.status)
        and (not needle or needle in t["title"].lower() or any(needle in tag for tag in t["tags"]))
    ]
    return SearchTasksOutput(matches=hits, count=len(hits))


# --- Tool 2: create_calendar_event ------------------------------------------
class CreateCalendarEventInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(description="Short title for the event.")
    date: DateType = Field(description="Event date as YYYY-MM-DD.")
    start_time: TimeType = Field(description="Start time in 24-hour HH:MM.")
    duration_minutes: int = Field(ge=5, le=480, description="Length of the event in minutes.")


class CreateCalendarEventOutput(BaseModel):
    created: bool
    event: CalendarEvent | None = None
    conflict_with: str | None = Field(default=None, description="Title of the clashing event, if any.")


def create_calendar_event(args: CreateCalendarEventInput) -> CreateCalendarEventOutput:
    """Refuse to double-book, then delegate the write to the active backend.

    The conflict check lives here rather than in a backend because it is the
    agent's rule, not the calendar's: Google would accept an overlapping event
    without complaint. Step 3 lets the agent retry another slot.
    """
    cal = get_backend()
    begin = datetime.combine(args.date, args.start_time)
    block = (begin, begin + timedelta(minutes=args.duration_minutes))

    for booked in cal.list_events(args.date):
        if overlaps(block, booked.span()):
            return CreateCalendarEventOutput(created=False, conflict_with=booked.title)

    event = cal.create_event(args.title, args.date, args.start_time, args.duration_minutes)
    return CreateCalendarEventOutput(created=True, event=event)


# --- Tool 3: find_free_slots ------------------------------------------------
class FindFreeSlotsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    date: DateType = Field(description="Day to check, as YYYY-MM-DD.")
    duration_minutes: int = Field(ge=5, le=480, description="Length of the block you need.")


class FindFreeSlotsOutput(BaseModel):
    slots: list[str] = Field(description="Free start times in HH:MM, earliest first.")
    count: int


def find_free_slots(args: FindFreeSlotsInput) -> FindFreeSlotsOutput:
    """Compute availability instead of making the model guess at it."""
    slots = get_backend().find_free_slots(args.date, args.duration_minutes)
    return FindFreeSlotsOutput(slots=slots, count=len(slots))


def day_agenda(day: str) -> list[dict]:
    """What is already on the calendar for one day, earliest first.

    Not a tool. The approval prompt uses it to show a human why a slot was
    chosen, which is a question the proposal alone cannot answer.
    """
    agenda = []
    for event in get_backend().list_events(DateType.fromisoformat(day)):
        start, end = event.span()
        agenda.append({"title": event.title, "from": start.strftime("%H:%M"), "to": end.strftime("%H:%M")})
    return agenda


# --- Registry ---------------------------------------------------------------
class ToolSpec(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    description: str
    input_model: type[BaseModel]
    fn: Callable[[Any], BaseModel]


REGISTRY: dict[str, ToolSpec] = {
    "search_tasks": ToolSpec(
        name="search_tasks",
        description="Search the task list by keyword and status. Call this before scheduling work, to find out what actually needs doing.",
        input_model=SearchTasksInput,
        fn=search_tasks,
    ),
    "find_free_slots": ToolSpec(
        name="find_free_slots",
        description="List the times a block of a given length could start on a given day. Call this before booking; do not guess at a time.",
        input_model=FindFreeSlotsInput,
        fn=find_free_slots,
    ),
    "create_calendar_event": ToolSpec(
        name="create_calendar_event",
        description="Book one event on the calendar. Returns created=false with conflict_with set when the slot is already taken.",
        input_model=CreateCalendarEventInput,
        fn=create_calendar_event,
    ),
}


def anthropic_tool_defs() -> list[dict]:
    """Pydantic schemas -> the `tools` array the Messages API expects.

    strict=True demands every property in `required` and no extras, so Claude
    can never hand you an input your model would reject.
    """
    defs = []
    for spec in REGISTRY.values():
        schema = spec.input_model.model_json_schema()
        schema["additionalProperties"] = False
        schema["required"] = list(schema.get("properties", {}))
        schema.pop("$defs", None)
        defs.append(
            {
                "name": spec.name,
                "description": spec.description,
                "strict": True,
                "input_schema": schema,
            }
        )
    return defs


class ToolCallResult(BaseModel):
    """Uniform envelope. The evals layer asserts against exactly this shape."""

    tool: str
    ok: bool
    output: dict | None = None
    error: str | None = None


def dispatch(name: str, raw_input: dict) -> ToolCallResult:
    """Validate raw model output, run the tool, never raise."""
    spec = REGISTRY.get(name)
    if spec is None:
        return ToolCallResult(tool=name, ok=False, error=f"Unknown tool: {name}")
    try:
        args = spec.input_model.model_validate(raw_input)
    except ValidationError as exc:
        first = exc.errors()[0]
        return ToolCallResult(tool=name, ok=False, error=f"Invalid input at {first['loc']}: {first['msg']}")
    return ToolCallResult(tool=name, ok=True, output=spec.fn(args).model_dump(mode="json"))


if __name__ == "__main__":
    import json

    print("=== TOOL SCHEMAS SENT TO CLAUDE ===")
    print(json.dumps(anthropic_tool_defs(), indent=2))

    print("\n=== DISPATCH SMOKE TESTS ===")
    cases = [
        ("search_tasks", {"query": "ai", "status": "open"}),
        ("create_calendar_event", {"title": "Finish evals", "date": "2026-09-11", "start_time": "10:00", "duration_minutes": 60}),
        ("create_calendar_event", {"title": "Clash", "date": "2026-09-11", "start_time": "14:30", "duration_minutes": 30}),
        ("create_calendar_event", {"title": "Bad input", "date": "not-a-date", "start_time": "10:00", "duration_minutes": 60}),
        ("delete_everything", {}),
    ]
    for tool_name, payload in cases:
        print(f"\n-> {tool_name} {payload}")
        print(dispatch(tool_name, payload).model_dump_json(indent=2))
