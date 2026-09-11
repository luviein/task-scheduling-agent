"""HTTP over the agent. Thin on purpose: every route is a few lines around session.py.

The interesting design constraint is the approval pause. A run does not finish in
one request, so the API is two calls rather than one:

    POST /api/runs                    -> {"status": "paused", "thread_id": ..., "approval": {...}}
    POST /api/runs/{id}/decision      -> {"status": "done", "final": {...}}   (or paused again)

The thread id is the session. The graph's checkpointer holds the paused run in
memory between the two, which means sessions do not survive a restart. Fine for a
single-user local tool; a real deployment swaps InMemorySaver for a durable one
and changes nothing else.

Routes are plain `def`, not `async def`. A run blocks for ten to twenty seconds on
model calls, and FastAPI runs sync routes in a threadpool, so one slow run does
not freeze the server.
"""

import json
from collections.abc import Iterator

from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent import BUSINESS_END, BUSINESS_START, MODEL, DailyQuotaExhausted
from calendar_backend import get_backend
from session import RunStep, resume, resume_streaming, start, start_streaming
from task_store import (
    Task,
    TaskDraft,
    TaskPatch,
    create_task,
    delete_task,
    list_tasks,
    resync_bookings,
    update_task,
)

app = FastAPI(title="Task Scheduling Agent", version="1.0")

# The Vite dev server runs on another port, so the browser treats it as a
# different origin. Local development only; a deployed build would be served
# from this same origin and need none of this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class StartRequest(BaseModel):
    instruction: str = Field(min_length=1, description="What to ask the agent to do.")


class DecisionRequest(BaseModel):
    decision: bool | str = Field(
        description="true approves, false is a hard no, and any string is a "
        "counter-proposal the agent will try to satisfy."
    )


class Config(BaseModel):
    """What the UI needs to describe itself honestly: which calendar, which day."""

    model: str
    calendar: str
    today: str
    business_hours: str


@app.get("/api/config")
def config() -> Config:
    backend = get_backend()
    return Config(
        model=MODEL,
        calendar=backend.name,
        today=backend.today().isoformat(),
        business_hours=f"{BUSINESS_START:%H:%M}-{BUSINESS_END:%H:%M}",
    )


# --- Tasks ------------------------------------------------------------------
# The list the agent searches, editable from the UI. Edits persist to tasks.json;
# the eval suite reseeds in memory and never reads that file.


@app.get("/api/tasks")
def tasks() -> list[Task]:
    return list_tasks()


class Resync(BaseModel):
    """What the refresh changed, so the UI can say something specific."""

    cleared: list[str] = Field(
        default_factory=list, description="Tasks whose calendar event is gone. Open again."
    )
    adopted: list[str] = Field(
        default_factory=list, description="Tasks matched to an event already on the calendar."
    )
    tasks: list[Task]


@app.post("/api/tasks/resync")
def resync() -> Resync:
    """Reconcile the task list against the calendar, both ways.

    Declared before the /{task_id} routes so "resync" is never read as an id.
    """
    changed = resync_bookings()
    return Resync(cleared=changed.cleared, adopted=changed.adopted, tasks=list_tasks())


@app.post("/api/tasks", status_code=201)
def add_task(draft: TaskDraft) -> Task:
    return create_task(draft)


@app.patch("/api/tasks/{task_id}")
def edit_task(task_id: str, patch: TaskPatch) -> Task:
    task = update_task(task_id, patch)
    if task is None:
        raise HTTPException(status_code=404, detail=f"No task {task_id}")
    return task


@app.delete("/api/tasks/{task_id}", status_code=204)
def remove_task(task_id: str) -> Response:
    if not delete_task(task_id):
        raise HTTPException(status_code=404, detail=f"No task {task_id}")
    return Response(status_code=204)


def _guard(fn, *args) -> RunStep:
    """One place to turn an exhausted free tier into an honest HTTP status."""
    try:
        return fn(*args)
    except DailyQuotaExhausted as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/runs")
def create_run(body: StartRequest) -> RunStep:
    """Start a run. Returns at the first approval request, or at the end."""
    return _guard(start, body.instruction)


def _sse(events: Iterator) -> StreamingResponse:
    """Server-sent events, one JSON object per message.

    POST rather than a GET an EventSource could consume, because the instruction
    is the user's own words and has no business sitting in a URL, where it would
    land in access logs and browser history.
    """

    def body():
        try:
            for event in events:
                yield f"data: {event.model_dump_json()}\n\n"
        except DailyQuotaExhausted as exc:
            # The stream has already started, so the status line is long gone.
            # An error event is the only way left to say what happened.
            yield f'data: {{"type": "error", "detail": {json.dumps(str(exc))}}}\n\n'

    return StreamingResponse(
        body(),
        media_type="text/event-stream",
        # Proxies that buffer would defeat the point of streaming at all.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/api/runs/stream")
def create_run_streaming(body: StartRequest) -> StreamingResponse:
    """Same run as POST /api/runs, but each step is reported as it finishes."""
    return _sse(start_streaming(body.instruction))


@app.post("/api/runs/{thread_id}/decision/stream")
def decide_streaming(thread_id: str, body: DecisionRequest) -> StreamingResponse:
    return _sse(resume_streaming(thread_id, body.decision))


@app.post("/api/runs/{thread_id}/decision")
def decide(thread_id: str, body: DecisionRequest) -> RunStep:
    """Answer a pending approval and carry the run on."""
    try:
        return _guard(resume, thread_id, body.decision)
    except ValueError as exc:
        # No checkpoint under that id: an unknown session, or one lost to a restart.
        raise HTTPException(status_code=404, detail=f"No paused run {thread_id}") from exc
