"""The task list, editable at runtime and persisted to disk.

`mock_data.TASKS` is the seed and stays frozen: the eval suite grounds its
hallucination check against exactly those titles, so a suite that read your
edited list would score differently every time you added a task.

So the same split the calendar has. Edits made through the UI go to `tasks.json`
and are loaded from there on start. The eval suite calls `reset_tasks()` and runs
against the seed, in memory, without touching the file.
"""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mock_data import TASKS as SEED_TASKS

STORE_FILE = Path("tasks.json")

Status = Literal["open", "done"]
Priority = Literal["low", "medium", "high"]


class Task(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str = Field(min_length=1)
    status: Status = "open"
    priority: Priority = "medium"
    tags: list[str] = Field(default_factory=list)


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
