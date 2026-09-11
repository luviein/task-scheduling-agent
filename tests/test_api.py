"""The HTTP surface, minus the two routes that would spend model quota.

Everything here runs against the mock backend through FastAPI's test client, so
the whole file is free and offline.
"""

from datetime import date, time

import pytest
from fastapi.testclient import TestClient

from api import app
from calendar_backend import MockCalendar

client = TestClient(app)


# --- config -----------------------------------------------------------------
def test_config_reports_the_active_calendar_and_clock():
    body = client.get("/api/config").json()
    assert body["calendar"] == "mock"
    assert body["today"] == "2026-09-10"
    assert body["business_hours"] == "09:00-18:00"
    assert body["model"]


# --- reading tasks ----------------------------------------------------------
def test_tasks_returns_the_seed():
    body = client.get("/api/tasks").json()
    assert len(body) == 5
    assert body[0]["id"] == "T-1"
    assert body[0]["booked"] is None


# --- creating ---------------------------------------------------------------
def test_create_returns_201_and_the_new_task():
    response = client.post("/api/tasks", json={"title": "Write the case study"})
    assert response.status_code == 201
    assert response.json()["id"] == "T-6"
    assert len(client.get("/api/tasks").json()) == 6


def test_create_rejects_an_empty_title():
    assert client.post("/api/tasks", json={"title": ""}).status_code == 422


def test_create_rejects_an_unknown_priority():
    response = client.post("/api/tasks", json={"title": "Nope", "priority": "urgent"})
    assert response.status_code == 422
    assert "priority" in response.text


def test_create_rejects_a_client_supplied_id():
    """The server assigns ids. extra=forbid is what makes that true."""
    assert client.post("/api/tasks", json={"id": "T-99", "title": "Sneaky"}).status_code == 422


def test_create_rejects_a_client_supplied_booking():
    """`booked` is the agent's to write, never the form's."""
    payload = {
        "title": "Pretend",
        "booked": {"date": "2026-09-11", "start_time": "10:00", "duration_minutes": 30},
    }
    assert client.post("/api/tasks", json=payload).status_code == 422


# --- editing ----------------------------------------------------------------
def test_patch_updates_one_field():
    body = client.patch("/api/tasks/T-1", json={"status": "done"}).json()
    assert body["status"] == "done"
    assert body["title"] == "Write LangGraph agent draft"


def test_patch_of_a_missing_task_is_404():
    assert client.patch("/api/tasks/T-99", json={"status": "done"}).status_code == 404


def test_patch_with_no_fields_is_allowed_and_changes_nothing():
    before = client.get("/api/tasks").json()[0]
    assert client.patch("/api/tasks/T-1", json={}).json() == before


# --- deleting ---------------------------------------------------------------
def test_delete_returns_204_then_404():
    assert client.delete("/api/tasks/T-1").status_code == 204
    assert client.delete("/api/tasks/T-1").status_code == 404


def test_deleting_a_task_leaves_the_calendar_alone():
    """Stated plainly because people reasonably expect the opposite."""
    cal = MockCalendar()
    cal.create_event("Write LangGraph agent draft", date(2026, 9, 11), time(11, 0), 60)
    before = [event.id for event in cal.list_events(date(2026, 9, 11))]

    client.delete("/api/tasks/T-1")

    assert [event.id for event in cal.list_events(date(2026, 9, 11))] == before


# --- resync -----------------------------------------------------------------
def test_resync_on_an_untouched_calendar_changes_nothing():
    body = client.post("/api/tasks/resync").json()
    assert body["cleared"] == []
    assert body["adopted"] == []
    assert len(body["tasks"]) == 5


def test_resync_adopts_a_matching_event_and_returns_the_fresh_list():
    MockCalendar().create_event("Publish DeepEval results", date(2026, 9, 11), time(11, 0), 30)

    body = client.post("/api/tasks/resync").json()

    assert body["adopted"] == ["T-3"]
    adopted = next(task for task in body["tasks"] if task["id"] == "T-3")
    assert adopted["status"] == "done"
    assert adopted["booked"]["start_time"] == "11:00"


def test_resync_is_not_read_as_a_task_id():
    """POST /api/tasks/resync must not be routed to the /{task_id} handlers."""
    assert client.post("/api/tasks/resync").status_code == 200


# --- run routes -------------------------------------------------------------
def test_answering_an_unknown_session_is_404():
    """The one run-route case that costs no quota: there is nothing to resume."""
    response = client.post("/api/runs/does-not-exist/decision", json={"decision": True})
    assert response.status_code == 404


@pytest.mark.parametrize("decision", [True, False, "make it after lunch"])
def test_a_decision_may_be_yes_no_or_a_counter_proposal(decision):
    """Three answers, not two. A 422 here would mean the union stopped accepting one."""
    response = client.post("/api/runs/does-not-exist/decision", json={"decision": decision})
    assert response.status_code == 404  # rejected for the session, not the payload


def test_starting_a_run_needs_an_instruction():
    assert client.post("/api/runs", json={"instruction": ""}).status_code == 422
