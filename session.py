"""Turn the agent's blocking approval loop into something a server can drive.

`agent.run()` answers the graph's pause itself, through a callback, and does not
return until the run is over. An HTTP handler cannot do that: the answer arrives
in a later request, from a browser, minutes later.

So the same graph is driven one step at a time instead. `start()` runs until the
graph either finishes or pauses, and returns which. `resume()` carries the human's
answer back into the paused run. The thread id is the session.

The graph, the nodes, and the approval rules are untouched. This is a different
way of calling them, not a second implementation.
"""

from typing import Literal
from uuid import uuid4

from langgraph.types import Command
from pydantic import BaseModel, Field

from agent import AgentState, FinalAnswer, Usage, build_agent
from tools import ToolCallResult


class Proposal(BaseModel):
    """One pending write, with the context a human needs to judge it."""

    tool: str
    input: dict
    already_booked: list[dict] = Field(
        default_factory=list, description="That day's events; each carries a `clashes` flag."
    )
    clashes_with: list[str] = Field(
        default_factory=list,
        description="Titles the proposed slot would run into. Empty when the slot is free.",
    )
    other_options: list[str] = Field(default_factory=list)


class Approval(BaseModel):
    question: str
    proposals: list[Proposal]


class RunStep(BaseModel):
    """What the caller gets back: either a question to answer, or a result."""

    thread_id: str
    status: Literal["paused", "done"]
    approval: Approval | None = None
    final: FinalAnswer | None = None
    steps: list[str] = Field(default_factory=list)
    tool_log: list[ToolCallResult] = Field(default_factory=list)
    turns: int = 0
    usage: Usage = Field(default_factory=Usage, description="Tokens, model calls and seconds so far.")


# One compiled graph for the process. Compiling per request would hand each one a
# fresh checkpointer, and the resume would find no run to resume.
_APP = None


def _app():
    global _APP
    if _APP is None:
        _APP = build_agent()
    return _APP


def _advance(thread_id: str, payload: object) -> RunStep:
    result = _app().invoke(payload, {"configurable": {"thread_id": thread_id}})
    state = AgentState(**result)

    paused = result.get("__interrupt__")
    if paused:
        return RunStep(
            thread_id=thread_id,
            status="paused",
            approval=Approval(**paused[0].value),
            steps=state.steps,
            tool_log=state.tool_log,
            turns=state.turns,
            usage=state.usage,
        )

    return RunStep(
        thread_id=thread_id,
        status="done",
        final=state.final,
        steps=state.steps,
        tool_log=state.tool_log,
        turns=state.turns,
        usage=state.usage,
    )


def start(instruction: str) -> RunStep:
    """Begin a run. Returns at the first approval request, or at the end."""
    return _advance(uuid4().hex, AgentState(user_input=instruction))


def resume(thread_id: str, decision: bool | str) -> RunStep:
    """Answer a pause and carry on.

    Three answers, matching the terminal approver: True approves, False is a hard
    no, and any string is a counter-proposal the agent will try to satisfy.
    """
    return _advance(thread_id, Command(resume=decision))


if __name__ == "__main__":
    step = start("book an hour tomorrow for the Playwright login suite refactor")
    print(f"status: {step.status}  steps: {' -> '.join(step.steps)}")

    if step.status == "paused":
        print(f"\n{step.approval.question}")
        for proposal in step.approval.proposals:
            args = proposal.input
            print(f"  {args.get('title')} on {args.get('date')} at {args.get('start_time')}")
            print(f"  other options: {proposal.other_options[:6]}")

        print("\n-- declining with a counter-proposal --")
        step = resume(step.thread_id, "too early, make it after lunch")
        print(f"status: {step.status}  steps: {' -> '.join(step.steps)}")

        if step.status == "paused":
            args = step.approval.proposals[0].input
            print(f"  revised to {args.get('date')} at {args.get('start_time')}")
            print("\n-- approving the revision --")
            step = resume(step.thread_id, True)

    print(f"\nfinal status: {step.status}")
    print("\nTOOL CALLS")
    for entry in step.tool_log:
        detail = str(entry.output) if entry.ok else f"REFUSED {entry.error}"
        print(f"  {entry.tool}: {detail[:110]}")

    print("\nFINAL")
    print(step.final.model_dump_json(indent=2) if step.final else "no final answer")

    from mock_data import CALENDAR
    print("\nMOCK CALENDAR NOW")
    for event in CALENDAR:
        print(f"  {event['date']} {event['start_time']} {event['title']}")
