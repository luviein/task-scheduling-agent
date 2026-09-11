"""Stateful agent: reason -> act -> reason -> ... -> summarize.

The graph owns the loop. The model only ever decides "call a tool" or "I'm done".
Everything the evals layer needs is recorded in AgentState.tool_log.

Provider is Gemini, but nothing below the node bodies depends on that.
"""

import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime, timedelta
from uuid import uuid4

from dotenv import load_dotenv
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, ConfigDict, Field

from calendar_backend import BUSINESS_END, BUSINESS_START, get_backend, overlaps
from task_store import mark_booked
from tools import ToolCallResult, anthropic_tool_defs, day_agenda, dispatch

load_dotenv()

# The SDK warns about automatic function calling whenever tools are declared here,
# even though we disable it. We drive the loop ourselves, so the advice doesn't apply.
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

# Pinned on purpose. Evals are only comparable across runs if the model is fixed.
# Run `python agent.py --models` to see what your key can reach.
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
MAX_TURNS = 8  # hard stop so a confused agent can't loop forever
# Raised from 6 when the revision path proved to need the headroom: a declined
# booking costs a turn to refuse, one to look up slots again, and one to re-book,
# on top of the search and availability turns every run already spends.

# Reads run freely; writes need a human. Adding a destructive tool later means
# adding it here, not remembering to gate it at the call site.
WRITE_TOOLS = {"create_calendar_event"}

# How many times the user may send a proposal back for revision before the
# agent must stop and explain itself instead of proposing again.
MAX_REVISIONS = 2

SYSTEM_TEMPLATE = """You are a productivity agent. Today is {today}.

Rules:
- Look up real data with search_tasks before you schedule anything.
- If search_tasks returns no matches, search again with a shorter or more general
  keyword before concluding the task does not exist. One empty result is not proof.
- Only book an event whose title is copied exactly from a task returned by search_tasks.
- If the request never says which task to work on, book nothing. List the candidates,
  ask which one, and put the question in `unresolved`. A request that identifies the
  task indirectly, by priority or by topic, is not ambiguous: act on it.
- If no task matches what the user asked for, book nothing. Say the task does not
  exist and list what you could not do in `unresolved`. Do not invent a task to book.
- Before booking, call find_free_slots to see what is actually available. Never guess a time.
- Book the earliest slot it returns that suits the request. Business hours are {opens} to {closes}.
- If a booking still returns created=false, pick the next free slot and try again.
- If a booking is refused because the user asked for a different time, that is an
  instruction, not an ending. Call find_free_slots again if you need to, then call
  create_calendar_event again with the revised time. Do not summarize until you have.
- Never report an event as booked unless create_calendar_event returned created=true
  for it. A refused booking goes in `unresolved`, with the reason it was refused.
- Nothing can be booked outside business hours. If the user asks for a time outside
  them, do not keep proposing alternatives: say plainly that the time is outside
  business hours and what the range is.
- When every part of the request is handled, stop calling tools and give a short summary."""


def config_fingerprint() -> dict:
    """What a score depends on besides the agent's own cleverness.

    Comparing runs is meaningless without this. The prompt changed twice and the
    turn ceiling once in a single afternoon, and nothing in the report said so,
    which made every earlier number quietly incomparable.

    The prompt is hashed from the template, not the rendered prompt, so the
    fingerprint does not change just because it is a different day.
    """
    return {
        "model": MODEL,
        "prompt_sha": hashlib.sha256(SYSTEM_TEMPLATE.encode("utf-8")).hexdigest()[:12],
        "max_turns": MAX_TURNS,
        "max_revisions": MAX_REVISIONS,
    }


def system_prompt() -> str:
    """Built per run, not at import: the date comes from whichever calendar is active."""
    return SYSTEM_TEMPLATE.format(
        today=get_backend().today().isoformat(),
        opens=BUSINESS_START.strftime("%H:%M"),
        closes=BUSINESS_END.strftime("%H:%M"),
    )


# --- Structured final output ------------------------------------------------
class FinalAnswer(BaseModel):
    """The agent's contract with the caller. Evals assert against these fields."""

    summary: str = Field(description="One or two sentences on what was done.")
    tasks_found: list[str] = Field(description="Titles of tasks returned by search_tasks. Empty if none.")
    events_created: list[str] = Field(description="Titles of events actually booked. Empty if none.")
    unresolved: list[str] = Field(
        description="Anything requested but not completed. Each entry must say what "
        "failed AND the specific reason why, so the reader needs no further explanation. "
        "When everything succeeded this list MUST be empty: never write a placeholder "
        "such as 'none' or 'nothing outstanding', because callers test whether the list "
        "is empty."
    )


# --- Schema adapter ---------------------------------------------------------
def _strip_for_gemini(node: dict) -> dict:
    """Gemini's schema validator is fussier than the one Step 2 targeted.

    Drop the keywords it rejects and fold `format` into the description so the
    model still knows the expected shape. One tool schema, two dialects.
    """
    dropped = ("title", "additionalProperties", "strict", "format")
    clean: dict = {}
    for key, value in node.items():
        if key in dropped:
            continue
        # Recurse only into schema positions. Keys under `properties` are field
        # names, and one of ours is literally called "title".
        if key == "properties" and isinstance(value, dict):
            clean[key] = {name: _strip_for_gemini(sub) for name, sub in value.items()}
        elif key == "items" and isinstance(value, dict):
            clean[key] = _strip_for_gemini(value)
        else:
            clean[key] = value
    if "format" in node:
        clean["description"] = f"{node.get('description', '')} Format: {node['format']}.".strip()
    return clean


def gemini_tools() -> list[types.Tool]:
    declarations = [
        types.FunctionDeclaration(
            name=spec["name"],
            description=spec["description"],
            parameters_json_schema=_strip_for_gemini(spec["input_schema"]),
        )
        for spec in anthropic_tool_defs()
    ]
    return [types.Tool(function_declarations=declarations)]


# --- Cost and latency -------------------------------------------------------
# Rates are not hardcoded. Free-tier usage costs nothing, paid rates change, and
# a made-up number in a report is worse than no number. Set these from the
# provider's pricing page when you want dollars; tokens are recorded regardless.
INPUT_COST_PER_MTOK = float(os.getenv("INPUT_COST_PER_MTOK", "0") or 0)
OUTPUT_COST_PER_MTOK = float(os.getenv("OUTPUT_COST_PER_MTOK", "0") or 0)


class Usage(BaseModel):
    """What a run spent. Tool Efficiency counts calls; this is what calls cost."""

    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    seconds: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens * INPUT_COST_PER_MTOK + self.output_tokens * OUTPUT_COST_PER_MTOK
        ) / 1_000_000

    def plus(self, response, seconds: float) -> "Usage":
        """This ledger plus one model call. Returns a new Usage; never mutates."""
        meta = getattr(response, "usage_metadata", None)
        return Usage(
            calls=self.calls + 1,
            # A provider that reports nothing must not zero what came before, and
            # must not crash the run either. Missing counts are simply not added.
            input_tokens=self.input_tokens + (getattr(meta, "prompt_token_count", 0) or 0),
            output_tokens=self.output_tokens + (getattr(meta, "candidates_token_count", 0) or 0),
            seconds=round(self.seconds + seconds, 2),
        )

    def line(self) -> str:
        money = f", ${self.cost_usd:.4f}" if self.cost_usd else ""
        return (
            f"{self.calls} model calls, {self.total_tokens} tokens "
            f"({self.input_tokens} in / {self.output_tokens} out), {self.seconds:.1f}s{money}"
        )


# --- State ------------------------------------------------------------------
class AgentState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    user_input: str
    contents: list[types.Content] = Field(default_factory=list, description="Full conversation, model's native format.")
    tool_log: list[ToolCallResult] = Field(default_factory=list, description="Every tool call, in order.")
    steps: list[str] = Field(default_factory=list)
    turns: int = 0
    wants_tools: bool = False
    approved: bool | None = Field(default=None, description="None when nothing needed approval.")
    declined: bool = Field(default=False, description="Sticky: once you say no outright, the graph stops asking.")
    revisions: int = Field(default=0, description="Counter-proposals so far, capped at MAX_REVISIONS.")
    decline_reason: str | None = None
    awaiting_revision: bool = Field(
        default=False,
        description="A write was refused with a counter-proposal and nothing has been booked since.",
    )
    final: FinalAnswer | None = None
    usage: Usage = Field(default_factory=Usage, description="Tokens, calls and seconds so far.")


_CLIENT: genai.Client | None = None


def _client() -> genai.Client:
    """One client for the process.

    Building a fresh one per call looks harmless but isn't: nothing holds a
    reference once `.models` is read, the object is collected mid-call, and its
    destructor closes the underlying HTTP connection out from under the request.
    """
    global _CLIENT
    if _CLIENT is None:
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise SystemExit("GEMINI_API_KEY is not set. Put it in .env (see .env.example).")
        _CLIENT = genai.Client(api_key=key)
    return _CLIENT


class DailyQuotaExhausted(RuntimeError):
    """Raised instead of retrying, because tomorrow is not a retry interval."""


_RETRY_SECONDS = re.compile(r"retry in ([\d.]+)s", re.IGNORECASE)


def call_model(**kwargs) -> types.GenerateContentResponse:
    """generate_content with free-tier rate limiting handled.

    The free tier allows a handful of requests per minute and one agent run
    spends several, so 429s are expected traffic here, not failures. Google
    tells us how long to wait; obey it instead of guessing.
    """
    for attempt in range(4):
        try:
            return _client().models.generate_content(**kwargs)
        except ServerError as exc:
            # 503 means the model is momentarily oversubscribed. Back off and retry.
            if attempt == 3:
                raise
            wait = 5 * (attempt + 1)
            print(f"  [model busy, retrying in {wait}s]", flush=True)
            time.sleep(wait)
        except ClientError as exc:
            if exc.code != 429 or attempt == 3:
                raise
            # Two different 429s wear the same status code. A per-minute limit
            # clears in a minute; a per-day one does not clear today, so waiting
            # on it burns three minutes per case and still fails.
            if "PerDay" in str(exc):
                raise DailyQuotaExhausted(
                    f"Daily free-tier quota for {kwargs.get('model')} is used up. "
                    f"Set GEMINI_MODEL in .env to a different model, or wait for the reset."
                ) from exc
            match = _RETRY_SECONDS.search(str(exc))
            wait = float(match.group(1)) + 1 if match else 30.0
            print(f"  [rate limited, waiting {wait:.0f}s]", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


# --- Nodes ------------------------------------------------------------------
def reason(state: AgentState) -> dict:
    """Ask the model what to do next. It either emits function calls or stops."""
    contents = state.contents or [types.Content(role="user", parts=[types.Part.from_text(text=state.user_input)])]

    started = time.perf_counter()
    response = call_model(
        model=MODEL,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_prompt(),
            tools=gemini_tools(),
            # We drive the loop through LangGraph, so the SDK must not call tools itself.
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        ),
    )
    # Measured around call_model, so retries and rate-limit waits are included.
    # They are latency the caller actually waited through.
    elapsed = time.perf_counter() - started

    reply = response.candidates[0].content
    return {
        "contents": [*contents, reply],
        "wants_tools": bool(response.function_calls),
        "turns": state.turns + 1,
        "usage": state.usage.plus(response, elapsed),
        "steps": [*state.steps, "reason"],
    }


def overlaps_slot(day: str, start: str | None, minutes: int, booked: dict) -> bool:
    """Does a proposed block run into one already-booked entry?

    Same overlap rule the tool layer enforces, reused rather than restated, so
    the warning a human sees and the refusal the tool would give cannot disagree.
    """
    if not day or not start:
        return False
    try:
        begin = datetime.fromisoformat(f"{day}T{start}")
        block = (begin, begin + timedelta(minutes=int(minutes)))
        taken = (
            datetime.fromisoformat(f"{day}T{booked['from']}"),
            datetime.fromisoformat(f"{day}T{booked['to']}"),
        )
    except (ValueError, KeyError, TypeError):
        # A malformed proposal is the dispatcher's problem, not this warning's.
        return False
    return overlaps(block, taken)


def confirm(state: AgentState) -> dict:
    """Pause for human approval before anything is written.

    Reads are never gated; only WRITE_TOOLS trigger the interrupt. This node has
    no side effects on purpose: LangGraph re-runs it from the top when the graph
    resumes, so anything done before `interrupt()` would happen twice.
    """
    pending = [p.function_call for p in state.contents[-1].parts if p.function_call]
    writes = [call for call in pending if call.name in WRITE_TOOLS]

    if not writes:
        return {"approved": None, "steps": [*state.steps, "confirm(skipped)"]}

    # A hard no is final: told only in the prompt not to retry, the model kept
    # proposing new slots. Enforce it here instead.
    if state.declined:
        return {
            "approved": False,
            "decline_reason": "already declined; do not propose this again",
            "steps": [*state.steps, "confirm(auto-declined)"],
        }

    # Show the reasoning, not just the conclusion. A bare proposal cannot answer
    # "why that time?", and a human being asked to approve a write deserves to know.
    proposals = []
    for call in writes:
        args = dict(call.args or {})
        day = args.get("date", "")
        duration = args.get("duration_minutes", 60)
        free = dispatch("find_free_slots", {"date": day, "duration_minutes": duration})

        # Which existing events the proposal would run into. Asked for a specific
        # time, the model will propose it even when the day is busy; the write
        # would then be refused, but only after a human had already approved it.
        # Work this out here so the approval prompt can say so up front.
        agenda = day_agenda(day)
        for booked in agenda:
            booked["clashes"] = overlaps_slot(day, args.get("start_time"), duration, booked)

        proposals.append(
            {
                "tool": call.name,
                "input": args,
                "already_booked": agenda,
                "clashes_with": [item["title"] for item in agenda if item["clashes"]],
                "other_options": [slot for slot in (free.output or {}).get("slots", []) if slot != args.get("start_time")],
            }
        )

    decision = interrupt({"question": "Approve these before they are written?", "proposals": proposals})

    if decision is True:
        return {"approved": True, "steps": [*state.steps, "confirm(approved)"]}

    # A string is a counter-proposal, not a refusal. Hand it back as an
    # instruction and let the agent revise, but cap the round trips so a user
    # who keeps asking for the impossible does not loop until MAX_TURNS.
    if isinstance(decision, str) and state.revisions < MAX_REVISIONS:
        return {
            "approved": False,
            "awaiting_revision": True,
            "revisions": state.revisions + 1,
            "decline_reason": f"Not approved. The user said: {decision}. Nothing has been booked yet. Call create_calendar_event again with a time that matches what they asked for, or explain why no such time exists. Do not stop without doing one of those.",
            "steps": [*state.steps, "confirm(revise)"],
        }

    exhausted = isinstance(decision, str)
    return {
        "approved": False,
        "declined": True,
        "decline_reason": (
            f"The user said: {decision}. You have revised enough; stop proposing and explain the outcome."
            if exhausted
            else "The user declined outright."
        ),
        "steps": [*state.steps, "confirm(declined)"],
    }


def act(state: AgentState) -> dict:
    """Run every function call from the last model turn through dispatch."""
    calls = [p.function_call for p in state.contents[-1].parts if p.function_call]

    results, parts = [], []
    for call in calls:
        if call.name in WRITE_TOOLS and state.approved is False:
            # A refusal the model can read and respond to, not a crash.
            result = ToolCallResult(
                tool=call.name,
                ok=False,
                error=state.decline_reason or "The user declined this.",
            )
        else:
            result = dispatch(call.name, dict(call.args or {}))
        results.append(result)
        # Parallel calls go back in ONE turn, in the order they were requested.
        parts.append(types.Part.from_function_response(name=call.name, response=result.model_dump(mode="json")))

    booked = False
    for result in results:
        if result.tool != "create_calendar_event" or not result.ok:
            continue
        event = (result.output or {}).get("event")
        if not (result.output or {}).get("created") or not event:
            continue
        booked = True
        # The calendar is the source of truth for what was booked, so the task
        # list follows it rather than the other way round.
        mark_booked(
            event["title"],
            event["date"],
            event["start_time"],
            event["duration_minutes"],
            event.get("id"),
        )

    return {
        "contents": [*state.contents, types.Content(role="user", parts=parts)],
        "tool_log": [*state.tool_log, *results],
        "awaiting_revision": False if booked else state.awaiting_revision,
        "steps": [*state.steps, "act"],
    }


def summarize(state: AgentState) -> dict:
    """Second call, no tools, schema-constrained. This is the structured output."""
    closing = types.Content(
        role="user",
        parts=[types.Part.from_text(text="Summarize what you did as structured output.")],
    )
    started = time.perf_counter()
    response = call_model(
        model=MODEL,
        contents=[*state.contents, closing],
        config=types.GenerateContentConfig(
            system_instruction=system_prompt(),
            response_mime_type="application/json",
            response_schema=FinalAnswer,
        ),
    )
    usage = state.usage.plus(response, time.perf_counter() - started)
    final = response.parsed
    if final is not None:
        # The model is not the authority on what it booked. A run that had its
        # write refused once reported the booking as done anyway, so take this
        # field from the tool log, which cannot be talked into anything.
        final.events_created = [
            entry.output["event"]["title"]
            for entry in state.tool_log
            if entry.tool == "create_calendar_event"
            and entry.ok
            and (entry.output or {}).get("created")
            and (entry.output or {}).get("event")
        ]

    if final is not None and state.awaiting_revision:
        # Reached the turn ceiling with the revision unfinished. Silence here would
        # read as success, since events_created is empty either way.
        final.unresolved = [
            *final.unresolved,
            "The booking was declined and no replacement was made. Nothing was written to the calendar.",
        ]

    return {"final": final, "usage": usage, "steps": [*state.steps, "summarize"]}


def nudge(state: AgentState) -> dict:
    """Refuse to let the agent stop halfway through a revision.

    Asked in the prompt to re-propose after a counter-proposal, the model instead
    looked up free slots and then wrote a summary claiming it had booked one. The
    prompt could not hold it; this does. Same reasoning as the hard-no rule in
    `confirm`: state a requirement in the prompt, enforce it in the graph.
    """
    reminder = types.Content(
        role="user",
        parts=[
            types.Part.from_text(
                text=(
                    "Nothing has been booked. The user declined your first time and asked "
                    "for a different one. Call create_calendar_event now with a time that "
                    "matches what they asked for. If no such time is free, say so plainly "
                    "and put it in unresolved. Do not claim a booking you have not made."
                )
            )
        ],
    )
    return {"contents": [*state.contents, reminder], "steps": [*state.steps, "nudge"]}


# --- Routing ----------------------------------------------------------------
def route_after_reason(state: AgentState) -> str:
    if state.wants_tools and state.turns < MAX_TURNS:
        return "confirm"
    # Stopping is only allowed once the revision is settled, one way or the other.
    # MAX_TURNS still bounds the loop, so this cannot spin forever.
    if state.awaiting_revision and state.turns < MAX_TURNS:
        return "nudge"
    return "summarize"


def build_agent():
    graph = StateGraph(AgentState)
    graph.add_node("reason", reason)
    graph.add_node("confirm", confirm)
    graph.add_node("act", act)
    graph.add_node("nudge", nudge)
    graph.add_node("summarize", summarize)

    graph.add_edge(START, "reason")
    graph.add_conditional_edges(
        "reason", route_after_reason, {"confirm": "confirm", "nudge": "nudge", "summarize": "summarize"}
    )
    graph.add_edge("nudge", "reason")
    graph.add_edge("confirm", "act")
    graph.add_edge("act", "reason")  # the loop
    graph.add_edge("summarize", END)

    # A checkpointer is what makes the pause durable: the graph stops mid-run,
    # the state survives, and Command(resume=...) picks up exactly where it left off.
    #
    # State holds two non-builtin types. Naming them explicitly silences the
    # deserialization warning and, more usefully, opts into the strict allowlist
    # that a future LangGraph will enforce anyway.
    serde = JsonPlusSerializer(
        allowed_msgpack_modules=[("google.genai.types", "Content"), ("tools", "ToolCallResult")]
    )
    return graph.compile(checkpointer=InMemorySaver(serde=serde))


def ask_terminal(request: dict) -> bool | str:
    """Interactive approver. Anything that isn't yes counts as a decline."""
    print(f"\n  {request['question']}")
    for proposal in request["proposals"]:
        args = proposal["input"]
        print(f"    {args.get('title', proposal['tool'])}")
        print(f"      {args.get('date')} at {args.get('start_time')} for {args.get('duration_minutes')} min")

        booked = proposal.get("already_booked") or []
        print("      that day: " + (", ".join(f"{e['from']}-{e['to']} {e['title']}" for e in booked) or "nothing booked"))

        options = proposal.get("other_options") or []
        if options:
            shown = ", ".join(options[:8]) + (" ..." if len(options) > 8 else "")
            print(f"      other free starts: {shown}")
    try:
        answer = input("  approve? [y / N / a reason to decline]: ").strip()
    except EOFError:
        # Piped or non-interactive stdin. Refusing is the safe read of silence.
        return "no answer given, stdin closed"
    if answer.lower() in {"y", "yes"}:
        return True
    # Three outcomes, not two. A bare "n" is a hard no. Anything else is a
    # counter-proposal ("too early, make it 2pm") and the agent should act on it
    # rather than treat it as a refusal.
    return False if answer.lower() in {"", "n", "no"} else answer


def approve_everything(request: dict) -> bool:
    """For the eval harness. The graph still asks; the harness always says yes.

    Keeping the question in the graph and the answer in the caller is the point:
    the same code path is exercised in tests and in front of a human.
    """
    return True


def run(instruction: str, decide=ask_terminal) -> AgentState:
    """Drive the graph, answering any pause via `decide`, until it finishes."""
    app = build_agent()
    config = {"configurable": {"thread_id": uuid4().hex}}
    payload: object = AgentState(user_input=instruction)

    while True:
        result = app.invoke(payload, config)
        paused = result.get("__interrupt__")
        if not paused:
            return AgentState(**result)
        payload = Command(resume=decide(paused[0].value))


if __name__ == "__main__":
    import json
    import sys

    args = sys.argv[1:]
    if args and args[0] == "--models":
        # Model names change; ask the key what it can actually reach.
        for model in _client().models.list():
            if "generateContent" in (model.supported_actions or []):
                print(model.name)
        raise SystemExit(0)

    prompt = " ".join(args) or "Find my open AI tasks and book an hour tomorrow to work on the highest priority one."
    state = run(prompt)

    print(f"\nPATH:  {' -> '.join(state.steps)}")
    print(f"TURNS: {state.turns}")
    print(f"SPENT: {state.usage.line()}")
    print("\nTOOL CALLS")
    for entry in state.tool_log:
        detail = json.dumps(entry.output) if entry.ok else f"FAIL {entry.error}"
        print(f"  {entry.tool}: {detail[:120]}")
    print("\nFINAL")
    print(state.final.model_dump_json(indent=2) if state.final else "none")
