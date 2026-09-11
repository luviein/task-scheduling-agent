# Task Scheduling Agent

A stateful scheduling agent built on LangGraph, and an automated evaluation
suite that measures whether it actually works. The eval layer is the point of
the project.

The agent takes a multi-step instruction, searches a task list, computes real
calendar availability, pauses for human approval before it writes anything, and
returns a schema-validated result. Everything is typed with Pydantic end to end:
tool inputs, tool outputs, graph state, and the final answer.

```
  Approve these before they are written?
    Write LangGraph agent draft
      2026-09-11 at 09:30 for 120 min
      that day: 09:00-09:15 Standup, 14:00-15:00 Design review
      other free starts: 10:00, 10:30, 11:00, 11:30, 12:00, 15:00, 15:30, 16:00
  approve? [y / N / a reason to decline]:
```

---

## The finding

On the first full eval run, one case failed. Asked to *"schedule two hours
tomorrow to do my tax return"*, where no such task exists, the agent invented a
task called "Tax Return", booked two hours for it, and reported nothing
unresolved. A fabrication that reached the calendar, not just the prose.

**The LLM judge scored that same case a perfect 1.00.**

It was not wrong to. Its verdict read: *"the summary is fully supported by the
provided context."* And it was. The judge sees the tool results, and the booking
tool genuinely did report success. The fabrication happened *upstream* of the
tool call, in the arguments the agent chose. That is outside what the judge can
see.

A five-line set comparison caught it immediately:

```python
invented = [title for title in claimed if title not in REAL_TASK_TITLES]
```

**The lesson this project is built around.** Use an exact assertion wherever the
correct answer is knowable. Reserve LLM-as-judge for questions that are
genuinely fuzzy. Reaching for a judge on something a set comparison can decide
buys you cost, latency, and false confidence.

This is also why `TaskExecutionMetric` scores against the calendar's actual
contents rather than the agent's own summary. An agent that claims success while
booking nothing must not be able to pass.

---

## Results

Six cases, five metrics. Scores are 0 to 1, averaged across cases.

| Metric | Judged by | v1 baseline | v2 stricter | v3 broadened | current |
|---|---|---|---|---|---|
| Tool Selection | rule | 1.00 | 0.90 | 1.00 | **1.00** |
| Tool Efficiency | rule | — | — | — | 0.96 |
| Task Execution | rule | 0.80 | 0.80 | 1.00 | **1.00** |
| Task Grounding | rule | 0.80 | 1.00 | 1.00 | **1.00** |
| Summary Faithfulness | LLM judge | 1.00 | 0.86 | 0.90 | **1.00** |

The first three columns are the prompt-tuning progression described below, on the
original five cases. `current` is the finished system: six cases, an availability
tool, and the approval gate.

Tool Efficiency sits at 0.96 by choice. One case searches four times against a
ceiling of three, and the ceiling was not raised to match. Moving the bar to fit
the behavior is how a suite stops meaning anything.

**v2** added a rule: book nothing when no task matches, and report it as
unresolved. It fixed the hallucination and grounding reached 1.00.

**It also broke something else.** The conflict-recovery case regressed. Told to
book time for the Playwright refactor, the agent searched once with a narrow
query, got zero matches, and declared the task nonexistent. It exists. In v1 the
agent had searched again with a broader keyword and found it. The new rule taught
it to stop, and it stopped too early.

That trade is the argument for the whole suite. A single-case check on the
hallucination would have shown a clean fix and shipped a worse agent.

**v3** added one more line: *one empty result is not proof, search again with a
more general keyword first.* Every rule-based metric reached 1.00. The
hallucination stayed fixed and conflict recovery came back.

Every run is preserved under `runs/history/` with a matching report, so the
progression is reproducible rather than recalled. The current run lives in
`runs/`.

**The suite kept earning its place after that.** Adding the ambiguity case caught
the agent silently choosing a task you never named and booking it. Tool Efficiency
caught two drifts no correctness metric noticed. And a late change to a field
description made the agent write `["None. All parts were handled."]` into the
`unresolved` list on a fully successful run, which would break any caller doing
`if unresolved:`. Every one of those was a change aimed somewhere else.

That is the whole argument for the suite. Five of the six regressions in this
project's history were side effects, not the thing being worked on.

### The judge failed in both directions

The same LLM judge that passed a fabricated booking in v1 produced a **false
negative** in v2 and again in v3: it marked the two-booking case down for
*"falsely claims that both tasks were successfully scheduled."* The calendar
shows both tasks booked, at 15:00 and 16:00, and `TaskExecutionMetric` confirms
it against that ground truth. The summary was accurate. The judge penalised it
for not narrating an intermediate conflict the summary never claimed had
succeeded, and it made the same mistake on two independent runs, so this is a
systematic flaw rather than sampling noise.

Across three runs the judge missed a real failure once and invented a fake one
twice, while the rule-based metrics were correct every time. This is not an
argument against LLM judges. It is an argument for knowing which questions they
are for, and for never letting one adjudicate something you can check exactly.

It is also why the judge's threshold sits at 0.7 while the rule-based metrics
demand a perfect score. A metric you know to be noisy should not be given the
authority to fail a build on its own.

---

## Architecture

```
START -> reason ---(wants tools)---> confirm ---> act
           ^                            |          |
           |                       (interrupt)     |
           |________________________________________|
           |
           +---(done)---> summarize -> END
```

- **`reason`** asks the model for the next move. It either emits tool calls or stops.
- **`confirm`** pauses for human approval, but only when a tool in `WRITE_TOOLS` is proposed. Reads are never gated.
- **`act`** runs the calls through a validating dispatcher and writes results back into state.
- **`summarize`** makes one tool-free, schema-constrained call, producing a validated `FinalAnswer`.

The graph owns the loop, not the model. `MAX_TURNS` caps it.

### Human in the loop

`confirm` uses LangGraph's `interrupt` with a checkpointer, so the graph genuinely
stops mid-run and resumes where it left off rather than replaying. The design rule:
**the graph always asks, the caller decides how to answer.** A terminal session asks
a person; the eval harness passes `approve_everything`. Tests therefore exercise the
same code path a human does, rather than a bypass.

Approval has three outcomes, not two. `y` approves. A bare `n` is a hard no and is
sticky, enforced in code because the model ignored a prompt rule telling it not to
retry. Anything else is a counter-proposal, passed back as an instruction with a
capped number of revision rounds.

The approval prompt shows the day's existing events and the other free starts,
computed by the same functions the tools use. Asking the model to explain its
choice would let the explanation drift from what actually happened, which is
exactly the failure mode the judge metric exists to catch and got wrong three
times below.

### Availability is computed, not guessed

Early versions had no availability tool. The model picked a plausible time, tried
to book it, and only learned about a clash when the tool refused, so it burned a
call per wrong guess and landed on arbitrary slots. `find_free_slots` computes
openings in ordinary Python. Bookings now land on the first attempt, on the
genuinely earliest opening.

Never make a model guess what code can calculate exactly.

### Type safety

Pydantic input models generate the JSON Schema sent to the provider, so the
model cannot produce an argument set the validator would reject. `dispatch()`
validates, executes, and returns a uniform `ToolCallResult` envelope. It never
raises. The eval layer asserts against that envelope rather than against prose,
which is what makes tool-calling accuracy measurable instead of anecdotal.

### One calendar interface, two implementations

The tool layer never touches a calendar. It asks `get_backend()`, which returns
whichever `CalendarBackend` the `CALENDAR_BACKEND` variable names: the in-memory
mock, or a real Google Calendar over OAuth. A backend supplies two primitives,
listing a day and creating an event; free-slot search has a default derived from
listing, so a third implementation would be about forty lines.

Two decisions worth defending:

**Conflict detection lives in the tool layer, not in a backend.** Refusing to
double-book is the agent's rule, not the calendar's. Google accepts overlapping
events without complaint, so pushing the check down would make the agent behave
differently depending on where its events happened to be stored.

**The clock belongs to the calendar.** `backend.today()` returns the mock's
frozen date or the real date in the real calendar's timezone. A frozen clock is
what makes evals reproducible; a live calendar with a frozen clock would have
the agent reasoning confidently about the wrong day.

**The eval suite is pinned to the mock permanently.** `eval_run.py` overrides the
backend at import regardless of configuration. Scoring against a live calendar
would create real events, defeat `reset_calendar()`, and make every score depend
on that week's meetings.

### Editable tasks, frozen expectations

The task list is editable from the UI and persists to `tasks.json`. The eval
suite does not read it. `mock_data.TASKS` stays the seed, `reset_tasks()` runs
per case alongside `reset_calendar()`, and the hallucination check grounds
against the seed titles rather than the live list.

Without that split, adding a task through the UI would quietly change what
"invented a task title" means, and a suite that scored 1.00 yesterday would
score differently today for reasons nothing recorded. Same reasoning as pinning
the evals to the mock calendar: anything a human can edit between runs cannot
also be the yardstick.

### The gate held and the report lied

Every eval case auto-approved, so writes always succeeded and the agent's claim
about what it booked always happened to match reality. Building a session API
meant exercising the decline path for the first time, and it failed:

```
create_calendar_event: REFUSED  Not approved. The user said: too early, make it after lunch.
find_free_slots:       {...}
FINAL events_created:  ["Refactor Playwright login suite"]
CALENDAR:              unchanged
```

Nothing was written without consent, so the approval gate worked. The report did
not. The user was told their afternoon was blocked while the calendar was empty.

The root cause was mundane. `MAX_TURNS` was 6, and the decline path needs more:
a turn to refuse, one to look up slots again, one to re-book, on top of the
search and availability turns every run spends. Out of turns, the graph routed
to `summarize`, and the model wrote up the booking it had intended to make.

Two changes came out of it. `events_created` is no longer the model's claim; the
summarize node computes it from bookings that actually returned `created=true`,
because the tool log cannot be talked into anything. And `revise_after_decline`
now covers the path, with scripted answers to the approval gate so a case can
decline as easily as approve.

The lesson is about eval coverage rather than about this bug. Four metrics scored
1.00 across six cases while a whole branch of the graph went untested. A suite
only measures the paths it walks.

### Provider independence

The project was built against one provider and moved to another mid-build. Only
the two node bodies that make model calls changed. The graph, state, tool layer,
and dispatcher were untouched. One adapter (`_strip_for_gemini`) reconciles the
two JSON Schema dialects.

---

## Eval design

**Runs and scores are separate programs.** `eval_run.py` executes the suite and
saves traces to `runs/traces.json`. `eval_score.py` reads them back. Model calls are slow,
rate limited, and non-deterministic, so you pay for them once and then iterate on
metrics for free. It also means scoring is reproducible: the same traces score
the same way every time.

**Expectations are written before the run.** Each case in `eval_cases.py`
declares which tools must fire, which must not, what should be booked, and
whether anything should be reported unresolved. Editing an expectation to make a
case pass is the moment to stop and ask which one is actually wrong.

**Failed runs are distinguished from bad runs.** An early suite attempt died on
an API quota, and four cases returned nothing. Scored naively, that reads as an
agent quality collapse. `eval_run.py` records infrastructure errors in their own
field so they cannot be mistaken for behavior.

**State is reset between cases.** The calendar is module-level and mutable, so
`reset_calendar()` runs before each case. Without it, case five inherits case
four's bookings and results depend on execution order.

### The metrics

| Metric | Judged by | Asks |
|---|---|---|
| Tool Selection | rule | Did it call the required tools, and avoid the forbidden ones? |
| Tool Efficiency | rule | Did it get there without wasting calls, against a per-case ceiling? |
| Task Execution | rule | Did the booking actually land, checked against the calendar? |
| Task Grounding | rule | Is every task title it reports real? |
| Summary Faithfulness | LLM judge | Does its prose match what the tools returned? |

Tool Efficiency was added after a run reached the right answer by searching three
times for a task the first search had already found. Correctness metrics are blind
to that; on a metered API it is money, and on a free tier with a daily cap it is
the ability to run the suite twice. It has since caught two separate drifts that
no correctness metric noticed.

---

## Running it

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Put a Gemini API key in `.env`:

```
GEMINI_API_KEY=...
```

```bash
# tool layer, no API key needed
python tools.py

# one agent run
python agent.py "book two hours friday for the QA refactor"

# record eval traces (spends API quota)
python eval_run.py

# score them (free, repeatable)
python eval_score.py --no-judge   # rule-based metrics only
python eval_score.py              # adds the LLM judge
```

### Running the web UI

Two processes. The API on 8000, the client on 5173, which proxies `/api` to the
first so both look like one origin.

```bash
# terminal 1
.venv/Scripts/python.exe -m uvicorn api:app --reload --port 8000

# terminal 2
cd web && npm install && npm run dev
```

Open http://localhost:5173. The approval card shows the proposed slot, what else
is on that day, and the other free starts. Clicking one of those sends it back as
a counter-proposal rather than booking it directly, so the agent re-checks
availability instead of being overridden.

### Running against a real calendar

Optional. The mock is the default and needs none of this.

Create a Google Cloud project, enable the Google Calendar API, and configure the
OAuth consent screen as External with your own address added as a test user.
Request the `calendar.events` scope, which covers reading and writing events and
nothing wider. Create an OAuth client of type **Desktop app**, download the JSON,
and save it in the project root as `credentials.json`.

```bash
# first run opens a browser for consent and writes token.json
python google_calendar.py 2026-09-11

# then point the agent at it
CALENDAR_BACKEND=google python agent.py "book an hour tomorrow for the QA refactor"
```

`credentials.json` and `token.json` are both gitignored. While the app sits in
Testing status Google expires the refresh token after seven days, so sign-in
repeats about weekly until the app is published.

---

## Files

| File | Role |
|---|---|
| `agent.py` | LangGraph state machine, nodes, provider calls |
| `tools.py` | Three tools, their Pydantic schemas, and the validating dispatcher |
| `calendar_backend.py` | The calendar interface, the mock implementation, backend selection |
| `google_calendar.py` | Google Calendar over OAuth, imported only when selected |
| `mock_data.py` | Seed task list and calendar, frozen clock, state reset |
| `task_store.py` | The editable task list, persisted to tasks.json, reseedable for evals |
| `eval_cases.py` | The eval set and its written expectations |
| `session.py` | start/resume wrapper so a server can drive the approval pause |
| `api.py` | FastAPI routes over that wrapper |
| `web/` | Vite and React client, typed against the API |
| `eval_run.py` | Executes cases, saves traces |
| `eval_metrics.py` | Four rule-based metrics, one LLM-judged |
| `eval_score.py` | Scores saved traces, writes the report |

---

## Notes on the free tier

The default model is pinned rather than using a `-latest` alias, because eval
scores are only comparable across runs if the model is fixed. Free-tier quotas
are per model and include a daily cap, which one eval run can exhaust. The
client distinguishes per-minute throttling, which it waits out, from daily
exhaustion, which it fails fast on rather than sleeping through a retry loop that
cannot succeed.
