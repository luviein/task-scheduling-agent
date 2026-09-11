import { useEffect, useState } from "react";
import {
  getConfig,
  getTasks,
  sendDecision,
  startRun,
  type Config,
  type Decision,
  type Proposal,
  type RunStep,
  type Task,
} from "./api";

const EXAMPLES = [
  "Book an hour tomorrow morning for the Playwright login suite refactor.",
  "Find my open AI tasks and book an hour tomorrow for the highest priority one.",
  "What QA work do I have open? Do not book anything.",
];

function ProposalCard({
  proposal,
  onDecide,
  busy,
}: {
  proposal: Proposal;
  onDecide: (decision: Decision) => void;
  busy: boolean;
}) {
  const [reason, setReason] = useState("");
  const { title, date, start_time, duration_minutes } = proposal.input;

  return (
    <div className="card approval">
      <div className="approval-head">
        <span className="badge warn">Needs your approval</span>
        <span className="muted">nothing is written until you say so</span>
      </div>

      <h3>{title ?? proposal.tool}</h3>
      <p className="when">
        {date} at {start_time} for {duration_minutes} minutes
      </p>

      <div className="context">
        <div>
          <h4>That day</h4>
          {proposal.already_booked.length === 0 ? (
            <p className="muted">Nothing booked</p>
          ) : (
            <ul>
              {proposal.already_booked.map((slot) => (
                <li key={`${slot.from}-${slot.title}`}>
                  <code>
                    {slot.from}-{slot.to}
                  </code>{" "}
                  {slot.title}
                </li>
              ))}
            </ul>
          )}
        </div>

        <div>
          <h4>Other free starts</h4>
          {proposal.other_options.length === 0 ? (
            <p className="muted">No other openings</p>
          ) : (
            <div className="chips">
              {/* Each chip is a counter-proposal, not a direct booking: it goes
                  back through the agent, which re-checks availability. */}
              {proposal.other_options.map((slot) => (
                <button
                  key={slot}
                  className="chip"
                  disabled={busy}
                  onClick={() => onDecide(`make it ${slot} instead`)}
                >
                  {slot}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="actions">
        <button className="primary" disabled={busy} onClick={() => onDecide(true)}>
          Approve
        </button>
        <button className="danger" disabled={busy} onClick={() => onDecide(false)}>
          Decline
        </button>
      </div>

      <form
        className="counter"
        onSubmit={(event) => {
          event.preventDefault();
          if (reason.trim()) onDecide(reason.trim());
        }}
      >
        <input
          value={reason}
          disabled={busy}
          placeholder="Or say what to change, e.g. make it after lunch"
          onChange={(event) => setReason(event.target.value)}
        />
        <button disabled={busy || !reason.trim()}>Send</button>
      </form>
    </div>
  );
}

function Result({ step }: { step: RunStep }) {
  const final = step.final;
  if (!final) return null;

  return (
    <div className="card">
      <h3>Result</h3>
      <p>{final.summary}</p>

      <div className="result-grid">
        <div>
          <h4>Booked</h4>
          {final.events_created.length === 0 ? (
            <p className="muted">Nothing was written to the calendar</p>
          ) : (
            <ul>
              {final.events_created.map((title) => (
                <li key={title}>{title}</li>
              ))}
            </ul>
          )}
        </div>

        <div>
          <h4>Tasks found</h4>
          {final.tasks_found.length === 0 ? (
            <p className="muted">None</p>
          ) : (
            <ul>
              {final.tasks_found.map((title) => (
                <li key={title}>{title}</li>
              ))}
            </ul>
          )}
        </div>
      </div>

      {final.unresolved.length > 0 && (
        <div className="unresolved">
          <h4>Not done</h4>
          <ul>
            {final.unresolved.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

function ToolLog({ step }: { step: RunStep }) {
  if (step.tool_log.length === 0) return null;

  return (
    <details className="card log">
      <summary>
        {step.tool_log.length} tool calls, {step.turns} turns
      </summary>
      <ol>
        {step.tool_log.map((entry, index) => (
          <li key={index} className={entry.ok ? "" : "failed"}>
            <code>{entry.tool}</code>
            <span>{entry.ok ? JSON.stringify(entry.output) : entry.error}</span>
          </li>
        ))}
      </ol>
      <p className="muted path">{step.steps.join(" -> ")}</p>
    </details>
  );
}

export default function App() {
  const [config, setConfig] = useState<Config | null>(null);
  const [tasks, setTasks] = useState<Task[]>([]);
  const [instruction, setInstruction] = useState(EXAMPLES[0]);
  const [step, setStep] = useState<RunStep | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getConfig()
      .then(setConfig)
      .catch((e) => setError(e.message));
    getTasks()
      .then(setTasks)
      .catch(() => undefined);
  }, []);

  async function guard(work: () => Promise<RunStep>) {
    setBusy(true);
    setError(null);
    try {
      setStep(await work());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  const live = config?.calendar === "google";

  return (
    <div className="page">
      <header>
        <div>
          <h1>Task Scheduling Agent</h1>
          <p className="muted">
            Searches your tasks, checks real availability, and asks before it writes.
          </p>
        </div>
        {config && (
          <div className="badges">
            <span className={live ? "badge live" : "badge"}>
              {live ? "live calendar" : "mock calendar"}
            </span>
            <span className="badge">{config.model}</span>
            <span className="badge">today {config.today}</span>
            <span className="badge">{config.business_hours}</span>
          </div>
        )}
      </header>

      {live && <p className="banner">Approvals write real events to your Google Calendar.</p>}

      <div className="columns">
        <aside>
          <h2>Tasks</h2>
          <ul className="tasks">
            {tasks.map((task) => (
              <li key={task.id} className={task.status}>
                <span>{task.title}</span>
                <span className="meta">
                  {task.priority} - {task.status}
                </span>
              </li>
            ))}
          </ul>
        </aside>

        <main>
          <form
            className="card"
            onSubmit={(event) => {
              event.preventDefault();
              if (instruction.trim()) guard(() => startRun(instruction.trim()));
            }}
          >
            <label htmlFor="instruction">What should it do?</label>
            <textarea
              id="instruction"
              rows={3}
              value={instruction}
              disabled={busy}
              onChange={(event) => setInstruction(event.target.value)}
            />
            <div className="chips">
              {EXAMPLES.map((example) => (
                <button
                  key={example}
                  type="button"
                  className="chip"
                  disabled={busy}
                  onClick={() => setInstruction(example)}
                >
                  {example.slice(0, 38)}...
                </button>
              ))}
            </div>
            <button className="primary" disabled={busy || !instruction.trim()}>
              {busy ? "Working..." : "Run"}
            </button>
          </form>

          {busy && <p className="muted">Thinking. Model calls take a few seconds each.</p>}

          {error && <p className="card error">{error}</p>}

          {step?.status === "paused" &&
            step.approval?.proposals.map((proposal, index) => (
              <ProposalCard
                key={index}
                proposal={proposal}
                busy={busy}
                onDecide={(decision) => guard(() => sendDecision(step.thread_id, decision))}
              />
            ))}

          {step?.status === "done" && <Result step={step} />}

          {step && <ToolLog step={step} />}
        </main>
      </div>
    </div>
  );
}
