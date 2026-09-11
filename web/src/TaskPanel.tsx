import { useState } from "react";
import {
  createTask,
  patchTask,
  removeTask,
  type Priority,
  type Status,
  type Task,
  type TaskDraft,
} from "./api";

const PRIORITIES: Priority[] = ["low", "medium", "high"];

const EMPTY: TaskDraft = { title: "", status: "open", priority: "medium", tags: [] };

/** One form, used both for adding a task and for editing an existing one. */
function TaskForm({
  initial,
  submitLabel,
  onSubmit,
  onCancel,
  busy,
}: {
  initial: TaskDraft;
  submitLabel: string;
  onSubmit: (draft: TaskDraft) => void;
  onCancel: () => void;
  busy: boolean;
}) {
  const [title, setTitle] = useState(initial.title);
  const [priority, setPriority] = useState<Priority>(initial.priority);
  const [tags, setTags] = useState(initial.tags.join(", "));

  return (
    <form
      className="task-form"
      onSubmit={(event) => {
        event.preventDefault();
        if (!title.trim()) return;
        onSubmit({
          title: title.trim(),
          status: initial.status,
          priority,
          tags: tags
            .split(",")
            .map((tag) => tag.trim().toLowerCase())
            .filter(Boolean),
        });
      }}
    >
      <input
        autoFocus
        value={title}
        disabled={busy}
        placeholder="Task title"
        onChange={(event) => setTitle(event.target.value)}
      />
      <select
        value={priority}
        disabled={busy}
        onChange={(event) => setPriority(event.target.value as Priority)}
      >
        {PRIORITIES.map((level) => (
          <option key={level} value={level}>
            {level}
          </option>
        ))}
      </select>
      <input
        value={tags}
        disabled={busy}
        placeholder="tags, comma separated"
        onChange={(event) => setTags(event.target.value)}
      />
      <div className="task-form-actions">
        <button className="primary" disabled={busy || !title.trim()}>
          {submitLabel}
        </button>
        <button type="button" disabled={busy} onClick={onCancel}>
          Cancel
        </button>
      </div>
    </form>
  );
}

function TaskRow({
  task,
  onChanged,
  onError,
}: {
  task: Task;
  onChanged: () => void;
  onError: (message: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  // Two-step delete rather than a browser confirm dialog: the second click on
  // the same button is the confirmation.
  const [confirming, setConfirming] = useState(false);
  const [busy, setBusy] = useState(false);

  async function run(work: () => Promise<unknown>) {
    setBusy(true);
    try {
      await work();
      onChanged();
    } catch (e) {
      onError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  if (editing) {
    return (
      <li className="task editing">
        <TaskForm
          initial={task}
          submitLabel="Save"
          busy={busy}
          onCancel={() => setEditing(false)}
          onSubmit={(draft) =>
            run(async () => {
              await patchTask(task.id, draft);
              setEditing(false);
            })
          }
        />
      </li>
    );
  }

  const done = task.status === "done";

  return (
    <li className={done ? "task done" : "task"}>
      <label className="task-check">
        <input
          type="checkbox"
          checked={done}
          disabled={busy}
          onChange={() =>
            run(() => patchTask(task.id, { status: (done ? "open" : "done") as Status }))
          }
        />
        <span className="task-title">{task.title}</span>
      </label>

      <div className="task-meta">
        <span className={`pill ${task.priority}`}>{task.priority}</span>
        {task.tags.map((tag) => (
          <span key={tag} className="pill tag">
            {tag}
          </span>
        ))}
      </div>

      <div className="task-actions">
        <button className="link" disabled={busy} onClick={() => setEditing(true)}>
          Edit
        </button>
        {confirming ? (
          <>
            <button
              className="link danger"
              disabled={busy}
              onClick={() => run(() => removeTask(task.id))}
            >
              Really delete
            </button>
            <button className="link" disabled={busy} onClick={() => setConfirming(false)}>
              Keep
            </button>
          </>
        ) : (
          <button className="link" disabled={busy} onClick={() => setConfirming(true)}>
            Delete
          </button>
        )}
      </div>
    </li>
  );
}

export default function TaskPanel({
  tasks,
  onChanged,
}: {
  tasks: Task[];
  onChanged: () => void;
}) {
  const [adding, setAdding] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function add(draft: TaskDraft) {
    setBusy(true);
    setError(null);
    try {
      await createTask(draft);
      setAdding(false);
      onChanged();
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <aside>
      <div className="panel-head">
        <h2>Tasks</h2>
        {!adding && (
          <button className="link" onClick={() => setAdding(true)}>
            Add
          </button>
        )}
      </div>

      <p className="muted small">
        The list the agent searches. Edits are saved and survive a restart.
      </p>

      {error && <p className="card error small">{error}</p>}

      <ul className="tasks">
        {adding && (
          <li className="task editing">
            <TaskForm
              initial={EMPTY}
              submitLabel="Add"
              busy={busy}
              onCancel={() => setAdding(false)}
              onSubmit={add}
            />
          </li>
        )}
        {tasks.map((task) => (
          <TaskRow key={task.id} task={task} onChanged={onChanged} onError={setError} />
        ))}
      </ul>
    </aside>
  );
}
