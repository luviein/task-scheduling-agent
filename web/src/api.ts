// Types mirror the Pydantic models in session.py and api.py. Both ends are typed;
// this file is the seam where that has to be restated by hand.

export type Config = {
  model: string;
  calendar: string;
  today: string;
  business_hours: string;
};

export type Status = "open" | "done";
export type Priority = "low" | "medium" | "high";

export type Task = {
  id: string;
  title: string;
  status: Status;
  priority: Priority;
  tags: string[];
};

// Creating sends the whole task minus the id, which the server assigns.
export type TaskDraft = Omit<Task, "id">;

// Patching sends only the fields that changed, so an omitted field keeps its value.
export type TaskPatch = Partial<TaskDraft>;

export type BookedSlot = { title: string; from: string; to: string };

export type Proposal = {
  tool: string;
  input: {
    title?: string;
    date?: string;
    start_time?: string;
    duration_minutes?: number;
  };
  already_booked: BookedSlot[];
  other_options: string[];
};

export type Approval = { question: string; proposals: Proposal[] };

export type Final = {
  summary: string;
  tasks_found: string[];
  events_created: string[];
  unresolved: string[];
};

export type ToolCall = {
  tool: string;
  ok: boolean;
  output: Record<string, unknown> | null;
  error: string | null;
};

export type RunStep = {
  thread_id: string;
  status: "paused" | "done";
  approval: Approval | null;
  final: Final | null;
  steps: string[];
  tool_log: ToolCall[];
  turns: number;
};

// `decision` is the same three-way answer the terminal approver takes:
// true approves, false is a hard no, a string is a counter-proposal.
export type Decision = boolean | string;

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!response.ok) {
    // FastAPI puts the readable message in `detail`, which is what the quota
    // and unknown-session errors rely on.
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? `${response.status} ${response.statusText}`);
  }
  return response.json() as Promise<T>;
}

export const getConfig = () => request<Config>("/api/config");

export const getTasks = () => request<Task[]>("/api/tasks");

export const createTask = (draft: TaskDraft) =>
  request<Task>("/api/tasks", { method: "POST", body: JSON.stringify(draft) });

export const patchTask = (id: string, patch: TaskPatch) =>
  request<Task>(`/api/tasks/${id}`, { method: "PATCH", body: JSON.stringify(patch) });

export const removeTask = async (id: string) => {
  const response = await fetch(`/api/tasks/${id}`, { method: "DELETE" });
  // 204 has no body, so this one cannot go through request().
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
};

export const startRun = (instruction: string) =>
  request<RunStep>("/api/runs", {
    method: "POST",
    body: JSON.stringify({ instruction }),
  });

export const sendDecision = (threadId: string, decision: Decision) =>
  request<RunStep>(`/api/runs/${threadId}/decision`, {
    method: "POST",
    body: JSON.stringify({ decision }),
  });
