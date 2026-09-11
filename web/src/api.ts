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

// Where a task ended up on the calendar. Written by the agent, never by the form.
export type Booking = {
  date: string;
  start_time: string;
  duration_minutes: number;
};

export type Task = {
  id: string;
  title: string;
  status: Status;
  priority: Priority;
  tags: string[];
  booked: Booking | null;
};

// Creating sends the editable fields only. The server assigns the id, and the
// agent owns `booked`.
export type TaskDraft = Omit<Task, "id" | "booked">;

// Patching sends only the fields that changed, so an omitted field keeps its value.
export type TaskPatch = Partial<TaskDraft>;

export type BookedSlot = {
  title: string;
  from: string;
  to: string;
  // True when the proposed slot would run into this entry.
  clashes: boolean;
};

export type Proposal = {
  tool: string;
  input: {
    title?: string;
    date?: string;
    start_time?: string;
    duration_minutes?: number;
  };
  already_booked: BookedSlot[];
  // Titles the proposed slot would run into. Empty when the slot is free.
  clashes_with: string[];
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

// What the run spent. Tool Efficiency counts calls; this is what the calls cost.
export type Usage = {
  calls: number;
  input_tokens: number;
  output_tokens: number;
  seconds: number;
};

export type RunStep = {
  thread_id: string;
  status: "paused" | "done";
  approval: Approval | null;
  final: Final | null;
  steps: string[];
  tool_log: ToolCall[];
  turns: number;
  usage: Usage;
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

export type Resync = { cleared: string[]; adopted: string[]; tasks: Task[] };

// Reconciles the task list against the calendar both ways: clears a booking whose
// event has gone, and adopts an event whose title matches an unticked task.
// Never creates, moves or deletes an event.
export const resyncTasks = () => request<Resync>("/api/tasks/resync", { method: "POST" });

export const createTask = (draft: TaskDraft) =>
  request<Task>("/api/tasks", { method: "POST", body: JSON.stringify(draft) });

export const patchTask = (id: string, patch: TaskPatch) =>
  request<Task>(`/api/tasks/${id}`, { method: "PATCH", body: JSON.stringify(patch) });

export const removeTask = async (id: string) => {
  const response = await fetch(`/api/tasks/${id}`, { method: "DELETE" });
  // 204 has no body, so this one cannot go through request().
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
};

// One node of the graph finished. Arrives while the run is still going.
export type Progress = {
  type: "step";
  node: string;
  label: string;
  turns: number;
  tools: string[];
};

type StreamEvent = Progress | (RunStep & { type?: undefined }) | { type: "error"; detail: string };

/** POST and read back server-sent events, calling onProgress until the result arrives.
 *
 *  POST rather than EventSource: the instruction is the user's own words and
 *  should not travel in a URL, where it lands in logs and browser history.
 */
async function streamRun(
  path: string,
  payload: unknown,
  onProgress: (step: Progress) => void,
): Promise<RunStep> {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  if (!response.ok || !response.body) {
    const body = await response.json().catch(() => null);
    throw new Error(body?.detail ?? `${response.status} ${response.statusText}`);
  }

  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  let result: RunStep | null = null;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += value;

    // Messages are separated by a blank line, and a chunk can split one in half.
    const messages = buffer.split("\n\n");
    buffer = messages.pop() ?? "";

    for (const message of messages) {
      const line = message.split("\n").find((part) => part.startsWith("data: "));
      if (!line) continue;
      const event = JSON.parse(line.slice(6)) as StreamEvent;

      if (event.type === "step") onProgress(event);
      else if (event.type === "error") throw new Error(event.detail);
      else result = event as RunStep;
    }
  }

  if (!result) throw new Error("The run ended without a result.");
  return result;
}

export const startRunStreaming = (instruction: string, onProgress: (step: Progress) => void) =>
  streamRun("/api/runs/stream", { instruction }, onProgress);

export const sendDecisionStreaming = (
  threadId: string,
  decision: Decision,
  onProgress: (step: Progress) => void,
) => streamRun(`/api/runs/${threadId}/decision/stream`, { decision }, onProgress);

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
