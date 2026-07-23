/**
 * Leaderboard API client.
 *
 * Requests go to a same-origin /api path — the Vite dev server proxies it to
 * the Python server (see vite.config.ts), so there is no base URL to configure.
 *
 * Note what is NOT sent for scoring: the browser's own accuracy. The server
 * rebuilds the split from the seed and retrains its own model, and that score
 * is the only one ranked. We pass ours along purely so the two can be compared.
 */

export interface Student {
  student_id: number;
  name: string;
  entry_number: string;
  returning?: boolean;
  stats?: {
    total_runs: number;
    algorithms_tried: number;
    best_macro_f1: number | null;
  };
}

export interface OfficialScore {
  accuracy: number;
  macro_f1: number;
  train_accuracy: number;
  n_train: number;
  n_test: number;
}

export interface SubmitResult {
  run_id: number;
  student: Student;
  algorithm: string;
  algorithm_label: string;
  hyperparams: Record<string, number | string>;
  summary: string;
  seed: number;
  official: OfficialScore;
  client: { accuracy: number | null; macro_f1: number | null };
  /** null when this run is not the student's best — an earlier run still holds their slot. */
  position: number | null;
  total: number;
}

export interface BoardRow {
  position: number;
  run_id: number;
  student_id: number;
  name: string;
  entry_number: string;
  algorithm: string;
  summary: string;
  seed: number;
  accuracy: number;
  macro_f1: number;
  train_accuracy: number | null;
  timestamp: number;
}

export class ApiError extends Error {}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`/api${path}`, {
      headers: { "Content-Type": "application/json" },
      ...init,
    });
  } catch {
    throw new ApiError("Could not reach the leaderboard server. Is it running?");
  }

  let body: unknown = null;
  try {
    body = await res.json();
  } catch {
    // fall through — an empty/non-JSON body is handled below
  }

  if (!res.ok) {
    const detail =
      body && typeof body === "object" && "detail" in body
        ? String((body as { detail: unknown }).detail)
        : `Request failed (${res.status}).`;
    throw new ApiError(detail);
  }
  return body as T;
}

export function register(name: string, entryNumber: string): Promise<Student> {
  return request<Student>("/register", {
    method: "POST",
    body: JSON.stringify({ name, entry_number: entryNumber }),
  });
}

export function submitRun(payload: {
  studentId: number;
  algorithm: string;
  seed: number;
  hyperparams: Record<string, number | string>;
  clientAccuracy: number;
  clientMacroF1: number;
}): Promise<SubmitResult> {
  return request<SubmitResult>("/submit", {
    method: "POST",
    body: JSON.stringify({
      student_id: payload.studentId,
      algorithm: payload.algorithm,
      seed: payload.seed,
      hyperparams: payload.hyperparams,
      client_accuracy: payload.clientAccuracy,
      client_macro_f1: payload.clientMacroF1,
    }),
  });
}

export function fetchLeaderboard(algorithm?: string): Promise<BoardRow[]> {
  const query = algorithm ? `?algorithm=${encodeURIComponent(algorithm)}` : "";
  return request<BoardRow[]>(`/leaderboard${query}`);
}

// --- local persistence of the signed-in student -----------------------------

const STUDENT_KEY = "sylva:student";

export function cacheStudent(student: Student) {
  try {
    localStorage.setItem(STUDENT_KEY, JSON.stringify(student));
  } catch {
    // private mode / quota — the session still works, it just won't persist
  }
}

export function readCachedStudent(): Student | null {
  try {
    const raw = localStorage.getItem(STUDENT_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed.student_id === "number" ? (parsed as Student) : null;
  } catch {
    return null;
  }
}

export function clearCachedStudent() {
  try {
    localStorage.removeItem(STUDENT_KEY);
  } catch {
    // nothing to do
  }
}
