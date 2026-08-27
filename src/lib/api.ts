const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

/** Thin fetch wrapper for calling paaral-student-api. Throws on both
 * network failure and a non-2xx response, so callers can wrap a single
 * try/catch rather than checking `res.ok` themselves at every call site. */
export async function apiPost<TResponse>(
  path: string,
  body: unknown
): Promise<TResponse> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`API request to ${path} failed with ${res.status}`);
  }
  return res.json() as Promise<TResponse>;
}
