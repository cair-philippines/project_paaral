const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function request<TResponse>(
  path: string,
  init?: RequestInit
): Promise<TResponse> {
  const res = await fetch(`${API_BASE_URL}${path}`, init);
  if (!res.ok) {
    throw new Error(`API request to ${path} failed with ${res.status}`);
  }
  return res.json() as Promise<TResponse>;
}

/** Thin fetch wrapper for calling paaral-student-api. Throws on both
 * network failure and a non-2xx response, so callers can wrap a single
 * try/catch rather than checking `res.ok` themselves at every call site. */
export function apiPost<TResponse>(
  path: string,
  body: unknown
): Promise<TResponse> {
  return request<TResponse>(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** Same contract as `apiPost`, for read-only GET calls. */
export function apiGet<TResponse>(path: string): Promise<TResponse> {
  return request<TResponse>(path);
}

/** Same contract as `apiPost`, for whole-resource-replace calls. */
export function apiPut<TResponse>(
  path: string,
  body: unknown
): Promise<TResponse> {
  return request<TResponse>(path, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}

/** Same contract as `apiPost`, for partial-update calls. */
export function apiPatch<TResponse>(
  path: string,
  body: unknown
): Promise<TResponse> {
  return request<TResponse>(path, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
}
