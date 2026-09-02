const API_BASE_URL =
  process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

async function request<TResponse>(
  path: string,
  init?: RequestInit
): Promise<TResponse> {
  const res = await fetch(`${API_BASE_URL}${path}`, init);
  if (!res.ok) {
    // FastAPI's HTTPException bodies look like {"detail": "..."} with a
    // plain-language message (e.g. document-upload validation errors) -
    // surface that to the caller instead of a generic status-code string,
    // so the UI can show it directly rather than a technical fallback.
    const detail = await res
      .json()
      .then((body: { detail?: string }) => body.detail)
      .catch(() => undefined);
    throw new Error(
      detail ?? `API request to ${path} failed with ${res.status}`
    );
  }
  if (res.status === 204) return undefined as TResponse;
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

/** Same contract as `apiPost`, for deleting a resource - no body. */
export function apiDelete<TResponse = void>(path: string): Promise<TResponse> {
  return request<TResponse>(path, { method: "DELETE" });
}

/** Uploads a file as `multipart/form-data` (document uploads, Chunk 18).
 * No `Content-Type` header is set here - the browser fills in the
 * multipart boundary itself when the body is a `FormData` instance;
 * setting it manually would omit the boundary and break parsing. */
export function apiUpload<TResponse>(
  path: string,
  file: File
): Promise<TResponse> {
  const formData = new FormData();
  formData.append("file", file);
  return request<TResponse>(path, { method: "PUT", body: formData });
}
