// Calls to the Python server. Every function either returns data or throws an
// ApiError whose `message` is safe to show to the user.

export class ApiError extends Error {
  constructor(message, { code = "unknown", status = 0 } = {}) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.status = status;
  }
}

async function request(path, options = {}) {
  let response;
  try {
    response = await fetch(path, options);
  } catch {
    throw new ApiError("Could not reach the server. Is it running?", { code: "network" });
  }

  if (response.ok) {
    return response.status === 204 ? null : response.json();
  }

  // The server reports errors as {"error": {"code": ..., "message": ...}}. Anything
  // else (a proxy error page, for example) falls back to a generic message.
  let message = `Request failed (${response.status}).`;
  let code = "http_error";
  try {
    const body = await response.json();
    if (body && body.error) {
      message = body.error.message || message;
      code = body.error.code || code;
    }
  } catch {
    // not JSON; keep the generic message
  }
  throw new ApiError(message, { code, status: response.status });
}

export const getHealth = () => request("/api/health");

export async function listDocuments() {
  const body = await request("/api/documents");
  return body.documents;
}

export function uploadDocument(file) {
  const form = new FormData();
  form.append("file", file);
  return request("/api/documents", { method: "POST", body: form });
}

export const deleteDocument = (name) =>
  request(`/api/documents/${encodeURIComponent(name)}`, { method: "DELETE" });

export function askQuestion(question, k) {
  const payload = k === undefined ? { question } : { question, k };
  return request("/api/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}
