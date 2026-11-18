import assert from "node:assert/strict";
import { afterEach, test } from "node:test";

import { ApiError, askQuestion, deleteDocument, getHealth, listDocuments, uploadDocument } from "./api.js";

const realFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = realFetch;
});

function mockFetch(status, body, { json = true } = {}) {
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    return {
      ok: status >= 200 && status < 300,
      status,
      json: async () => {
        if (!json) throw new SyntaxError("not json");
        return body;
      },
    };
  };
  return calls;
}

test("listDocuments returns the documents array", async () => {
  mockFetch(200, { documents: [{ name: "a.pdf", chunks: 2, pages: 1 }] });
  assert.deepEqual(await listDocuments(), [{ name: "a.pdf", chunks: 2, pages: 1 }]);
});

test("getHealth returns the health object", async () => {
  mockFetch(200, { status: "ok", llm_ready: true });
  assert.equal((await getHealth()).llm_ready, true);
});

test("uploadDocument posts the file as multipart form data under the 'file' field", async () => {
  const calls = mockFetch(201, { name: "a.pdf", chunks: 3, replaced: false });
  const file = new File(["%PDF"], "a.pdf", { type: "application/pdf" });

  const result = await uploadDocument(file);

  assert.equal(result.chunks, 3);
  assert.equal(calls[0].url, "/api/documents");
  assert.equal(calls[0].options.method, "POST");
  assert.ok(calls[0].options.body instanceof FormData);
  assert.equal(calls[0].options.body.get("file").name, "a.pdf");
});

test("deleteDocument encodes the file name and resolves on 204", async () => {
  const calls = mockFetch(204, null);
  assert.equal(await deleteDocument("my report #1.pdf"), null);
  assert.equal(calls[0].url, "/api/documents/my%20report%20%231.pdf");
  assert.equal(calls[0].options.method, "DELETE");
});

test("askQuestion sends JSON, and only includes k when given", async () => {
  const calls = mockFetch(200, { answer: "x", sources: [] });
  await askQuestion("Why?");
  await askQuestion("Why?", 6);
  assert.deepEqual(JSON.parse(calls[0].options.body), { question: "Why?" });
  assert.deepEqual(JSON.parse(calls[1].options.body), { question: "Why?", k: 6 });
  assert.equal(calls[0].options.headers["Content-Type"], "application/json");
});

test("server errors become ApiError with the server's message and code", async () => {
  mockFetch(422, { error: { code: "no_text", message: "It may be a scanned document." } });
  await assert.rejects(uploadDocument(new File(["x"], "s.pdf")), (error) => {
    assert.ok(error instanceof ApiError);
    assert.equal(error.message, "It may be a scanned document.");
    assert.equal(error.code, "no_text");
    assert.equal(error.status, 422);
    return true;
  });
});

test("a non-JSON error response falls back to a generic message", async () => {
  mockFetch(502, null, { json: false });
  await assert.rejects(listDocuments(), (error) => {
    assert.equal(error.message, "Request failed (502).");
    assert.equal(error.code, "http_error");
    return true;
  });
});

test("a network failure becomes a friendly ApiError", async () => {
  globalThis.fetch = async () => {
    throw new TypeError("fetch failed");
  };
  await assert.rejects(getHealth(), (error) => {
    assert.equal(error.code, "network");
    assert.match(error.message, /Could not reach the server/);
    return true;
  });
});
