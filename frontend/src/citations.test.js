import assert from "node:assert/strict";
import { test } from "node:test";

import { splitCitations } from "./citations.js";

test("text without citations is a single text segment", () => {
  assert.deepEqual(splitCitations("Plain answer."), [{ type: "text", value: "Plain answer." }]);
});

test("a citation splits the text into text, cite, text", () => {
  assert.deepEqual(splitCitations("Fans [1] sit."), [
    { type: "text", value: "Fans " },
    { type: "cite", numbers: [1], raw: "[1]" },
    { type: "text", value: " sit." },
  ]);
});

test("grouped citations keep every number", () => {
  const [segment] = splitCitations("[1, 2,3]");
  assert.deepEqual(segment, { type: "cite", numbers: [1, 2, 3], raw: "[1, 2,3]" });
});

test("adjacent citations are separate segments", () => {
  const segments = splitCitations("Claim [1][2].");
  assert.deepEqual(
    segments.filter((s) => s.type === "cite").map((s) => s.numbers),
    [[1], [2]],
  );
  assert.equal(segments.at(-1).value, ".");
});

test("brackets that are not numbers are left as text", () => {
  assert.deepEqual(splitCitations("See [note] and [a, b]."), [
    { type: "text", value: "See [note] and [a, b]." },
  ]);
});

test("empty text gives no segments", () => {
  assert.deepEqual(splitCitations(""), []);
});
