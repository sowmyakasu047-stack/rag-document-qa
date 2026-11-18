// The model cites excerpts as [1], [2] or [1, 2]. This turns an answer string into
// segments so the interface can render each citation as a link to its source.

const CITATION = /\[(\d+(?:\s*,\s*\d+)*)\]/g;

/**
 * @param {string} text
 * @returns {({type: "text", value: string} | {type: "cite", numbers: number[], raw: string})[]}
 */
export function splitCitations(text) {
  const segments = [];
  let last = 0;
  for (const match of text.matchAll(CITATION)) {
    if (match.index > last) {
      segments.push({ type: "text", value: text.slice(last, match.index) });
    }
    segments.push({
      type: "cite",
      numbers: match[1].split(",").map((part) => Number(part.trim())),
      raw: match[0],
    });
    last = match.index + match[0].length;
  }
  if (last < text.length) {
    segments.push({ type: "text", value: text.slice(last) });
  }
  return segments;
}
