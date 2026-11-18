import { useId, useState } from "react";

function plural(count, word) {
  return `${count} ${word}${count === 1 ? "" : "s"}`;
}

export default function DocumentsPanel({ documents, status, maxMb, onFiles, onRemove }) {
  const inputId = useId();
  const [dragging, setDragging] = useState(false);
  const busy = status.kind === "busy";

  function handleChange(event) {
    onFiles(event.target.files);
    event.target.value = ""; // allow choosing the same file again
  }

  function handleDrop(event) {
    event.preventDefault();
    setDragging(false);
    if (!busy) onFiles(event.dataTransfer.files);
  }

  return (
    <section className="panel documents" aria-labelledby="documents-heading">
      <h2 id="documents-heading">Documents</h2>

      <label
        htmlFor={inputId}
        className={`dropzone${dragging ? " dropzone-active" : ""}${busy ? " dropzone-busy" : ""}`}
        onDragOver={(event) => {
          event.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
      >
        <span className="dropzone-title">{busy ? "Working…" : "Choose PDFs or drop them here"}</span>
        <span className="dropzone-hint">Up to {maxMb} MB each. Re-uploading a name replaces it.</span>
        <input
          id={inputId}
          type="file"
          accept=".pdf,application/pdf"
          multiple
          disabled={busy}
          onChange={handleChange}
        />
      </label>

      <p
        className={`status status-${status.kind}`}
        role={status.kind === "error" ? "alert" : "status"}
        aria-live="polite"
      >
        {status.message}
      </p>

      {documents.length === 0 ? (
        <p className="empty">No documents yet.</p>
      ) : (
        <ul className="doc-list">
          {documents.map((doc) => (
            <li key={doc.name} className="doc">
              <div className="doc-info">
                <span className="doc-name" title={doc.name}>
                  {doc.name}
                </span>
                <span className="doc-meta">
                  {plural(doc.pages, "page")} · {plural(doc.chunks, "passage")}
                </span>
              </div>
              <button
                type="button"
                className="link-button"
                onClick={() => onRemove(doc.name)}
                disabled={busy}
                aria-label={`Remove ${doc.name}`}
              >
                Remove
              </button>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
