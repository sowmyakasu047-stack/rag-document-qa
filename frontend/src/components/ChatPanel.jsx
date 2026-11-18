import { useEffect, useRef, useState } from "react";

import Message from "./Message.jsx";

const MAX_QUESTION_CHARS = 2000;

export default function ChatPanel({ messages, asking, hasDocuments, llmNotice, onAsk }) {
  const [draft, setDraft] = useState("");
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView?.({ block: "end", behavior: "smooth" });
  }, [messages, asking]);

  const question = draft.trim();
  const canSend = hasDocuments && !asking && question.length > 0;

  function submit(event) {
    event.preventDefault();
    if (!canSend) return;
    setDraft("");
    onAsk(question);
  }

  function handleKeyDown(event) {
    // Enter sends; Shift+Enter inserts a new line.
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      submit(event);
    }
  }

  return (
    <section className="panel chat" aria-labelledby="chat-heading">
      <h2 id="chat-heading">Ask a question</h2>

      {llmNotice && (
        <p className="banner banner-warning" role="note">
          Answers are unavailable: {llmNotice}
        </p>
      )}

      <div className="messages" role="log" aria-live="polite" aria-label="Conversation">
        {messages.length === 0 && (
          <p className="empty">
            {hasDocuments
              ? "Ask something about your documents, for example “What are the main conclusions?”"
              : "Upload a PDF on the left, then ask a question about it."}
          </p>
        )}
        {messages.map((message) => (
          <Message key={message.id} message={message} />
        ))}
        {asking && (
          <p className="thinking" role="status">
            Searching your documents…
          </p>
        )}
        <div ref={endRef} />
      </div>

      <form className="composer" onSubmit={submit}>
        <label htmlFor="question" className="visually-hidden">
          Your question
        </label>
        <textarea
          id="question"
          rows={2}
          value={draft}
          maxLength={MAX_QUESTION_CHARS}
          placeholder={hasDocuments ? "Type your question…" : "Upload a document first"}
          disabled={!hasDocuments}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={handleKeyDown}
        />
        <button type="submit" disabled={!canSend}>
          {asking ? "Asking…" : "Ask"}
        </button>
      </form>
    </section>
  );
}
