import { splitCitations } from "../citations.js";

function openSource(event, domId) {
  event.preventDefault();
  const item = document.getElementById(domId);
  if (!item) return;
  item.querySelector("details")?.setAttribute("open", "");
  item.scrollIntoView({ block: "nearest", behavior: "smooth" });
  item.focus({ preventScroll: true });
}

function AnswerText({ text, sources, messageId }) {
  const known = new Set(sources.map((source) => source.number));
  return (
    <p className="answer-text">
      {splitCitations(text).map((segment, index) => {
        if (segment.type === "text") return segment.value;
        // Only numbers that match a returned source become links; anything else stays plain text.
        if (!segment.numbers.every((number) => known.has(number))) return segment.raw;
        return segment.numbers.map((number) => (
          <a
            key={`${index}-${number}`}
            className="cite"
            href={`#source-${messageId}-${number}`}
            onClick={(event) => openSource(event, `source-${messageId}-${number}`)}
            aria-label={`Go to source ${number}`}
          >
            {number}
          </a>
        ));
      })}
    </p>
  );
}

function Sources({ sources, messageId }) {
  return (
    <div className="sources">
      <h3>Sources</h3>
      <ol>
        {sources.map((source) => (
          <li key={source.number} id={`source-${messageId}-${source.number}`} tabIndex={-1}>
            <details>
              <summary>
                <span className="source-ref">{source.number}</span>
                <span className="source-name">{source.source}</span>
                <span className="source-page">page {source.page}</span>
                <span className="source-score" title="Similarity between the question and this passage">
                  relevance {source.score.toFixed(2)}
                </span>
              </summary>
              <blockquote>{source.text}</blockquote>
            </details>
          </li>
        ))}
      </ol>
    </div>
  );
}

export default function Message({ message }) {
  if (message.role === "user") {
    return (
      <article className="message message-user" aria-label="You asked">
        <p>{message.text}</p>
      </article>
    );
  }

  if (message.error) {
    return (
      <article className="message message-error" role="alert" aria-label="Error">
        <p>{message.text}</p>
      </article>
    );
  }

  return (
    <article className="message message-assistant" aria-label="Answer">
      <AnswerText text={message.text} sources={message.sources} messageId={message.id} />
      {!message.answered && (
        <p className="note">Nothing in your documents supports an answer to this question.</p>
      )}
      {message.warnings.map((warning) => (
        <p key={warning} className="note note-warning" role="note">
          {warning}
        </p>
      ))}
      {message.sources.length > 0 && <Sources sources={message.sources} messageId={message.id} />}
    </article>
  );
}
