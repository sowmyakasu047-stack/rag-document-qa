import { useCallback, useEffect, useRef, useState } from "react";

import { askQuestion, deleteDocument, getHealth, listDocuments, uploadDocument } from "./api.js";
import ChatPanel from "./components/ChatPanel.jsx";
import DocumentsPanel from "./components/DocumentsPanel.jsx";

const DEFAULT_MAX_UPLOAD_MB = 20;

export default function App() {
  const [documents, setDocuments] = useState([]);
  const [health, setHealth] = useState(null);
  const [startupError, setStartupError] = useState(null);
  const [uploadStatus, setUploadStatus] = useState({ kind: "idle", message: "" });
  const [messages, setMessages] = useState([]);
  const [asking, setAsking] = useState(false);
  const nextId = useRef(1);

  const refreshDocuments = useCallback(async () => {
    setDocuments(await listDocuments());
  }, []);

  useEffect(() => {
    Promise.all([getHealth(), listDocuments()])
      .then(([healthInfo, docs]) => {
        setHealth(healthInfo);
        setDocuments(docs);
      })
      .catch((error) => setStartupError(error.message));
  }, []);

  const maxMb = health?.max_upload_mb ?? DEFAULT_MAX_UPLOAD_MB;

  async function handleFiles(fileList) {
    const files = Array.from(fileList);
    if (files.length === 0) return;

    const problems = [];
    let indexed = 0;
    for (const [position, file] of files.entries()) {
      setUploadStatus({
        kind: "busy",
        message: `Indexing ${file.name} (${position + 1} of ${files.length})…`,
      });
      if (!file.name.toLowerCase().endsWith(".pdf")) {
        problems.push(`${file.name}: only PDF files can be uploaded.`);
        continue;
      }
      if (file.size > maxMb * 1024 * 1024) {
        problems.push(`${file.name}: larger than the ${maxMb} MB limit.`);
        continue;
      }
      try {
        await uploadDocument(file);
        indexed += 1;
      } catch (error) {
        problems.push(`${file.name}: ${error.message}`);
      }
    }

    try {
      await refreshDocuments();
    } catch (error) {
      problems.push(error.message);
    }

    if (problems.length > 0) {
      setUploadStatus({ kind: "error", message: problems.join(" ") });
    } else {
      setUploadStatus({
        kind: "ok",
        message: indexed === 1 ? "Document added." : `${indexed} documents added.`,
      });
    }
  }

  async function handleRemove(name) {
    try {
      await deleteDocument(name);
      await refreshDocuments();
      setUploadStatus({ kind: "ok", message: `Removed ${name}.` });
    } catch (error) {
      setUploadStatus({ kind: "error", message: error.message });
    }
  }

  async function handleAsk(question) {
    const userId = nextId.current++;
    setMessages((current) => [...current, { id: userId, role: "user", text: question }]);
    setAsking(true);
    const replyId = nextId.current++;
    try {
      const result = await askQuestion(question);
      setMessages((current) => [
        ...current,
        {
          id: replyId,
          role: "assistant",
          text: result.answer,
          answered: result.answered,
          sources: result.sources,
          warnings: result.warnings,
        },
      ]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        { id: replyId, role: "assistant", text: error.message, error: true },
      ]);
    } finally {
      setAsking(false);
    }
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1>Document Q&amp;A</h1>
        <p>Ask questions about your PDFs. Every answer shows the page it came from.</p>
      </header>

      {startupError && (
        <p className="banner banner-error" role="alert">
          {startupError}
        </p>
      )}

      <main className="layout">
        <DocumentsPanel
          documents={documents}
          status={uploadStatus}
          maxMb={maxMb}
          onFiles={handleFiles}
          onRemove={handleRemove}
        />
        <ChatPanel
          messages={messages}
          asking={asking}
          hasDocuments={documents.length > 0}
          llmNotice={health && !health.llm_ready ? health.llm_message : null}
          onAsk={handleAsk}
        />
      </main>
    </div>
  );
}
