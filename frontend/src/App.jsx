import { useState } from "react";

const API_BASE = "";

const STAGES = [
  { id: 1, label: "1 · Ingest", subtitle: "Read the corpus" },
  { id: 2, label: "2 · Chunk", subtitle: "Split into chunks" },
  { id: 3, label: "3 · Embed", subtitle: "Vectorize query" },
  { id: 4, label: "4 · Index", subtitle: "Search vector store" },
  { id: 5, label: "5 · Retrieve", subtitle: "Dense + BM25 (before rerank)" },
  { id: 6, label: "6 · Fuse", subtitle: "RRF merge" },
  { id: 7, label: "7 · Rerank", subtitle: "Cross-encoder (after rerank)" },
  { id: 8, label: "8 · Contextualize", subtitle: "Context preamble" },
  { id: 9, label: "9 · Generate", subtitle: "Answer with citations" },
  { id: 10, label: "10 · Evaluate", subtitle: "Faithfulness + relevance" },
];

const STATUS_COLORS = {
  ok: "#22c55e",
  skipped: "#f59e0b",
  error: "#ef4444",
  idle: "#94a3b8",
  running: "#3b82f6",
};

function escapeHtml(text) {
  if (!text) return "";
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function truncate(text, len = 200) {
  if (!text) return "";
  text = String(text).trim();
  if (text.length <= len) return text;
  return text.slice(0, len) + "…";
}

export default function App() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [response, setResponse] = useState(null);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;
    setLoading(true);
    setError(null);
    setResponse(null);
    try {
      const url = `${API_BASE}/query`;
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, top_k: 5 }),
      });
      if (!res.ok) {
        throw new Error(
          `Server error ${res.status} on ${url}. ` +
            `If the UI loaded but this fails: 1) hard-refresh (Ctrl+Shift+R) to clear a cached bundle, ` +
            `2) check Firefox proxy settings — add localhost,127.0.0.1 to "No proxy for", ` +
            `3) confirm the backend is running (python api/server.py).`
        );
      }
      const data = await res.json();
      setResponse(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const stageColor = (status) => STATUS_COLORS[status] || STATUS_COLORS.idle;

  return (
    <div style={styles.page}>
      <header style={styles.header}>
        <h1 style={styles.title}>IncidentLens</h1>
        <p style={styles.subtitle}>
          Production RAG pipeline — stage-by-stage view
        </p>
      </header>

      <form onSubmit={handleSubmit} style={styles.form}>
        <input
          type="text"
          placeholder="Ask an incident question…"
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          style={styles.input}
        />
        <button type="submit" disabled={loading} style={styles.button}>
          {loading ? "Running pipeline…" : "Run pipeline"}
        </button>
      </form>

      {error && (
        <div style={styles.errorBox}>
          ⚠️ {escapeHtml(error)}
          <br />
          Make sure the API server is running: <code>python api/server.py</code>
        </div>
      )}

      {response && (
        <section style={styles.results}>
          <div style={styles.diagnosis}>
            <strong>Diagnosis:</strong> {escapeHtml(response.diagnosis)}
          </div>

          {response.answer && (
            <div style={styles.answerCard}>
              <h3 style={styles.answerTitle}>Answer</h3>
              <div
                dangerouslySetInnerHTML={{
                  __html: escapeHtml(response.answer).replace(/\n/g, "<br />"),
                }}
              />
              {response.citations.length > 0 && (
                <div style={styles.citations}>
                  <strong>Sources:</strong> {response.citations.join(", ")}
                </div>
              )}
            </div>
          )}

          <div style={styles.pipeline}>
            <h3 style={styles.pipelineTitle}>Pipeline stages</h3>
            {STAGES.map((stage) => {
              const result = response.stages.find((s) => s.name === stage.label);
              const status = result ? result.status : "idle";
              return (
                <div key={stage.id} style={styles.stageCard(status)}>
                  <div style={styles.stageHeader}>
                    <span
                      style={{
                        ...styles.stageBadge,
                        backgroundColor: stageColor(status),
                      }}
                    >
                      {status === "running" ? "⟳" : status === "ok" ? "✓" : status === "skipped" ? "–" : "✕"}
                    </span>
                    <div>
                      <div style={styles.stageName}>{stage.label}</div>
                      <div style={styles.stageSub}>{stage.subtitle}</div>
                    </div>
                  </div>
                  {result && result.detail && (
                    <div style={styles.stageDetail}>
                      {escapeHtml(result.detail)}
                    </div>
                  )}
                  {result && result.data && (
                    <div style={styles.stageData}>
                      {renderStageData(stage.id, result.data)}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </section>
      )}
    </div>
  );
}

function renderStageData(stageId, data) {
  switch (stageId) {
    case 1:
      return (
        <ul style={styles.metaList}>
          <li>Documents: {data.documents}</li>
          <li>Organisations: {data.organisations}</li>
        </ul>
      );
    case 2:
      return (
        <ul style={styles.metaList}>
          <li>Chunks: {data.chunks}</li>
          <li>Size / overlap: {data.size_chars} / {data.overlap_chars} chars</li>
        </ul>
      );
    case 3:
    case 4:
      return (
        <ul style={styles.metaList}>
          <li>Dimension: {data.dimension}</li>
          <li>Query vector: {data.query_vector_len} floats</li>
          <li>Index vectors: {data.index_vectors}</li>
        </ul>
      );
    case 5: {
      const dense = data.dense || [];
      const bm25 = data.bm25 || [];
      return (
        <div style={styles.comparePanel}>
          <div style={styles.panel}>
            <div style={styles.panelTitle}>Dense (before rerank)</div>
            <div style={styles.panelMeta}>
              {dense.length} chunks · {data.dense_latency_ms} ms
            </div>
            {dense.map((h, i) => (
              <div key={i} style={styles.chip}>
                {h.document_id} · sim {h.similarity}
              </div>
            ))}
          </div>
          <div style={styles.panel}>
            <div style={styles.panelTitle}>BM25 (before rerank)</div>
            <div style={styles.panelMeta}>
              {bm25.length} chunks · {data.bm25_latency_ms} ms
            </div>
            {bm25.map((h, i) => (
              <div key={i} style={styles.chip}>
                {h.document_id} · score {h.similarity}
              </div>
            ))}
          </div>
        </div>
      );
    }
    case 6: {
      const fused = data.fused || [];
      return (
        <div>
          <div style={styles.panelMeta}>
            {fused.length} fused chunks · {data.latency_ms} ms · sources:{" "}
            {JSON.stringify(data.contributions)}
          </div>
          <div style={styles.chipRow}>
            {fused.map((h, i) => (
              <div key={i} style={styles.chip}>
                {h.document_id} · sim {h.similarity}
              </div>
            ))}
          </div>
        </div>
      );
    }
    case 7: {
      const reranked = data.reranked || [];
      return (
        <div>
          <div style={styles.panelMeta}>
            {reranked.length} reranked chunks · {data.latency_ms} ms ·
            {data.candidates} candidates
          </div>
          <div style={styles.chipRow}>
            {reranked.map((h, i) => (
              <div key={i} style={styles.chip}>
                {h.document_id} · sim {h.similarity}
              </div>
            ))}
          </div>
        </div>
      );
    }
    case 9:
      return (
        <ul style={styles.metaList}>
          <li>
            Latency: {data.latency_ms} ms · Tokens: {data.tokens}
          </li>
          <li>
            Blocked: {String(data.blocked)} {data.block_reason ? `— ${data.block_reason}` : ""}
          </li>
        </ul>
      );
    case 10:
      return (
        <ul style={styles.metaList}>
          <li>
            Faithfulness:{data.faithfulness === null ? " n/a" : ` ${data.faithfulness.toFixed(3)}`}
          </li>
          <li>
            Answer relevance:{data.answer_relevance === null ? " n/a" : ` ${data.answer_relevance.toFixed(3)}`}
          </li>
        </ul>
      );
    default:
      return null;
  }
}

const styles = {
  page: {
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    maxWidth: 900,
    margin: "0 auto",
    padding: 24,
    color: "#1e293b",
    lineHeight: 1.5,
  },
  header: {
    textAlign: "center",
    marginBottom: 32,
  },
  title: {
    fontSize: 32,
    margin: 0,
    color: "#0f172a",
  },
  subtitle: {
    color: "#64748b",
    marginTop: 8,
  },
  form: {
    display: "flex",
    gap: 12,
    marginBottom: 32,
  },
  input: {
    flex: 1,
    padding: "10px 14px",
    fontSize: 16,
    border: "1px solid #cbd5e1",
    borderRadius: 8,
    outline: "none",
  },
  button: {
    padding: "10px 24px",
    fontSize: 16,
    backgroundColor: "#0f172a",
    color: "#fff",
    border: "none",
    borderRadius: 8,
    cursor: "pointer",
    fontWeight: 600,
  },
  errorBox: {
    backgroundColor: "#fef2f2",
    border: "1px solid #fecaca",
    color: "#991b1b",
    padding: 14,
    borderRadius: 8,
    marginBottom: 24,
  },
  results: {},
  diagnosis: {
    backgroundColor: "#eff6ff",
    border: "1px solid #bfdbfe",
    color: "#1e3a8a",
    padding: 12,
    borderRadius: 8,
    marginBottom: 20,
    fontSize: 15,
  },
  answerCard: {
    backgroundColor: "#f8fafc",
    border: "1px solid #e2e8f0",
    borderRadius: 10,
    padding: 16,
    marginBottom: 24,
  },
  answerTitle: {
    margin: "0 0 8px 0",
    fontSize: 18,
    color: "#0f172a",
  },
  citations: {
    marginTop: 10,
    fontSize: 13,
    color: "#475569",
  },
  pipeline: {},
  pipelineTitle: {
    margin: "0 0 12px 0",
    fontSize: 18,
    color: "#0f172a",
  },
  stageCard: (status) => ({
    border: `1px solid ${status === "error" ? "#fca5a5" : status === "skipped" ? "#fcd34d" : "#e2e8f0"}`,
    borderLeft: `4px solid ${stageColor(status)}`,
    borderRadius: 8,
    padding: 14,
    marginBottom: 10,
    backgroundColor:
      status === "ok" ? "#f0fdf4" : status === "skipped" ? "#fffbeb" : "#ffffff",
  }),
  stageHeader: {
    display: "flex",
    alignItems: "center",
    gap: 10,
    marginBottom: 6,
  },
  stageBadge: {
    width: 26,
    height: 26,
    borderRadius: 6,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    color: "#fff",
    fontWeight: 700,
    fontSize: 14,
    flexShrink: 0,
  },
  stageName: {
    fontWeight: 700,
    fontSize: 15,
    color: "#0f172a",
  },
  stageSub: {
    fontSize: 12,
    color: "#64748b",
  },
  stageDetail: {
    fontSize: 13,
    color: "#475569",
    marginBottom: 8,
  },
  stageData: {
    marginTop: 8,
  },
  metaList: {
    margin: 0,
    paddingLeft: 18,
    fontSize: 13,
    color: "#475569",
  },
  comparePanel: {
    display: "flex",
    gap: 12,
    marginTop: 8,
  },
  panel: {
    flex: 1,
    border: "1px solid #e2e8f0",
    borderRadius: 6,
    padding: 10,
    backgroundColor: "#ffffff",
  },
  panelTitle: {
    fontWeight: 700,
    fontSize: 13,
    color: "#0f172a",
    marginBottom: 4,
  },
  panelMeta: {
    fontSize: 12,
    color: "#64748b",
    marginBottom: 6,
  },
  chip: {
    display: "inline-block",
    backgroundColor: "#e2e8f0",
    color: "#334155",
    padding: "4px 8px",
    borderRadius: 4,
    fontSize: 12,
    marginRight: 6,
    marginBottom: 4,
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, monospace",
  },
  chipRow: {
    marginTop: 8,
    display: "flex",
    flexWrap: "wrap",
    gap: 6,
  },
};
