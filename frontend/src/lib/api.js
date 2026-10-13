/**
 * src/lib/api.js
 * ==============
 * Every call this UI makes to the IncidentLens FastAPI service (modules/20_api_server.py).
 * No mocking, no fixtures: if the API is down the UI says so.
 *
 * Shapes are documented with JSDoc "types by convention" (this is a plain-JS project).
 *
 * Exported surface:
 *   API_BASE                       resolved base URL
 *   OFFLINE_HINT                   exact command an operator should run when the API is down
 *   ApiError                       thrown for every non-2xx / network failure
 *   fetchHealth()                  GET  /health
 *   postQuery()                    POST /query
 *   streamQuery()                  POST /query/stream  (SSE, via fetch + ReadableStream)
 *   postEvaluate()                 POST /evaluate
 *   fetchMetrics()                 GET  /metrics   (raw Prometheus text)
 *   parsePrometheus()              raw text -> { samples, meta, order }
 *   formatUptime()                 seconds -> "1h 04m 12s"
 */

/**
 * Base URL of the backend. Override with VITE_API_BASE in a .env file.
 *
 * Vite statically replaces `import.meta.env.VITE_API_BASE` at build time, so the value is
 * inlined. The `?.` guard is what lets this same module be imported by plain Node (the
 * scripts in scripts/ do that) without an env object:
 *   VITE_API_BASE=http://127.0.0.1:8000   # .env, next to package.json
 */
export const API_BASE = (
  import.meta.env?.VITE_API_BASE || "http://127.0.0.1:8000"
).replace(/\/+$/, "");

/** The exact start command, surfaced in the offline state. */
export const OFFLINE_HINT = ".\\venv\\Scripts\\python.exe modules\\20_api_server.py";

/**
 * @typedef {Object} ApiErrorShape
 * @property {string} message     human-readable message
 * @property {number} status      HTTP status, or 0 for a network/transport failure
 * @property {string} type        backend error type, or "network_error"
 * @property {string|null} requestId  X-Request-ID echoed by the server
 * @property {Array} details      validation details when present
 */

export class ApiError extends Error {
  /**
   * @param {ApiErrorShape} shape
   */
  constructor({ message, status = 0, type = "error", requestId = null, details = [] }) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.type = type;
    this.requestId = requestId;
    this.details = details;
  }

  /** True when the request never reached the server (or was blocked by the browser). */
  get isNetworkError() {
    return this.status === 0;
  }
}

/**
 * Read a fetch Response as JSON, converting any failure mode into an ApiError.
 *
 * The server always answers errors with `{error: {type, message, details, request_id}}`
 * (see `_error_payload` in module 20), so that envelope is unwrapped here.
 *
 * @param {Response} res
 * @returns {Promise<any>}
 */
async function readJson(res) {
  const text = await res.text();
  let payload = null;
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = null;
    }
  }

  if (res.ok) {
    if (payload === null) {
      throw new ApiError({
        message: "The server returned a non-JSON body.",
        status: res.status,
        type: "bad_response",
        requestId: res.headers.get("X-Request-ID"),
      });
    }
    return payload;
  }

  const envelope = payload && typeof payload === "object" ? payload.error : null;
  throw new ApiError({
    status: res.status,
    type: envelope?.type || "http_error",
    message:
      envelope?.message ||
      (text ? text.slice(0, 300) : `Request failed with HTTP ${res.status}.`),
    details: envelope?.details || [],
    requestId: envelope?.request_id || res.headers.get("X-Request-ID"),
  });
}

/**
 * Wrap fetch so transport-level failures become ApiError(status 0).
 * @param {string} path
 * @param {RequestInit} [init]
 */
async function request(path, init) {
  let res;
  try {
    res = await fetch(`${API_BASE}${path}`, init);
  } catch (err) {
    throw new ApiError({
      message: `Cannot reach the IncidentLens API at ${API_BASE}.`,
      status: 0,
      type: "network_error",
      details: [String(err && err.message ? err.message : err)],
    });
  }
  return res;
}

const JSON_HEADERS = { "Content-Type": "application/json" };

/* ------------------------------------------------------------------ *
 * 1. POST /query
 * ------------------------------------------------------------------ */

/**
 * @typedef {Object} QueryRequest
 * @property {string} question
 * @property {string} [thread_id]  reuse to keep agent memory across turns
 *
 * @typedef {Object} QueryResponse
 * @property {string} thread_id
 * @property {string} answer
 * @property {boolean} blocked
 * @property {string[]} citations
 * @property {number|null} relevance_score
 * @property {string|null} relevance_source
 * @property {number|null} grounding_score
 * @property {boolean} grounding_checked
 * @property {string|null} security_note
 * @property {number} latency_ms
 * @property {string[]} path
 */

/**
 * One grounded answer through the full pipeline.
 * @param {QueryRequest} body
 * @param {{ signal?: AbortSignal }} [opts]
 * @returns {Promise<QueryResponse>}
 */
export async function postQuery(body, opts = {}) {
  const res = await request("/query", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(body),
    signal: opts.signal,
  });
  return readJson(res);
}

/* ------------------------------------------------------------------ *
 * 2. POST /query/stream  (Server-Sent Events over fetch)
 * ------------------------------------------------------------------ */

/**
 * @typedef {Object} StreamEvent
 * @property {"start"|"node"|"answer"|"done"|"error"} event
 * @property {string} [thread_id]
 * @property {string} [question]
 * @property {string} [node]
 * @property {string[]} [steps]
 * @property {string} [answer]
 * @property {boolean} [blocked]
 * @property {string[]} [citations]
 * @property {number} [latency_ms]
 * @property {string} [type]
 * @property {string} [message]
 */

/**
 * Parse a buffer of SSE text into complete events, returning the unterminated remainder.
 *
 * The server writes exactly `data: {json}\n\n` per event (see `sse()` in module 20).
 * EventSource cannot POST, so we read the body stream ourselves. A chunk boundary can
 * split an event anywhere, hence the buffering.
 *
 * @param {string} buffer
 * @returns {{ events: StreamEvent[], rest: string }}
 */
export function parseSseChunk(buffer) {
  const events = [];
  // Split on a blank line; tolerate both \n\n and \r\n\r\n.
  const parts = buffer.split(/\r?\n\r?\n/);
  const rest = parts.pop() ?? "";

  for (const part of parts) {
    const dataLines = part
      .split(/\r?\n/)
      .filter((line) => line.startsWith("data:"))
      .map((line) => line.slice(5).trimStart());

    if (dataLines.length === 0) continue;
    const raw = dataLines.join("\n").trim();
    if (!raw) continue;
    try {
      events.push(JSON.parse(raw));
    } catch {
      // A malformed frame is not worth killing the stream over; skip it.
      events.push({ event: "error", type: "parse_error", message: raw.slice(0, 200) });
    }
  }
  return { events, rest };
}

/**
 * Stream the agent's node-by-node progress.
 *
 * `onEvent` is called for every parsed event, in order. The promise resolves once the
 * stream ends (or the `done` event arrives) and rejects only on transport/HTTP errors.
 *
 * @param {QueryRequest} body
 * @param {{ onEvent?: (e: StreamEvent) => void, signal?: AbortSignal }} [opts]
 * @returns {Promise<void>}
 */
export async function streamQuery(body, opts = {}) {
  const { onEvent, signal } = opts;
  const res = await request("/query/stream", {
    method: "POST",
    headers: { ...JSON_HEADERS, Accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  });

  if (!res.ok || !res.body) {
    // Reuse the JSON error envelope path when the server refused before streaming.
    if (res.body) await readJson(res);
    throw new ApiError({
      message: `The streaming endpoint answered HTTP ${res.status} with no body.`,
      status: res.status,
      type: "stream_error",
      requestId: res.headers.get("X-Request-ID"),
    });
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const { events, rest } = parseSseChunk(buffer);
      buffer = rest;
      for (const event of events) {
        if (typeof onEvent === "function") onEvent(event);
      }
    }
    // Flush whatever is left if the server closed without a trailing blank line.
    buffer += decoder.decode();
    const tail = parseSseChunk(buffer.endsWith("\n\n") ? buffer : `${buffer}\n\n`);
    for (const event of tail.events) {
      if (typeof onEvent === "function") onEvent(event);
    }
  } finally {
    // Release the lock so an aborted request does not leak the reader.
    reader.releaseLock?.();
  }
}

/* ------------------------------------------------------------------ *
 * 3. GET /health
 * ------------------------------------------------------------------ */

/**
 * @typedef {Object} HealthCheck
 * @property {string} name
 * @property {"ok"|"degraded"|"error"} status
 * @property {string} detail
 *
 * @typedef {Object} HealthResponse
 * @property {"ok"|"degraded"} status
 * @property {string} service
 * @property {string} version
 * @property {string} timestamp
 * @property {HealthCheck[]} checks
 */

/**
 * Dependency status. `deep=true` adds a live LLM probe (costs tokens), so it is opt-in.
 * @param {{ deep?: boolean, signal?: AbortSignal }} [opts]
 * @returns {Promise<HealthResponse>}
 */
export async function fetchHealth(opts = {}) {
  const query = opts.deep ? "?deep=true" : "";
  const res = await request(`/health${query}`, { signal: opts.signal });
  return readJson(res);
}

/* ------------------------------------------------------------------ *
 * 4. POST /evaluate
 * ------------------------------------------------------------------ */

/**
 * @typedef {Object} PerQuestion
 * @property {string} question
 * @property {number} faithfulness
 * @property {number} answer_relevancy
 * @property {number} context_precision
 * @property {number} context_recall
 *
 * @typedef {Object} EvaluateResponse
 * @property {number} questions_evaluated
 * @property {Record<string, number>} metrics
 * @property {Record<string, number>} targets
 * @property {Record<string, boolean>} passed
 * @property {number} judge_failures
 * @property {boolean} complete
 * @property {PerQuestion[]} per_question
 */

/**
 * Run the 4-metric judge over N golden questions (each costs 5 LLM calls).
 * @param {number} limit 1..5
 * @param {{ signal?: AbortSignal }} [opts]
 * @returns {Promise<EvaluateResponse>}
 */
export async function postEvaluate(limit, opts = {}) {
  const res = await request("/evaluate", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ limit }),
    signal: opts.signal,
  });
  return readJson(res);
}

/* ------------------------------------------------------------------ *
 * 5. GET /metrics  (Prometheus text)
 * ------------------------------------------------------------------ */

/**
 * Raw Prometheus exposition text.
 * @param {{ signal?: AbortSignal }} [opts]
 * @returns {Promise<string>}
 */
export async function fetchMetrics(opts = {}) {
  const res = await request("/metrics", { signal: opts.signal });
  if (!res.ok) await readJson(res);
  return res.text();
}

/**
 * Parse Prometheus exposition text into a plain map.
 *
 * Only unlabelled samples are used as keys (`incidentlens_query_total 3` -> 3); labelled
 * samples keep their full `name{labels}` key so nothing is silently merged. HELP/TYPE
 * comment lines are collected separately for the raw view.
 *
 * @param {string} text
 * @returns {{ samples: Record<string, number>, meta: Record<string, {help?: string, type?: string}>, order: string[] }}
 */
export function parsePrometheus(text) {
  /** @type {Record<string, number>} */
  const samples = {};
  /** @type {Record<string, {help?: string, type?: string}>} */
  const meta = {};
  const order = [];

  for (const line of String(text || "").split(/\r?\n/)) {
    const trimmed = line.trim();
    if (!trimmed) continue;

    if (trimmed.startsWith("#")) {
      const help = trimmed.match(/^#\s*HELP\s+(\S+)\s+(.*)$/);
      if (help) {
        meta[help[1]] = { ...(meta[help[1]] || {}), help: help[2] };
        continue;
      }
      const type = trimmed.match(/^#\s*TYPE\s+(\S+)\s+(\S+)$/);
      if (type) {
        meta[type[1]] = { ...(meta[type[1]] || {}), type: type[2] };
      }
      continue;
    }

    const match = trimmed.match(/^([a-zA-Z_:][a-zA-Z0-9_:]*(?:\{[^}]*\})?)\s+(-?[\d.eE+]+|NaN|\+Inf|-Inf)/);
    if (!match) continue;
    const value = Number(match[2]);
    samples[match[1]] = Number.isFinite(value) ? value : 0;
    if (!order.includes(match[1])) order.push(match[1]);
  }

  return { samples, meta, order };
}

/** Seconds -> compact "2h 04m 09s" / "4m 09s" / "9.4s". */
export function formatUptime(seconds) {
  const total = Math.max(0, Math.floor(Number(seconds) || 0));
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const pad = (n) => String(n).padStart(2, "0");
  if (h > 0) return `${h}h ${pad(m)}m ${pad(s)}s`;
  if (m > 0) return `${m}m ${pad(s)}s`;
  return `${s}s`;
}
