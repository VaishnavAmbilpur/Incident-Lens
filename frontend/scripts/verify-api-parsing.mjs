/**
 * scripts/verify-api-parsing.mjs
 * ==============================
 * Unit checks for the two parsers in src/lib/api.js, using byte-for-byte samples of what
 * module 20 actually emits. Run: node scripts/verify-api-parsing.mjs
 *
 * The import of src/lib/api.js is a plain .js ESM import and needs no transform.
 */

import assert from "node:assert/strict";
import { formatUptime, parsePrometheus, parseSseChunk } from "../src/lib/api.js";

let passed = 0;
function check(name, fn) {
  try {
    fn();
    passed += 1;
    console.log(`  ok   ${name}`);
  } catch (error) {
    console.error(`  FAIL ${name}`);
    console.error(`       ${error.message}`);
    process.exitCode = 1;
  }
}

/* ------------------------------------------------------------------ *
 * SSE framing. Frames copied from `sse()` in modules/20_api_server.py.
 * ------------------------------------------------------------------ */
const SSE_LINE_START =
  'data: {"event": "start", "thread_id": "api-1a2b3c4d", "question": "why did checkout fail?"}\n\n';
const SSE_LINE_NODE =
  'data: {"event": "node", "node": "retrieve", "steps": ["hybrid retrieval: 4 chunks", "rrf_k=60"]}\n\n';
const SSE_LINE_ANSWER =
  'data: {"event": "answer", "thread_id": "api-1a2b3c4d", "answer": "Root cause [c12]: a bad migration.", "blocked": false, "citations": ["c12"], "latency_ms": 4210}\n\n';
const SSE_LINE_DONE = 'data: {"event": "done"}\n\n';

console.log("SSE framing");

check("parses three whole frames and leaves no remainder", () => {
  const { events, rest } = parseSseChunk(SSE_LINE_START + SSE_LINE_NODE + SSE_LINE_ANSWER);
  assert.equal(rest, "");
  assert.equal(events.length, 3);
  assert.equal(events[0].event, "start");
  assert.equal(events[1].node, "retrieve");
  assert.deepEqual(events[1].steps, ["hybrid retrieval: 4 chunks", "rrf_k=60"]);
  assert.equal(events[2].event, "answer");
  assert.equal(events[2].latency_ms, 4210);
  assert.deepEqual(events[2].citations, ["c12"]);
});

check("keeps a partially received frame in the buffer", () => {
  const whole = SSE_LINE_START + SSE_LINE_NODE;
  const splitAt = whole.length - 25;
  const first = parseSseChunk(whole.slice(0, splitAt));
  assert.equal(first.events.length, 1, "only the first frame is complete");
  assert.equal(first.events[0].event, "start");
  assert.ok(first.rest.length > 0, "the partial frame must be retained");

  const second = parseSseChunk(first.rest + whole.slice(splitAt));
  assert.equal(second.events.length, 1);
  assert.equal(second.events[0].node, "retrieve");
  assert.equal(second.rest, "");
});

check("handles CRLF line endings", () => {
  const crlf = SSE_LINE_DONE.replace(/\n/g, "\r\n");
  const { events, rest } = parseSseChunk(crlf);
  assert.equal(rest, "");
  assert.equal(events.length, 1);
  assert.equal(events[0].event, "done");
});

check("an error frame keeps its type and message", () => {
  const { events } = parseSseChunk(
    'data: {"event": "error", "type": "pipeline_error", "message": "RuntimeError: boom"}\n\n'
  );
  assert.equal(events[0].event, "error");
  assert.equal(events[0].type, "pipeline_error");
  assert.equal(events[0].message, "RuntimeError: boom");
});

check("a malformed frame degrades to an error event instead of throwing", () => {
  const { events } = parseSseChunk("data: {not json}\n\n");
  assert.equal(events.length, 1);
  assert.equal(events[0].event, "error");
  assert.equal(events[0].type, "parse_error");
});

/* ------------------------------------------------------------------ *
 * Prometheus text. Copied from render_metrics() in modules/20_api_server.py.
 * ------------------------------------------------------------------ */
const PROM = `# HELP incidentlens_uptime_seconds Seconds since the service started.
# TYPE incidentlens_uptime_seconds gauge
incidentlens_uptime_seconds 821.4
# HELP incidentlens_http_requests_total Total HTTP requests received.
# TYPE incidentlens_http_requests_total counter
incidentlens_http_requests_total 17
# HELP incidentlens_query_total Total /query calls.
# TYPE incidentlens_query_total counter
incidentlens_query_total 3
# HELP incidentlens_query_blocked_total Queries refused by the security gate.
# TYPE incidentlens_query_blocked_total counter
incidentlens_query_blocked_total 1
# HELP incidentlens_query_failed_total Queries that ended in an error.
# TYPE incidentlens_query_failed_total counter
incidentlens_query_failed_total 0
# HELP incidentlens_query_latency_ms Query latency in milliseconds.
# TYPE incidentlens_query_latency_ms summary
incidentlens_query_latency_ms_sum 9120.5
incidentlens_query_latency_ms_count 3
incidentlens_query_latency_ms_avg 3040.2
incidentlens_query_latency_ms_max 4210.0
`;

console.log("Prometheus parsing");

check("extracts every unlabelled sample", () => {
  const { samples, order } = parsePrometheus(PROM);
  assert.equal(samples.incidentlens_query_total, 3);
  assert.equal(samples.incidentlens_query_blocked_total, 1);
  assert.equal(samples.incidentlens_query_failed_total, 0);
  assert.equal(samples.incidentlens_query_latency_ms_avg, 3040.2);
  assert.equal(samples.incidentlens_query_latency_ms_max, 4210);
  assert.equal(samples.incidentlens_uptime_seconds, 821.4);
  assert.equal(order.length, 9);
});

check("collects HELP and TYPE metadata keyed by metric", () => {
  const { meta } = parsePrometheus(PROM);
  assert.equal(meta.incidentlens_query_total.type, "counter");
  assert.match(meta.incidentlens_query_total.help, /Total \/query calls/);
  assert.equal(meta.incidentlens_uptime_seconds.type, "gauge");
});

check("keeps labelled samples under their full key", () => {
  const { samples } = parsePrometheus('incidentlens_http_requests_total{method="POST"} 4\n');
  assert.equal(samples['incidentlens_http_requests_total{method="POST"}'], 4);
});

check("empty input yields empty results, not an exception", () => {
  const { samples, order, meta } = parsePrometheus("");
  assert.deepEqual(samples, {});
  assert.deepEqual(order, []);
  assert.deepEqual(meta, {});
});

console.log("Formatting");

check("formatUptime covers all three magnitudes", () => {
  assert.equal(formatUptime(9.4), "9s");
  assert.equal(formatUptime(249), "4m 09s");
  assert.equal(formatUptime(7449), "2h 04m 09s");
  assert.equal(formatUptime(undefined), "0s");
});

console.log(`\n${passed} checks passed${process.exitCode ? " (with failures)" : ""}.`);
