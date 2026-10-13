import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";
import { BarChart3, Gauge, HeartPulse, MessageSquare, WifiOff } from "lucide-react";

import { Toaster } from "@/components/ui/sonner";
import { Badge } from "@/components/ui/badge";
import {
  Tabs,
  TabsContent,
  TabsList,
  TabsTrigger,
} from "@/components/ui/tabs";
import { TooltipProvider } from "@/components/ui/tooltip";

import { Header } from "@/components/header";
import { QueryPanel } from "@/components/query-panel";
import { AgentTrace } from "@/components/agent-trace";
import { AnswerCard } from "@/components/answer-card";
import { EvaluationPanel } from "@/components/evaluation-panel";
import { MetricsPanel } from "@/components/metrics-panel";
import { HealthPanel, OfflineNotice } from "@/components/health-panel";

import { useHealth } from "@/hooks/useHealth";
import {
  API_BASE,
  ApiError,
  fetchMetrics,
  parsePrometheus,
  postEvaluate,
  postQuery,
  streamQuery,
} from "@/lib/api";
import { applyTheme, getStoredTheme, persistTheme } from "@/lib/theme";
import { cn } from "@/lib/utils";

const TABS = [
  { value: "ask", label: "Ask", icon: MessageSquare },
  { value: "evaluate", label: "Evaluate", icon: Gauge },
  { value: "metrics", label: "Metrics", icon: BarChart3 },
  { value: "health", label: "Health", icon: HeartPulse },
];

/** Human message for any thrown error, preferring the backend's own wording. */
function describeError(error) {
  if (error instanceof ApiError) {
    if (error.isNetworkError) return error.message;
    const suffix = error.requestId ? ` (request ${error.requestId})` : "";
    return `${error.message}${suffix}`;
  }
  return error?.message || String(error);
}

export default function App() {
  /* ---------------- theme (dark-first, persisted) ---------------- */
  const [theme, setTheme] = useState(() => getStoredTheme());

  useEffect(() => {
    applyTheme(theme);
    persistTheme(theme);
  }, [theme]);

  const toggleTheme = useCallback(() => {
    setTheme((current) => (current === "dark" ? "light" : "dark"));
  }, []);

  /* ---------------- health drives the whole shell ---------------- */
  const health = useHealth();
  const isOffline = health.isOffline;

  const [activeTab, setActiveTab] = useState("ask");

  /* ---------------- Ask state ---------------- */
  const [question, setQuestion] = useState("");
  const [threadId, setThreadId] = useState("");
  const [useStream, setUseStream] = useState(true);
  const [trace, setTrace] = useState([]);
  const [isRunning, setIsRunning] = useState(false);
  const [result, setResult] = useState(null); // full POST /query payload
  const [streamPartial, setStreamPartial] = useState(null); // answer frame from SSE
  const [askError, setAskError] = useState(null);
  const abortRef = useRef(null);

  // The stream is "live" while a request is in flight; that flags the trailing trace node.
  const traceEntries = useMemo(
    () =>
      trace.map((entry, index) => ({
        ...entry,
        live: isRunning && index === trace.length - 1,
      })),
    [trace, isRunning]
  );

  const resetRun = useCallback(() => {
    setTrace([]);
    setResult(null);
    setStreamPartial(null);
    setAskError(null);
  }, []);

  const handleAsk = useCallback(async () => {
    const trimmed = question.trim();
    if (!trimmed || isRunning) return;

    resetRun();
    setIsRunning(true);

    // An empty thread_id means "new conversation": the server mints one and we adopt it.
    const body = threadId.trim() ? { question: trimmed, thread_id: threadId.trim() } : { question: trimmed };

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      if (useStream) {
        let answerFrame = null;
        await streamQuery(body, {
          signal: controller.signal,
          onEvent: (event) => {
            switch (event.event) {
              case "start":
                if (event.thread_id) setThreadId(event.thread_id);
                break;
              case "node":
                setTrace((prev) => [
                  ...prev,
                  { node: event.node, steps: Array.isArray(event.steps) ? event.steps : [] },
                ]);
                break;
              case "answer":
                answerFrame = {
                  answer: event.answer || "",
                  blocked: Boolean(event.blocked),
                  citations: Array.isArray(event.citations) ? event.citations : [],
                  latency_ms: event.latency_ms,
                  thread_id: event.thread_id,
                };
                setStreamPartial(answerFrame);
                if (event.thread_id) setThreadId(event.thread_id);
                break;
              case "error":
                setAskError(new ApiError({ message: event.message, type: event.type }));
                break;
              case "done":
                break;
              default:
                break;
            }
          },
        });

        if (!answerFrame) {
          // The stream ended without an answer frame: say so instead of showing a blank card.
          setAskError(
            new ApiError({
              message: "The stream ended before the answer frame arrived.",
              type: "stream_incomplete",
            })
          );
        }
      } else {
        const payload = await postQuery(body, { signal: controller.signal });
        if (payload.thread_id) setThreadId(payload.thread_id);
        setResult(payload);
        if (payload.blocked) {
          toast.warning("Blocked by the security gate", {
            description: "No LLM calls were made for this query.",
          });
        }
      }
    } catch (error) {
      if (error?.name === "AbortError") {
        toast.info("Query cancelled");
      } else {
        setAskError(error);
        toast.error("Query failed", { description: describeError(error) });
      }
    } finally {
      setIsRunning(false);
      abortRef.current = null;
    }
  }, [question, threadId, useStream, isRunning, resetRun]);

  const handleCancel = useCallback(() => {
    abortRef.current?.abort();
    setIsRunning(false);
  }, []);

  /* ---------------- Evaluate state ---------------- */
  const [evalLimit, setEvalLimit] = useState(1);
  const [evalResult, setEvalResult] = useState(null);
  const [isEvaluating, setIsEvaluating] = useState(false);
  const evalAbortRef = useRef(null);

  const handleEvaluate = useCallback(async () => {
    if (isEvaluating) return;
    setIsEvaluating(true);
    setEvalResult(null);
    const controller = new AbortController();
    evalAbortRef.current = controller;
    try {
      const payload = await postEvaluate(evalLimit, { signal: controller.signal });
      setEvalResult(payload);
      if (payload.complete === false || payload.judge_failures > 0) {
        toast.warning("Incomplete scorecard", {
          description: `${payload.judge_failures} judge call(s) failed.`,
        });
      } else {
        toast.success("Evaluation complete");
      }
    } catch (error) {
      if (error?.name === "AbortError") {
        toast.info("Evaluation cancelled");
      } else {
        toast.error("Evaluation failed", { description: describeError(error) });
      }
    } finally {
      setIsEvaluating(false);
      evalAbortRef.current = null;
    }
  }, [evalLimit, isEvaluating]);

  /* ---------------- Metrics state ---------------- */
  const [metricsRaw, setMetricsRaw] = useState("");
  const [metricsError, setMetricsError] = useState(null);
  const [isLoadingMetrics, setIsLoadingMetrics] = useState(false);
  const metricsAbortRef = useRef(null);

  const loadMetrics = useCallback(async () => {
    metricsAbortRef.current?.abort();
    const controller = new AbortController();
    metricsAbortRef.current = controller;
    setIsLoadingMetrics(true);
    try {
      const text = await fetchMetrics({ signal: controller.signal });
      setMetricsRaw(text);
      setMetricsError(null);
    } catch (error) {
      if (error?.name !== "AbortError") {
        setMetricsError(error);
        setMetricsRaw("");
      }
    } finally {
      setIsLoadingMetrics(false);
    }
  }, []);

  // Fetch when the Metrics tab is opened, not on every render of the shell.
  useEffect(() => {
    if (activeTab === "metrics") loadMetrics();
  }, [activeTab, loadMetrics]);

  const parsedMetrics = useMemo(() => parsePrometheus(metricsRaw), [metricsRaw]);

  // Abort anything in flight if the app unmounts.
  useEffect(
    () => () => {
      abortRef.current?.abort();
      evalAbortRef.current?.abort();
      metricsAbortRef.current?.abort();
    },
    []
  );

  const offlineNote =
    "The API is unreachable, so this action is disabled. Start it and re-check the status pill.";

  return (
    <TooltipProvider delayDuration={200}>
      <div className="min-h-screen bg-background">
        <Header health={health} theme={theme} onToggleTheme={toggleTheme} />

        <main className="mx-auto w-full max-w-7xl px-4 pb-16 pt-6 sm:px-6">
          {isOffline ? <OfflineNotice className="mb-6" message={health.error?.message} /> : null}

          <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <TabsList>
                {TABS.map(({ value, label, icon: Icon }) => (
                  <TabsTrigger key={value} value={value}>
                    <Icon className="h-3.5 w-3.5" />
                    {label}
                  </TabsTrigger>
                ))}
              </TabsList>

              {isOffline ? (
                <Badge variant="destructive">
                  <WifiOff className="h-3 w-3" />
                  offline
                </Badge>
              ) : null}
            </div>

            {/* ------------------------------ Ask ------------------------------ */}
            <TabsContent value="ask" className="space-y-5">
              <QueryPanel
                question={question}
                onQuestionChange={setQuestion}
                threadId={threadId}
                onThreadIdChange={setThreadId}
                useStream={useStream}
                onUseStreamChange={setUseStream}
                onSubmit={handleAsk}
                onCancel={handleCancel}
                isRunning={isRunning}
                disabled={isOffline}
                disabledReason={offlineNote}
              />

              {askError ? (
                <div className="rounded-xl border border-destructive/40 bg-destructive/10 p-4">
                  <p className="text-sm font-medium text-destructive">Request failed</p>
                  <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                    {describeError(askError)}
                  </p>
                </div>
              ) : null}

              {/*
                Two columns on wide screens: the trace stays beside the answer it produced.
                `lg:grid-cols-[minmax(0,20rem)_minmax(0,1fr)]` keeps the trace readable
                without letting it dominate the answer.
              */}
              <div
                className={cn(
                  "grid gap-5",
                  "lg:grid-cols-[minmax(0,21rem)_minmax(0,1fr)] lg:items-start"
                )}
              >
                <div className="rounded-xl border border-border bg-card/50 p-5">
                  <AgentTrace entries={traceEntries} isStreaming={isRunning} error={askError} />
                </div>

                <AnswerCard result={result} isStreaming={isRunning} streamPartial={streamPartial} />
              </div>
            </TabsContent>

            {/* --------------------------- Evaluate ---------------------------- */}
            <TabsContent value="evaluate">
              {isOffline ? (
                <OfflineNotice className="mb-5" message={health.error?.message} />
              ) : null}
              <EvaluationPanel
                limit={evalLimit}
                onLimitChange={setEvalLimit}
                onRun={handleEvaluate}
                isRunning={isEvaluating}
                result={evalResult}
                disabled={isOffline}
              />
            </TabsContent>

            {/* ---------------------------- Metrics ---------------------------- */}
            <TabsContent value="metrics">
              {isOffline ? (
                <OfflineNotice className="mb-5" message={health.error?.message} />
              ) : null}
              <MetricsPanel
                samples={parsedMetrics.samples}
                raw={metricsRaw}
                isLoading={isLoadingMetrics}
                error={metricsError}
                onRefresh={loadMetrics}
              />
            </TabsContent>

            {/* ----------------------------- Health ---------------------------- */}
            <TabsContent value="health">
              <HealthPanel health={health} />
            </TabsContent>
          </Tabs>
        </main>

        <footer className="border-t border-border">
          <div className="mx-auto flex w-full max-w-7xl flex-wrap items-center gap-x-3 gap-y-1 px-4 py-5 text-[11px] text-muted-foreground sm:px-6">
            <span>IncidentLens</span>
            <span className="text-muted-foreground/50">|</span>
            <span>
              Talking to <span className="font-mono">{API_BASE}</span>
            </span>
            <span className="text-muted-foreground/50">|</span>
            <span>Blocked queries cost zero LLM calls.</span>
          </div>
        </footer>

        <Toaster theme={theme} />
      </div>
    </TooltipProvider>
  );
}
