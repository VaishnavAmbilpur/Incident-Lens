import { AlertTriangle, Check, Play, X } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { cn } from "@/lib/utils";

/** The four judge metrics, in the order the scorecard reports them. */
export const METRIC_KEYS = [
  { key: "faithfulness", label: "Faithfulness" },
  { key: "answer_relevancy", label: "Answer relevancy" },
  { key: "context_precision", label: "Context precision" },
  { key: "context_recall", label: "Context recall" },
];

const LIMITS = [1, 2, 3, 4, 5];

function formatMetric(value) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "-";
}

/**
 * One metric as a card comparing the measured value with its target.
 * The bar is a plain div: no chart library for four numbers.
 */
function MetricCard({ label, value, target, passed }) {
  const pct = typeof value === "number" ? Math.min(100, Math.max(0, value * 100)) : 0;
  const targetPct = typeof target === "number" ? Math.min(100, Math.max(0, target * 100)) : 0;

  return (
    <div className="rounded-xl border border-border bg-card/50 p-4">
      <div className="flex items-center justify-between gap-2">
        <p className="meta-label">{label}</p>
        <Badge variant={passed ? "success" : "destructive"}>
          {passed ? <Check className="h-3 w-3" /> : <X className="h-3 w-3" />}
          {passed ? "pass" : "fail"}
        </Badge>
      </div>

      <p
        className={cn(
          "mt-2 font-mono text-2xl tabular-nums",
          passed ? "text-emerald-500" : "text-destructive"
        )}
      >
        {formatMetric(value)}
      </p>

      <div className="relative mt-3 h-1.5 w-full overflow-hidden rounded-full bg-muted">
        <div
          className={cn("h-full rounded-full", passed ? "bg-emerald-500" : "bg-destructive")}
          style={{ width: `${pct}%` }}
        />
        {/* Target marker: where the bar needed to reach. */}
        <span
          className="absolute top-[-2px] h-[10px] w-px bg-foreground/70"
          style={{ left: `${targetPct}%` }}
          aria-hidden="true"
        />
      </div>
      <p className="mt-1.5 font-mono text-[10px] text-muted-foreground">
        target {formatMetric(target)}
      </p>
    </div>
  );
}

function MetricSkeletons() {
  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
      {METRIC_KEYS.map(({ key }) => (
        <div key={key} className="rounded-xl border border-border bg-card/50 p-4">
          <Skeleton className="h-3 w-24" />
          <Skeleton className="mt-3 h-7 w-20" />
          <Skeleton className="mt-3 h-1.5 w-full" />
        </div>
      ))}
    </div>
  );
}

/**
 * @param {{
 *   limit: number, onLimitChange: (n:number)=>void,
 *   onRun: ()=>void, isRunning: boolean, result: import("@/lib/api").EvaluateResponse|null,
 *   disabled: boolean,
 * }} props
 */
export function EvaluationPanel({ limit, onLimitChange, onRun, isRunning, result, disabled }) {
  const incomplete = result ? result.complete === false || result.judge_failures > 0 : false;

  return (
    <div className="space-y-5">
      <div className="rounded-xl border border-border bg-card/50 p-5">
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div className="space-y-2">
            <p className="meta-label">Golden questions to evaluate</p>
            <div className="flex items-center gap-1.5">
              {LIMITS.map((n) => (
                <button
                  key={n}
                  type="button"
                  onClick={() => onLimitChange(n)}
                  disabled={isRunning || disabled}
                  aria-pressed={limit === n}
                  className={cn(
                    "h-9 w-9 rounded-lg border font-mono text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50",
                    limit === n
                      ? "border-primary/50 bg-primary/15 text-primary"
                      : "border-border bg-background/40 text-muted-foreground hover:bg-accent hover:text-foreground"
                  )}
                >
                  {n}
                </button>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-3">
            <p className="max-w-[16rem] text-[11px] leading-relaxed text-amber-500">
              <AlertTriangle className="mr-1 inline h-3 w-3 align-[-2px]" />
              Each question costs 5 LLM calls (1 answer + 4 judges). A run of {limit} costs{" "}
              <span className="font-mono">{limit * 5}</span> calls.
            </p>
            <Button type="button" onClick={onRun} disabled={isRunning || disabled}>
              <Play className={cn(isRunning && "animate-pulse")} />
              {isRunning ? "Running..." : "Run evaluation"}
            </Button>
          </div>
        </div>
      </div>

      {isRunning ? (
        <div className="space-y-5">
          <MetricSkeletons />
          <div className="rounded-xl border border-border bg-card/50 p-5">
            <p className="text-xs text-muted-foreground">
              Judge calls in flight. A 5-question run can take several minutes - this panel stays
              open.
            </p>
            <div className="mt-4 space-y-2">
              {Array.from({ length: limit }).map((_, i) => (
                <Skeleton key={i} className="h-8 w-full" />
              ))}
            </div>
          </div>
        </div>
      ) : null}

      {!isRunning && !result ? (
        <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-border px-6 py-12 text-center">
          <p className="text-sm font-medium">No scorecard yet</p>
          <p className="mt-1 max-w-md text-xs leading-relaxed text-muted-foreground">
            Pick a limit and run the judge. The scorecard is honest by design: if any judge call
            fails, the run is reported as incomplete rather than silently averaged down.
          </p>
        </div>
      ) : null}

      {!isRunning && result ? (
        <div className="space-y-5">
          {incomplete ? (
            <Alert variant="destructive">
              <AlertTriangle />
              <AlertTitle>INCOMPLETE scorecard</AlertTitle>
              <AlertDescription>
                <span className="font-mono">{result.judge_failures}</span> judge call
                {result.judge_failures === 1 ? "" : "s"} failed, so these numbers are a partial
                picture - treat them as a floor, not a verdict. Re-run before trusting the deltas.
              </AlertDescription>
            </Alert>
          ) : (
            <Alert variant="success">
              <Check />
              <AlertTitle>Complete scorecard</AlertTitle>
              <AlertDescription>
                All judge calls returned for {result.questions_evaluated} question
                {result.questions_evaluated === 1 ? "" : "s"}. No judge failures.
              </AlertDescription>
            </Alert>
          )}

          <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            {METRIC_KEYS.map(({ key, label }) => (
              <MetricCard
                key={key}
                label={label}
                value={result.metrics?.[key]}
                target={result.targets?.[key]}
                passed={Boolean(result.passed?.[key])}
              />
            ))}
          </div>

          <div className="overflow-hidden rounded-xl border border-border bg-card/50">
            <div className="flex items-center justify-between gap-2 border-b border-border px-5 py-3">
              <h3 className="text-sm font-semibold tracking-tight">Per-question scores</h3>
              <span className="meta-label">
                {result.per_question?.length || 0} row
                {(result.per_question?.length || 0) === 1 ? "" : "s"}
              </span>
            </div>

            {result.per_question?.length ? (
              <div className="scrollbar-thin overflow-x-auto">
                <table className="w-full min-w-[44rem] border-collapse text-sm">
                  <thead>
                    <tr className="border-b border-border text-left">
                      <th className="meta-label px-5 py-2 font-medium">Question</th>
                      {METRIC_KEYS.map(({ key, label }) => (
                        <th key={key} className="meta-label px-3 py-2 text-right font-medium">
                          {label}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {result.per_question.map((row, i) => (
                      <tr
                        key={`${row.question}-${i}`}
                        className="border-b border-border/60 last:border-0 hover:bg-accent/40"
                      >
                        <td className="max-w-[24rem] px-5 py-3 text-xs leading-relaxed text-foreground/90">
                          {row.question}
                        </td>
                        {METRIC_KEYS.map(({ key }) => {
                          const value = row[key];
                          const target = result.targets?.[key];
                          const failed = typeof value === "number" && typeof target === "number" && value < target;
                          return (
                            <td
                              key={key}
                              className={cn(
                                "px-3 py-3 text-right font-mono text-xs tabular-nums",
                                failed ? "text-destructive" : "text-foreground/90"
                              )}
                            >
                              {formatMetric(value)}
                            </td>
                          );
                        })}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="px-5 py-6 text-xs text-muted-foreground">
                The judge returned no per-question rows for this run.
              </p>
            )}
          </div>
        </div>
      ) : null}
    </div>
  );
}
