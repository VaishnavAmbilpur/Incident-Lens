import { Activity, RefreshCw } from "lucide-react";

import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { StatTile } from "@/components/stat-tile";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { formatUptime } from "@/lib/api";
import { cn } from "@/lib/utils";

/**
 * Stat card definitions. Each reads one unlabelled sample from the Prometheus text
 * rendered by `render_metrics()` in module 20.
 *
 * `tone` can be a function so a non-zero failure counter is coloured without extra markup.
 */
const STAT_CARDS = [
  {
    label: "HTTP requests",
    key: "incidentlens_http_requests_total",
    hint: "every route, including health probes",
    format: (v) => String(Math.round(v)),
  },
  {
    label: "Queries",
    key: "incidentlens_query_total",
    hint: "POST /query and /query/stream",
    format: (v) => String(Math.round(v)),
  },
  {
    label: "Blocked",
    key: "incidentlens_query_blocked_total",
    hint: "refused by the security gate, 0 LLM calls",
    format: (v) => String(Math.round(v)),
    tone: (v) => (v > 0 ? "warning" : "default"),
  },
  {
    label: "Failed",
    key: "incidentlens_query_failed_total",
    hint: "queries that ended in an error",
    format: (v) => String(Math.round(v)),
    tone: (v) => (v > 0 ? "danger" : "default"),
  },
  {
    label: "HTTP errors",
    key: "incidentlens_http_errors_total",
    hint: "responses with status >= 400",
    format: (v) => String(Math.round(v)),
    tone: (v) => (v > 0 ? "danger" : "default"),
  },
  {
    label: "Evaluations",
    key: "incidentlens_evaluate_total",
    hint: "scorecard runs",
    format: (v) => String(Math.round(v)),
  },
  {
    label: "Latency avg",
    key: "incidentlens_query_latency_ms_avg",
    hint: "mean query latency",
    format: (v) => `${v.toFixed(1)} ms`,
  },
  {
    label: "Latency max",
    key: "incidentlens_query_latency_ms_max",
    hint: "worst observed query",
    format: (v) => `${v.toFixed(1)} ms`,
  },
  {
    label: "Latency samples",
    key: "incidentlens_query_latency_ms_count",
    hint: "queries included in the average",
    format: (v) => String(Math.round(v)),
  },
  {
    label: "Uptime",
    key: "incidentlens_uptime_seconds",
    hint: "since the service started",
    format: (v) => formatUptime(v),
  },
];

function MetricsSkeleton() {
  return (
    <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
      {Array.from({ length: 10 }).map((_, i) => (
        <div key={i} className="rounded-lg border border-border bg-card/50 p-3">
          <Skeleton className="h-3 w-20" />
          <Skeleton className="mt-3 h-5 w-16" />
          <Skeleton className="mt-2 h-2 w-full" />
        </div>
      ))}
    </div>
  );
}

/**
 * @param {{
 *   samples: Record<string, number>, raw: string, isLoading: boolean,
 *   error: Error|null, onRefresh: ()=>void,
 * }} props
 */
export function MetricsPanel({ samples, raw, isLoading, error, onRefresh }) {
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold tracking-tight">Service counters</h2>
          <p className="mt-1 text-xs text-muted-foreground">
            Prometheus text from <span className="font-mono">GET /metrics</span>, parsed into
            counters. Values are per-process and reset when the API restarts.
          </p>
        </div>
        <Tooltip>
          <TooltipTrigger asChild>
            <Button type="button" variant="outline" size="sm" onClick={onRefresh} disabled={isLoading}>
              <RefreshCw className={cn(isLoading && "animate-spin")} />
              Refresh
            </Button>
          </TooltipTrigger>
          <TooltipContent>Re-fetch /metrics</TooltipContent>
        </Tooltip>
      </div>

      {error ? (
        <div className="rounded-xl border border-destructive/40 bg-destructive/10 p-4 text-sm">
          <p className="font-medium text-destructive">Could not read /metrics</p>
          <p className="mt-1 text-xs text-muted-foreground">{error.message || String(error)}</p>
        </div>
      ) : null}

      {isLoading && !raw ? <MetricsSkeleton /> : null}

      {!isLoading && !raw && !error ? (
        <div className="flex flex-col items-center justify-center rounded-xl border border-dashed border-border px-6 py-12 text-center">
          <Activity className="h-5 w-5 text-muted-foreground/60" />
          <p className="mt-3 text-sm font-medium">No metrics yet</p>
          <p className="mt-1 text-xs text-muted-foreground">
            The endpoint answered with an empty body.
          </p>
        </div>
      ) : null}

      {raw ? (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
          {STAT_CARDS.map((card) => {
            const value = samples[card.key];
            const missing = value === undefined;
            return (
              <StatTile
                key={card.key}
                label={card.label}
                value={missing ? "n/a" : card.format(value)}
                hint={missing ? "not present in the exposition" : card.hint}
                tone={missing ? "muted" : card.tone ? card.tone(value) : "default"}
              />
            );
          })}
        </div>
      ) : null}

      {raw ? (
        <div className="overflow-hidden rounded-xl border border-border bg-card/50">
          <div className="flex items-center justify-between gap-2 border-b border-border px-5 py-3">
            <h3 className="text-sm font-semibold tracking-tight">Raw exposition</h3>
            <span className="meta-label">{raw.split(/\r?\n/).filter(Boolean).length} lines</span>
          </div>
          <ScrollArea className="h-72">
            <pre className="scrollbar-thin whitespace-pre p-5 font-mono text-[11px] leading-relaxed text-muted-foreground">
              {raw}
            </pre>
          </ScrollArea>
        </div>
      ) : null}
    </div>
  );
}
