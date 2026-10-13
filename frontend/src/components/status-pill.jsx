import { RefreshCw } from "lucide-react";

import { cn } from "@/lib/utils";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";

/**
 * The colour language of the whole app:
 *   emerald = ok, amber = degraded / not verified, red = error / offline.
 */
const DOT_STYLES = {
  ok: "bg-emerald-500",
  degraded: "bg-amber-500",
  error: "bg-red-500",
  unknown: "bg-muted-foreground",
};

export function StatusDot({ status, pulse = false, className }) {
  return (
    <span className={cn("relative inline-flex h-2 w-2 shrink-0", className)}>
      {pulse ? (
        <span
          className={cn(
            "absolute inline-flex h-full w-full animate-ping rounded-full opacity-60",
            DOT_STYLES[status] || DOT_STYLES.unknown
          )}
        />
      ) : null}
      <span
        className={cn(
          "relative inline-flex h-2 w-2 rounded-full",
          DOT_STYLES[status] || DOT_STYLES.unknown
        )}
      />
    </span>
  );
}

const PILL_STYLES = {
  ok: "border-emerald-500/30 bg-emerald-500/10 text-emerald-500",
  degraded: "border-amber-500/30 bg-amber-500/10 text-amber-500",
  offline: "border-red-500/30 bg-red-500/10 text-red-500",
  checking: "border-border bg-card/50 text-muted-foreground",
};

const PILL_LABELS = {
  ok: "API online",
  degraded: "API degraded",
  offline: "API offline",
  checking: "Checking...",
};

/**
 * Live status pill driven entirely by /health. Clicking it re-probes.
 * `state` is one of "ok" | "degraded" | "offline" | "checking".
 */
export function StatusPill({ state, service, version, lastCheckedAt, isRefreshing, onRecheck }) {
  const styleKey = PILL_STYLES[state] ? state : "checking";
  const label = PILL_LABELS[styleKey];
  const checked = lastCheckedAt
    ? `checked ${lastCheckedAt.toLocaleTimeString()}`
    : "not checked yet";

  const tooltip =
    state === "offline"
      ? "The API did not answer. Click to re-check."
      : `${service || "IncidentLens RAG API"}${version ? ` v${version}` : ""} - ${checked}. Click to re-check.`;

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          onClick={onRecheck}
          aria-label={label}
          className={cn(
            "inline-flex h-7 items-center gap-2 rounded-full border px-3 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring",
            PILL_STYLES[styleKey]
          )}
        >
          <StatusDot status={styleKey === "checking" ? "unknown" : state} pulse={styleKey === "checking"} />
          <span>{label}</span>
          {isRefreshing && styleKey !== "checking" ? (
            <RefreshCw className="h-3 w-3 animate-spin opacity-70" />
          ) : null}
        </button>
      </TooltipTrigger>
      <TooltipContent className="max-w-xs">{tooltip}</TooltipContent>
    </Tooltip>
  );
}
