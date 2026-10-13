import { Terminal } from "lucide-react";

import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { OFFLINE_HINT } from "@/lib/api";
import { cn } from "@/lib/utils";

/**
 * The offline state. The command below is the one from module 20's interpreter guard,
 * so it is exactly what the operator needs to paste into a shell.
 */
export function OfflineNotice({ className, message }) {
  return (
    <div className={cn("rounded-xl border border-destructive/40 bg-destructive/10 p-5", className)}>
      <h3 className="text-sm font-semibold tracking-tight text-destructive">
        API offline
      </h3>
      <p className="mt-1.5 text-sm leading-relaxed text-muted-foreground">
        {message ||
          "The UI could not reach the IncidentLens API. Nothing else here works without it - start the service, then re-check the status pill."}
      </p>
      <p className="meta-label mt-4">Start it from the project root</p>
      <div className="mt-1.5 flex items-center gap-2 rounded-lg border border-border bg-background/60 px-3 py-2">
        <Terminal className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
        <code className="scrollbar-thin overflow-x-auto whitespace-pre font-mono text-xs text-foreground">
          {OFFLINE_HINT}
        </code>
      </div>
      <p className="mt-2 text-[11px] text-muted-foreground/80">
        Default address <span className="font-mono">http://127.0.0.1:8000</span>. Point the UI
        elsewhere with <span className="font-mono">VITE_API_BASE</span> in a{" "}
        <span className="font-mono">.env</span> file.
      </p>
    </div>
  );
}

/**
 * Health tab: the six dependency checks with status dots and their detail text.
 * `health` is the object returned by useHealth().
 */
export function HealthPanel({ health }) {
  const { state, data, error, lastCheckedAt, isRefreshing, recheck } = health;

  if (state === "offline") {
    return (
      <div className="space-y-5">
        <OfflineNotice message={error?.message} />
        <button
          type="button"
          onClick={recheck}
          className="text-xs text-muted-foreground underline underline-offset-4 hover:text-foreground"
        >
          Re-check now
        </button>
      </div>
    );
  }

  const checks = data?.checks || [];
  const summaryTone =
    state === "ok"
      ? "border-emerald-500/40 bg-emerald-500/10"
      : "border-amber-500/40 bg-amber-500/10";

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold tracking-tight">Dependencies</h2>
          <p className="mt-1 text-xs text-muted-foreground">
            {data?.service || "IncidentLens RAG API"}
            {data?.version ? <span className="font-mono"> v{data.version}</span> : null}
            {data?.timestamp ? (
              <>
                {" "}
                - reported <span className="font-mono">{data.timestamp}</span>
              </>
            ) : null}
          </p>
        </div>
        <button
          type="button"
          onClick={recheck}
          disabled={isRefreshing}
          className="inline-flex h-8 items-center gap-2 rounded-md border border-border px-3 text-xs font-medium transition-colors hover:bg-accent disabled:opacity-50"
        >
          <span className={cn("h-1.5 w-1.5 rounded-full", isRefreshing ? "animate-pulse bg-primary" : "bg-muted-foreground")} />
          {isRefreshing ? "Re-checking..." : "Re-check"}
        </button>
      </div>

      <div className={cn("rounded-xl border p-4", summaryTone)}>
        <p className="text-sm">
          Overall status{" "}
          <span className={cn("font-mono", state === "ok" ? "text-emerald-500" : "text-amber-500")}>
            {data?.status || state}
          </span>
        </p>
        <p className="mt-1 text-xs text-muted-foreground">
          {checks.filter((c) => c.status !== "ok").length} of {checks.length || "?"} checks are not
          fully green.
          {lastCheckedAt ? ` Last polled ${lastCheckedAt.toLocaleTimeString()}.` : ""}
        </p>
      </div>

      <ul className="divide-y divide-border overflow-hidden rounded-xl border border-border bg-card/50">
        {checks.map((check) => {
          const tone =
            check.status === "ok" ? "bg-emerald-500" : check.status === "degraded" ? "bg-amber-500" : "bg-red-500";
          const textTone =
            check.status === "ok"
              ? "text-emerald-500"
              : check.status === "degraded"
                ? "text-amber-500"
                : "text-red-500";
          return (
            <li key={check.name} className="flex items-start gap-3 px-5 py-3.5">
              <span className={cn("mt-1.5 h-2 w-2 shrink-0 rounded-full", tone)} aria-hidden="true" />
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-baseline gap-2">
                  <span className="font-mono text-xs font-medium">{check.name}</span>
                  <span className={cn("text-[10px] font-medium uppercase tracking-[0.14em]", textTone)}>
                    {check.status}
                  </span>
                </div>
                <p className="mt-1 break-words text-xs leading-relaxed text-muted-foreground">
                  {check.detail}
                </p>
              </div>
            </li>
          );
        })}
        {checks.length === 0 ? (
          <li className="px-5 py-6 text-xs text-muted-foreground">
            The health endpoint returned no checks.
          </li>
        ) : null}
      </ul>

      <Alert>
        <AlertTitle className="text-xs">Why some checks read degraded</AlertTitle>
        <AlertDescription className="text-xs">
          "No database yet (nothing has run)" is expected on a fresh start - the SQLite thread
          memory file appears after the first query. A missing <span className="font-mono">GROQ_API_KEY</span>{" "}
          turns the credentials check red, and every query will fail until it is set.
        </AlertDescription>
      </Alert>
    </div>
  );
}
