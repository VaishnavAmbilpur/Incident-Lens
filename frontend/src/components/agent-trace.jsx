import { Check, CircleDashed } from "lucide-react";

import { cn } from "@/lib/utils";
import { Skeleton } from "@/components/ui/skeleton";

/**
 * Human labels for the LangGraph node names, taken from the `add_node` calls in
 * modules/19_langgraph_agent.py:
 *   security_gate, retrieve, grade_relevance, rewrite_query, synthesize,
 *   check_hallucination, sanitize
 * Any name not listed falls through to a de-underscored version of itself, so a renamed or
 * newly added node still renders sensibly.
 */
const NODE_LABELS = {
  security_gate: "Security gate",
  retrieve: "Retrieve",
  grade_relevance: "Grade relevance",
  rewrite_query: "Rewrite query",
  synthesize: "Synthesize",
  check_hallucination: "Check hallucination",
  sanitize: "Sanitize",
};

export function nodeLabel(node) {
  if (!node) return "node";
  return NODE_LABELS[node] || node.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
}

/**
 * One row of the agent timeline: dot + connector, node name, and the step text the node
 * appended to graph state. The live node pulses; finished nodes show a check.
 *
 * @param {{ entry: {node: string, steps: string[], live: boolean, index: number}, isLast: boolean }} props
 */
function TraceRow({ entry, isLast }) {
  return (
    <li className="relative flex gap-3 pb-5 last:pb-0 animate-trace-in">
      {/* Connector rail */}
      {!isLast ? (
        <span className="absolute left-[5px] top-4 h-full w-px bg-border" aria-hidden="true" />
      ) : null}

      <span className="relative z-10 mt-1 flex h-[11px] w-[11px] shrink-0 items-center justify-center">
        {entry.live ? (
          <span className="relative flex h-[11px] w-[11px]">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-primary opacity-60" />
            <span className="relative inline-flex h-[11px] w-[11px] rounded-full bg-primary" />
          </span>
        ) : (
          <span className="flex h-[11px] w-[11px] items-center justify-center rounded-full border border-border bg-card">
            <Check className="h-2 w-2 text-emerald-500" strokeWidth={3} />
          </span>
        )}
      </span>

      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-baseline gap-2">
          <span className="font-mono text-xs font-medium text-foreground">{entry.node}</span>
          <span className="meta-label">{nodeLabel(entry.node)}</span>
          {entry.live ? (
            <span className="text-[10px] font-medium uppercase tracking-[0.14em] text-primary">
              running
            </span>
          ) : null}
        </div>

        {entry.steps.length > 0 ? (
          <ul className="mt-1.5 space-y-1">
            {entry.steps.map((step, i) => (
              <li key={`${entry.node}-${i}`} className="text-xs leading-relaxed text-muted-foreground">
                {step}
              </li>
            ))}
          </ul>
        ) : (
          <p className="mt-1.5 text-xs italic text-muted-foreground/70">
            No step text reported for this node.
          </p>
        )}
      </div>
    </li>
  );
}

/**
 * Live vertical timeline of `{"event":"node"}` frames from POST /query/stream.
 *
 * @param {{ entries: Array, isStreaming: boolean, error: (Error|null) }} props
 */
export function AgentTrace({ entries, isStreaming, error }) {
  const isEmpty = entries.length === 0;

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between gap-2 pb-3">
        <h2 className="text-sm font-semibold tracking-tight">Agent trace</h2>
        <span className="meta-label">
          {entries.length === 0 ? "no nodes yet" : `${entries.length} node${entries.length === 1 ? "" : "s"}`}
        </span>
      </div>

      {isEmpty && isStreaming ? (
        // Skeletons, not a spinner: the shape of what is coming.
        <div className="space-y-4">
          {[0, 1, 2].map((i) => (
            <div key={i} className="flex gap-3">
              <Skeleton className="mt-1 h-[11px] w-[11px] rounded-full" />
              <div className="flex-1 space-y-2">
                <Skeleton className="h-3 w-32" />
                <Skeleton className="h-3 w-full max-w-md" />
              </div>
            </div>
          ))}
        </div>
      ) : null}

      {isEmpty && !isStreaming ? (
        <div className="flex flex-1 flex-col items-center justify-center rounded-lg border border-dashed border-border px-4 py-10 text-center">
          <CircleDashed className="h-5 w-5 text-muted-foreground/60" />
          <p className="mt-3 text-sm text-muted-foreground">No trace yet.</p>
          <p className="mt-1 max-w-xs text-xs text-muted-foreground/70">
            Ask a question with stream progress enabled and each agent node appears here as it
            finishes.
          </p>
        </div>
      ) : null}

      {!isEmpty ? (
        <ol className="scrollbar-thin -mr-2 max-h-[28rem] overflow-y-auto pr-2">
          {entries.map((entry, i) => (
            <TraceRow
              key={`${entry.node}-${i}`}
              entry={entry}
              isLast={i === entries.length - 1}
            />
          ))}
        </ol>
      ) : null}

      {error ? (
        <p className={cn("mt-3 rounded-md border border-destructive/40 bg-destructive/10 px-3 py-2 text-xs text-destructive")}>
          {error.message || String(error)}
        </p>
      ) : null}
    </div>
  );
}
