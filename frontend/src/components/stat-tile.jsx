import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";

/**
 * Shared small pieces used by the answer card, evaluation and metrics panels.
 * Values are always monospace: this is a metrics-heavy UI.
 */

/** A metadata tile: small uppercase label + monospace value. */
export function StatTile({ label, value, hint, tone = "default", className }) {
  const toneClass = {
    default: "text-foreground",
    success: "text-emerald-500",
    warning: "text-amber-500",
    danger: "text-destructive",
    muted: "text-muted-foreground",
  }[tone];

  return (
    <div className={cn("min-w-0 rounded-lg border border-border bg-background/40 px-3 py-2", className)}>
      <p className="meta-label truncate">{label}</p>
      <p className={cn("mt-1 truncate font-mono text-sm", toneClass)} title={typeof value === "string" ? value : undefined}>
        {value}
      </p>
      {hint ? <p className="mt-0.5 truncate text-[10px] text-muted-foreground/70">{hint}</p> : null}
    </div>
  );
}

/**
 * Honesty badges. These states are normal in this project and are never hidden:
 * unavailable relevance, unchecked grounding, externally sourced answers.
 */
export function UncertaintyBadges({ relevanceSource, groundingChecked, answer, className }) {
  const relevanceUnknown = !relevanceSource || relevanceSource === "unavailable";
  const external = typeof answer === "string" && answer.trimStart().startsWith("[EXTERNAL / UNVERIFIED]");

  return (
    <div className={cn("flex flex-wrap items-center gap-1.5", className)}>
      {external ? (
        <Badge variant="warning" title="The model answered from outside the indexed corpus.">
          external / unverified
        </Badge>
      ) : null}
      {relevanceUnknown ? (
        <Badge variant="warning" title="The relevance signal did not come back from the retriever.">
          relevance unknown
        </Badge>
      ) : null}
      {!groundingChecked ? (
        <Badge variant="warning" title="The grounding check did not run for this answer.">
          grounding NOT CHECKED
        </Badge>
      ) : null}
    </div>
  );
}
