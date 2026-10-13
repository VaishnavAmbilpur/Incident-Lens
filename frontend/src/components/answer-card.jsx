import { AlertTriangle, Check, Copy, Quote, ShieldAlert } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { StatTile, UncertaintyBadges } from "@/components/stat-tile";
import { useCopyToClipboard } from "@/hooks/useCopyToClipboard";
import { cn } from "@/lib/utils";

/** Format an optional 0..1 score for display without inventing precision. */
function formatScore(value) {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(3) : "-";
}

/**
 * The security gate result. A blocked query never reaches the LLM, which is the single
 * most important thing to tell an operator looking at this screen.
 */
function BlockedCard({ note }) {
  return (
    <div className="rounded-xl border border-destructive/40 bg-destructive/10 p-5">
      <div className="flex items-start gap-3">
        <ShieldAlert className="mt-0.5 h-5 w-5 shrink-0 text-destructive" />
        <div className="min-w-0">
          <h3 className="text-sm font-semibold tracking-tight text-destructive">
            Blocked by security gate
          </h3>
          <p className="mt-1.5 text-sm leading-relaxed text-foreground/90">
            {note || "The request was refused before reaching the agent."}
          </p>
          <p className="mt-2 text-xs text-muted-foreground">
            The gate runs first, so a blocked query costs <span className="font-mono">zero</span> LLM
            calls. Nothing was retrieved, synthesized or scored.
          </p>
        </div>
      </div>
    </div>
  );
}

/** Citations as monospace badges, wrapped so long chunk ids stay readable. */
function CitationList({ citations }) {
  if (!citations || citations.length === 0) {
    return <p className="text-xs text-muted-foreground">No citation markers were found in the answer.</p>;
  }
  return (
    <div className="flex flex-wrap gap-1.5">
      {citations.map((citation) => (
        <Tooltip key={citation}>
          <TooltipTrigger asChild>
            <Badge variant="outline" className="font-mono text-[11px]">
              <Quote className="h-2.5 w-2.5" />
              {citation}
            </Badge>
          </TooltipTrigger>
          <TooltipContent className="font-mono text-[11px]">source chunk: {citation}</TooltipContent>
        </Tooltip>
      ))}
    </div>
  );
}

/**
 * @param {{
 *   result: import("@/lib/api").QueryResponse | null,
 *   isStreaming: boolean,
 *   streamPartial: {answer: string, blocked: boolean, citations: string[], latency_ms: number} | null,
 * }} props
 */
export function AnswerCard({ result, isStreaming, streamPartial }) {
  const { copied, copy } = useCopyToClipboard();

  // Prefer the full POST /query payload; fall back to the answer frame from the SSE stream.
  const answer = result?.answer ?? streamPartial?.answer ?? null;
  const blocked = Boolean(result?.blocked ?? streamPartial?.blocked);
  const citations = result?.citations ?? streamPartial?.citations ?? [];
  const latency = result?.latency_ms ?? streamPartial?.latency_ms;
  const isStreamOnly = !result && Boolean(streamPartial);

  if (!answer && !blocked && !isStreaming) {
    return (
      <div className="flex h-full min-h-[16rem] flex-col items-center justify-center rounded-xl border border-dashed border-border px-6 py-12 text-center">
        <Quote className="h-5 w-5 text-muted-foreground/60" />
        <p className="mt-3 text-sm font-medium">No answer yet</p>
        <p className="mt-1 max-w-sm text-xs leading-relaxed text-muted-foreground">
          Ask about an incident post-mortem. The answer, its citations and the grounding verdict
          appear here. Answers are only ever as good as the retrieved chunks - check the badges.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {blocked ? <BlockedCard note={result?.security_note} /> : null}

      {!blocked && answer ? (
        <div className="rounded-xl border border-border bg-card/50">
          <div className="flex items-center gap-2 border-b border-border px-5 py-3">
            <h3 className="text-sm font-semibold tracking-tight">Answer</h3>
            {result?.thread_id ? (
              <span className="meta-label font-mono normal-case tracking-normal">
                thread {result.thread_id}
              </span>
            ) : null}
            <Tooltip>
              <TooltipTrigger asChild>
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="ml-auto text-muted-foreground hover:text-foreground"
                  onClick={() => copy(answer)}
                >
                  {copied ? <Check className="text-emerald-500" /> : <Copy />}
                  {copied ? "Copied" : "Copy"}
                </Button>
              </TooltipTrigger>
              <TooltipContent>Copy the answer text</TooltipContent>
            </Tooltip>
          </div>

          <div className="space-y-4 px-5 py-4">
            <UncertaintyBadges
              relevanceSource={result?.relevance_source}
              groundingChecked={result?.grounding_checked}
              answer={answer}
            />

            {/* whitespace-pre-wrap keeps the model's paragraph breaks without a markdown dep. */}
            <div className="whitespace-pre-wrap break-words text-sm leading-relaxed text-foreground/95">
              {answer}
            </div>

            {isStreamOnly ? (
              <p className="flex items-start gap-1.5 text-[11px] leading-relaxed text-muted-foreground">
                <AlertTriangle className="mt-px h-3 w-3 shrink-0 text-amber-500" />
                Streamed answer. /query/stream reports the answer, citations and latency only -
                relevance and grounding come from POST /query, so they are not scored here.
              </p>
            ) : (
              <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
                <StatTile label="Latency" value={latency != null ? `${latency} ms` : "-"} />
                <StatTile
                  label="Relevance"
                  value={formatScore(result?.relevance_score)}
                  hint={result?.relevance_source || "source unavailable"}
                  tone={
                    !result?.relevance_source || result?.relevance_source === "unavailable"
                      ? "warning"
                      : "default"
                  }
                />
                <StatTile
                  label="Grounding"
                  value={result?.grounding_checked ? formatScore(result?.grounding_score) : "NOT CHECKED"}
                  tone={result?.grounding_checked ? "default" : "warning"}
                />
                <StatTile label="Citations" value={String(citations.length)} />
              </div>
            )}
          </div>

          {/* A blocked query produces no answer, so citations would be empty noise. */}
          {!blocked ? (
            <div className="space-y-2 border-t border-border px-5 py-4">
              <p className="meta-label">Citations</p>
              <CitationList citations={citations} />
            </div>
          ) : null}

          {result?.path?.length ? (
            <div className="space-y-2 border-t border-border px-5 py-4">
              <p className="meta-label">Node path</p>
              <div className="flex flex-wrap items-center gap-1.5">
                {result.path.map((node, i) => (
                  <span key={`${node}-${i}`} className="flex items-center gap-1.5">
                    {i > 0 ? <span className="text-muted-foreground/50">-&gt;</span> : null}
                    <span className="rounded-md border border-border bg-background/40 px-2 py-0.5 font-mono text-[11px]">
                      {node}
                    </span>
                  </span>
                ))}
              </div>
            </div>
          ) : null}
        </div>
      ) : null}

      {isStreaming && !answer ? (
        <p className={cn("text-xs text-muted-foreground")}>
          Waiting for the final answer frame...
        </p>
      ) : null}
    </div>
  );
}
