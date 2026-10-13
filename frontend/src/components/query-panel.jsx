import { useEffect, useRef } from "react";
import { CornerDownLeft, Loader2, Send, Square } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

const PLACEHOLDERS = [
  "What was the root cause of the checkout incident?",
  "Which change preceded the database failover, and how was it rolled back?",
  "What monitoring gap let this incident run undetected for 40 minutes?",
];

/**
 * The Ask panel: question, conversation id, stream switch, submit.
 *
 * Cmd/Ctrl+Enter submits (the textarea keeps plain Enter for newlines).
 */
export function QueryPanel({
  question,
  onQuestionChange,
  threadId,
  onThreadIdChange,
  useStream,
  onUseStreamChange,
  onSubmit,
  onCancel,
  isRunning,
  disabled,
  disabledReason,
}) {
  const textareaRef = useRef(null);

  // Focus the textarea on mount: this is the primary action of the whole app.
  useEffect(() => {
    textareaRef.current?.focus();
  }, []);

  const handleKeyDown = (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") {
      event.preventDefault();
      if (!isRunning && !disabled) onSubmit();
    }
  };

  const canSubmit = Boolean(question.trim()) && !isRunning && !disabled;

  return (
    <form
      className="rounded-xl border border-border bg-card/50 p-4 sm:p-5"
      onSubmit={(event) => {
        event.preventDefault();
        if (canSubmit) onSubmit();
      }}
    >
      <div className="flex items-baseline justify-between gap-3">
        <Label htmlFor="question" className="meta-label">
          Question
        </Label>
        <span className="hidden text-[10px] text-muted-foreground/70 sm:block">
          Ctrl/Cmd + Enter to send
        </span>
      </div>

      <Textarea
        id="question"
        ref={textareaRef}
        value={question}
        onChange={(e) => onQuestionChange(e.target.value)}
        onKeyDown={handleKeyDown}
        placeholder={PLACEHOLDERS[0]}
        maxLength={2000}
        rows={3}
        disabled={disabled}
        className="mt-2 min-h-[104px] text-[15px]"
        aria-label="Question about an incident post-mortem"
      />

      <div className="mt-3 grid gap-3 sm:grid-cols-[minmax(0,1fr)_auto] sm:items-end">
        <div className="space-y-1.5">
          <Label htmlFor="thread-id" className="meta-label">
            Conversation id (optional)
          </Label>
          <Input
            id="thread-id"
            value={threadId}
            onChange={(e) => onThreadIdChange(e.target.value)}
            placeholder="auto-filled after the first answer"
            maxLength={100}
            disabled={disabled}
            className="font-mono text-xs"
          />
          <p className="text-[10px] text-muted-foreground/70">
            Reuse it to keep the agent's memory across follow-up questions.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <Tooltip>
            <TooltipTrigger asChild>
              <div className="flex items-center gap-2">
                <Switch
                  id="stream"
                  checked={useStream}
                  onCheckedChange={onUseStreamChange}
                  disabled={disabled}
                />
                <Label htmlFor="stream" className="cursor-pointer whitespace-nowrap">
                  Stream progress
                </Label>
              </div>
            </TooltipTrigger>
            <TooltipContent className="max-w-xs">
              Off: one POST /query that returns the full scorecard. On: POST /query/stream and the
              agent trace appears node by node.
            </TooltipContent>
          </Tooltip>

          {isRunning ? (
            <Button type="button" variant="outline" onClick={onCancel} className="shrink-0">
              <Square className="fill-current" />
              Stop
            </Button>
          ) : (
            <Button type="submit" disabled={!canSubmit} className="shrink-0">
              {isRunning ? <Loader2 className="animate-spin" /> : <Send />}
              Ask
            </Button>
          )}
        </div>
      </div>

      {disabled ? (
        <p className="mt-3 rounded-md border border-amber-500/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-500">
          {disabledReason || "The API is unreachable, so asking is disabled."}
        </p>
      ) : null}

      <div className="mt-3 flex items-center gap-2 text-[10px] text-muted-foreground/70">
        <CornerDownLeft className="h-3 w-3" />
        <span className={cn("truncate")}>
          Blocked queries are refused by the security gate before any LLM call.
        </span>
      </div>
    </form>
  );
}
