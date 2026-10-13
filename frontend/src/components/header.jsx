import { Activity } from "lucide-react";

import { StatusPill } from "@/components/status-pill";
import { ThemeToggle } from "@/components/theme-toggle";
import { Separator } from "@/components/ui/separator";

/**
 * Sticky application header: identity, live health pill, theme toggle.
 * Backdrop blur keeps content legible as it scrolls underneath.
 */
export function Header({ health, theme, onToggleTheme }) {
  return (
    <header className="sticky top-0 z-40 border-b border-border bg-background/80 backdrop-blur supports-[backdrop-filter]:bg-background/60">
      <div className="mx-auto flex h-14 w-full max-w-7xl items-center gap-3 px-4 sm:px-6">
        <div className="flex items-center gap-2.5">
          <span className="flex h-7 w-7 items-center justify-center rounded-lg border border-border bg-card/60 text-primary">
            <Activity className="h-4 w-4" />
          </span>
          <div className="leading-none">
            <h1 className="text-sm font-semibold tracking-tight">IncidentLens</h1>
            <p className="meta-label mt-1 hidden sm:block">Grounded post-mortem answers</p>
          </div>
        </div>

        <div className="ml-auto flex items-center gap-2">
          <StatusPill
            state={health.state}
            service={health.data?.service}
            version={health.data?.version}
            lastCheckedAt={health.lastCheckedAt}
            isRefreshing={health.isRefreshing}
            onRecheck={health.recheck}
          />
          <Separator orientation="vertical" className="h-5" />
          <ThemeToggle theme={theme} onToggle={onToggleTheme} />
        </div>
      </div>
    </header>
  );
}
