import { Moon, Sun } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

/**
 * Light/dark toggle. The theme itself lives in <html class="dark">; App owns the state
 * and persistence, this component only renders and reports intent.
 */
export function ThemeToggle({ theme, onToggle, className }) {
  const isDark = theme === "dark";
  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <Button
          type="button"
          variant="ghost"
          size="icon"
          onClick={onToggle}
          aria-label={isDark ? "Switch to light theme" : "Switch to dark theme"}
          className={cn("text-muted-foreground hover:text-foreground", className)}
        >
          {/* Both icons are rendered and cross-faded so the swap is animated, not abrupt. */}
          <Sun
            className={cn(
              "transition-all duration-200",
              isDark ? "rotate-90 scale-0 opacity-0" : "rotate-0 scale-100 opacity-100"
            )}
          />
          <Moon
            className={cn(
              "absolute transition-all duration-200",
              isDark ? "rotate-0 scale-100 opacity-100" : "-rotate-90 scale-0 opacity-0"
            )}
          />
        </Button>
      </TooltipTrigger>
      <TooltipContent>{isDark ? "Light theme" : "Dark theme"}</TooltipContent>
    </Tooltip>
  );
}
