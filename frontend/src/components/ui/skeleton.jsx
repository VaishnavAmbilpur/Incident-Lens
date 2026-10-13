import { cn } from "@/lib/utils";

/** Shimmering placeholder used instead of spinners for content areas. */
function Skeleton({ className, ...props }) {
  return (
    <div
      className={cn("animate-pulse rounded-md bg-muted/70", className)}
      aria-hidden="true"
      {...props}
    />
  );
}

export { Skeleton };
