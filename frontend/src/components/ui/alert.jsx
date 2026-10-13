import * as React from "react";
import { cva } from "class-variance-authority";

import { cn } from "@/lib/utils";

const alertVariants = cva(
  "relative w-full rounded-xl border p-4 text-sm [&>svg]:h-4 [&>svg]:w-4",
  {
    variants: {
      variant: {
        default: "border-border bg-card/50 text-foreground [&>svg]:text-muted-foreground",
        destructive: "border-destructive/40 bg-destructive/10 text-foreground [&>svg]:text-destructive",
        warning: "border-amber-500/40 bg-amber-500/10 text-foreground [&>svg]:text-amber-500",
        success:
          "border-emerald-500/40 bg-emerald-500/10 text-foreground [&>svg]:text-emerald-500",
      },
    },
    defaultVariants: {
      variant: "default",
    },
  }
);

const Alert = React.forwardRef(({ className, variant, ...props }, ref) => (
  <div ref={ref} role="alert" className={cn(alertVariants({ variant }), className)} {...props} />
));
Alert.displayName = "Alert";

const AlertTitle = React.forwardRef(({ className, ...props }, ref) => (
  <h5
    ref={ref}
    className={cn("mb-1 font-medium leading-none tracking-tight", className)}
    {...props}
  />
));
AlertTitle.displayName = "AlertTitle";

const AlertDescription = React.forwardRef(({ className, ...props }, ref) => (
  <div ref={ref} className={cn("text-sm text-muted-foreground [&_p]:leading-relaxed", className)} {...props} />
));
AlertDescription.displayName = "AlertDescription";

export { Alert, AlertTitle, AlertDescription };
