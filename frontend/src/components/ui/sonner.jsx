import { Toaster as Sonner } from "sonner";

/**
 * shadcn's sonner wrapper.
 * Rich colors stay off and the palette is driven by our own CSS variables, so toasts
 * match the theme in both light and dark mode.
 */
const Toaster = ({ ...props }) => {
  return (
    <Sonner
      className="toaster group"
      position="bottom-right"
      closeButton
      toastOptions={{
        classNames: {
          toast:
            "group toast group-[.toaster]:bg-popover group-[.toaster]:text-popover-foreground group-[.toaster]:border-border group-[.toaster]:rounded-xl group-[.toaster]:shadow-none",
          description: "group-[.toast]:text-muted-foreground",
          actionButton:
            "group-[.toast]:bg-primary group-[.toast]:text-primary-foreground rounded-md",
          cancelButton:
            "group-[.toast]:bg-muted group-[.toast]:text-muted-foreground rounded-md",
          error: "group-[.toaster]:border-destructive/40",
          success: "group-[.toaster]:border-emerald-500/40",
        },
      }}
      {...props}
    />
  );
};

export { Toaster };
