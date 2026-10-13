import { clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/**
 * `cn` - the standard shadcn class merge helper.
 * clsx flattens conditionals; twMerge resolves conflicting Tailwind utilities
 * (so a caller-supplied `className` can override a component default).
 */
export function cn(...inputs) {
  return twMerge(clsx(inputs));
}
