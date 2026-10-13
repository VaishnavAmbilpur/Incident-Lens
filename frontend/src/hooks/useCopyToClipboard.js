import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Copy-to-clipboard with a short-lived "copied" confirmation flag.
 * Uses the async Clipboard API and falls back to a hidden textarea when the
 * page is not served over a secure context.
 */
export function useCopyToClipboard(resetAfterMs = 1600) {
  const [copied, setCopied] = useState(false);
  const timerRef = useRef(null);

  useEffect(() => () => window.clearTimeout(timerRef.current), []);

  const copy = useCallback(
    async (text) => {
      const value = String(text ?? "");
      if (!value) return false;
      try {
        if (navigator.clipboard && window.isSecureContext) {
          await navigator.clipboard.writeText(value);
        } else {
          const el = document.createElement("textarea");
          el.value = value;
          el.setAttribute("readonly", "");
          el.style.position = "fixed";
          el.style.opacity = "0";
          document.body.appendChild(el);
          el.select();
          document.execCommand("copy");
          document.body.removeChild(el);
        }
        setCopied(true);
        window.clearTimeout(timerRef.current);
        timerRef.current = window.setTimeout(() => setCopied(false), resetAfterMs);
        return true;
      } catch {
        return false;
      }
    },
    [resetAfterMs]
  );

  return { copied, copy };
}
