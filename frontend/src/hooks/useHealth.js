import { useCallback, useEffect, useRef, useState } from "react";

import { fetchHealth } from "@/lib/api";

const POLL_INTERVAL_MS = 15000;

/**
 * Poll /health and expose a single `state` the whole shell can trust.
 *
 * states:
 *   "checking"  first probe in flight
 *   "ok"        status === "ok" and every dependency check is "ok"
 *   "degraded"  the service answered, but at least one check is degraded/error
 *   "offline"   the request never reached the service
 *
 * A failed poll never throws into React: the offline state is data, not an exception.
 */
export function useHealth() {
  const [state, setState] = useState("checking");
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [lastCheckedAt, setLastCheckedAt] = useState(null);
  const [isRefreshing, setIsRefreshing] = useState(false);

  // Keeps a poll that resolves after unmount from setting state.
  const mountedRef = useRef(true);
  const abortRef = useRef(null);

  const check = useCallback(async ({ silent = false } = {}) => {
    // Cancel an in-flight probe so overlapping clicks cannot race.
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    if (!silent) setIsRefreshing(true);
    try {
      const payload = await fetchHealth({ signal: controller.signal });
      if (!mountedRef.current || controller.signal.aborted) return;
      setData(payload);
      setError(null);
      setState(payload?.status === "ok" ? "ok" : "degraded");
      setLastCheckedAt(new Date());
    } catch (err) {
      if (!mountedRef.current || controller.signal.aborted) return;
      setError(err);
      setData(null);
      setState("offline");
      setLastCheckedAt(new Date());
    } finally {
      if (mountedRef.current) setIsRefreshing(false);
    }
  }, []);

  useEffect(() => {
    mountedRef.current = true;
    check({ silent: true });
    const id = window.setInterval(() => check({ silent: true }), POLL_INTERVAL_MS);
    return () => {
      mountedRef.current = false;
      window.clearInterval(id);
      abortRef.current?.abort();
    };
  }, [check]);

  return {
    state,
    data,
    error,
    lastCheckedAt,
    isRefreshing,
    recheck: () => check({ silent: false }),
    isOffline: state === "offline",
  };
}
