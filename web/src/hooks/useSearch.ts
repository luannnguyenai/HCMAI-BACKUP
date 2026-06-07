// Implements SPEC-0027 SS 4 (run a query: WS hot path + REST fallback).
import { useCallback } from "react";

import { DEFAULT_RRF_K, DEFAULT_TOP_K, type QueryRequest } from "../api/types";
import { useServices } from "../services";
import { useStore } from "../store";

/**
 * Returns a `run()` that submits the current query. An empty / whitespace query
 * is a no-op and keeps the prior grid (AC2). The hot path is the WebSocket
 * channel; on any channel error the app flips to a "reconnecting" state and
 * completes the query via the REST fallback (AC7).
 */
export function useSearch(): () => Promise<void> {
  const { channel, api } = useServices();
  const query = useStore((s) => s.query);
  const lanes = useStore((s) => s.lanes);
  const fusion = useStore((s) => s.fusion);
  const setResults = useStore((s) => s.setResults);
  const setStatus = useStore((s) => s.setStatus);
  const setError = useStore((s) => s.setError);
  const setWsState = useStore((s) => s.setWsState);
  const pushHistory = useStore((s) => s.pushHistory);

  return useCallback(async () => {
    const q = query.trim();
    if (!q) return; // AC2: do not issue a request for an empty/blank query.

    const req: QueryRequest = {
      query_vi: q,
      lanes,
      top_k: DEFAULT_TOP_K,
      fusion,
      rrf_k: DEFAULT_RRF_K,
    };
    setStatus("loading");
    try {
      let resp;
      try {
        resp = await channel.send(req);
        setWsState("connected");
      } catch {
        // AC7: WS dropped -> show reconnecting + complete via REST.
        setWsState("reconnecting");
        resp = await api.query(req);
      }
      setResults(resp.results, resp.took_ms);
      setStatus("idle");
      pushHistory(q);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setStatus("error");
    }
  }, [query, lanes, fusion, channel, api, setResults, setStatus, setError, setWsState, pushHistory]);
}
