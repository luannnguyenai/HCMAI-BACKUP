// Implements SPEC-0027 SS 3 (Zustand UI state).
import { create } from "zustand";

import { DEFAULT_TOP_K, type FrameDetail, type FusionMode, type Lane, type RankedFrame } from "./api/types";

export type WsState = "connected" | "reconnecting";
export type Status = "idle" | "loading" | "error";

const HISTORY_LIMIT = 10; // Q3 RESOLVED: lightweight last-10 in-memory history.

export interface UiState {
  query: string;
  lanes: Lane[];
  fusion: FusionMode;
  topK: number;
  results: RankedFrame[];
  selectedPk: string | null;
  detail: FrameDetail | null;
  status: Status;
  tookMs: number | null;
  wsState: WsState;
  error: string | null;
  history: string[];

  setQuery: (q: string) => void;
  setLanes: (lanes: Lane[]) => void;
  setTopK: (k: number) => void;
  setResults: (results: RankedFrame[], tookMs: number) => void;
  setStatus: (s: Status) => void;
  setError: (e: string | null) => void;
  selectFrame: (pk: string | null) => void;
  // Step selection by +1 / -1 through the current result list (detail nav).
  stepSelection: (delta: number) => void;
  setWsState: (w: WsState) => void;
  pushHistory: (q: string) => void;
  reset: () => void;
}

export const INITIAL_STATE = {
  query: "",
  lanes: ["siglip2"] as Lane[],
  fusion: "single" as FusionMode,
  topK: DEFAULT_TOP_K,
  results: [] as RankedFrame[],
  selectedPk: null as string | null,
  detail: null as FrameDetail | null,
  status: "idle" as Status,
  tookMs: null as number | null,
  wsState: "connected" as WsState,
  error: null as string | null,
  history: [] as string[],
};

// Fusion is derived from the lane count: two lanes -> RRF, one -> single
// (SPEC-0027 AC5). qwen3vl is offline-only and never selectable here.
function deriveFusion(lanes: Lane[]): FusionMode {
  return lanes.length >= 2 ? "rrf" : "single";
}

export const useStore = create<UiState>((set) => ({
  ...INITIAL_STATE,
  setQuery: (query) => set({ query }),
  setLanes: (lanes) => set({ lanes, fusion: deriveFusion(lanes) }),
  setTopK: (topK) => set({ topK }),
  setResults: (results, tookMs) => set({ results, tookMs, error: null }),
  setStatus: (status) => set({ status }),
  setError: (error) => set({ error }),
  selectFrame: (selectedPk) => set({ selectedPk }),
  stepSelection: (delta) =>
    set((s) => {
      if (s.results.length === 0) return {};
      const idx = s.results.findIndex((r) => r.pk === s.selectedPk);
      if (idx === -1) return { selectedPk: s.results[0].pk };
      const next = Math.min(Math.max(idx + delta, 0), s.results.length - 1);
      return { selectedPk: s.results[next].pk };
    }),
  setWsState: (wsState) => set({ wsState }),
  pushHistory: (q) =>
    set((s) => ({
      history: [q, ...s.history.filter((h) => h !== q)].slice(0, HISTORY_LIMIT),
    })),
  reset: () => set({ ...INITIAL_STATE }),
}));
