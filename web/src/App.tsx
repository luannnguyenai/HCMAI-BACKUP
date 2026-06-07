// Implements SPEC-0027 SS 4 (KIS console composition).
import { FrameDetailPanel } from "./components/FrameDetailPanel";
import { HistoryList } from "./components/HistoryList";
import { IssueButton } from "./components/IssueButton";
import { LaneSelector } from "./components/LaneSelector";
import { QueryBox } from "./components/QueryBox";
import { ResultGrid } from "./components/ResultGrid";
import { useStore } from "./store";

export function App() {
  const wsState = useStore((s) => s.wsState);
  const tookMs = useStore((s) => s.tookMs);
  const results = useStore((s) => s.results);

  return (
    <div className="max-w-6xl mx-auto p-4 space-y-3">
      <header className="flex items-center justify-between gap-4">
        <h1 className="text-lg font-semibold">AIC2026 - KIS console (MVP)</h1>
        {wsState === "reconnecting" && (
          <span data-testid="ws-reconnecting" className="text-amber-600 text-sm">
            WebSocket mat ket noi - dang dung REST
          </span>
        )}
      </header>

      <QueryBox />
      <div className="flex items-center justify-between gap-4 flex-wrap">
        <LaneSelector />
        <IssueButton />
      </div>
      <HistoryList />

      {tookMs != null && (
        <div data-testid="took-ms" className="text-xs opacity-60">
          {results.length} ket qua - {tookMs} ms
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-[2fr_1fr] gap-4">
        <ResultGrid />
        <FrameDetailPanel />
      </div>
    </div>
  );
}
