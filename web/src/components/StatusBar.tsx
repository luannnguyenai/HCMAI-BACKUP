// Implements SPEC-0027 SS 4 (result/status strip: count, latency, echo, errors).
import { useStore } from "../store";

export function StatusBar() {
  const status = useStore((s) => s.status);
  const tookMs = useStore((s) => s.tookMs);
  const results = useStore((s) => s.results);
  const error = useStore((s) => s.error);
  const wsState = useStore((s) => s.wsState);
  const lastQuery = useStore((s) => s.query);

  return (
    <div className="flex min-h-[1.75rem] flex-wrap items-center gap-x-4 gap-y-1 text-xs">
      {tookMs != null && (
        <span data-testid="took-ms" className="flex items-center gap-3 text-fg-muted">
          <span>
            <span className="font-mono text-fg">{results.length}</span> ket qua
          </span>
          <span className="text-fg-faint">|</span>
          <span>
            <span className="font-mono text-accent">{Math.round(tookMs)}</span> ms
          </span>
        </span>
      )}

      {tookMs != null && lastQuery.trim() && (
        <span className="truncate text-fg-faint">
          truy van: <span className="text-fg-muted">"{lastQuery.trim()}"</span>
        </span>
      )}

      {wsState === "reconnecting" && (
        <span data-testid="ws-reconnecting" className="flex items-center gap-1.5 text-warn">
          <span className="h-1.5 w-1.5 rounded-full bg-warn animate-pulse-soft" aria-hidden="true" />
          WebSocket mat ket noi - dang dung REST
        </span>
      )}

      {status === "error" && error && (
        <span data-testid="query-error" className="flex items-center gap-1.5 text-bad" role="alert">
          <span className="h-1.5 w-1.5 rounded-full bg-bad" aria-hidden="true" />
          Loi: {error}
        </span>
      )}
    </div>
  );
}
