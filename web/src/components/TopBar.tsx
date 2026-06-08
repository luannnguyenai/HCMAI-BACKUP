// Implements SPEC-0027 SS 4 (top bar: wordmark, readiness, active lane, report).
import { useStore } from "../store";
import { ReadinessIndicator } from "./ReadinessIndicator";

export function TopBar({ onReport }: { onReport: () => void }) {
  const lanes = useStore((s) => s.lanes);
  const fusion = useStore((s) => s.fusion);

  return (
    <header className="sticky top-0 z-30 border-b border-line bg-ink-950/85 backdrop-blur supports-[backdrop-filter]:bg-ink-950/70">
      <div className="mx-auto flex max-w-[1600px] flex-wrap items-center gap-x-4 gap-y-2 px-4 py-2.5">
        <div className="flex items-center gap-2.5">
          <span className="grid h-7 w-7 place-items-center rounded-md bg-accent/15 font-mono text-sm font-semibold text-accent ring-1 ring-accent/30">
            K
          </span>
          <div className="leading-tight">
            <div className="text-sm font-semibold tracking-tight">
              AIC2026 <span className="text-fg-muted">KIS console</span>
            </div>
            <div className="text-[11px] text-fg-faint">Vietnamese keyframe retrieval - operator</div>
          </div>
        </div>

        <div className="flex flex-1 items-center justify-end gap-2.5">
          <span className="pill text-fg-muted" title="Lane dang dung va che do fusion">
            <span className="label">lane</span>
            <span className="font-mono text-fg">{lanes.join(" + ")}</span>
            <span className="text-fg-faint">/</span>
            <span className="font-mono text-accent">{fusion}</span>
          </span>
          <ReadinessIndicator />
          <button
            data-testid="report-issue-btn"
            type="button"
            onClick={onReport}
            className="btn"
            title="Bao loi ket qua truy van (Shift+R)"
          >
            Bao loi
          </button>
        </div>
      </div>
    </header>
  );
}
