// Implements SPEC-0027 SS 4 (top_k page-size control).
import { TOP_K_OPTIONS } from "../api/types";
import { useStore } from "../store";

export function TopKControl() {
  const topK = useStore((s) => s.topK);
  const setTopK = useStore((s) => s.setTopK);

  return (
    <div className="flex items-center gap-1.5" data-testid="topk-control">
      <span className="label mr-1">top_k</span>
      <div className="inline-flex overflow-hidden rounded-md border border-line">
        {TOP_K_OPTIONS.map((k) => {
          const active = topK === k;
          return (
            <button
              key={k}
              type="button"
              data-testid={`topk-${k}`}
              aria-pressed={active}
              onClick={() => setTopK(k)}
              className={`px-3 py-1.5 font-mono text-xs transition-colors ${
                active
                  ? "bg-accent text-ink-950 font-semibold"
                  : "bg-ink-900 text-fg-muted hover:bg-ink-800 hover:text-fg"
              }`}
            >
              {k}
            </button>
          );
        })}
      </div>
    </div>
  );
}
