// Implements SPEC-0027 SS 4 (virtualised ranked keyframe grid, AC3).
import { forwardRef, type HTMLAttributes } from "react";
import { VirtuosoGrid } from "react-virtuoso";

import type { RankedFrame } from "../api/types";
import { useStore } from "../store";

// Initial mounted count for first paint / jsdom (well below the top_k ceiling
// of 500, so the grid never mounts all nodes -> AC3).
const GRID_INITIAL_ITEM_COUNT = 24;
const SKELETON_COUNT = 18;

const GridList = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
  function GridList({ children, style, ...props }, ref) {
    return (
      <div
        ref={ref}
        {...props}
        style={style}
        className="grid grid-cols-[repeat(auto-fill,minmax(150px,1fr))] gap-2.5 p-0.5"
      >
        {children}
      </div>
    );
  },
);

function ScoreBadge({ frame }: { frame: RankedFrame }) {
  return (
    <span className="rounded bg-ink-950/80 px-1.5 py-0.5 font-mono text-[10px] text-accent-strong ring-1 ring-inset ring-white/5">
      {frame.score.toFixed(3)}
    </span>
  );
}

export function ResultGrid() {
  const results = useStore((s) => s.results);
  const status = useStore((s) => s.status);
  const selectedPk = useStore((s) => s.selectedPk);
  const selectFrame = useStore((s) => s.selectFrame);

  if (results.length === 0) {
    if (status === "loading") {
      return (
        <div
          data-testid="grid-skeleton"
          className="grid grid-cols-[repeat(auto-fill,minmax(150px,1fr))] gap-2.5"
        >
          {Array.from({ length: SKELETON_COUNT }).map((_, i) => (
            <div key={i} className="skeleton aspect-video rounded-md" />
          ))}
        </div>
      );
    }
    return (
      <div
        data-testid="no-results"
        className="flex flex-col items-center justify-center gap-2 rounded-lg border border-dashed border-line py-20 text-center"
      >
        <div className="text-3xl opacity-30" aria-hidden="true">
          &#9906;
        </div>
        <div className="text-sm font-medium text-fg-muted">Chua co ket qua</div>
        <div className="max-w-xs text-xs text-fg-faint">
          Nhap mo ta tieng Viet roi nhan Enter de tim khung hinh.
        </div>
      </div>
    );
  }

  return (
    <VirtuosoGrid
      data-testid="result-grid"
      style={{ height: "100%" }}
      data={results}
      totalCount={results.length}
      initialItemCount={Math.min(GRID_INITIAL_ITEM_COUNT, results.length)}
      components={{ List: GridList }}
      itemContent={(_index: number, frame: RankedFrame) => {
        const selected = frame.pk === selectedPk;
        return (
          <button
            data-testid="thumb"
            data-pk={frame.pk}
            aria-pressed={selected}
            className={`group relative block aspect-video w-full overflow-hidden rounded-md border bg-ink-900 text-left transition-all ${
              selected
                ? "border-accent shadow-glow"
                : "border-line hover:border-line-strong hover:-translate-y-0.5"
            }`}
            onClick={() => selectFrame(frame.pk)}
            title={`#${frame.rank} ${frame.pk} (${frame.score.toFixed(3)})`}
          >
            <img
              src={frame.thumb_url}
              alt={frame.pk}
              loading="lazy"
              className="h-full w-full object-cover transition-transform duration-200 group-hover:scale-[1.03]"
            />
            <span className="absolute left-1.5 top-1.5 rounded bg-ink-950/80 px-1.5 py-0.5 font-mono text-[10px] font-semibold text-fg ring-1 ring-inset ring-white/5">
              #{frame.rank}
            </span>
            <span className="absolute right-1.5 top-1.5">
              <ScoreBadge frame={frame} />
            </span>
            <span className="absolute inset-x-0 bottom-0 truncate bg-gradient-to-t from-ink-950/95 via-ink-950/70 to-transparent px-1.5 pb-1 pt-4 font-mono text-[10px] text-fg-muted">
              {frame.video_id}/{frame.frame_id}
            </span>
          </button>
        );
      }}
    />
  );
}
