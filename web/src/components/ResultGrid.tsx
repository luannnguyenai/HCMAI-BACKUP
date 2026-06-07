// Implements SPEC-0027 SS 4 (virtualised ranked keyframe grid, AC3).
import { forwardRef, type HTMLAttributes } from "react";
import { VirtuosoGrid } from "react-virtuoso";

import type { RankedFrame } from "../api/types";
import { useStore } from "../store";

// Initial mounted count for first paint / jsdom (well below the top_k ceiling
// of 500, so the grid never mounts all nodes -> AC3).
const GRID_INITIAL_ITEM_COUNT = 24;

const GridList = forwardRef<HTMLDivElement, HTMLAttributes<HTMLDivElement>>(
  function GridList({ children, ...props }, ref) {
    return (
      <div
        ref={ref}
        {...props}
        style={{ display: "flex", flexWrap: "wrap", gap: 8, ...props.style }}
      >
        {children}
      </div>
    );
  },
);

export function ResultGrid() {
  const results = useStore((s) => s.results);
  const status = useStore((s) => s.status);
  const selectFrame = useStore((s) => s.selectFrame);

  if (results.length === 0) {
    return (
      <div data-testid="no-results" className="py-8 text-center opacity-60">
        {status === "loading" ? "Dang tim..." : "Chua co ket qua."}
      </div>
    );
  }

  return (
    <VirtuosoGrid
      data-testid="result-grid"
      style={{ height: "70vh" }}
      data={results}
      totalCount={results.length}
      initialItemCount={Math.min(GRID_INITIAL_ITEM_COUNT, results.length)}
      components={{ List: GridList }}
      itemContent={(_index: number, frame: RankedFrame) => (
        <button
          data-testid="thumb"
          data-pk={frame.pk}
          className="border rounded overflow-hidden w-40"
          onClick={() => selectFrame(frame.pk)}
          title={`#${frame.rank} ${frame.pk} (${frame.score.toFixed(3)})`}
        >
          <img src={frame.thumb_url} alt={frame.pk} loading="lazy" className="w-full" />
          <span className="block text-xs px-1 py-0.5">
            #{frame.rank} {frame.video_id}/{frame.frame_id}
          </span>
        </button>
      )}
    />
  );
}
