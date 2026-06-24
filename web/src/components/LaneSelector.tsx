// Implements SPEC-0027 SS 4 (lane selector, AC5).
import type { Lane } from "../api/types";
import { useStore } from "../store";

// siglip2 is online today; metaclip2 is wired and future-proofed (selecting it
// flips fusion single -> rrf). Other lanes are offline-only and not shown.
const ALL_LANES: Lane[] = ["siglip2", "metaclip2"];

export function LaneSelector() {
  const lanes = useStore((s) => s.lanes);
  const fusion = useStore((s) => s.fusion);
  const setLanes = useStore((s) => s.setLanes);

  const toggle = (lane: Lane) => {
    const next = lanes.includes(lane)
      ? lanes.filter((l) => l !== lane)
      : [...lanes, lane];
    // Always keep at least one lane selected.
    setLanes(next.length === 0 ? [lane] : next);
  };

  return (
    <fieldset className="flex items-center gap-1.5" data-testid="lane-selector">
      <legend className="sr-only">Chon lane</legend>
      <span className="label mr-1">lane</span>
      {ALL_LANES.map((lane) => {
        const active = lanes.includes(lane);
        return (
          <label
            key={lane}
            className={`flex cursor-pointer items-center gap-1.5 rounded-md border px-2.5 py-1.5 text-xs font-medium transition-colors ${
              active
                ? "border-accent/50 bg-accent/10 text-accent-strong"
                : "border-line bg-ink-900 text-fg-muted hover:border-line-strong hover:text-fg"
            }`}
          >
            <input
              type="checkbox"
              data-testid={`lane-${lane}`}
              className="sr-only"
              checked={active}
              onChange={() => toggle(lane)}
            />
            <span
              className={`h-1.5 w-1.5 rounded-full ${active ? "bg-accent" : "bg-fg-faint"}`}
              aria-hidden="true"
            />
            <span className="font-mono">{lane}</span>
          </label>
        );
      })}
      <span
        data-testid="fusion-mode"
        className="ml-1 font-mono text-[11px] text-fg-faint"
        title="Hai lane -> RRF, mot lane -> single"
      >
        fusion: <span className="text-fg-muted">{fusion}</span>
      </span>
    </fieldset>
  );
}
