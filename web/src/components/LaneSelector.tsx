// Implements SPEC-0027 SS 4 (lane selector, AC5).
import type { Lane } from "../api/types";
import { useStore } from "../store";

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
    <fieldset className="flex gap-3 items-center" data-testid="lane-selector">
      <legend className="sr-only">Chon lane</legend>
      {ALL_LANES.map((lane) => (
        <label key={lane} className="flex gap-1 items-center">
          <input
            type="checkbox"
            data-testid={`lane-${lane}`}
            checked={lanes.includes(lane)}
            onChange={() => toggle(lane)}
          />
          {lane}
        </label>
      ))}
      <span data-testid="fusion-mode" className="text-sm opacity-70">
        fusion: {fusion}
      </span>
    </fieldset>
  );
}
