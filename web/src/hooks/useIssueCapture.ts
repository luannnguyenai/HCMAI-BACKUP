// Implements SPEC-0027 SS 4 (in-UI issue capture).
import { useCallback } from "react";

import type { IssueReport, IssueResponse } from "../api/types";
import { useServices } from "../services";
import { useStore } from "../store";

/**
 * Returns `report(note?)` that bundles the current query, lanes, fusion, the
 * pks currently shown, a PNG screenshot, and an ISO-8601 client timestamp into
 * an `IssueReport` and POSTs it to `/api/issues` (SPEC-0027 AC6).
 */
export function useIssueCapture(): (note?: string) => Promise<IssueResponse> {
  const { api, captureScreenshot, now } = useServices();
  const query = useStore((s) => s.query);
  const lanes = useStore((s) => s.lanes);
  const fusion = useStore((s) => s.fusion);
  const results = useStore((s) => s.results);

  return useCallback(
    async (note?: string) => {
      const screenshot = await captureScreenshot();
      const report: IssueReport = {
        query_vi: query,
        lanes,
        fusion,
        returned_frame_ids: results.map((r) => r.pk),
        screenshot_png_b64: screenshot,
        client_timestamp: now(),
        note,
      };
      return api.reportIssue(report);
    },
    [api, captureScreenshot, now, query, lanes, fusion, results],
  );
}
