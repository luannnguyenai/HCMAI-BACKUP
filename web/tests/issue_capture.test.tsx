// Implements SPEC-0027 SS 5 AC6 (issue capture: full IssueReport POSTed to /api/issues).
import { fireEvent, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";

import { App } from "../src/App";
import { useStore } from "../src/store";
import { makeFrame, makeServices, renderApp } from "../src/test/utils";

describe("AC6: issue capture", () => {
  beforeEach(() => {
    useStore.getState().setQuery("nguoi dan ong ao do");
    useStore.getState().setResults([makeFrame(1), makeFrame(2), makeFrame(3)], 7);
  });

  it("produces an IssueReport with query/lanes/fusion/shown pks/screenshot/ISO timestamp", async () => {
    const services = makeServices({
      screenshot: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==",
      now: "2026-06-07T12:34:56.000Z",
    });
    renderApp(<App />, services);

    fireEvent.click(screen.getByTestId("report-issue-btn"));

    await waitFor(() => expect(services.api.reportIssue).toHaveBeenCalledTimes(1));
    const report = services.api.reportIssue.mock.calls[0][0];
    expect(report.query_vi).toBe("nguoi dan ong ao do");
    expect(report.lanes).toEqual(["siglip2"]);
    expect(report.fusion).toBe("single");
    expect(report.returned_frame_ids).toEqual([
      makeFrame(1).pk,
      makeFrame(2).pk,
      makeFrame(3).pk,
    ]);
    expect(report.screenshot_png_b64.length).toBeGreaterThan(0);
    expect(report.client_timestamp).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}/);
  });
});
