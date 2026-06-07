// Implements SPEC-0027 SS 5 AC4 (frame detail on click; OCR/ASR hidden when null).
import { fireEvent, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { App } from "../src/App";
import { useStore } from "../src/store";
import { makeDetail, makeFrame, makeServices, renderApp } from "../src/test/utils";

describe("AC4: frame detail", () => {
  it("loads FrameDetail with video_id/frame_id + full image and hides null OCR/ASR rows", async () => {
    useStore.getState().setResults([makeFrame(1), makeFrame(2)], 5);
    const detail = makeDetail({ ocr_text: null, asr_text: null });
    const services = makeServices({ frameDetail: detail });
    renderApp(<App />, services);

    fireEvent.click(screen.getAllByTestId("thumb")[0]);

    const panel = await screen.findByTestId("frame-detail");
    expect(services.api.frameDetail).toHaveBeenCalledTimes(1);
    expect(within(panel).getByTestId("detail-ids").textContent).toContain(detail.video_id);
    expect(within(panel).getByTestId("detail-ids").textContent).toContain(detail.frame_id);
    expect(within(panel).getByRole("img")).toHaveAttribute("src", detail.full_url);
    expect(within(panel).queryByTestId("detail-ocr")).toBeNull();
    expect(within(panel).queryByTestId("detail-asr")).toBeNull();
  });

  it("shows the OCR row when ocr_text is present", async () => {
    useStore.getState().setResults([makeFrame(1)], 5);
    const detail = makeDetail({ ocr_text: "VTV1" });
    renderApp(<App />, makeServices({ frameDetail: detail }));

    fireEvent.click(screen.getAllByTestId("thumb")[0]);

    const panel = await screen.findByTestId("frame-detail");
    expect(within(panel).getByTestId("detail-ocr").textContent).toContain("VTV1");
  });
});
