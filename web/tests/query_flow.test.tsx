// Implements SPEC-0027 SS 5 AC1 (query flow: one QueryRequest, rank-ordered grid).
import { fireEvent, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { App } from "../src/App";
import { makeResponse, makeServices, renderApp } from "../src/test/utils";

describe("AC1: query flow", () => {
  it("sends one QueryRequest with selected lanes/fusion and renders results in rank order", async () => {
    const response = makeResponse(3);
    const services = makeServices({ query: response });
    renderApp(<App />, services);

    fireEvent.change(screen.getByTestId("query-input"), {
      target: { value: "canh quay bien dem" },
    });
    fireEvent.click(screen.getByTestId("search-btn"));

    const thumbs = await screen.findAllByTestId("thumb");

    expect(services.channel.send).toHaveBeenCalledTimes(1);
    expect(services.channel.send).toHaveBeenCalledWith(
      expect.objectContaining({
        query_vi: "canh quay bien dem",
        lanes: ["siglip2"],
        fusion: "single",
        top_k: 48,
        rrf_k: 60,
      }),
    );
    expect(thumbs.map((t) => t.dataset.pk)).toEqual(response.results.map((r) => r.pk));
  });
});
