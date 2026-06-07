// Implements SPEC-0027 SS 5 AC7 (WS error -> reconnecting state + REST fallback).
import { fireEvent, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { App } from "../src/App";
import { makeResponse, makeServices, renderApp } from "../src/test/utils";

describe("AC7: WebSocket fallback", () => {
  it("shows a reconnecting state and completes the query via REST when the channel errors", async () => {
    const restResponse = makeResponse(2);
    const services = makeServices({
      query: restResponse,
      channelSend: async () => {
        throw new Error("websocket error");
      },
    });
    renderApp(<App />, services);

    fireEvent.change(screen.getByTestId("query-input"), { target: { value: "duong pho" } });
    fireEvent.click(screen.getByTestId("search-btn"));

    await waitFor(() => expect(screen.getByTestId("ws-reconnecting")).toBeInTheDocument());
    const thumbs = await screen.findAllByTestId("thumb");

    expect(services.channel.send).toHaveBeenCalledTimes(1);
    expect(services.api.query).toHaveBeenCalledTimes(1);
    expect(thumbs.map((t) => t.dataset.pk)).toEqual(restResponse.results.map((r) => r.pk));
  });
});
