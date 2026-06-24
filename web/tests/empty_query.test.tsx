// Implements SPEC-0027 SS 5 AC2 (empty/whitespace query is a no-op, grid kept).
import { fireEvent, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it } from "vitest";

import { App } from "../src/App";
import { useStore } from "../src/store";
import { makeFrame, makeServices, renderApp } from "../src/test/utils";

describe("AC2: empty query", () => {
  beforeEach(() => {
    // Seed a prior grid so we can assert it is preserved.
    useStore.getState().setResults([makeFrame(1), makeFrame(2)], 10);
  });

  it("does not issue a request and keeps the prior grid for a whitespace query", () => {
    const services = makeServices();
    renderApp(<App />, services);

    fireEvent.change(screen.getByTestId("query-input"), { target: { value: "   " } });
    expect(screen.getByTestId("search-btn")).toBeDisabled();
    fireEvent.submit(screen.getByTestId("query-form"));

    expect(services.channel.send).not.toHaveBeenCalled();
    expect(services.api.query).not.toHaveBeenCalled();
    expect(screen.getAllByTestId("thumb").length).toBe(2);
  });
});
