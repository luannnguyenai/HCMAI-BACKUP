// Implements SPEC-0027 SS 5 AC3 (grid virtualised: 200 results -> bounded mounts).
import { screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { App } from "../src/App";
import { useStore } from "../src/store";
import { makeFrame, makeServices, renderApp } from "../src/test/utils";

describe("AC3: grid virtualisation", () => {
  it("mounts fewer than 200 thumbnail nodes for 200 results", () => {
    const frames = Array.from({ length: 200 }, (_, i) => makeFrame(i + 1));
    useStore.getState().setResults(frames, 99);

    renderApp(<App />, makeServices());

    const mounted = screen.getAllByTestId("thumb");
    expect(mounted.length).toBeGreaterThan(0);
    expect(mounted.length).toBeLessThan(200);
  });
});
