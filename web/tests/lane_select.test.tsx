// Implements SPEC-0027 SS 5 AC5 (lane selector drives lanes/fusion on next query).
import { fireEvent, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import { App } from "../src/App";
import { makeServices, renderApp } from "../src/test/utils";

describe("AC5: lane selection", () => {
  it("switches single -> rrf when a second lane is chosen and sends it on next query", async () => {
    const services = makeServices();
    renderApp(<App />, services);

    expect(screen.getByTestId("fusion-mode").textContent).toContain("single");

    fireEvent.click(screen.getByTestId("lane-metaclip2"));
    expect(screen.getByTestId("fusion-mode").textContent).toContain("rrf");

    fireEvent.change(screen.getByTestId("query-input"), { target: { value: "xe cuu hoa" } });
    fireEvent.click(screen.getByTestId("search-btn"));
    await screen.findAllByTestId("thumb");

    expect(services.channel.send).toHaveBeenCalledWith(
      expect.objectContaining({
        lanes: ["siglip2", "metaclip2"],
        fusion: "rrf",
      }),
    );
  });
});
