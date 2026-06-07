// Implements SPEC-0027 SS 6 (Vitest setup).
import "@testing-library/jest-dom/vitest";
import { afterEach } from "vitest";
import { cleanup } from "@testing-library/react";

import { useStore } from "../store";

afterEach(() => {
  cleanup();
  // Each AC test starts from a clean UI state (the store is module-global).
  useStore.getState().reset();
});
