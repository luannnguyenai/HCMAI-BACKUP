// Implements SPEC-0027 SS 3 (injectable services: API, WS channel, screenshot).
import { createContext, useContext, type ReactNode } from "react";

import { createApiClient, type ApiClient } from "./api/client";
import { createQueryChannel, type QueryChannel } from "./api/ws";

export interface Services {
  api: ApiClient;
  channel: QueryChannel;
  // Capture a PNG screenshot of the current UI, returned as base64 (no data:
  // prefix). Injected so tests need not run html2canvas in jsdom (Q2: html2canvas).
  captureScreenshot: () => Promise<string>;
  // ISO-8601 clock; injectable so tests can assert a deterministic timestamp.
  now: () => string;
}

async function captureScreenshot(): Promise<string> {
  // Lazy-import so the heavy canvas lib is only pulled when a report is filed.
  const html2canvas = (await import("html2canvas")).default;
  const canvas = await html2canvas(document.body);
  const dataUrl = canvas.toDataURL("image/png");
  return dataUrl.replace(/^data:image\/png;base64,/, "");
}

export function createDefaultServices(secret?: string | null): Services {
  return {
    api: createApiClient({ secret }),
    channel: createQueryChannel({ secret }),
    captureScreenshot,
    now: () => new Date().toISOString(),
  };
}

const ServicesContext = createContext<Services | null>(null);

export function ServicesProvider({
  services,
  children,
}: {
  services: Services;
  children: ReactNode;
}) {
  return <ServicesContext.Provider value={services}>{children}</ServicesContext.Provider>;
}

export function useServices(): Services {
  const ctx = useContext(ServicesContext);
  if (!ctx) throw new Error("useServices must be used within a ServicesProvider");
  return ctx;
}
