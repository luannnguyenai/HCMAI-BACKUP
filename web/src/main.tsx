// Implements SPEC-0027 (app entry: providers + render).
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";

import { App } from "./App";
import { createDefaultServices, ServicesProvider } from "./services";
import "./index.css";

// Shared secret is injected at build/serve time (SPEC-0026 SS 9 Q1). On the
// shared MVP server it is baked into the served bundle behind the reverse proxy.
const secret = import.meta.env.VITE_AIC_SHARED_SECRET ?? null;

const queryClient = new QueryClient({
  defaultOptions: { queries: { staleTime: 60_000, retry: 1 } },
});
const services = createDefaultServices(secret);

const root = document.getElementById("root");
if (!root) throw new Error("missing #root");

createRoot(root).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <ServicesProvider services={services}>
        <App />
      </ServicesProvider>
    </QueryClientProvider>
  </StrictMode>,
);
