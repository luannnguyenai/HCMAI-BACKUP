// Implements SPEC-0027 (Vite + React + Vitest config, ADR-0004).
/// <reference types="vitest/config" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react()],
  server: {
    // Dev proxy: the API + static image tiers are served by the SPEC-0026
    // FastAPI app (nginx in production). In dev, proxy /api, /ws, /thumbs,
    // /frames, /readyz, /healthz to the backend.
    proxy: {
      "/api": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
      "/thumbs": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/frames": { target: "http://127.0.0.1:8000", changeOrigin: true },
      "/readyz": { target: "http://127.0.0.1:8000", changeOrigin: true },
    },
  },
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: ["./src/test/setup.ts"],
    css: false,
  },
});
