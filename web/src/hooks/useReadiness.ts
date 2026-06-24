// Implements SPEC-0027 SS 4 (top-bar readiness indicator; polls SPEC-0026 /readyz).
import { useQuery } from "@tanstack/react-query";

import type { ReadyStatus } from "../api/types";
import { useServices } from "../services";

// Poll cadence for the readiness pill. The collection load is a one-time boot
// step, so a slow poll keeps the indicator fresh without load on the server.
const READYZ_POLL_MS = 5_000;

export function useReadiness() {
  const { api } = useServices();
  return useQuery<ReadyStatus>({
    queryKey: ["readyz"],
    queryFn: () => api.readiness(),
    refetchInterval: READYZ_POLL_MS,
    retry: false,
    staleTime: 0,
  });
}
