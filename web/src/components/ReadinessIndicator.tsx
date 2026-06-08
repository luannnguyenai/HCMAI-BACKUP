// Implements SPEC-0027 SS 4 (live readiness indicator; polls SPEC-0026 /readyz).
import { useReadiness } from "../hooks/useReadiness";

export function ReadinessIndicator() {
  const { data, isLoading, isError } = useReadiness();

  let tone: "ok" | "bad" | "warn" = "warn";
  let dot = "bg-warn";
  let label = "dang kiem tra...";

  if (isLoading) {
    tone = "warn";
    dot = "bg-warn animate-pulse-soft";
    label = "dang kiem tra...";
  } else if (isError || !data) {
    tone = "bad";
    dot = "bg-bad";
    label = "khong ket noi duoc server";
  } else if (data.ready) {
    tone = "ok";
    dot = "bg-ok";
    label = `san sang - ${data.row_count.toLocaleString("en-US")} khung hinh`;
  } else {
    tone = "bad";
    dot = "bg-bad";
    label = data.collection_loaded ? "chua co thumbnail" : "chua nap collection";
  }

  const ring =
    tone === "ok"
      ? "border-ok/40 text-ok"
      : tone === "bad"
        ? "border-bad/40 text-bad"
        : "border-warn/40 text-warn";

  return (
    <span
      data-testid="readiness"
      data-ready={data?.ready ? "true" : "false"}
      className={`pill ${ring}`}
      title="Trang thai san sang cua server (GET /readyz)"
    >
      <span className={`h-2 w-2 rounded-full ${dot}`} aria-hidden="true" />
      {label}
    </span>
  );
}
