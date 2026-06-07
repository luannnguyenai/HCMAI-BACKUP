// Implements SPEC-0027 SS 4 (in-UI issue capture, AC6).
import { useState } from "react";

import { useIssueCapture } from "../hooks/useIssueCapture";

export function IssueButton() {
  const report = useIssueCapture();
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const onClick = async () => {
    setBusy(true);
    setMessage(null);
    try {
      const resp = await report();
      setMessage(resp.issue_url ?? resp.fallback_path ?? "Da ghi nhan.");
    } catch (e) {
      setMessage(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex gap-2 items-center">
      <button
        data-testid="report-issue-btn"
        onClick={onClick}
        disabled={busy}
        className="border rounded px-3 py-2 disabled:opacity-50"
      >
        {busy ? "Dang gui..." : "Bao loi"}
      </button>
      {message && (
        <span data-testid="issue-result" className="text-sm opacity-70">
          {message}
        </span>
      )}
    </div>
  );
}
