// Implements SPEC-0027 SS 4 (in-UI issue capture form, AC6).
import { useState, type FormEvent } from "react";

import { useIssueCapture } from "../hooks/useIssueCapture";
import { useStore } from "../store";

type Severity = "thap" | "trung binh" | "cao";
const SEVERITIES: Severity[] = ["thap", "trung binh", "cao"];

// The SPEC-0026 IssueReport schema is extra="forbid" and carries a single free
// text `note`; the form's title + severity + description + extra context
// (top_k, selected pk) are folded into that note so the backend contract is
// unchanged.
function composeNote(args: {
  title: string;
  severity: Severity;
  description: string;
  topK: number;
  selectedPk: string | null;
  resultCount: number;
}): string {
  const lines = [
    `[muc do: ${args.severity}] ${args.title}`.trim(),
    "",
    args.description.trim(),
    "",
    `--- ngu canh ---`,
    `top_k=${args.topK}`,
    `so ket qua=${args.resultCount}`,
    `khung hinh dang chon=${args.selectedPk ?? "(khong)"}`,
  ];
  return lines.join("\n").trim();
}

export function IssueDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  const report = useIssueCapture();
  const topK = useStore((s) => s.topK);
  const query = useStore((s) => s.query);
  const lanes = useStore((s) => s.lanes);
  const fusion = useStore((s) => s.fusion);
  const results = useStore((s) => s.results);
  const selectedPk = useStore((s) => s.selectedPk);

  const [title, setTitle] = useState("");
  const [severity, setSeverity] = useState<Severity>("trung binh");
  const [description, setDescription] = useState("");
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [ok, setOk] = useState<boolean | null>(null);

  if (!open) return null;

  const onSubmit = async (e: FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setMessage(null);
    setOk(null);
    try {
      const note = composeNote({
        title,
        severity,
        description,
        topK,
        selectedPk,
        resultCount: results.length,
      });
      const resp = await report(note);
      setOk(true);
      setMessage(resp.issue_url ?? resp.fallback_path ?? "Da ghi nhan bao loi.");
    } catch (err) {
      setOk(false);
      setMessage(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-ink-950/70 p-4 backdrop-blur-sm animate-fade-in"
      onClick={onClose}
      role="presentation"
    >
      <form
        data-testid="issue-form"
        onSubmit={onSubmit}
        onClick={(e) => e.stopPropagation()}
        className="panel w-full max-w-lg space-y-4 bg-ink-850 p-5 shadow-modal animate-scale-in"
        role="dialog"
        aria-modal="true"
        aria-label="Bao loi"
      >
        <div className="flex items-center justify-between">
          <h2 className="text-base font-semibold">Bao loi ket qua truy van</h2>
          <button type="button" onClick={onClose} className="btn btn-ghost px-2 py-1 text-xs">
            Dong
          </button>
        </div>

        <div className="space-y-1.5">
          <label className="label" htmlFor="issue-title">
            Tieu de
          </label>
          <input
            id="issue-title"
            data-testid="issue-title"
            className="input"
            placeholder="Vi du: ket qua hang dau khong lien quan"
            value={title}
            onChange={(e) => setTitle(e.target.value)}
          />
        </div>

        <div className="space-y-1.5">
          <span className="label">Muc do</span>
          <div className="inline-flex overflow-hidden rounded-md border border-line">
            {SEVERITIES.map((s) => {
              const active = severity === s;
              return (
                <button
                  key={s}
                  type="button"
                  data-testid={`severity-${s.replace(/\s+/g, "-")}`}
                  aria-pressed={active}
                  onClick={() => setSeverity(s)}
                  className={`px-3 py-1.5 text-xs capitalize transition-colors ${
                    active
                      ? "bg-accent text-ink-950 font-semibold"
                      : "bg-ink-900 text-fg-muted hover:bg-ink-800 hover:text-fg"
                  }`}
                >
                  {s}
                </button>
              );
            })}
          </div>
        </div>

        <div className="space-y-1.5">
          <label className="label" htmlFor="issue-desc">
            Mo ta
          </label>
          <textarea
            id="issue-desc"
            data-testid="issue-description"
            className="input min-h-[88px] resize-y"
            placeholder="Mo ta van de quan sat duoc..."
            value={description}
            onChange={(e) => setDescription(e.target.value)}
          />
        </div>

        <div className="rounded-md border border-line bg-ink-900 p-3 text-[11px] text-fg-faint">
          <div className="label mb-1.5">Ngu canh tu dong dinh kem</div>
          <div className="grid grid-cols-2 gap-x-3 gap-y-1 font-mono text-fg-muted">
            <span>truy van</span>
            <span className="truncate text-fg">"{query.trim() || "(trong)"}"</span>
            <span>lane / fusion</span>
            <span className="text-fg">
              {lanes.join("+")} / {fusion}
            </span>
            <span>top_k</span>
            <span className="text-fg">{topK}</span>
            <span>so ket qua</span>
            <span className="text-fg">{results.length}</span>
            <span>pk dang chon</span>
            <span className="truncate text-fg">{selectedPk ?? "(khong)"}</span>
          </div>
          <div className="mt-1.5 text-fg-faint">Anh chup man hinh duoc dinh kem khi gui.</div>
        </div>

        <div className="flex items-center justify-between gap-3">
          {message ? (
            <span
              data-testid="issue-result"
              className={`truncate text-xs ${ok ? "text-ok" : "text-bad"}`}
            >
              {message}
            </span>
          ) : (
            <span className="text-xs text-fg-faint">POST /api/issues</span>
          )}
          <button
            data-testid="issue-submit"
            type="submit"
            disabled={busy}
            className="btn btn-primary px-4 py-2"
          >
            {busy ? "Dang gui..." : "Gui bao loi"}
          </button>
        </div>
      </form>
    </div>
  );
}
