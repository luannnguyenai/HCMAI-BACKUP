// Implements SPEC-0027 SS 4 (frame-detail view as a modal, AC4).
import { useQuery } from "@tanstack/react-query";
import { useState } from "react";

import type { Lane } from "../api/types";
import { useServices } from "../services";
import { useStore } from "../store";

function CopyPkButton({ pk }: { pk: string }) {
  const [copied, setCopied] = useState(false);
  const onCopy = async () => {
    try {
      await navigator.clipboard?.writeText(pk);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1200);
    } catch {
      setCopied(false);
    }
  };
  return (
    <button type="button" data-testid="copy-pk" onClick={onCopy} className="btn px-2 py-1 text-xs">
      {copied ? "Da chep" : "Chep pk"}
    </button>
  );
}

export function FrameDetailModal() {
  const { api } = useServices();
  const selectedPk = useStore((s) => s.selectedPk);
  const results = useStore((s) => s.results);
  const selectFrame = useStore((s) => s.selectFrame);
  const stepSelection = useStore((s) => s.stepSelection);

  const { data: detail, isLoading } = useQuery({
    queryKey: ["frame", selectedPk],
    queryFn: () => api.frameDetail(selectedPk as string),
    enabled: selectedPk != null,
  });

  if (selectedPk == null) return null;

  const frame = results.find((r) => r.pk === selectedPk) ?? null;
  const idx = results.findIndex((r) => r.pk === selectedPk);
  const close = () => selectFrame(null);

  const shell =
    "panel relative flex max-h-[90vh] w-full max-w-5xl flex-col overflow-hidden bg-ink-850 shadow-modal animate-scale-in md:flex-row";

  return (
    <div
      className="fixed inset-0 z-40 flex items-center justify-center bg-ink-950/70 p-4 backdrop-blur-sm animate-fade-in"
      onClick={close}
      role="presentation"
    >
      {isLoading || !detail ? (
        <div
          data-testid="frame-detail-loading"
          role="dialog"
          aria-modal="true"
          aria-label="Dang tai chi tiet khung hinh"
          className={shell}
          onClick={(e) => e.stopPropagation()}
        >
          <div className="skeleton h-[60vh] w-full" />
        </div>
      ) : (
        <div
          data-testid="frame-detail"
          role="dialog"
          aria-modal="true"
          aria-label="Chi tiet khung hinh"
          className={shell}
          onClick={(e) => e.stopPropagation()}
        >
          {/* Image pane */}
          <div className="relative flex min-h-[240px] flex-1 items-center justify-center bg-ink-950 p-3">
            <img
              src={detail.full_url}
              alt={detail.pk}
              className="max-h-[78vh] w-auto max-w-full rounded-md object-contain"
            />
            {idx >= 0 && (
              <span className="absolute left-4 top-4 rounded bg-ink-950/80 px-2 py-1 font-mono text-xs text-fg-muted ring-1 ring-inset ring-white/5">
                #{frame?.rank ?? idx + 1} / {results.length}
              </span>
            )}
          </div>

          {/* Meta pane */}
          <aside className="flex w-full shrink-0 flex-col gap-3 overflow-y-auto border-t border-line p-4 md:w-80 md:border-l md:border-t-0">
            <div className="flex items-start justify-between gap-2">
              <div className="label">Chi tiet khung hinh</div>
              <button
                type="button"
                data-testid="detail-close"
                onClick={close}
                className="btn btn-ghost px-2 py-1 text-xs"
                aria-label="Dong"
              >
                Dong (Esc)
              </button>
            </div>

            <div className="space-y-1">
              <div data-testid="detail-ids" className="font-mono text-sm text-fg">
                {detail.video_id} / {detail.frame_id}
              </div>
              <div className="font-mono text-[11px] text-fg-faint">idx {detail.frame_idx}</div>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <span className="chip font-mono">{detail.pk}</span>
              <CopyPkButton pk={detail.pk} />
            </div>

            {frame && (
              <div className="space-y-1.5 rounded-md border border-line bg-ink-900 p-2.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-fg-faint">score</span>
                  <span className="font-mono text-accent-strong">{frame.score.toFixed(4)}</span>
                </div>
                {Object.entries(frame.per_lane).length > 0 && (
                  <div data-testid="detail-per-lane" className="space-y-1 border-t border-line pt-1.5">
                    {(Object.entries(frame.per_lane) as [Lane, number][]).map(([lane, s]) => (
                      <div key={lane} className="flex items-center justify-between text-[11px]">
                        <span className="font-mono text-fg-muted">{lane}</span>
                        <span className="font-mono text-fg">{s.toFixed(4)}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}

            {detail.youtube_url && (
              <a
                data-testid="detail-youtube"
                href={detail.youtube_url}
                target="_blank"
                rel="noreferrer"
                className="btn px-2 py-1.5 text-xs"
              >
                Mo YouTube
              </a>
            )}
            {detail.description && (
              <p data-testid="detail-description" className="text-sm text-fg-muted">
                {detail.description}
              </p>
            )}
            {detail.od_tags.length > 0 && (
              <div className="flex flex-wrap gap-1.5">
                {detail.od_tags.map((t) => (
                  <span key={t} className="chip">
                    {t}
                  </span>
                ))}
              </div>
            )}
            {/* OCR/ASR rows render only when present (Q4): null on the proxy today. */}
            {detail.ocr_text != null && (
              <p data-testid="detail-ocr" className="text-xs text-fg-muted">
                <span className="label mr-1">OCR</span>
                {detail.ocr_text}
              </p>
            )}
            {detail.asr_text != null && (
              <p data-testid="detail-asr" className="text-xs text-fg-muted">
                <span className="label mr-1">ASR</span>
                {detail.asr_text}
              </p>
            )}
            {detail.neighbours.length > 0 && (
              <div data-testid="detail-neighbours" className="space-y-1">
                <span className="label">lan can</span>
                <div className="flex flex-wrap gap-1.5">
                  {detail.neighbours.map((n) => (
                    <button
                      key={n}
                      type="button"
                      onClick={() => selectFrame(n)}
                      className="chip font-mono"
                    >
                      {n}
                    </button>
                  ))}
                </div>
              </div>
            )}

            <div className="mt-auto flex items-center justify-between gap-2 pt-2">
              <button
                type="button"
                data-testid="detail-prev"
                onClick={() => stepSelection(-1)}
                disabled={idx <= 0}
                className="btn flex-1 px-2 py-1.5 text-xs"
              >
                &#8592; Truoc
              </button>
              <button
                type="button"
                data-testid="detail-next"
                onClick={() => stepSelection(1)}
                disabled={idx === -1 || idx >= results.length - 1}
                className="btn flex-1 px-2 py-1.5 text-xs"
              >
                Sau &#8594;
              </button>
            </div>
          </aside>
        </div>
      )}
    </div>
  );
}
