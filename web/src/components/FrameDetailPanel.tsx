// Implements SPEC-0027 SS 4 (frame-detail view, AC4).
import { useQuery } from "@tanstack/react-query";

import { useServices } from "../services";
import { useStore } from "../store";

export function FrameDetailPanel() {
  const { api } = useServices();
  const selectedPk = useStore((s) => s.selectedPk);

  const { data: detail, isLoading } = useQuery({
    queryKey: ["frame", selectedPk],
    queryFn: () => api.frameDetail(selectedPk as string),
    enabled: selectedPk != null,
  });

  if (selectedPk == null) {
    return (
      <aside data-testid="frame-detail-empty" className="opacity-60 p-3">
        Chon mot khung hinh de xem chi tiet.
      </aside>
    );
  }
  if (isLoading || !detail) {
    return (
      <aside data-testid="frame-detail-loading" className="p-3">
        Dang tai chi tiet...
      </aside>
    );
  }

  return (
    <aside data-testid="frame-detail" className="p-3 space-y-2">
      <img src={detail.full_url} alt={detail.pk} className="w-full rounded" />
      <div data-testid="detail-ids" className="font-mono text-sm">
        {detail.video_id} / {detail.frame_id} (idx {detail.frame_idx})
      </div>
      {detail.youtube_url && (
        <a data-testid="detail-youtube" href={detail.youtube_url} target="_blank" rel="noreferrer">
          Mo YouTube
        </a>
      )}
      {detail.description && <p data-testid="detail-description">{detail.description}</p>}
      {/* OCR/ASR rows render only when present (Q4): null on the proxy today. */}
      {detail.ocr_text != null && <p data-testid="detail-ocr">OCR: {detail.ocr_text}</p>}
      {detail.asr_text != null && <p data-testid="detail-asr">ASR: {detail.asr_text}</p>}
      {detail.neighbours.length > 0 && (
        <div data-testid="detail-neighbours" className="text-xs opacity-70">
          Lan can: {detail.neighbours.join(", ")}
        </div>
      )}
    </aside>
  );
}
