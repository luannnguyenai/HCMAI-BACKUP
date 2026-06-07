// Implements SPEC-0027 SS 3 (TypeScript mirror of the SPEC-0026 Pydantic models).
// Keep in lockstep with src/aic2026/serving/models.py.

export type Lane = "siglip2" | "metaclip2";
export type FusionMode = "single" | "rrf";

export interface QueryRequest {
  query_vi: string;
  lanes: Lane[];
  top_k: number; // default 48
  fusion: FusionMode; // default "single"
  rrf_k?: number; // default 60, used only when fusion === "rrf"
}

export interface RankedFrame {
  pk: string;
  video_id: string;
  frame_id: string;
  rank: number;
  score: number;
  thumb_url: string;
  full_url: string;
  per_lane: Partial<Record<Lane, number>>;
}

export interface QueryResponse {
  query_vi: string;
  lanes: Lane[];
  fusion: FusionMode;
  results: RankedFrame[];
  took_ms: number;
}

export interface FrameDetail {
  pk: string;
  video_id: string;
  frame_id: string;
  frame_idx: number;
  youtube_url: string | null;
  description: string | null;
  od_tags: string[];
  ocr_text: string | null;
  asr_text: string | null;
  full_url: string;
  neighbours: string[];
}

export interface IssueReport {
  query_vi: string;
  lanes: Lane[];
  fusion: FusionMode;
  returned_frame_ids: string[]; // pks shown when the report was filed
  screenshot_png_b64: string;
  client_timestamp: string; // ISO-8601
  note?: string;
}

export interface IssueResponse {
  issue_url: string | null;
  fallback_path: string | null;
}

// Default request knobs (SPEC-0026 SS 3 defaults).
export const DEFAULT_TOP_K = 48;
export const DEFAULT_RRF_K = 60;
