// Implements SPEC-0027 SS 6 (test render helper + mock services/data).
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, type RenderResult } from "@testing-library/react";
import type { ReactElement } from "react";
import { vi } from "vitest";

import type { ApiClient } from "../api/client";
import type { QueryChannel } from "../api/ws";
import type {
  FrameDetail,
  IssueResponse,
  QueryResponse,
  RankedFrame,
} from "../api/types";
import { ServicesProvider, type Services } from "../services";

export function makeFrame(i: number): RankedFrame {
  const pk = `vid${String(i).padStart(2, "0")}_${String(i).padStart(4, "0")}`;
  return {
    pk,
    video_id: `vid${String(i).padStart(2, "0")}`,
    frame_id: String(i).padStart(4, "0"),
    rank: i,
    score: 1 / (i + 1),
    thumb_url: `/thumbs/${pk}.jpg`,
    full_url: `/frames/${pk}.jpg`,
    per_lane: { siglip2: 1 / (i + 1) },
  };
}

export function makeResponse(n: number, over?: Partial<QueryResponse>): QueryResponse {
  return {
    query_vi: "test",
    lanes: ["siglip2"],
    fusion: "single",
    results: Array.from({ length: n }, (_, i) => makeFrame(i + 1)),
    took_ms: 42,
    ...over,
  };
}

export function makeDetail(over?: Partial<FrameDetail>): FrameDetail {
  return {
    pk: "vid01_0001",
    video_id: "vid01",
    frame_id: "0001",
    frame_idx: 1,
    youtube_url: "https://youtu.be/abc?t=1",
    description: null,
    od_tags: [],
    ocr_text: null,
    asr_text: null,
    full_url: "/frames/vid01_0001.jpg",
    neighbours: [],
    ...over,
  };
}

export interface MockServices extends Services {
  api: {
    query: ReturnType<typeof vi.fn>;
    frameDetail: ReturnType<typeof vi.fn>;
    reportIssue: ReturnType<typeof vi.fn>;
  } & ApiClient;
  channel: { send: ReturnType<typeof vi.fn>; onError: ReturnType<typeof vi.fn>; close: ReturnType<typeof vi.fn> } & QueryChannel;
  captureScreenshot: ReturnType<typeof vi.fn>;
  now: ReturnType<typeof vi.fn>;
}

export function makeServices(over?: {
  query?: QueryResponse;
  frameDetail?: FrameDetail;
  reportIssue?: IssueResponse;
  channelSend?: (...args: unknown[]) => Promise<QueryResponse>;
  screenshot?: string;
  now?: string;
}): MockServices {
  const query = vi.fn(async () => over?.query ?? makeResponse(3));
  const frameDetail = vi.fn(async () => over?.frameDetail ?? makeDetail());
  const reportIssue = vi.fn(
    async () => over?.reportIssue ?? ({ issue_url: "https://github.com/x/y/issues/1", fallback_path: null } as IssueResponse),
  );
  const send = vi.fn(over?.channelSend ?? (async () => over?.query ?? makeResponse(3)));
  return {
    api: { query, frameDetail, reportIssue },
    channel: { send, onError: vi.fn(), close: vi.fn() },
    captureScreenshot: vi.fn(async () => over?.screenshot ?? "iVBORw0KGgoAAAANS"),
    now: vi.fn(() => over?.now ?? "2026-06-07T12:00:00.000Z"),
  } as MockServices;
}

export function renderApp(ui: ReactElement, services: Services): RenderResult {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <ServicesProvider services={services}>{ui}</ServicesProvider>
    </QueryClientProvider>,
  );
}
