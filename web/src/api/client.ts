// Implements SPEC-0027 SS 3 (typed REST client for the SPEC-0026 API).
import type {
  FrameDetail,
  IssueReport,
  IssueResponse,
  QueryRequest,
  QueryResponse,
} from "./types";

export interface ApiClient {
  // REST fallback + non-hot-path fetches.
  query(req: QueryRequest): Promise<QueryResponse>;
  frameDetail(pk: string): Promise<FrameDetail>;
  reportIssue(report: IssueReport): Promise<IssueResponse>;
}

const SECRET_HEADER = "X-AIC-Secret";

export interface ApiClientOptions {
  baseUrl?: string; // default same-origin
  secret?: string | null; // shared-secret gate (SPEC-0026 SS 9 Q1)
  fetchImpl?: typeof fetch;
}

function headers(secret?: string | null): Record<string, string> {
  const h: Record<string, string> = { "Content-Type": "application/json" };
  if (secret) h[SECRET_HEADER] = secret;
  return h;
}

async function asJson<T>(resp: Response): Promise<T> {
  if (!resp.ok) {
    const body = await resp.text().catch(() => "");
    throw new Error(`HTTP ${resp.status}: ${body || resp.statusText}`);
  }
  return (await resp.json()) as T;
}

export function createApiClient(opts: ApiClientOptions = {}): ApiClient {
  const base = (opts.baseUrl ?? "").replace(/\/$/, "");
  const doFetch = opts.fetchImpl ?? fetch;
  const url = (path: string) => `${base}${path}`;
  return {
    async query(req) {
      const resp = await doFetch(url("/api/query"), {
        method: "POST",
        headers: headers(opts.secret),
        body: JSON.stringify(req),
      });
      return asJson<QueryResponse>(resp);
    },
    async frameDetail(pk) {
      const resp = await doFetch(url(`/api/frame/${encodeURIComponent(pk)}`), {
        headers: headers(opts.secret),
      });
      return asJson<FrameDetail>(resp);
    },
    async reportIssue(report) {
      const resp = await doFetch(url("/api/issues"), {
        method: "POST",
        headers: headers(opts.secret),
        body: JSON.stringify(report),
      });
      return asJson<IssueResponse>(resp);
    },
  };
}
