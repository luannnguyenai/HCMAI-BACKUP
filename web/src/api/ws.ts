// Implements SPEC-0027 SS 3-4 (WebSocket query channel, ADR-0004 hot path).
import type { QueryRequest, QueryResponse } from "./types";

export interface QueryChannel {
  // Hot path: send a QueryRequest, resolve on the QueryResponse frame.
  send(req: QueryRequest): Promise<QueryResponse>;
  onError(handler: (message: string) => void): void;
  close(): void;
}

interface Pending {
  resolve: (r: QueryResponse) => void;
  reject: (e: Error) => void;
}

export interface WsChannelOptions {
  url?: string; // default derives from window.location
  secret?: string | null;
  socketFactory?: (url: string) => WebSocket; // injectable for tests
}

function defaultWsUrl(secret?: string | null): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  const base = `${proto}//${window.location.host}/ws`;
  return secret ? `${base}?secret=${encodeURIComponent(secret)}` : base;
}

/**
 * One persistent WebSocket with a FIFO queue of pending requests. The server
 * answers each `QueryRequest` with exactly one `QueryResponse` (SPEC-0026 AC8),
 * so responses are matched to requests in send order. Any socket error or an
 * error frame rejects the in-flight request; the caller (useSearch) then falls
 * back to the REST `query` endpoint (SPEC-0027 AC7).
 */
export function createQueryChannel(opts: WsChannelOptions = {}): QueryChannel {
  const url = opts.url ?? defaultWsUrl(opts.secret);
  const make = opts.socketFactory ?? ((u: string) => new WebSocket(u));
  let sock: WebSocket | null = null;
  const pending: Pending[] = [];
  let errorHandler: ((m: string) => void) | null = null;

  function rejectAll(message: string): void {
    while (pending.length) pending.shift()?.reject(new Error(message));
  }

  function ensure(): WebSocket {
    if (sock && (sock.readyState === WebSocket.OPEN || sock.readyState === WebSocket.CONNECTING)) {
      return sock;
    }
    const s = make(url);
    s.onmessage = (ev: MessageEvent) => {
      const data = JSON.parse(typeof ev.data === "string" ? ev.data : "");
      if (data && typeof data === "object" && "error" in data) {
        errorHandler?.(String(data.error));
        pending.shift()?.reject(new Error(String(data.error)));
        return;
      }
      pending.shift()?.resolve(data as QueryResponse);
    };
    s.onerror = () => {
      errorHandler?.("websocket error");
      rejectAll("websocket error");
    };
    s.onclose = () => {
      rejectAll("websocket closed");
      if (sock === s) sock = null;
    };
    sock = s;
    return s;
  }

  return {
    send(req) {
      return new Promise<QueryResponse>((resolve, reject) => {
        let s: WebSocket;
        try {
          s = ensure();
        } catch (e) {
          reject(e instanceof Error ? e : new Error(String(e)));
          return;
        }
        pending.push({ resolve, reject });
        const payload = JSON.stringify(req);
        if (s.readyState === WebSocket.OPEN) {
          s.send(payload);
        } else {
          s.addEventListener("open", () => s.send(payload), { once: true });
        }
      });
    },
    onError(handler) {
      errorHandler = handler;
    },
    close() {
      sock?.close();
      sock = null;
    },
  };
}
