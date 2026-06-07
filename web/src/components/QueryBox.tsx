// Implements SPEC-0027 SS 4 (Vietnamese query box, AC1/AC2).
import type { FormEvent } from "react";

import { useSearch } from "../hooks/useSearch";
import { useStore } from "../store";

export function QueryBox() {
  const query = useStore((s) => s.query);
  const setQuery = useStore((s) => s.setQuery);
  const status = useStore((s) => s.status);
  const run = useSearch();
  const disabled = query.trim().length === 0;

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (disabled) return; // AC2: Enter is a no-op on an empty query.
    void run();
  };

  return (
    <form data-testid="query-form" onSubmit={onSubmit} className="flex gap-2 items-center">
      <input
        data-testid="query-input"
        aria-label="Truy van tieng Viet"
        className="flex-1 border rounded px-3 py-2"
        placeholder="Nhap mo ta canh quay..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
      />
      <button
        data-testid="search-btn"
        type="submit"
        disabled={disabled}
        className="border rounded px-4 py-2 disabled:opacity-50"
      >
        {status === "loading" ? "Dang tim..." : "Tim"}
      </button>
    </form>
  );
}
