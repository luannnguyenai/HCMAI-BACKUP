// Implements SPEC-0027 SS 4 (Vietnamese query box, AC1/AC2).
import { useEffect, useRef, type FormEvent } from "react";

import { useSearch } from "../hooks/useSearch";
import { useStore } from "../store";

export function QueryBox() {
  const query = useStore((s) => s.query);
  const setQuery = useStore((s) => s.setQuery);
  const status = useStore((s) => s.status);
  const run = useSearch();
  const disabled = query.trim().length === 0;
  const inputRef = useRef<HTMLInputElement>(null);

  // Focus the search box on load so an operator can type immediately (SS 4).
  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const onSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (disabled) return; // AC2: Enter is a no-op on an empty query.
    void run();
  };

  return (
    <form data-testid="query-form" onSubmit={onSubmit} className="flex gap-2">
      <div className="relative flex-1">
        <span
          className="pointer-events-none absolute left-3 top-1/2 -translate-y-1/2 text-fg-faint"
          aria-hidden="true"
        >
          {/* magnifier glyph */}
          &#9906;
        </span>
        <input
          ref={inputRef}
          data-testid="query-input"
          aria-label="Truy van tieng Viet"
          className="input py-2.5 pl-9 pr-3 text-[15px]"
          placeholder="Mo ta canh quay can tim, vi du: nguoi dan ong mac ao do dang chay tren bai bien"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
        />
      </div>
      <button
        data-testid="search-btn"
        type="submit"
        disabled={disabled}
        className="btn btn-primary px-5 py-2.5"
      >
        {status === "loading" ? "Dang tim..." : "Tim"}
      </button>
    </form>
  );
}
