// Implements SPEC-0027 SS 4 (last-10 query history, Q3).
import { useSearch } from "../hooks/useSearch";
import { useStore } from "../store";

export function HistoryList() {
  const history = useStore((s) => s.history);
  const setQuery = useStore((s) => s.setQuery);
  const run = useSearch();

  if (history.length === 0) return null;

  return (
    <div data-testid="history" className="flex flex-wrap items-center gap-1.5">
      <span className="label mr-1">gan day</span>
      {history.map((q) => (
        <button
          key={q}
          type="button"
          className="chip max-w-[16rem] truncate"
          title={q}
          onClick={() => {
            setQuery(q);
            void run();
          }}
        >
          {q}
        </button>
      ))}
    </div>
  );
}
