// Implements SPEC-0027 SS 4 (last-10 query history, Q3).
import { useSearch } from "../hooks/useSearch";
import { useStore } from "../store";

export function HistoryList() {
  const history = useStore((s) => s.history);
  const setQuery = useStore((s) => s.setQuery);
  const run = useSearch();

  if (history.length === 0) return null;

  return (
    <div data-testid="history" className="flex flex-wrap gap-2 text-sm">
      {history.map((q) => (
        <button
          key={q}
          className="border rounded px-2 py-1 opacity-80"
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
