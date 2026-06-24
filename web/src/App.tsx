// Implements SPEC-0027 SS 4 (KIS console composition + keyboard ergonomics).
import { useEffect, useState } from "react";

import { FrameDetailModal } from "./components/FrameDetailModal";
import { HistoryList } from "./components/HistoryList";
import { IssueDialog } from "./components/IssueDialog";
import { LaneSelector } from "./components/LaneSelector";
import { QueryBox } from "./components/QueryBox";
import { ResultGrid } from "./components/ResultGrid";
import { StatusBar } from "./components/StatusBar";
import { TopBar } from "./components/TopBar";
import { TopKControl } from "./components/TopKControl";
import { useStore } from "./store";

function isTypingTarget(el: EventTarget | null): boolean {
  if (!(el instanceof HTMLElement)) return false;
  const tag = el.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || el.isContentEditable;
}

export function App() {
  const [issueOpen, setIssueOpen] = useState(false);
  const selectedPk = useStore((s) => s.selectedPk);
  const selectFrame = useStore((s) => s.selectFrame);
  const stepSelection = useStore((s) => s.stepSelection);

  // Operator keyboard ergonomics (SS 4): Esc closes overlays; arrows step the
  // selection while the detail view is open; typing is never hijacked.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        if (issueOpen) setIssueOpen(false);
        else if (selectedPk != null) selectFrame(null);
        return;
      }
      if (issueOpen || isTypingTarget(e.target)) return;
      if (selectedPk != null) {
        if (e.key === "ArrowRight" || e.key === "ArrowDown") {
          e.preventDefault();
          stepSelection(1);
        } else if (e.key === "ArrowLeft" || e.key === "ArrowUp") {
          e.preventDefault();
          stepSelection(-1);
        }
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [issueOpen, selectedPk, selectFrame, stepSelection]);

  return (
    <div className="flex h-full flex-col">
      <TopBar onReport={() => setIssueOpen(true)} />

      <main className="mx-auto flex w-full max-w-[1600px] flex-1 flex-col gap-3 overflow-hidden px-4 py-3">
        <section className="panel space-y-3 p-3" aria-label="Truy van">
          <QueryBox />
          <div className="flex flex-wrap items-center justify-between gap-x-5 gap-y-2">
            <div className="flex flex-wrap items-center gap-x-5 gap-y-2">
              <LaneSelector />
              <TopKControl />
            </div>
            <StatusBar />
          </div>
          <HistoryList />
        </section>

        <section
          className="panel min-h-0 flex-1 overflow-y-auto p-3"
          aria-label="Ket qua"
        >
          <ResultGrid />
        </section>
      </main>

      <FrameDetailModal />
      <IssueDialog open={issueOpen} onClose={() => setIssueOpen(false)} />
    </div>
  );
}
