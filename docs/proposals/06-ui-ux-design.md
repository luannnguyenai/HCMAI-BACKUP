# Proposal 06 - UI / UX Design

> The UI is where 30% of the points come from. PraK1 vs PraK2 differed by 30 points on the same engine purely from operator speed. This proposal locks down the design.

## 1. Guiding principles

1. **Speed first, beauty second.** Every animation longer than 100ms is a bug.
2. **Eyes on the images.** ?80% of viewport is the image grid (SnapSeek 3.0's rule).
3. **Keyboard everything.** Hotkeys for every action; mouse is a fallback.
4. **Verify before submit.** No accidental wrong submissions allowed.
5. **Show the planner's reasoning.** Operator should see why the system thinks a frame is relevant.

## 2. Layout (single-monitor 1920x1080)

```
+-----------------+--------------------------------------------------+
|                 |                                                  |
|  Query box      |  Image grid (8x6 thumbnails @ 240x135 px)        |
|  + filters      |                                                  |
|  + planner JSON |  Hover: enlarge to 480x270                       |
|  panel          |  Click: open frame detail panel (right slide)    |
|  (220 px wide)  |                                                  |
|                 |                                                  |
+-----------------+--------------------------------------------------+
|                 |                                                  |
|  Query history  |  Keyframe scrubber (timeline view of selected    |
|  + saved frames |  frame's video context, +/- 60 sec)              |
|  (220 px)       |                                                  |
+-----------------+--------------------------------------------------+
|  Submission verification bar (full width, 80 px)                   |
|  shows pending submission with thumbnail + countdown + Confirm/    |
|  Cancel buttons                                                    |
+--------------------------------------------------------------------+
```

## 3. Components

### 3.1 Query box (top-left)
- Single text input (Vietnamese IME-friendly).
- Below it: 3 filter chips ("Time:", "Place:", "Object:") that expand on click.
- "Temporal" toggle for KIS-style sequence queries: switches input to a 2-line box for `q1 < q2`.
- TRAKE mode: 4-tab input for the 4 scene descriptions, each independently submittable for review.

### 3.2 Planner JSON panel (below query)
- Shows the LLM planner's JSON output for the current query.
- Operator can edit it (override paraphrases, change top_k) and rerun.
- Optional: collapsible.

### 3.3 Image grid (main)
- 8 columns x 6 rows = 48 thumbnails per page.
- Each thumbnail: image + tiny badges showing OCR/ASR/place/ADL hits.
- Hover-to-enlarge (300ms delay, 480x270 popup).
- Single-click: opens frame detail panel.
- Double-click: queues for submission (goes to verification bar).
- Hotkey **1-9**: select thumbnail at that position in current row.

### 3.4 Frame detail panel (right slide, 600 px)
- Large preview (1024x576 or original aspect).
- All metadata: video_id, ts, place, ADL, objects detected, OCR text, ASR transcript snippet.
- Buttons: "+/- relevance feedback", "save to query history", "submit this".
- "Neighbours" tab: prev/next 5 keyframes in the same video.

### 3.5 Keyframe scrubber (bottom-left)
- Visualises the selected frame's video context: +/- 60 seconds.
- 10-frame strip at 80x45 px each.
- Shift+hover: scrubs a low-GOP video preview (diveXplore trick).
- Click any frame to swap selection.

### 3.6 Query history (bottom-left)
- Last 10 queries with their result counts and any submissions.
- Click to re-run.

### 3.7 Submission verification bar (bottom, full width)
**Critical anti-foot-gun feature.**

When operator triggers a submission:
1. Bar slides up showing: pending frame thumbnail | query | confidence score | 3-second countdown | Confirm button | Cancel button.
2. During countdown, the bar stays interactive.
3. If operator hits Enter or clicks Confirm: submit to DRES.
4. Cancel returns to grid.
5. After submit, bar shows result (correct/incorrect + score delta) for 2 seconds.

This single component prevents accidental submissions and gives the operator a forced 1-3 second sanity check.

## 4. Hotkeys

| Key | Action |
|---|---|
| `Ctrl+L` | Focus query box |
| `Enter` (in query) | Submit query |
| `Ctrl+Enter` | Submit current selection to DRES |
| `Esc` | Cancel current modal / submission |
| `1-9` | Select thumbnail in current row |
| `Tab` / `Shift+Tab` | Next / prev row |
| `+` / `-` | Relevance feedback +/- on currently selected |
| `[` / `]` | Prev / next neighbour frame (in detail panel) |
| `Ctrl+1..4` | Switch TRAKE scene tab |
| `Ctrl+H` | Toggle query history |
| `Ctrl+P` | Toggle planner JSON panel |

## 5. Visual design

- Dark mode default (reduces eye strain in finals room).
- Monospace font for IDs/timestamps; sans-serif (Inter) for everything else.
- Vietnamese diacritics tested at all sizes.
- Tailwind CSS. shadcn/ui components.
- Performance: virtualised grid (react-window); no layout shifts.

## 6. TRAKE-specific UX

Because TRAKE asks for 4 scenes in correct order:

1. Each TRAKE query has 4 scene tabs (Ctrl+1..4 to switch).
2. Each tab is an independent retrieval, but the bottom of the screen shows a **TRAKE staging tray** with 4 slots.
3. Operator drags frames from the grid into the slots (or hotkey `S1`-`S4` to assign).
4. The staging tray shows the temporal-consistency score (red if `t1 > t2` etc.).
5. Submit button on the staging tray triggers the 4-frame submission.

## 7. Operator-mode vs novice-mode

The same UI has two modes (toggleable in settings):

| Feature | Operator mode | Novice mode |
|---|---|---|
| Planner JSON panel | shown | hidden |
| Keyboard hints | overlay on hover | always shown for first 30 min |
| Default filters | empty | one-click "common defaults" |
| Verification bar countdown | 1s (fast operator) | 5s (newbie) |

This is critical because if novice users run our system in the *novice track* (rules don't currently say there will be one, but historically LSC has had it), we shouldn't waste their cognitive bandwidth.

## 8. Tech stack

- React 18 + Vite + TypeScript
- TailwindCSS + shadcn/ui
- Zustand for state
- TanStack Query for data fetching
- WebSocket (`ws://localhost:8000/ws`) for real-time updates from the planner LLM
- Native HTML5 drag-and-drop (no library)

## 9. Performance budget

| Metric | Target |
|---|---|
| Initial load | <500 ms |
| Time to first thumbnail painted | <300 ms |
| Grid scroll FPS | 60 |
| Thumbnail render p50 | <16 ms |
| Hotkey response | <50 ms |
| Query submission round-trip | <2 s |

## 10. Accessibility

- All actions reachable by keyboard.
- Screen-reader labels on every clickable element (even though we don't expect visually-impaired operators).
- High-contrast mode for finals room with bad lighting.
- Font scaling 80%-150% via Ctrl+/-.

## 11. Localisation

- UI language toggle: Vietnamese (default) / English.
- Vietnamese strings live in `src/i18n/vi.json`; English in `en.json`.
- All user-facing text is keyed; never hardcoded.

## 12. Testing

- Playwright e2e for the critical flows: search -> select -> verify -> submit.
- Storybook for component-level visual regression.
- Real operators do a "speed run" once a week timing the same 10 mock queries; we record their time and improvement.
