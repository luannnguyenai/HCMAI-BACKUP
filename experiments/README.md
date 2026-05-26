# experiments/

> Per-experiment workspaces. One folder per experiment, named for its purpose.
>
> Contents may include throwaway notebooks, hyperparameter sweeps, ablation data, and decision artefacts. Final results are persisted here (not in `eval-results/`, which is reserved for the standard harness output).

Conventions:

- One folder per experiment: `experiments/<short-name>/`
- Each folder contains a `README.md` describing what was tried, why, and where the final numbers live.
- Per-PR work that is not gated by a spec lives here (see [`CONTRIBUTING.md`](../CONTRIBUTING.md) "What is NOT spec-driven").
- Anything `experiments/` imports from `src/` becomes spec-driven by transitivity; if you need a `src/` import, write a spec.

Reserved subfolders referenced by current specs:

- `experiments/llm-path-bakeoff/` — output of SPEC-0002 (bakeoff runner). `REPORT.md`, `summary.json`, and `raw/*.parquet` will live here after the June run.

Anything else here is fair game for individual contributors.
