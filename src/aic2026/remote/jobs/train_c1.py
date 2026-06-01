# Implements SPEC-0014 (train-c1 remote job) + SPEC-0022 job contract.
"""`train-c1`: build the C1 corpus, train the DiacriticBERT head, bank artefacts.

Runs on the GPU box via `bin/remote run train-c1`. Writes everything under
`ctx.local_run_dir` (corpus Parquet, `head.pt`, `train_meta.json`, a summary),
which the runner uploads to R2 under `runs/<run_id>/` for cross-lease durability.

Heavy training deps (torch / transformers / datasets / pyarrow) are imported
**inside** the job so importing the jobs package for the registry stays light
(CI has none of them).

Config keys (all optional strings, forwarded via `--config k=v`):
  backbone, max_steps, batch_size, lr, k, seed, max_per_source, pairs_path.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from aic2026.remote.context import RunContext
from aic2026.remote.registry import register

logger = logging.getLogger(__name__)


@register("train-c1")
def train_c1(ctx: RunContext, config: dict[str, Any]) -> None:
    """Build corpus -> train head -> write artefacts under ``ctx.local_run_dir``."""
    from aic2026.train.diacritic_bert import TrainConfig, train_diacritic_head
    from aic2026.train.diacritic_corpus import build_corpus

    out = Path(ctx.local_run_dir)
    out.mkdir(parents=True, exist_ok=True)

    # k = 0 (default) -> "one of each NoiseMode" in the v2 schedule (diacritic + OCR).
    # An explicit positive k truncates / extends the cycle.
    k_raw = int(config.get("k", "0"))
    k = k_raw or None
    seed = int(config.get("seed", "0"))

    pairs_path = config.get("pairs_path")
    if pairs_path:
        pairs = Path(pairs_path)
        logger.info("train-c1: using existing corpus %s", pairs)
    else:
        pairs = out / "diacritic_pairs.parquet"
        mps_raw = config.get("max_per_source")
        max_per_source = int(mps_raw) if mps_raw else None
        cres = build_corpus(out=pairs, k=k, max_per_source=max_per_source, seed=seed)
        logger.info(
            "train-c1 corpus: %d clean -> %d pairs (used=%s skipped=%s)",
            cres.n_clean,
            cres.n_pairs,
            cres.sources_used,
            cres.sources_skipped,
        )

    cfg = TrainConfig(
        backbone=config.get("backbone", "BAAI/bge-m3"),
        batch_size=int(config.get("batch_size", "64")),
        max_steps=int(config.get("max_steps", "20000")),
        lr=float(config.get("lr", "2e-4")),
        seed=seed,
    )
    res = train_diacritic_head(pairs, cfg, out_dir=out)

    summary = {
        "run_id": ctx.run_id,
        "git_sha": ctx.git_sha,
        "backbone": cfg.backbone,
        "checkpoint": res.checkpoint.name,
        "train_meta": res.meta.name,
        "initial_loss": res.initial_loss,
        "final_loss": res.final_loss,
        "steps": res.steps,
    }
    (out / "train_c1_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        "train-c1 done: loss %.4f -> %.4f over %d steps -> %s",
        res.initial_loss,
        res.final_loss,
        res.steps,
        res.checkpoint,
    )
