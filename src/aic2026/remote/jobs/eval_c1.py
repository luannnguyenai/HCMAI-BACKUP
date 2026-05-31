# Implements SPEC-0014 (eval-c1 remote job) + SPEC-0022 job contract.
"""`eval-c1`: degradation@10 ship-gate run for the C1 DiacriticBERT head.

Runs on a GPU lease via `bin/remote run eval-c1`. Restores ``head.pt`` and the
training corpus from R2 (``c1-baseline/<sha>/``), harvests a held-out query set
disjoint from the training anchors, runs the three-way comparison (C1 on vs raw
BGE-M3 MaxSim vs BGE-M3 dense), and writes ``c1_eval.json`` under
``ctx.local_run_dir`` for the runner to upload to R2 under ``runs/<run_id>/``.

Heavy deps (torch / transformers / pyarrow / datasets) are imported **inside**
the job so importing the jobs package for the registry stays light (CI has none
of them).

Config keys (all optional strings, forwarded via `--config k=v`):
  baseline_prefix (R2 prefix holding head.pt + pairs.parquet; default
                   ``c1-baseline/<git_sha[:7]>``),
  checkpoint      (local override path to head.pt),
  pairs_path      (local override path to pairs.parquet),
  backbone        (default ``BAAI/bge-m3``),
  k               (default ``10``),
  n_heldout       (default ``200``),
  max_per_source  (default ``5000``),
  seed            (default ``0``),
  target          (ship-gate absolute target; default ``0.85``).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from aic2026.remote.context import RunContext
from aic2026.remote.registry import register

logger = logging.getLogger(__name__)


def _restore_from_r2(prefix: str, *, want: dict[str, Path]) -> None:
    """Restore each ``suffix -> dest`` from R2 ``<prefix>/...endswith(suffix)``.

    Only the entries in ``want`` whose ``dest`` does not yet exist are pulled. The
    bank-script layout puts ``pairs.parquet`` at the top level and ``head.pt``
    under ``run/``; suffix matching tolerates both layouts. Raises if the prefix
    is empty or any wanted suffix is missing.
    """
    from aic2026.remote.r2 import R2Client

    missing = {suffix: dest for suffix, dest in want.items() if not dest.exists()}
    if not missing:
        return

    client = R2Client()
    prefix = prefix.rstrip("/")
    keys = client.list(prefix)
    if not keys:
        raise FileNotFoundError(
            f"eval-c1: no objects under R2 prefix {prefix!r}; pass --config "
            "baseline_prefix=... or --config checkpoint=...,pairs_path=...",
        )

    for suffix, dest in missing.items():
        key = next((k for k in keys if k.endswith(suffix)), None)
        if key is None:
            raise FileNotFoundError(f"eval-c1: no {suffix} under {prefix}")
        dest.write_bytes(client.get_bytes(key))
        logger.info("eval-c1: restored %s -> %s", key, dest)


def _ensure_baseline(
    ctx: RunContext, prefix: str, *, checkpoint_override: str | None, pairs_override: str | None
) -> tuple[Path, Path]:
    """Resolve the local paths to ``head.pt`` and ``pairs.parquet``.

    Order:
      1. Explicit ``--config checkpoint=...`` / ``pairs_path=...`` overrides.
      2. Already-on-disk under ``ctx.local_run_dir/baseline/``.
      3. Pull from R2 ``s3://<bucket>/<prefix>/`` (the bank-script layout).
    """
    base_dir = Path(ctx.local_run_dir) / "baseline"
    base_dir.mkdir(parents=True, exist_ok=True)

    ckpt = Path(checkpoint_override) if checkpoint_override else base_dir / "head.pt"
    pairs = Path(pairs_override) if pairs_override else base_dir / "pairs.parquet"

    if ckpt.exists() and pairs.exists():
        logger.info("eval-c1: baseline already local (%s, %s)", ckpt, pairs)
        return ckpt, pairs

    logger.info("eval-c1: restoring missing baseline files from R2 prefix %s", prefix)
    _restore_from_r2(prefix, want={"head.pt": ckpt, "pairs.parquet": pairs})
    return ckpt, pairs


@register("eval-c1")
def eval_c1(ctx: RunContext, config: dict[str, Any]) -> None:
    """Run the C1 degradation@10 ship-gate and write ``c1_eval.json``."""
    from aic2026.eval.diacritic_robustness import (
        build_heldout_queries,
        compare_c1_vs_baselines,
    )
    from aic2026.eval.retrievers import load_head
    from aic2026.train.diacritic_bert import BgeM3Backbone

    out = Path(ctx.local_run_dir)
    out.mkdir(parents=True, exist_ok=True)

    prefix = config.get("baseline_prefix") or f"c1-baseline/{ctx.git_sha[:7]}"
    ckpt, pairs = _ensure_baseline(
        ctx,
        prefix,
        checkpoint_override=config.get("checkpoint"),
        pairs_override=config.get("pairs_path"),
    )

    k = int(config.get("k", "10"))
    n_heldout = int(config.get("n_heldout", "200"))
    seed = int(config.get("seed", "0"))
    max_per_source = int(config.get("max_per_source", "5000"))
    backbone_id = config.get("backbone", "BAAI/bge-m3")
    target = float(config.get("target", "0.85"))

    logger.info("eval-c1: harvesting %d held-out queries (exclude %s)", n_heldout, pairs)
    queries = build_heldout_queries(
        n_heldout, exclude_corpus=pairs, seed=seed, max_per_source=max_per_source
    )
    logger.info("eval-c1: %d queries gathered", len(queries))

    logger.info("eval-c1: loading backbone %s", backbone_id)
    backbone = BgeM3Backbone(backbone_id)
    logger.info("eval-c1: loading head %s", ckpt)
    head = load_head(ckpt)

    logger.info("eval-c1: running degradation@%d on %d queries", k, len(queries))
    result = compare_c1_vs_baselines(
        queries, backbone=backbone, head=head, k=k, seed=seed, target=target
    )

    summary = {
        "run_id": ctx.run_id,
        "git_sha": ctx.git_sha,
        "backbone": backbone_id,
        "checkpoint": str(ckpt),
        "pairs": str(pairs),
        "result": result,
    }
    (out / "c1_eval.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    sg = result["ship_gate"]
    logger.info(
        "eval-c1 done: c1=%.4f maxsim=%.4f dense=%.4f target=%.2f -> %s",
        sg["c1_overall"],
        sg["baseline_maxsim_overall"],
        sg["baseline_dense_overall"],
        sg["target"],
        "PASS" if sg["passes_ship_gate"] else "FAIL",
    )
