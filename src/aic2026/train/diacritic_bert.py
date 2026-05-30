# Implements SPEC-0014 section 3 + AC3 (frozen-backbone late-interaction training).
"""Train the C1 DiacriticBERT head: a projection over a frozen BGE-M3.

Architecture (proposal 08 part 3.2 step 3): the backbone (BGE-M3) is **frozen**;
a small 2-layer MLP projects its per-token last-hidden states, and a ColBERT-style
**MaxSim** late-interaction score drives an **InfoNCE** objective with in-batch
negatives. Only the head trains.

SPEC-0014 Q1: the head input width is **read from the loaded model's hidden size**,
never hardcoded (the SPEC-0004 1024->1152 bug is the cautionary precedent). MaxSim
is averaged over query tokens so the score sits in [-1, 1] and the temperature 0.05
behaves like standard normalized-embedding InfoNCE.

The backbone is **injectable** (`backbone=`) so tests train on CPU against a tiny
stub without downloading the ~2 GB BGE-M3. `import torch` is top-level here; the
module is only imported once torch is present (the GPU box, or a test after
`importorskip`). The `train-c1` job imports it lazily.
"""

from __future__ import annotations

import json
import logging
import random
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import torch
import torch.nn.functional as F
from torch import nn

from aic2026.train.diacritic_corpus import read_pairs

logger = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    backbone: str = "BAAI/bge-m3"
    proj_dims: tuple[int, int] = (384, 384)
    temperature: float = 0.05
    batch_size: int = 64
    max_steps: int = 250_000
    lr: float = 2e-4
    weight_decay: float = 0.01
    seed: int = 0
    max_length: int = 64


@dataclass
class TrainResult:
    checkpoint: Path
    meta: Path
    initial_loss: float
    final_loss: float
    steps: int


@runtime_checkable
class TokenEncoder(Protocol):
    """A frozen token-level encoder: text -> (token embeddings, attention mask)."""

    hidden_size: int

    def encode(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(emb (B, T, H), mask (B, T))``; mask is 1 for real tokens."""


class DiacriticHead(nn.Module):
    """2-layer MLP projection producing L2-normalised per-token vectors."""

    def __init__(self, in_dim: int, proj_dims: tuple[int, int] = (384, 384)) -> None:
        super().__init__()
        h0, h1 = proj_dims
        self.l1 = nn.Linear(in_dim, h0)
        self.l2 = nn.Linear(h0, h1)

    def forward(self, tok_emb: torch.Tensor) -> torch.Tensor:
        x = self.l2(F.gelu(self.l1(tok_emb)))
        return F.normalize(x, dim=-1)


def maxsim_scores(
    q: torch.Tensor, qmask: torch.Tensor, d: torch.Tensor, dmask: torch.Tensor
) -> torch.Tensor:
    """ColBERT MaxSim averaged over query tokens -> ``(B, B)`` score matrix in [-1, 1].

    ``score[a, b] = mean_{i in query a} max_{j in doc b} <q[a,i], d[b,j]>`` over
    real (unpadded) tokens.
    """
    sim = torch.einsum("aih,bjh->abij", q, d)  # (B, B, Tq, Td)
    dm = dmask.bool()[None, :, None, :]  # (1, B, 1, Td)
    sim = sim.masked_fill(~dm, float("-inf"))
    maxd = sim.max(dim=-1).values  # (B, B, Tq) max over doc tokens
    qm = qmask.bool()[:, None, :]  # (B, 1, Tq)
    maxd = maxd.masked_fill(~qm, 0.0)
    qlen = qmask.sum(dim=-1).clamp(min=1.0)  # (B,)
    return maxd.sum(dim=-1) / qlen[:, None]  # (B, B)


def info_nce(scores: torch.Tensor, temperature: float) -> torch.Tensor:
    """Symmetric in-batch InfoNCE: the diagonal (clean, its-own-noisy) is positive."""
    n = scores.size(0)
    labels = torch.arange(n, device=scores.device)
    logits = scores / temperature
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


class BgeM3Backbone(nn.Module):
    """Frozen BGE-M3 token encoder (the real backbone; needs the ``train`` extra)."""

    def __init__(self, model_id: str = "BAAI/bge-m3", *, max_length: int = 64) -> None:
        super().__init__()
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id)
        # SPEC-0014 Q1: read the width from the model, never hardcode.
        self.hidden_size = int(self.model.config.hidden_size)
        self.max_length = max_length
        self.model.eval()

    @torch.no_grad()
    def encode(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        dev = next(self.model.parameters()).device
        enc = {k: v.to(dev) for k, v in enc.items()}
        out = self.model(**enc)
        return out.last_hidden_state, enc["attention_mask"]


def _iter_step_batches(
    data: list[tuple[str, str]], batch_size: int, max_steps: int, rng: random.Random
) -> Iterator[list[tuple[str, str]]]:
    """Yield up to ``max_steps`` shuffled batches (skipping any of size < 2)."""
    idx = list(range(len(data)))
    steps = 0
    while steps < max_steps:
        rng.shuffle(idx)
        for start in range(0, len(idx), batch_size):
            chunk = idx[start : start + batch_size]
            if len(chunk) < 2:  # InfoNCE needs >= 2 in-batch examples
                continue
            yield [data[i] for i in chunk]
            steps += 1
            if steps >= max_steps:
                return


def _freeze(backbone: object, device: torch.device) -> None:
    if isinstance(backbone, nn.Module):
        backbone.to(device)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)


def train_diacritic_head(
    pairs: Path,
    cfg: TrainConfig,
    *,
    out_dir: Path,
    device: str | None = None,
    backbone: TokenEncoder | None = None,
) -> TrainResult:
    """Frozen-backbone InfoNCE training of the projection head.

    Reads the contrastive Parquet, encodes (clean, noisy) pairs through the frozen
    backbone, trains the head with in-batch negatives, and writes ``head.pt`` +
    ``train_meta.json`` (with ``in_dim`` read from the model). ``backbone`` is
    injectable for tests; when ``None`` the real BGE-M3 is loaded from ``cfg.backbone``.
    """
    torch.manual_seed(cfg.seed)
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    rows = read_pairs(pairs)
    data = [(str(r["anchor_clean"]), str(r["positive_noisy"])) for r in rows]
    if len(data) < 2:
        raise ValueError(f"need >=2 pairs to train; got {len(data)}")

    if backbone is None:
        backbone = BgeM3Backbone(cfg.backbone, max_length=cfg.max_length)
    _freeze(backbone, dev)
    in_dim = int(backbone.hidden_size)

    head = DiacriticHead(in_dim, cfg.proj_dims).to(dev)
    opt = torch.optim.Adam(head.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    rng = random.Random(cfg.seed)

    def _loss(batch: list[tuple[str, str]]) -> torch.Tensor:
        anchors = [a for a, _ in batch]
        positives = [p for _, p in batch]
        with torch.no_grad():
            ae, am = backbone.encode(anchors)
            pe, pm = backbone.encode(positives)
        ae, am, pe, pm = ae.to(dev), am.to(dev), pe.to(dev), pm.to(dev)
        scores = maxsim_scores(head(ae), am, head(pe), pm)
        return info_nce(scores, cfg.temperature)

    initial_loss: float | None = None
    final_loss = float("nan")
    steps = 0
    for batch in _iter_step_batches(data, cfg.batch_size, cfg.max_steps, rng):
        loss = _loss(batch)
        if initial_loss is None:
            initial_loss = float(loss.detach())
        opt.zero_grad()
        loss.backward()
        opt.step()
        final_loss = float(loss.detach())
        steps += 1

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt = out_dir / "head.pt"
    torch.save({"head_state": head.state_dict(), "in_dim": in_dim, "config": asdict(cfg)}, ckpt)
    meta = out_dir / "train_meta.json"
    meta_obj = {
        "backbone": cfg.backbone,
        "in_dim": in_dim,
        "backbone_frozen": True,
        "steps": steps,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "n_pairs": len(data),
        "temperature": cfg.temperature,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "batch_size": cfg.batch_size,
        "proj_dims": list(cfg.proj_dims),
        "corpus": str(pairs),
        "device": str(dev),
    }
    meta.write_text(json.dumps(meta_obj, indent=2), encoding="utf-8")
    logger.info(
        "trained head: %d steps, loss %.4f -> %.4f, in_dim=%d -> %s",
        steps,
        initial_loss if initial_loss is not None else float("nan"),
        final_loss,
        in_dim,
        ckpt,
    )
    return TrainResult(
        checkpoint=ckpt,
        meta=meta,
        initial_loss=initial_loss if initial_loss is not None else float("nan"),
        final_loss=final_loss,
        steps=steps,
    )
