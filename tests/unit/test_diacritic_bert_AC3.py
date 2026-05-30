# Proves SPEC-0014 AC3: train_diacritic_head (a) keeps the backbone frozen,
# (b) reduces the loss, (c) writes a checkpoint + train_meta.json whose in_dim
# was read from the (stub) model. Trains on CPU against a tiny frozen stub
# backbone - no BGE-M3 download. Skipped in CI (torch not installed).

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("pyarrow")

import json

import torch
from torch import nn

from aic2026.train.diacritic_bert import TrainConfig, train_diacritic_head
from aic2026.train.diacritic_corpus import build_corpus

_CLEAN = [
    "con chó chạy trong công viên",
    "Hà Nội mùa thu lá rơi nhiều",
    "phở bò tái nạm gầu nóng",
    "Đà Nẵng có biển rất đẹp",
    "chợ Bến Thành ở Sài Gòn",
    "trời mưa to ở miền Trung",
    "cà phê sữa đá buổi sáng",
    "xe máy chạy trên đường phố",
]


class StubBackbone(nn.Module):
    """A frozen char-embedding token encoder; hidden_size 16, no download."""

    hidden_size = 16

    def __init__(self, vocab: int = 512) -> None:
        super().__init__()
        self.vocab = vocab
        self.emb = nn.Embedding(vocab, self.hidden_size)

    def encode(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        maxlen = max((len(t) for t in texts), default=1) or 1
        ids = torch.zeros(len(texts), maxlen, dtype=torch.long)
        mask = torch.zeros(len(texts), maxlen)
        for i, t in enumerate(texts):
            chars = t[:maxlen] or " "
            for j, ch in enumerate(chars):
                ids[i, j] = ord(ch) % self.vocab
                mask[i, j] = 1.0
        return self.emb(ids), mask


def test_train_head_freezes_backbone_and_reduces_loss_AC3(tmp_path: Path) -> None:
    pairs = tmp_path / "pairs.parquet"
    build_corpus(out=pairs, k=4, seed=0, clean_strings=_CLEAN)

    stub = StubBackbone()
    before = stub.emb.weight.detach().clone()

    cfg = TrainConfig(proj_dims=(16, 16), batch_size=8, max_steps=120, lr=1e-2, seed=0)
    out_dir = tmp_path / "run"
    res = train_diacritic_head(pairs, cfg, out_dir=out_dir, device="cpu", backbone=stub)

    # (a) backbone frozen: no grads, weights unchanged.
    assert all(not p.requires_grad for p in stub.parameters())
    assert torch.equal(stub.emb.weight.detach(), before)

    # (b) loss decreased.
    assert res.final_loss < res.initial_loss

    # (c) artefacts written; in_dim read from the (stub) model.
    assert res.checkpoint.exists()
    assert res.meta.exists()
    meta = json.loads(res.meta.read_text())
    assert meta["in_dim"] == StubBackbone.hidden_size
    assert meta["backbone_frozen"] is True
    assert meta["steps"] == 120

    ckpt = torch.load(res.checkpoint, weights_only=False)
    assert ckpt["in_dim"] == StubBackbone.hidden_size
    assert "head_state" in ckpt
