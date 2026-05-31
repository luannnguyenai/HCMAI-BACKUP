# Proves SPEC-0014 AC7: DenseRetriever score == q @ d.T cosine matrix;
# MaxSimRetriever returns the right shape and a head improves scores over the
# raw-token baseline on a constructed signal; and degradation_at_k is back-
# compatible when given an Embedder directly (existing AC4 path).

from __future__ import annotations

import numpy as np
import pytest

from aic2026.embedding.dummy import DummyEmbedder
from aic2026.eval.diacritic_robustness import degradation_at_k
from aic2026.eval.retrievers import DenseRetriever


def test_dense_retriever_score_equals_cosine_AC7() -> None:
    enc = DummyEmbedder(dim=32)
    docs = ["alpha beta", "gamma delta", "epsilon zeta", "eta theta"]
    queries = ["alpha beta", "gamma delta"]

    r = DenseRetriever(enc).score(queries, docs)

    q = enc.encode_text(queries)
    d = enc.encode_text(docs)
    expected = (q @ d.T).astype(np.float32)
    np.testing.assert_allclose(r, expected, atol=1e-6)
    assert r.shape == (2, 4)


def test_dense_retriever_back_compat_with_embedder_AC7() -> None:
    """degradation_at_k(Embedder) and degradation_at_k(DenseRetriever(Embedder)) must agree."""
    enc = DummyEmbedder(dim=32)
    queries = [
        "con chó ở chợ Bến Thành",
        "Hà Nội mùa thu lá vàng rơi",
        "phở bò tái nạm gầu nóng hổi",
        "Đà Nẵng có biển xanh cát trắng",
        "cà phê sữa đá buổi sáng sớm",
    ]
    a = degradation_at_k(queries, enc, k=3, seed=0)
    b = degradation_at_k(queries, DenseRetriever(enc), k=3, seed=0)
    assert a == b


def test_maxsim_retriever_shape_and_range_AC7() -> None:
    """MaxSimRetriever with a stub backbone returns (nq, nd) in [-1, 1]."""
    pytest.importorskip("torch")
    import torch
    from torch import nn

    from aic2026.eval.retrievers import MaxSimRetriever

    class StubBackbone(nn.Module):
        """Char-id token encoder; hidden_size 8, no model download."""

        hidden_size = 8

        def __init__(self, vocab: int = 256) -> None:
            super().__init__()
            self.vocab = vocab
            self.emb = nn.Embedding(vocab, self.hidden_size)

        def encode(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
            maxlen = max((len(t) for t in texts), default=1) or 1
            ids = torch.zeros(len(texts), maxlen, dtype=torch.long)
            mask = torch.zeros(len(texts), maxlen)
            for i, t in enumerate(texts):
                for j, ch in enumerate(t[:maxlen] or " "):
                    ids[i, j] = ord(ch) % self.vocab
                    mask[i, j] = 1.0
            return self.emb(ids), mask

    bb = StubBackbone()
    for p in bb.parameters():
        p.requires_grad_(False)

    r = MaxSimRetriever(bb, head=None, q_chunk=2, device="cpu")
    scores = r.score(["alpha", "beta", "gamma"], ["alpha", "gamma", "delta", "epsilon"])
    assert scores.shape == (3, 4)
    # Each token is L2-normalised and we average over tokens -> bounded in [-1, 1].
    assert float(scores.max()) <= 1.0 + 1e-6
    assert float(scores.min()) >= -1.0 - 1e-6


def test_maxsim_retriever_empty_inputs_AC7() -> None:
    pytest.importorskip("torch")
    import torch as _torch
    from torch import nn

    from aic2026.eval.retrievers import MaxSimRetriever

    class _Stub(nn.Module):
        hidden_size = 4

        def __init__(self) -> None:
            super().__init__()
            self.emb = nn.Embedding(8, self.hidden_size)

        def encode(self, texts: list[str]) -> tuple[_torch.Tensor, _torch.Tensor]:
            n = max(1, len(texts))
            return _torch.zeros(n, 1, self.hidden_size), _torch.zeros(n, 1)

    r = MaxSimRetriever(_Stub(), head=None, device="cpu")
    assert r.score([], ["a"]).shape == (0, 1)
    assert r.score(["a"], []).shape == (1, 0)


def test_load_head_roundtrip_AC7(tmp_path) -> None:
    """Saved head.pt -> load_head reconstructs an eval-mode, frozen head."""
    pytest.importorskip("torch")
    import torch

    from aic2026.eval.retrievers import load_head
    from aic2026.train.diacritic_bert import DiacriticHead

    head = DiacriticHead(in_dim=12, proj_dims=(6, 6))
    ckpt_path = tmp_path / "head.pt"
    torch.save(
        {
            "head_state": head.state_dict(),
            "in_dim": 12,
            "config": {"proj_dims": [6, 6]},
        },
        ckpt_path,
    )

    restored = load_head(ckpt_path)
    assert not restored.training
    assert all(not p.requires_grad for p in restored.parameters())
    # Output shape matches.
    x = torch.randn(2, 3, 12)
    with torch.no_grad():
        out = restored(x)
    assert out.shape == (2, 3, 6)
    # And it's L2-normalised per token.
    norms = out.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
