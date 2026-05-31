# Proves SPEC-0014 AC8: compare_c1_vs_baselines runs degradation@k for the three
# retrievers (c1_on / baseline_maxsim / baseline_dense), returns per-mode +
# overall blocks, and emits a ship-gate verdict that's the conjunction of the
# absolute target + beats-each-baseline checks. Uses a stub backbone + a freshly
# constructed head: the goal here is *contract correctness*, not the real model
# numbers (those are the lease-run AC).

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch
from torch import nn

from aic2026.eval.diacritic_robustness import compare_c1_vs_baselines


class StubBackbone(nn.Module):
    """Char-id token encoder; deterministic, no download, hidden_size 8."""

    hidden_size = 8

    def __init__(self, vocab: int = 256) -> None:
        super().__init__()
        self.vocab = vocab
        self.emb = nn.Embedding(vocab, self.hidden_size)
        torch.manual_seed(0)
        nn.init.normal_(self.emb.weight, std=0.5)

    def encode(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        maxlen = max((len(t) for t in texts), default=1) or 1
        ids = torch.zeros(len(texts), maxlen, dtype=torch.long)
        mask = torch.zeros(len(texts), maxlen)
        for i, t in enumerate(texts):
            for j, ch in enumerate(t[:maxlen] or " "):
                ids[i, j] = ord(ch) % self.vocab
                mask[i, j] = 1.0
        return self.emb(ids), mask


_QUERIES = [
    "con chó ở chợ Bến Thành",
    "Hà Nội mùa thu lá vàng rơi",
    "phở bò tái nạm gầu nóng hổi",
    "Đà Nẵng có biển xanh cát trắng",
    "cà phê sữa đá buổi sáng sớm",
]


def _make_head(in_dim: int, proj: tuple[int, int]) -> nn.Module:
    """A randomly initialised DiacriticHead - the contract test doesn't need it trained."""
    from aic2026.train.diacritic_bert import DiacriticHead

    torch.manual_seed(1)
    head = DiacriticHead(in_dim, proj)
    head.eval()
    for p in head.parameters():
        p.requires_grad_(False)
    return head


def test_compare_returns_three_blocks_plus_verdict_AC8() -> None:
    bb = StubBackbone()
    for p in bb.parameters():
        p.requires_grad_(False)
    head = _make_head(bb.hidden_size, (4, 4))

    res = compare_c1_vs_baselines(_QUERIES, backbone=bb, head=head, k=3, seed=0)

    # The three retriever blocks are present + carry the same mode keys.
    assert set(res.keys()) == {
        "k",
        "n_queries",
        "modes",
        "c1_on",
        "baseline_maxsim",
        "baseline_dense",
        "ship_gate",
    }
    assert res["n_queries"] == len(_QUERIES)
    expected_mode_keys = set(res["modes"]) | {"overall"}
    for block in ("c1_on", "baseline_maxsim", "baseline_dense"):
        assert set(res[block].keys()) == expected_mode_keys
        for v in res[block].values():
            assert 0.0 <= float(v) <= 1.0

    # Verdict block: conjunction of the three checks against the target.
    sg = res["ship_gate"]
    assert sg["target"] == 0.85
    assert sg["c1_overall"] == res["c1_on"]["overall"]
    assert sg["baseline_maxsim_overall"] == res["baseline_maxsim"]["overall"]
    assert sg["baseline_dense_overall"] == res["baseline_dense"]["overall"]
    assert sg["passes_absolute"] == (sg["c1_overall"] >= sg["target"])
    assert sg["beats_baseline_maxsim"] == (sg["c1_overall"] > sg["baseline_maxsim_overall"])
    assert sg["beats_baseline_dense"] == (sg["c1_overall"] > sg["baseline_dense_overall"])
    assert sg["passes_ship_gate"] == (
        sg["passes_absolute"] and sg["beats_baseline_maxsim"] and sg["beats_baseline_dense"]
    )


def test_compare_target_override_AC8() -> None:
    """A trivially-low target makes passes_absolute True regardless of the head quality."""
    bb = StubBackbone()
    for p in bb.parameters():
        p.requires_grad_(False)
    head = _make_head(bb.hidden_size, (4, 4))

    res = compare_c1_vs_baselines(_QUERIES, backbone=bb, head=head, k=3, seed=0, target=0.0)
    assert res["ship_gate"]["target"] == 0.0
    assert res["ship_gate"]["passes_absolute"] is True


def test_compare_is_deterministic_AC8() -> None:
    bb = StubBackbone()
    for p in bb.parameters():
        p.requires_grad_(False)
    head = _make_head(bb.hidden_size, (4, 4))

    a = compare_c1_vs_baselines(_QUERIES, backbone=bb, head=head, k=3, seed=42)
    b = compare_c1_vs_baselines(_QUERIES, backbone=bb, head=head, k=3, seed=42)
    assert a == b
