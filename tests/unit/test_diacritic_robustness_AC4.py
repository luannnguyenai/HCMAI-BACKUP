# Proves SPEC-0014 AC4: degradation_at_k returns per-mode + overall values in
# [0, 1], is deterministic per seed, and returns 1.0 when noise is a no-op
# (ASCII queries -> identity). Uses DummyEmbedder - no torch.

from __future__ import annotations

import pytest

from aic2026.embedding.dummy import DummyEmbedder
from aic2026.eval.diacritic_robustness import DEFAULT_MODES, degradation_at_k

_VI = [
    "con chó ở chợ Bến Thành",
    "Hà Nội mùa thu lá vàng rơi",
    "phở bò tái nạm gầu nóng hổi",
    "Đà Nẵng có biển xanh cát trắng",
    "cà phê sữa đá buổi sáng sớm",
]


def test_values_in_range_and_deterministic_AC4() -> None:
    enc = DummyEmbedder(dim=64)
    a = degradation_at_k(_VI, enc, k=3, seed=0)
    b = degradation_at_k(_VI, enc, k=3, seed=0)
    assert a == b  # deterministic
    keys = {m.value for m in DEFAULT_MODES} | {"overall"}
    assert set(a.keys()) == keys
    assert all(0.0 <= v <= 1.0 for v in a.values())


def test_noop_noise_is_one_AC4() -> None:
    # ASCII queries have no diacritics, so every mode is the identity -> the
    # noisy vector equals the clean one -> perfect retrieval -> degradation 1.0.
    ascii_q = [
        "dog at ben thanh market",
        "ha noi autumn leaves",
        "pho bo for breakfast",
        "da nang has a nice beach",
    ]
    enc = DummyEmbedder(dim=32)
    res = degradation_at_k(ascii_q, enc, k=10, seed=1)
    assert res["overall"] == pytest.approx(1.0)
    assert all(v == pytest.approx(1.0) for v in res.values())


def test_invalid_inputs_raise_AC4() -> None:
    enc = DummyEmbedder(dim=8)
    with pytest.raises(ValueError, match="empty"):
        degradation_at_k([], enc)
    with pytest.raises(ValueError, match="k must be positive"):
        degradation_at_k(["a", "b"], enc, k=0)
