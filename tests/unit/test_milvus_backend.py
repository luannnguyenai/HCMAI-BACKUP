# Exercises the SPEC-0006 MilvusBackend single-lane Backend (SPEC-0001):
# encode a MockTask query with an injected DummyEmbedder, ANN one dense field,
# adapt to Submissions. Full fusion + task-type routing is SPEC-0015.

from __future__ import annotations

from collections.abc import Callable

from aic2026.embedding.dummy import DummyEmbedder
from aic2026.index.milvus_schema import DenseField
from aic2026.index.milvus_store import MilvusBackend, MilvusKeyframeStore
from aic2026.models.submission import Submission
from aic2026.models.task import GroundTruth, MockTask, TaskType

from .conftest import make_encoder_source

# Backend encodes query text with this dim, so the lane dim must match.
_DIM = 16
_FIELDS = (DenseField("siglip2", _DIM),)


def test_backend_returns_ranked_submissions(
    milvus_lite_store: Callable[..., MilvusKeyframeStore],
    tmp_path,
) -> None:
    frame_ids = [f"{i:04d}" for i in range(5)]  # per-video stems
    store = milvus_lite_store(fields=_FIELDS)
    store.ingest(
        {"siglip2": make_encoder_source(tmp_path, "siglip2", frame_ids, dim=_DIM, video="L25_V011")}
    )

    backend = MilvusBackend(
        store,
        DummyEmbedder(dim=_DIM, model_id="dummy-siglip2"),
        field="siglip2",
        top_k=3,
    )
    task = MockTask(
        task_id="KIS-0001",
        task_type=TaskType.KIS,
        query_vi="mot con meo tren ban",
        time_limit_seconds=300,
        ground_truth=GroundTruth(kis_frame_ids=["L25_V011_0000"]),
    )

    subs = backend.search(task, time_budget_ms=300_000)
    assert len(subs) == 3
    assert all(isinstance(s, Submission) for s in subs)
    assert [s.rank for s in subs] == [1, 2, 3]
    assert all(s.frame_id is not None for s in subs)
