# Proves SPEC-0028 AC3: an empty required-prefix list is a no-op success that
# makes no R2 call (so the opt-in `run` guard is safe to leave unset), and
# duplicate prefixes are probed once.

from __future__ import annotations

from aic2026.remote.preflight import check_prefixes


class _CountingClient:
    """Stand-in for R2Client that records every list() call.

    Returns one fake key for any prefix in `present`, else nothing. Used to
    assert call counts without a network or moto server.
    """

    def __init__(self, present: set[str]) -> None:
        self.present = present
        self.calls: list[str] = []

    def list(self, prefix: str) -> list[str]:
        self.calls.append(prefix)
        return [f"{prefix}/obj_0.bin"] if prefix in self.present else []


def test_empty_prefix_list_is_noop_success_AC3() -> None:
    client = _CountingClient(present=set())
    result = check_prefixes(client, [])  # type: ignore[arg-type]
    assert result.ok is True
    assert result.statuses == ()
    assert client.calls == []  # no R2 call at all


def test_duplicate_prefixes_probed_once_AC3() -> None:
    client = _CountingClient(present={"index/present"})
    result = check_prefixes(  # type: ignore[arg-type]
        client, ["index/present", "index/present", "index/present"]
    )
    assert result.ok is True
    assert len(result.statuses) == 1
    assert client.calls == ["index/present"]  # de-duped to a single probe
