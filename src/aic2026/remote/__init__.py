# Implements SPEC-0022 (remote GPU job runner).
"""`bin/remote` package: ship code to an ephemeral GPU cluster, run a job,
sync the results to Cloudflare R2 ([`ADR-0011`](../../../docs/adr/ADR-0011-r2-artifact-store-and-lease-rollover.md)).

The four-tier persistence model is enforced by who owns what:

- `context.RunContext`: the provenance + paths.
- `manifest.ManifestEntry`: the R2 ledger row.
- `r2.R2Client`: the only thing that touches the artifact store.
- `ssh.ssh_exec`: the only thing that touches the cluster.
- `launchers.*`: turn a job spec into a remote command line.
- `registry.register / resolve`: the job table the CLI dispatches on.

Importing this package must not require `boto3` to authenticate to R2 or `ssh`
to reach the cluster - both are deferred to method-call time so unit tests
(using `moto` + subprocess mocks) can exercise the layers in isolation.
"""

from aic2026.remote.context import RunContext
from aic2026.remote.manifest import ManifestEntry, append_to_r2, read_all
from aic2026.remote.r2 import R2Client
from aic2026.remote.registry import register, resolve

__all__ = [
    "ManifestEntry",
    "R2Client",
    "RunContext",
    "append_to_r2",
    "read_all",
    "register",
    "resolve",
]
