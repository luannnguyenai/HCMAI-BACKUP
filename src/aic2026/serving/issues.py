# Implements SPEC-0026 SS 4 (issue capture to GitHub + local fallback).
"""Capture a tester's in-UI issue report.

`IssueSink.capture` writes a GitHub issue when a repo + token are configured,
uploading the screenshot to R2 (ADR-0011) and linking it in the body (SPEC-0026
Q2 RESOLVED). When GitHub is unavailable - no repo, no token, or any error on
the GitHub round-trip - it writes the same payload (JSON + the decoded PNG) to a
local fallback directory and returns that path. A report is never lost and the
endpoint never 5xxes (SPEC-0026 SS 4, AC6).

Collaborators are injected so the GitHub path is testable against a mocked HTTP
transport and the screenshot upload against `moto` (SPEC-0026 SS 8): pass an
`http_client` (an `httpx.Client`) and/or a `screenshot_uploader`.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from aic2026.serving.config import ServingConfig
from aic2026.serving.models import IssueReport, IssueResponse

if TYPE_CHECKING:  # pragma: no cover - typing only
    import httpx

logger = logging.getLogger(__name__)

GITHUB_API_ROOT: str = "https://api.github.com"
GITHUB_TOKEN_ENV: str = "GITHUB_TOKEN"
# Optional public base for the R2 screenshot link; when unset the R2 key is
# recorded instead (still a usable reference for a reader with bucket access).
R2_PUBLIC_BASE_ENV: str = "R2_PUBLIC_BASE_URL"
ISSUE_LABEL: str = "mvp-feedback"
_GITHUB_TIMEOUT_S: float = 10.0
_TITLE_MAX: int = 60

# Callable[[png_bytes, r2_key], url_or_key]: uploads the screenshot, returns a
# link to embed (a public URL when available, else the R2 key).
ScreenshotUploader = Callable[[bytes, str], str | None]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _decode_png(b64: str) -> bytes:
    """Decode a base64 PNG, tolerating a `data:` URL prefix; never raises."""
    payload = b64.strip()
    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")
    try:
        return base64.b64decode(payload, validate=False)
    except (binascii.Error, ValueError):
        logger.warning("issue screenshot was not valid base64; storing empty bytes")
        return b""


def _default_uploader(png: bytes, key: str) -> str | None:
    """Upload the screenshot to R2 and return a link (public URL or key)."""
    from aic2026.remote.r2 import R2Client

    client = R2Client()
    client.put_bytes(key, png, content_type="image/png")
    base = os.environ.get(R2_PUBLIC_BASE_ENV)
    return f"{base.rstrip('/')}/{key}" if base else key


class IssueSink:
    """Persists issue reports to GitHub, with a never-lose local fallback."""

    def __init__(
        self,
        config: ServingConfig,
        *,
        http_client: httpx.Client | None = None,
        screenshot_uploader: ScreenshotUploader | None = None,
        github_token: str | None = None,
        timestamp: Callable[[], str] = _now_iso,
    ) -> None:
        self.config = config
        self._http_client = http_client
        self._uploader = screenshot_uploader
        self._token = github_token if github_token is not None else os.environ.get(GITHUB_TOKEN_ENV)
        self._timestamp = timestamp

    def capture(self, report: IssueReport) -> IssueResponse:
        """File the report to GitHub, or to the local fallback on any failure."""
        ts = self._timestamp()
        png = _decode_png(report.screenshot_png_b64)
        if self._github_enabled():
            try:
                return self._capture_github(report, png, ts)
            except Exception:  # a report must never be lost; fall back to local sink
                logger.warning("GitHub issue capture failed; using local fallback", exc_info=True)
        return self._capture_local(report, png, ts)

    # --- GitHub path -------------------------------------------------------

    def _github_enabled(self) -> bool:
        return bool(self.config.github_repo and self._token)

    def _capture_github(self, report: IssueReport, png: bytes, ts: str) -> IssueResponse:
        shot_link = self._upload_screenshot(png, ts)
        body = _render_body(report, shot_link)
        title = _render_title(report)
        client = self._http_client or _build_http_client()
        resp = client.post(
            f"{GITHUB_API_ROOT}/repos/{self.config.github_repo}/issues",
            headers={
                "Authorization": f"Bearer {self._token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
            json={"title": title, "body": body, "labels": [ISSUE_LABEL]},
        )
        resp.raise_for_status()
        issue_url = str(resp.json().get("html_url") or "") or None
        return IssueResponse(issue_url=issue_url, fallback_path=None)

    def _upload_screenshot(self, png: bytes, ts: str) -> str | None:
        key = f"issues/{_safe_ts(ts)}/screenshot.png"
        uploader = self._uploader or _default_uploader
        try:
            return uploader(png, key)
        except Exception:  # the issue body just omits the link on upload failure
            logger.warning("screenshot upload failed; issue will omit the link", exc_info=True)
            return None

    # --- local fallback ----------------------------------------------------

    def _capture_local(self, report: IssueReport, png: bytes, ts: str) -> IssueResponse:
        out_dir = self.config.issue_fallback_dir / _safe_ts(ts)
        out_dir.mkdir(parents=True, exist_ok=True)
        if png:
            (out_dir / "screenshot.png").write_bytes(png)
        payload = report.model_dump()
        payload.pop("screenshot_png_b64", None)  # the PNG is written as a file
        payload["captured_at"] = ts
        json_path = out_dir / "issue.json"
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return IssueResponse(issue_url=None, fallback_path=str(json_path))


def _build_http_client() -> httpx.Client:
    import httpx

    return httpx.Client(timeout=_GITHUB_TIMEOUT_S)


def _safe_ts(ts: str) -> str:
    """Filesystem/key-safe form of an ISO-8601 timestamp."""
    return ts.replace(":", "").replace("/", "").replace(" ", "_")


def _render_title(report: IssueReport) -> str:
    q = report.query_vi.strip().replace("\n", " ")
    if len(q) > _TITLE_MAX:
        q = q[: _TITLE_MAX - 1] + "\u2026"
    return f"[MVP feedback] {q}"


def _render_body(report: IssueReport, screenshot_link: str | None) -> str:
    lanes = ", ".join(lane.value for lane in report.lanes)
    frames = "\n".join(f"- {pk}" for pk in report.returned_frame_ids) or "- (none)"
    note = report.note or "(no note)"
    shot = f"![screenshot]({screenshot_link})" if screenshot_link else "(screenshot upload failed)"
    return (
        f"**Query (vi):** {report.query_vi}\n\n"
        f"**Lanes:** {lanes}\n\n"
        f"**Fusion:** {report.fusion.value}\n\n"
        f"**Client timestamp:** {report.client_timestamp}\n\n"
        f"**Note:**\n{note}\n\n"
        f"**Returned frames (pk):**\n{frames}\n\n"
        f"**Screenshot:**\n{shot}\n"
    )
