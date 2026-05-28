# Implements SPEC-0001 SS 3.3 (report.html) and AC1.
"""Render the per-run HTML report via jinja2."""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from aic2026.models.metrics import AggregateMetrics

_TEMPLATE_DIR = Path(__file__).parent / "templates"


def _env() -> Environment:
    return Environment(
        loader=FileSystemLoader(_TEMPLATE_DIR),
        autoescape=select_autoescape(("html", "j2")),
    )


def render_report_html(metrics: AggregateMetrics) -> str:
    template = _env().get_template("report.html.j2")
    return template.render(metrics=metrics)


def write_report_html(metrics: AggregateMetrics, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report_html(metrics), encoding="utf-8")
