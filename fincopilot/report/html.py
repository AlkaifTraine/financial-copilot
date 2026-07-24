"""
HTML rendering.

Charts are inlined as base64 data URIs rather than referenced by path, so the
rendered document is a single self-contained file: it displays correctly inside
Streamlit, survives being downloaded and emailed, and does not break when the
``charts/`` directory is cleaned.

Formatting helpers are passed into the template rather than written in it.
Jinja should place values, not decide how a currency or a percentage looks.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path

from .models import ReportModel

log = logging.getLogger(__name__)

_TEMPLATE_PATH = Path(__file__).parent / "template.html"


def _data_uri(path: str | None) -> str:
    """Inline an image file as a data URI. Empty string when unavailable."""
    if not path:
        return ""
    try:
        raw = Path(path).read_bytes()
    except OSError:
        log.warning("chart not readable: %s", path)
        return ""
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


def _formatters(report: ReportModel) -> dict:
    currency = report.currency

    def money(value) -> str:
        if value is None:
            return "-"
        magnitude = abs(value)
        for threshold, suffix in ((1e12, "tn"), (1e9, "bn"), (1e6, "m")):
            if magnitude >= threshold:
                return f"{value / threshold:,.1f}{suffix}"
        return f"{value:,.0f}"

    def pct(value) -> str:
        return "-" if value is None else f"{value * 100:.1f}%"

    def num(value) -> str:
        return "-" if value is None else f"{value:,.1f}x"

    def fmt_price(value) -> str:
        return "-" if value is None else f"{currency} {value:,.2f}"

    return {"money": money, "pct": pct, "num": num, "fmt_price": fmt_price}


def render_html(report: ReportModel) -> str:
    """Render the report to a self-contained HTML fragment."""
    from jinja2 import Environment, select_autoescape

    environment = Environment(autoescape=select_autoescape(["html"]))
    template = environment.from_string(_TEMPLATE_PATH.read_text(encoding="utf-8"))

    # Swap file paths for inline data so the output stands alone.
    inlined = {name: _data_uri(path) for name, path in report.charts.items()}
    view = type(report)(**{**report.__dict__, "charts": inlined})

    return template.render(r=view, **_formatters(report))


def render_document(report: ReportModel) -> str:
    """A complete standalone HTML page, for download."""
    body = render_html(report)
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        f"<title>{report.company_name} ({report.ticker}) - Equity Research</title>\n"
        "</head>\n<body style=\"margin:0;background:#f6f8fa\">\n"
        f"{body}\n</body>\n</html>\n"
    )
