"""Equity research report: typed model rendered to HTML and PDF."""

from .builder import build_report
from .html import render_document, render_html
from .models import KPI, Evidence, ReportModel, Section

__all__ = [
    "build_report",
    "render_html",
    "render_document",
    "render_pdf",
    "ReportModel",
    "Section",
    "Evidence",
    "KPI",
]


def render_pdf(report, output_path):
    """Render the report to PDF. Imported lazily to keep reportlab optional."""
    from .pdf import render_pdf as _render

    return _render(report, output_path)
