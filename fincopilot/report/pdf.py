"""
PDF rendering with ReportLab.

Renders the same :class:`ReportModel` the HTML renderer consumes, so the two
outputs cannot disagree about substance. Nothing here parses text or computes
values: every field arrives ready to place.

ReportLab rather than an HTML-to-PDF converter because it is pure Python with
no system dependencies. WeasyPrint needs GTK/Pango, which does not install on
Windows without extra work, and headless Chromium is a ~150MB download that the
deployment target cannot afford.
"""

from __future__ import annotations

import logging
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from .models import ReportModel

log = logging.getLogger(__name__)

NAVY = colors.HexColor("#0f2942")
ACCENT = colors.HexColor("#2a78d6")
POSITIVE = colors.HexColor("#137333")
NEGATIVE = colors.HexColor("#b3261e")
INK = colors.HexColor("#14181f")
INK_2 = colors.HexColor("#4a5460")
MUTED = colors.HexColor("#7c8794")
RULE = colors.HexColor("#e3e7ec")
SURFACE_2 = colors.HexColor("#f6f8fa")

PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 16 * mm
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN


def _styles() -> dict:
    base = getSampleStyleSheet()
    return {
        "company": ParagraphStyle(
            "company", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=20, leading=24, textColor=NAVY, alignment=TA_LEFT, spaceAfter=0,
        ),
        "ident": ParagraphStyle(
            "ident", parent=base["Normal"], fontSize=8.5, leading=11, textColor=MUTED,
        ),
        "h2": ParagraphStyle(
            "h2", parent=base["Heading2"], fontName="Helvetica-Bold", fontSize=10.5,
            leading=13, textColor=NAVY, spaceBefore=14, spaceAfter=5,
        ),
        "standfirst": ParagraphStyle(
            "standfirst", parent=base["Normal"], fontName="Helvetica-Oblique",
            fontSize=9.5, leading=13, textColor=INK_2, spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "body", parent=base["BodyText"], fontSize=9, leading=13,
            textColor=INK, spaceAfter=6,
        ),
        "bullet": ParagraphStyle(
            "bullet", parent=base["BodyText"], fontSize=9, leading=12.5,
            textColor=INK, spaceAfter=2,
        ),
        "small": ParagraphStyle(
            "small", parent=base["Normal"], fontSize=7.5, leading=10, textColor=MUTED,
        ),
        "cell": ParagraphStyle(
            "cell", parent=base["Normal"], fontSize=7.8, leading=10, textColor=INK,
        ),
        "cellb": ParagraphStyle(
            "cellb", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=7.8, leading=10, textColor=INK,
        ),
    }


def _footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setStrokeColor(RULE)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN, 12 * mm, PAGE_WIDTH - MARGIN, 12 * mm)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, 8 * mm, doc.report_title)
    canvas.drawRightString(PAGE_WIDTH - MARGIN, 8 * mm, f"Page {canvas.getPageNumber()}")
    canvas.restoreState()


def _money(value, blank: str = "-") -> str:
    if value is None:
        return blank
    magnitude = abs(value)
    for threshold, suffix in ((1e12, "tn"), (1e9, "bn"), (1e6, "m")):
        if magnitude >= threshold:
            return f"{value / threshold:,.1f}{suffix}"
    return f"{value:,.0f}"


def _pct(value, blank: str = "-") -> str:
    return blank if value is None else f"{value * 100:.1f}%"


def _table(rows: list[list], widths: list[float], styles: dict) -> Table:
    """A consistently styled data table."""
    body = [
        [Paragraph(str(c), styles["cellb"] if i == 0 else styles["cell"]) for c in row]
        for i, row in enumerate(rows)
    ]
    table = Table(body, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), SURFACE_2),
                ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 0), (-1, -1), "RIGHT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
            ]
        )
    )
    return table


def _chart(path: str | None, max_height: float = 62 * mm):
    """Scale a chart to the content width, preserving aspect ratio."""
    if not path or not Path(path).exists():
        return None
    try:
        from reportlab.lib.utils import ImageReader

        width_px, height_px = ImageReader(path).getSize()
    except Exception:
        return None

    width = CONTENT_WIDTH
    height = width * height_px / width_px
    if height > max_height:
        height = max_height
        width = height * width_px / height_px
    return Image(path, width=width, height=height)


def render_pdf(report: ReportModel, output_path: str | Path) -> Path:
    """Render ``report`` to a PDF file and return its path."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    styles = _styles()

    document = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=14 * mm, bottomMargin=18 * mm,
        title=f"{report.company_name} ({report.ticker}) Equity Research",
        author="Financial Copilot",
    )
    document.report_title = f"{report.company_name} ({report.ticker}) - equity research"

    story: list = []
    currency = report.currency

    # -- masthead ---------------------------------------------------------
    rating_colour = {"positive": POSITIVE, "negative": NEGATIVE}.get(
        report.rating_tone, MUTED
    )
    identity = f"{report.ticker}"
    if report.exchange:
        identity += f" &middot; {report.exchange}"
    if report.sector:
        identity += f" &middot; {report.sector}"
    identity += f" &middot; {report.as_of}"

    rating_cell = Table(
        [[Paragraph(
            f'<font size="14" color="#ffffff"><b>{report.rating}</b></font>',
            styles["cell"],
        )],
         [Paragraph('<font size="6.5" color="#ffffff">DCF-derived</font>', styles["cell"])]],
        colWidths=[34 * mm],
    )
    rating_cell.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), rating_colour),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, 0), 6),
        ("BOTTOMPADDING", (0, 1), (-1, 1), 6),
    ]))

    masthead = Table(
        [[
            [Paragraph(report.company_name, styles["company"]),
             Paragraph(identity, styles["ident"])],
            rating_cell,
        ]],
        colWidths=[CONTENT_WIDTH - 36 * mm, 36 * mm],
    )
    masthead.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("LINEBELOW", (0, 0), (-1, -1), 1.6, NAVY),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 8),
    ]))
    story += [masthead, Spacer(1, 8)]

    # -- verdict ----------------------------------------------------------
    upside_colour = "#137333" if (report.upside or 0) > 0 else "#b3261e"

    def stat(label: str, value: str, colour: str = "#14181f") -> list:
        return [
            Paragraph(f'<font size="6.5" color="#7c8794">{label.upper()}</font>',
                      styles["cell"]),
            Paragraph(f'<font size="11" color="{colour}"><b>{value}</b></font>',
                      styles["cell"]),
        ]

    verdict_cells = [
        stat("Market price", f"{currency} {report.money(report.share_price)}"),
        stat("DCF fair value", f"{currency} {report.money(report.fair_value)}"),
        stat("Upside", report.upside_display, upside_colour),
    ]
    if report.market_implied_growth is not None:
        verdict_cells.append(
            stat("Growth priced in", f"{report.market_implied_growth * 100:.0f}% p.a.")
        )

    verdict = Table([verdict_cells], colWidths=[CONTENT_WIDTH / len(verdict_cells)] * len(verdict_cells))
    verdict.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), SURFACE_2),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story += [verdict, Spacer(1, 8)]

    # -- KPI strip --------------------------------------------------------
    if report.kpis:
        cells = [
            [Paragraph(f'<font size="6.5" color="#7c8794">{k.label.upper()}</font>', styles["cell"]),
             Paragraph(f'<font size="10"><b>{k.value}</b></font>', styles["cell"]),
             Paragraph(f'<font size="6.5" color="#4a5460">{k.caption}</font>', styles["cell"])]
            for k in report.kpis
        ]
        kpi_table = Table([cells], colWidths=[CONTENT_WIDTH / len(cells)] * len(cells))
        kpi_table.setStyle(TableStyle([
            ("BOX", (0, 0), (-1, -1), 0.4, RULE),
            ("INNERGRID", (0, 0), (-1, -1), 0.4, RULE),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("TOPPADDING", (0, 0), (-1, -1), 6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ]))
        story += [kpi_table, Spacer(1, 6)]

    # -- how to read ------------------------------------------------------
    if report.market_implied_growth is not None and report.share_price:
        story.append(Paragraph(
            f"<b>Reading the valuation.</b> The model values {report.ticker} at "
            f"{currency} {report.money(report.fair_value)} against a market price of "
            f"{currency} {report.money(report.share_price)}. Holding every other "
            f"assumption fixed, today's price implies first-year revenue growth of about "
            f"{report.market_implied_growth * 100:.0f}%, decaying toward the terminal "
            f"rate. Judge that against the reported history below rather than treating "
            f"the rating as a forecast.",
            styles["body"],
        ))

    # -- narrative sections -----------------------------------------------
    for section in report.sections:
        block: list = [Paragraph(section.title.upper(), styles["h2"])]
        if section.summary:
            block.append(Paragraph(section.summary, styles["standfirst"]))
        story += block

        for paragraph in section.paragraphs:
            story.append(Paragraph(paragraph, styles["body"]))

        if section.bullets:
            story.append(ListFlowable(
                [ListItem(Paragraph(b, styles["bullet"]), leftIndent=10)
                 for b in section.bullets],
                bulletType="bullet", start="–", leftIndent=10, bulletFontSize=7,
            ))

        if section.evidence:
            # Numbered so the inline [n] markers in the prose above resolve to a
            # specific document and page the reader can open.
            listing = "<br/>".join(
                f"[{i}] {e.label}" for i, e in enumerate(section.evidence, start=1)
            )
            story.append(Paragraph(f"<b>Sources cited above.</b><br/>{listing}",
                                   styles["small"]))

        # Financial exhibits belong with the financial narrative.
        if section.key == "financials":
            for name in ("revenue", "margin", "fcf"):
                chart = _chart(report.charts.get(name))
                if chart:
                    story += [Spacer(1, 4), chart]

            if report.financial_table:
                rows = [["Fiscal year", "Revenue", "Growth", "Op. income",
                         "Op. margin", "Net income", "FCF"]]
                for row in report.financial_table:
                    rows.append([
                        row["fiscal_year"], _money(row["revenue"]),
                        _pct(row["revenue_growth"]), _money(row["operating_income"]),
                        _pct(row["operating_margin"]), _money(row["net_income"]),
                        _money(row["free_cash_flow"]),
                    ])
                widths = [CONTENT_WIDTH * w for w in (0.16, 0.15, 0.12, 0.15, 0.13, 0.14, 0.15)]
                story += [Spacer(1, 4), _table(rows, widths, styles),
                          Paragraph(f"Figures in {currency}. Source: company filings.",
                                    styles["small"])]

    # -- DCF --------------------------------------------------------------
    if report.forecast_table:
        story.append(PageBreak())
        story.append(Paragraph("DISCOUNTED CASH FLOW", styles["h2"]))
        rows = [["Year", "Revenue", "Growth", "Op. margin", "Free cash flow", "Present value"]]
        for row in report.forecast_table:
            rows.append([
                row["year"], _money(row["revenue"]), _pct(row["revenue_growth"]),
                _pct(row["operating_margin"]), _money(row["free_cash_flow"]),
                _money(row["present_value"]),
            ])
        widths = [CONTENT_WIDTH * w for w in (0.14, 0.17, 0.14, 0.17, 0.19, 0.19)]
        story.append(_table(rows, widths, styles))

    # -- assumptions ------------------------------------------------------
    if report.assumptions:
        story.append(Paragraph("ASSUMPTIONS", styles["h2"]))
        story.append(Paragraph(
            "Every input to the model, how it was derived, and why. Inputs marked "
            "<b>bounded</b> were proposed by the language model and constrained to a "
            "range supported by reported history before entering the calculation.",
            styles["small"],
        ))
        story.append(Spacer(1, 4))

        rows = [["Assumption", "Value", "Basis", "Derivation"]]
        for item in report.assumptions:
            basis = item["source_label"] + (" (bounded)" if item["clamped"] else "")
            rows.append([item["label"], item["display"], basis, item["derivation"]])
        widths = [CONTENT_WIDTH * w for w in (0.22, 0.11, 0.19, 0.48)]
        table = _table(rows, widths, styles)
        table.setStyle(TableStyle([("ALIGN", (2, 0), (-1, -1), "LEFT")]))
        story.append(table)

    # -- sensitivity ------------------------------------------------------
    chart = _chart(report.charts.get("sensitivity"), max_height=78 * mm)
    if chart:
        story += [
            Paragraph("SENSITIVITY", styles["h2"]),
            Paragraph(
                "Fair value per share across the discount rate and terminal growth "
                "rate. Cells above the current market price are blue, below are red.",
                styles["small"],
            ),
            Spacer(1, 4), chart,
        ]

    # -- comparables ------------------------------------------------------
    if report.comps and report.comps.get("peers"):
        rows = [["Company", "Market cap", "P/E", "EV/EBITDA", "EV/Sales"]]
        for peer in report.comps["peers"]:
            rows.append([
                f"{peer['name']} ({peer['ticker']})", _money(peer["market_cap"]),
                f"{peer['pe']:,.1f}x" if peer.get("pe") else "-",
                f"{peer['ev_to_ebitda']:,.1f}x" if peer.get("ev_to_ebitda") else "-",
                f"{peer['ev_to_sales']:,.1f}x" if peer.get("ev_to_sales") else "-",
            ])
        widths = [CONTENT_WIDTH * w for w in (0.36, 0.18, 0.14, 0.16, 0.16)]
        story += [Paragraph("COMPARABLE COMPANIES", styles["h2"]),
                  _table(rows, widths, styles)]
        for note in report.comps.get("notes", []):
            story.append(Paragraph(note, styles["small"]))

    # -- limitations ------------------------------------------------------
    if report.warnings:
        story.append(Paragraph("MODEL NOTES AND LIMITATIONS", styles["h2"]))
        story.append(ListFlowable(
            [ListItem(Paragraph(w, styles["bullet"]), leftIndent=10)
             for w in report.warnings],
            bulletType="bullet", start="–", leftIndent=10, bulletFontSize=7,
        ))

    # -- sources ----------------------------------------------------------
    if report.sources:
        story.append(Paragraph("SOURCE DOCUMENTS", styles["h2"]))
        story.append(Paragraph(
            "Every statement in this report is drawn from the documents below.",
            styles["small"],
        ))
        story.append(Spacer(1, 3))
        for source in report.sources:
            filed = f" &middot; filed {source['filed']}" if source.get("filed") else ""
            story.append(Paragraph(
                f"<b>{source['label']}</b>{filed}<br/>"
                f'<font size="7" color="#2a78d6">{source["url"]}</font>',
                styles["small"],
            ))
            story.append(Spacer(1, 3))

    story += [
        Spacer(1, 10),
        Paragraph(report.disclaimer, styles["small"]),
    ]

    document.build(story, onFirstPage=_footer, onLaterPages=_footer)
    log.info("PDF written to %s", output_path)
    return output_path
