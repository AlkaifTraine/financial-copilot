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
        stat("Our fair value (DCF)", f"{currency} {report.money(report.fair_value)}"),
        stat("Upside to our value", report.upside_display, upside_colour),
    ]
    if report.consensus_target:
        verdict_cells.append(
            stat("Consensus (market)", f"{currency} {report.money(report.consensus_target)}")
        )
    elif report.market_implied_growth is not None:
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
        consensus_bit = (
            f" and a Wall Street consensus target of {currency} "
            f"{report.money(report.consensus_target)} (the market's view, which our thesis "
            f"weighs against)"
            if report.consensus_target
            else ""
        )
        lead = (
            f"<b>Reading the valuation.</b> Our intrinsic DCF values {report.ticker} at "
            f"{currency} {report.money(report.fair_value)}, against a market price of "
            f"{currency} {report.money(report.share_price)}{consensus_bit}."
        )
        story.append(Paragraph(
            f"{lead} Holding every other assumption fixed, today's price implies "
            f"first-year revenue growth of about {report.market_implied_growth * 100:.0f}%, "
            f"decaying toward the terminal rate. Judge that against the reported history "
            f"below rather than treating the rating as a forecast.",
            styles["body"],
        ))

    # -- investment thesis (the analytical core) --------------------------
    thesis = report.thesis
    if thesis:
        story.append(Paragraph("INVESTMENT THESIS", styles["h2"]))
        if thesis.get("headline"):
            story.append(Paragraph(
                f'<b>{thesis["headline"]}</b>', styles["standfirst"]
            ))
        quality = thesis.get("business_quality")
        if quality:
            story.append(Paragraph(
                f'<b>Business quality:</b> {quality} &nbsp;·&nbsp; '
                f'<b>Stock:</b> {report.rating}'
                + (f' — {thesis["business_quality_note"]}' if thesis.get("business_quality_note") else ""),
                styles["small"],
            ))
        # The causal chain: market -> us -> fundamental difference -> financial
        # -> valuation -> recommendation. Falls back to the flat gap if the model
        # did not produce the chain fields.
        chain = [
            ("what_market_prices_in", "1 · What the market prices in."),
            ("what_we_expect", "2 · What we expect."),
            ("fundamental_difference", "3 · Why they differ (fundamental)."),
            ("financial_impact", "4 · Financial impact."),
            ("valuation_impact", f"5 · Valuation impact -> {report.rating}."),
        ]
        rendered_chain = False
        for key, lead in chain:
            if thesis.get(key):
                story.append(Paragraph(f"<b>{lead}</b> {thesis[key]}", styles["body"]))
                rendered_chain = True
        if not rendered_chain and thesis.get("the_gap"):
            story.append(Paragraph(f"<b>The gap.</b> {thesis['the_gap']}", styles["body"]))
        if thesis.get("thesis_points"):
            story.append(Paragraph(f"<b>Why {report.rating}</b>", styles["body"]))
            story.append(ListFlowable(
                [ListItem(Paragraph(p, styles["bullet"]), leftIndent=10)
                 for p in thesis["thesis_points"]],
                bulletType="1", leftIndent=12,
            ))
        cat_risk = []
        if thesis.get("biggest_catalyst"):
            cat_risk.append(f"<b>Biggest catalyst.</b> {thesis['biggest_catalyst']}")
        if thesis.get("biggest_risk"):
            cat_risk.append(f"<b>Biggest risk.</b> {thesis['biggest_risk']}")
        for line in cat_risk:
            story.append(Paragraph(line, styles["body"]))

        if thesis.get("bull_triggers"):
            story.append(Paragraph("<b>We would turn more positive if:</b>", styles["small"]))
            story.append(ListFlowable(
                [ListItem(Paragraph(x, styles["bullet"]), leftIndent=10) for x in thesis["bull_triggers"]],
                bulletType="bullet", start="–", leftIndent=12, bulletFontSize=7,
            ))
        if thesis.get("bear_triggers"):
            story.append(Paragraph("<b>We would turn more negative if:</b>", styles["small"]))
            story.append(ListFlowable(
                [ListItem(Paragraph(x, styles["bullet"]), leftIndent=10) for x in thesis["bear_triggers"]],
                bulletType="bullet", start="–", leftIndent=12, bulletFontSize=7,
            ))
        if thesis.get("red_team"):
            story.append(Paragraph(
                f"<b>The case against our view.</b> {thesis['red_team']}", styles["small"]
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

        if section.implication:
            story.append(Paragraph(
                f"<b>What it means.</b> {section.implication}", styles["body"]
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

    # -- quantified risks -------------------------------------------------
    if report.risks:
        story.append(Paragraph("KEY RISKS", styles["h2"]))
        story.append(Paragraph(
            "The risks most material to the investment case, ranked by how much damage each "
            "would do — not by filing order. Each carries what it hits, what it does to our "
            "fair value, and the single indicator to watch for it starting.",
            styles["small"],
        ))
        story.append(Spacer(1, 4))

        header = ["Risk", "Likelihood", "Financial impact", "Valuation impact", "What to watch"]
        rows = [[Paragraph(h, styles["cellb"]) for h in header]]
        for risk in report.risks:
            name = risk.get("risk", "")
            description = risk.get("description", "")
            risk_cell = f"<b>{name}</b>"
            if description:
                risk_cell += f'<br/><font size="6.8" color="#7c8794">{description}</font>'
            valuation_cell = risk.get("valuation_impact", "")
            if risk.get("basis"):
                valuation_cell += f'<br/><font size="6.5" color="#7c8794">Basis: {risk["basis"]}</font>'
            rows.append([
                Paragraph(risk_cell, styles["cell"]),
                Paragraph(risk.get("probability", ""), styles["cell"]),
                Paragraph(risk.get("financial_impact", ""), styles["cell"]),
                Paragraph(valuation_cell, styles["cell"]),
                Paragraph(risk.get("early_warning", ""), styles["cell"]),
            ])
        widths = [CONTENT_WIDTH * w for w in (0.22, 0.11, 0.23, 0.22, 0.22)]
        table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), SURFACE_2),
            ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (0, 0), (-1, -1), "LEFT"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(table)

    # -- competitive moat + management vs us ------------------------------
    cp = report.competitive
    if cp:
        story.append(Paragraph("COMPETITIVE MOAT", styles["h2"]))
        story.append(Paragraph(
            f"<b>Moat strength (today):</b> {cp.get('moat_strength', 'Undetermined')} "
            f"&nbsp;·&nbsp; <b>Direction:</b> {cp.get('moat_direction', 'Stable')}",
            styles["small"],
        ))
        if cp.get("moat_summary"):
            story.append(Paragraph(cp["moat_summary"], styles["body"]))
        if cp.get("strength_basis"):
            story.append(Paragraph(cp["strength_basis"], styles["small"]))
        if cp.get("dimensions"):
            story.append(Spacer(1, 3))
            header = ["Source of advantage", "Assessment", "Strength"]
            rows = [[Paragraph(h, styles["cellb"]) for h in header]]
            for dim in cp["dimensions"]:
                assessment = dim.get("assessment", "")
                if dim.get("evidence"):
                    assessment += f'<br/><font size="6.5" color="#7c8794">Evidence: {dim["evidence"]}</font>'
                strength = dim.get("strength", "")
                if dim.get("confidence"):
                    strength += f'<br/><font size="6.5" color="#7c8794">{dim["confidence"]} conf.</font>'
                rows.append([
                    Paragraph(f"<b>{dim.get('dimension', '')}</b>", styles["cell"]),
                    Paragraph(assessment, styles["cell"]),
                    Paragraph(strength, styles["cell"]),
                ])
            widths = [CONTENT_WIDTH * w for w in (0.28, 0.55, 0.17)]
            table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), SURFACE_2),
                ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(table)
        if cp.get("competitive_threats"):
            story.append(Paragraph("<b>Threats to the moat:</b>", styles["small"]))
            story.append(ListFlowable(
                [ListItem(Paragraph(s, styles["bullet"]), leftIndent=10) for s in cp["competitive_threats"]],
                bulletType="bullet", start="–", leftIndent=12, bulletFontSize=7,
            ))
        if cp.get("management_vs_us"):
            story.append(Paragraph("MANAGEMENT SAYS VS OUR VIEW", styles["h2"]))
            story.append(Paragraph(
                "Only actual, sourced management statements appear — never a forecast put in "
                "management's mouth. Each runs from what they said to what we model to the "
                "difference; the rows where we are more cautious are where the thesis lives.",
                styles["small"],
            ))
            story.append(Spacer(1, 4))
            header = ["Topic", "Management statement (source)", "Our interpretation / model", "Difference"]
            rows = [[Paragraph(h, styles["cellb"]) for h in header]]
            for claim in cp["management_vs_us"]:
                statement = claim.get("management_statement", "")
                if claim.get("source"):
                    dated = claim["source"]
                    if claim.get("source_date"):
                        dated += f' &middot; {claim["source_date"]}'
                    statement += f'<br/><font size="6.5" color="#7c8794">Source: {dated}</font>'
                view = claim.get("our_interpretation", "")
                if claim.get("our_model"):
                    view += f'<br/><font size="6.5" color="#7c8794">We model: {claim["our_model"]}</font>'
                rows.append([
                    Paragraph(f"<b>{claim.get('topic', '')}</b>", styles["cell"]),
                    Paragraph(statement, styles["cell"]),
                    Paragraph(view, styles["cell"]),
                    Paragraph(claim.get("difference", ""), styles["cell"]),
                ])
            widths = [CONTENT_WIDTH * w for w in (0.15, 0.32, 0.32, 0.21)]
            table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), SURFACE_2),
                ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(table)

    # -- upcoming catalysts + monitoring dashboard ------------------------
    fw = report.forward
    if fw and (fw.get("catalysts") or fw.get("watch_items")):
        if fw.get("catalysts"):
            story.append(Paragraph("UPCOMING CATALYSTS", styles["h2"]))
            story.append(Paragraph(
                "The dated events most likely to move the stock, the direction each points, and "
                "the figure it would move. These are when the thesis gets tested.",
                styles["small"],
            ))
            story.append(Spacer(1, 4))
            header = ["Event", "Timing", "Direction", "Moves"]
            rows = [[Paragraph(h, styles["cellb"]) for h in header]]
            for cat in fw["catalysts"]:
                event_cell = f"<b>{cat.get('event', '')}</b>"
                if cat.get("source"):
                    event_cell += f'<br/><font size="6.5" color="#7c8794">{cat["source"]}</font>'
                rows.append([
                    Paragraph(event_cell, styles["cell"]),
                    Paragraph(cat.get("timing", ""), styles["cell"]),
                    Paragraph(cat.get("direction", ""), styles["cell"]),
                    Paragraph(cat.get("metric", ""), styles["cell"]),
                ])
            widths = [CONTENT_WIDTH * w for w in (0.30, 0.22, 0.16, 0.32)]
            table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), SURFACE_2),
                ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(table)

        if fw.get("watch_items"):
            story.append(Paragraph("WHAT TO WATCH", styles["h2"]))
            story.append(Paragraph(
                "Each metric tests a specific assumption in our model — a falsifiability check on "
                "our own thesis. It states what our assumption expects and the value or trend that "
                "would push us more bullish or more bearish.",
                styles["small"],
            ))
            story.append(Spacer(1, 4))
            header = ["Metric (tests)", "Current", "Trend", "What we expect", "More bullish / bearish if"]
            rows = [[Paragraph(h, styles["cellb"]) for h in header]]
            for item in fw["watch_items"]:
                metric = f"<b>{item.get('metric', '')}</b>"
                if item.get("assumption"):
                    metric += f'<br/><font size="6.5" color="#7c8794">tests: {item["assumption"]}</font>'
                rows.append([
                    Paragraph(metric, styles["cell"]),
                    Paragraph(item.get("current", ""), styles["cell"]),
                    Paragraph(item.get("trend", ""), styles["cell"]),
                    Paragraph(item.get("expected", ""), styles["cell"]),
                    Paragraph(item.get("bull_bear", ""), styles["cell"]),
                ])
            widths = [CONTENT_WIDTH * w for w in (0.19, 0.15, 0.11, 0.25, 0.30)]
            table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
            table.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), SURFACE_2),
                ("LINEBELOW", (0, 0), (-1, -1), 0.4, RULE),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]))
            story.append(table)

    # -- valuation triangulation ------------------------------------------
    if report.blended and len(report.blended.get("estimates", [])) > 1:
        bl = report.blended
        story.append(PageBreak())
        story.append(Paragraph("VALUATION SUMMARY", styles["h2"]))
        story.append(Paragraph(
            "The target price is our <b>intrinsic DCF (base case)</b>. The scenario expected "
            "value, the comparable-companies value and the Wall Street consensus are shown as "
            "independent <b>cross-checks</b>, not blended into the target: the scenario set is "
            "built from the same DCF (blending would double-count it), a peer multiple is a "
            "different valuation basis, and the consensus is the market's external view.",
            styles["small"],
        ))
        story.append(Spacer(1, 4))

        def _role(est: dict) -> str:
            if est.get("key") == "dcf":
                return "Target (intrinsic)"
            if est.get("key") == "analyst_consensus":
                return "External reference"
            return "Cross-check"

        rows = [["Method", "Value / share", "Role", "Detail"]]
        for est in bl["estimates"]:
            rows.append([
                est["label"], f"{currency} {est['value_per_share']:,.2f}", _role(est),
                est.get("detail") or est.get("source_name") or "",
            ])
        widths = [CONTENT_WIDTH * w for w in (0.22, 0.18, 0.16, 0.44)]
        table = _table(rows, widths, styles)
        table.setStyle(TableStyle([("ALIGN", (3, 0), (-1, -1), "LEFT")]))
        story.append(table)

        upside_text = (
            f" ({bl['upside'] * 100:+.1f}% vs price)"
            if bl.get("upside") is not None else ""
        )
        story.append(Paragraph(
            f"<b>Target price (intrinsic DCF) {currency} {bl['blended_value']:,.2f}{upside_text}.</b> "
            f"The cross-checks above are shown for triangulation; they are not averaged into the target.",
            styles["small"],
        ))
        if report.valuation_confidence:
            drivers = (
                f" Lowered by: {'; '.join(report.confidence_drivers)}. Read the fair value as a "
                f"central estimate within a wide range, not a precise number."
                if report.confidence_drivers else ""
            )
            story.append(Paragraph(
                f"<b>Valuation confidence: {report.valuation_confidence}.</b>{drivers}", styles["small"],
            ))

    # -- what is priced in (reverse DCF) ----------------------------------
    if report.priced_in and report.priced_in.get("rows"):
        pi = report.priced_in
        price_text = f"{currency} {report.money(pi.get('share_price'))}"
        story.append(Paragraph("WHAT IS PRICED IN", styles["h2"]))
        story.append(Paragraph(
            "A DCF whose fair value sits below the price is easy to dismiss as the model "
            "being wrong. Inverting it is harder to dismiss: holding every other driver at our "
            f"base case, we solve for the level of each one that would return today's price of "
            f"{price_text}. Each row is a single, checkable claim about what the market must "
            "believe — judge it against the company's own history.",
            styles["small"],
        ))
        story.append(Spacer(1, 4))

        rows = [["Driver", "Our base case", f"Priced in at {price_text}", "Gap"]]
        for row in pi["rows"]:
            rows.append([
                row["label"], row["base_display"], row["implied_display"], row["gap_display"],
            ])
        widths = [CONTENT_WIDTH * w for w in (0.34, 0.22, 0.28, 0.16)]
        story.append(_table(rows, widths, styles))

        for row in pi["rows"]:
            if row.get("note"):
                story.append(Paragraph(
                    f"<b>{row['label']}.</b> {row['note']}", styles["small"]
                ))
        if pi.get("summary"):
            story.append(Paragraph(f"<b>What's doing the work.</b> {pi['summary']}", styles["body"]))
        story.append(Paragraph(
            "Read together, these are what the price assumes if the market is right about "
            "everything else. A gap the company has never closed in its own history is the risk "
            "in owning it here; a dash means the price cannot be reached on that lever alone.",
            styles["small"],
        ))

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

    # -- segment forecast (bottom-up cross-check) -------------------------
    sf = report.segment_forecast
    if sf and sf.get("segments"):
        end_year = sf["base_year"] + sf["horizon"]
        story.append(Paragraph("SEGMENT FORECAST: A BOTTOM-UP CROSS-CHECK", styles["h2"]))
        story.append(Paragraph(
            "The headline DCF grows total revenue top-down. This forecasts each reportable "
            "segment on its own trajectory and sums them, as a check on that single blended "
            "number. The headline valuation is unchanged; this is the stress-test on its "
            "central assumption.",
            styles["small"],
        ))
        story.append(Spacer(1, 4))

        rows = [["Segment", f"Revenue FY{sf['base_year']}", "Yr-1 growth",
                 f"{sf['horizon']}y CAGR", f"Revenue FY{end_year}"]]
        for seg in sf["segments"]:
            rows.append([
                seg["name"], _money(seg["latest_revenue"]), _pct(seg["year_one_growth"]),
                _pct(seg["implied_cagr"]), _money(seg["terminal_revenue"]),
            ])
        rows.append([
            "Bottom-up total", _money(sf["latest_segment_sum"]), "-",
            _pct(sf["bottom_up_cagr"]), _money(sf["bottom_up_terminal_revenue"]),
        ])
        totals_row = len(rows) - 1
        if sf.get("top_down_cagr") is not None:
            rows.append([
                "Top-down (headline DCF)", "-", "-", _pct(sf["top_down_cagr"]), "-",
            ])
        widths = [CONTENT_WIDTH * w for w in (0.32, 0.19, 0.15, 0.15, 0.19)]
        table = _table(rows, widths, styles)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, totals_row), (-1, totals_row), SURFACE_2),
            ("FONTNAME", (0, totals_row), (0, totals_row), "Helvetica-Bold"),
        ]))
        story.append(table)
        if sf.get("top_down_cagr") is not None:
            gap = (sf["bottom_up_cagr"] - sf["top_down_cagr"]) * 100
            story.append(Paragraph(
                f"<b>Reconciliation:</b> bottom-up {_pct(sf['bottom_up_cagr'])} vs top-down "
                f"{_pct(sf['top_down_cagr'])} — a {gap:+.0f}pp gap. The headline valuation uses the "
                f"more conservative TOP-DOWN path; the bottom-up sum is the cross-check.",
                styles["small"],
            ))
        story.append(Paragraph(sf["note"], styles["small"]))

    # -- scenarios --------------------------------------------------------
    if report.scenarios and report.scenarios.get("cases"):
        sc = report.scenarios
        story.append(Paragraph("SCENARIOS: BEAR, BASE, BULL", styles["h2"]))
        story.append(Paragraph(
            "Three coherent states of the world. Unlike the sensitivity grid, each "
            "scenario moves growth, margin, terminal growth and the discount rate "
            "<b>together</b>. The bear-to-bull spread is sized from this company's own "
            "historical volatility, not a fixed percentage. The probabilities are "
            "<b>analyst-assigned</b> (a deliberately conservative 25/50/25 prior), not "
            "statistically estimated.",
            styles["small"],
        ))
        story.append(Spacer(1, 4))

        rows = [["Scenario", "Yr-1 growth", "Term. margin", "Term. growth",
                 "WACC", "Fair value", "vs price", "Weight"]]
        for case in sc["cases"]:
            drivers = case["drivers"]
            upside = case.get("upside")
            rows.append([
                case["label"],
                _pct(drivers[0]["value"]),
                _pct(drivers[1]["value"]),
                _pct(drivers[2]["value"]),
                _pct(drivers[3]["value"]),
                f"{currency} {case['fair_value_per_share']:,.2f}",
                f"{upside * 100:+.0f}%" if upside is not None else "-",
                f"{case['probability'] * 100:.0f}%",
            ])
        widths = [CONTENT_WIDTH * w for w in (0.14, 0.13, 0.13, 0.13, 0.10, 0.16, 0.11, 0.10)]
        story.append(_table(rows, widths, styles))

        range_text = ""
        if sc.get("value_range"):
            low, high = sc["value_range"]
            range_text = f" Range {currency} {low:,.2f} to {currency} {high:,.2f}"
            if sc.get("dispersion") is not None:
                range_text += f", a spread of {sc['dispersion'] * 100:.0f}% of the base value"
            range_text += "."
        weighted = (
            f" ({sc['expected_upside'] * 100:+.1f}% vs price)"
            if sc.get("expected_upside") is not None else ""
        )
        story.append(Paragraph(
            f"<b>Probability-weighted value {currency} {sc['expected_value']:,.2f}"
            f"{weighted}.</b>{range_text} Read as a range, not a point.",
            styles["small"],
        ))
        story.append(Spacer(1, 3))
        for case in sc["cases"]:
            story.append(Paragraph(
                f"<b>{case['label']}.</b> {case['narrative']}", styles["small"]
            ))

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
            derivation = item["derivation"]
            prov = item.get("provenance") or {}
            if prov:
                chain = " &nbsp; ".join(
                    f'<font color="#7c8794">{k.capitalize()}:</font> {prov[k]}'
                    for k in ("historical", "guidance", "industry", "competitive", "management")
                    if prov.get(k)
                )
                derivation += (
                    f'<br/><font size="6.5">Provenance: {chain} '
                    f'<b>&#8594; {item["display"]}</b></font>'
                )
            bridge = item.get("bridge") or []
            if bridge:
                steps = " ".join(
                    f'{step["component"]}: {step["value"]:+.1f}pp' if i else
                    f'{step["component"]}: {step["value"]:.1f}pp'
                    for i, step in enumerate(bridge)
                )
                conf = f' (confidence: {item["confidence"]})' if item.get("confidence") else ""
                derivation += (
                    f'<br/><font size="6.5">Margin bridge{conf}: {steps} '
                    f'<b>&#8594; {item["display"]}</b></font>'
                )
            rows.append([item["label"], item["display"], basis, derivation])
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
        rows = [["Company", "Role", "Market cap", "P/E", "EV/EBITDA", "EV/Sales"]]
        for peer in report.comps["peers"]:
            name = f"{peer['name']} ({peer['ticker']})"
            if peer.get("rationale"):
                name += f'<br/><font size="6.5" color="#7c8794">{peer["rationale"]}</font>'
            rows.append([
                name,
                "Direct" if peer.get("tier", "direct") == "direct" else "Adjacent*",
                _money(peer["market_cap"]),
                f"{peer['pe']:,.1f}x" if peer.get("pe") else "-",
                f"{peer['ev_to_ebitda']:,.1f}x" if peer.get("ev_to_ebitda") else "-",
                f"{peer['ev_to_sales']:,.1f}x" if peer.get("ev_to_sales") else "-",
            ])
        widths = [CONTENT_WIDTH * w for w in (0.34, 0.12, 0.14, 0.12, 0.14, 0.14)]
        story += [Paragraph("COMPARABLE COMPANIES", styles["h2"]),
                  _table(rows, widths, styles)]
        for note in report.comps.get("notes", []):
            story.append(Paragraph(note, styles["small"]))

    # -- QA status + limitations ------------------------------------------
    status_note = (
        " — no critical issues; the notes below are non-blocking."
        if report.qa_status == "PASSED" else ""
    )
    story.append(Paragraph(f"<b>QA STATUS: {report.qa_status}</b>{status_note}", styles["small"]))
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
            fresh = (
                f' &middot; <b>{source["freshness"]}</b>' if source.get("freshness") else ""
            )
            story.append(Paragraph(
                f"<b>{source['label']}</b>{filed}{fresh}<br/>"
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
