"""
Narrative section generation.

Each section runs its own targeted retrieval before its own model call. That
matters: a single retrieval for "write me a report" returns a generic blend,
whereas the risk section searching Item 1A and the growth section searching
MD&A each get evidence actually relevant to what they are writing.

The model returns **JSON matching a fixed schema**, never free prose. The
renderer then lays out known fields. This is the inversion of the old design,
where the model emitted markdown and the PDF layer tried to parse structure
back out of it with ``report_text.split("###")``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from .. import config
from ..llm import complete_json
from ..retrieve import retrieve
from .models import Evidence, Section

log = logging.getLogger(__name__)


@dataclass
class SectionSpec:
    key: str
    title: str
    query: str                 # what to retrieve before writing
    brief: str                 # what the section must cover
    bullet_label: str = "Key points"


# Ordered as they appear in the report.
SPECS: list[SectionSpec] = [
    SectionSpec(
        key="business",
        title="Business Overview",
        query=(
            "business description operating segments products customers "
            "revenue by segment geography what the company does"
        ),
        brief=(
            "What the company sells, to whom, and how it makes money. Name the "
            "operating segments and their relative size. Be concrete and specific."
        ),
    ),
    SectionSpec(
        key="financials",
        title="Financial Performance",
        query=(
            "revenue growth gross margin operating margin operating expenses "
            "cash flow from operations year over year change drivers"
        ),
        brief=(
            "How revenue, margins and cash flow moved, and WHY. Attribute changes "
            "to the drivers management gives. Quote figures with their periods."
        ),
    ),
    SectionSpec(
        key="growth",
        title="Growth Drivers",
        query=(
            "growth drivers demand outlook new products capacity expansion "
            "total addressable market strategy investment"
        ),
        brief=(
            "What is expected to drive revenue from here, grounded in management's "
            "own commentary. Distinguish drivers already visible in results from "
            "those still prospective."
        ),
        bullet_label="Drivers",
    ),
    SectionSpec(
        key="competition",
        title="Competitive Position",
        query=(
            "competition competitors market share competitive advantages "
            "barriers to entry pricing pressure differentiation"
        ),
        brief=(
            "Who the company competes with and what protects its position. Note "
            "where the filings concede competitive pressure."
        ),
        bullet_label="Competitive factors",
    ),
    # NOTE: the prose "Key Risks" section was replaced by a quantified risk table
    # (report/risks.py), rendered separately in the report. The risk-factor
    # retrieval query it used lives on there now.
    SectionSpec(
        key="outlook",
        title="Outlook",
        query=(
            "outlook guidance next quarter next fiscal year expectations "
            "capital allocation dividend buyback capital expenditure plans"
        ),
        brief=(
            "Management's stated expectations and capital allocation plans. Be "
            "explicit about what is guidance versus inference."
        ),
        bullet_label="What to watch",
    ),
]

_SYSTEM = """You are an equity research analyst writing one section of a research report.

Rules:
- Use ONLY the numbered sources provided. No outside knowledge.
- Cite inline as [n] after each factual claim.
- Quote figures exactly, with their period.
- Write densely and analytically. No filler, no hedging, no restating the brief.
- If the sources do not support a claim, leave it out.
- If the sources cover the topic thinly, write less rather than padding."""

_PROMPT = """Company: {company}
Section: {title}

Brief: {brief}

Sources:
{context}

Return JSON:
{{"summary": "one sentence standfirst, under 25 words",
  "paragraphs": ["2-4 analytical paragraphs"],
  "bullets": ["3-6 short specific points, each with a figure or a named fact where possible"]}}"""


_CITATION = re.compile(r"\[\s*(?:SOURCE\s*)?((?:\d+\s*[,;]\s*)*\d+)\s*\]", re.I)


def _normalise_citations(text: str, evidence_count: int) -> str:
    """Rewrite citation markers to a single numbered form, dropping dead ones.

    The model emits `[SOURCE 1,4]`, `[1,4]` and `[4][5]` interchangeably. All
    three are normalised to `[1][4]`, and any number beyond the evidence list is
    removed — a marker that points at nothing is worse than no marker, because
    it invites a reader to check a source that does not exist.
    """

    def replace(match: re.Match) -> str:
        numbers = [int(n) for n in re.split(r"[,;]", match.group(1)) if n.strip()]
        keep = [n for n in numbers if 1 <= n <= evidence_count]
        return "".join(f"[{n}]" for n in keep)

    return _CITATION.sub(replace, text)


def _evidence_from(result) -> list[Evidence]:
    return [
        Evidence(
            doc_title=passage.chunk.doc_title,
            section=passage.chunk.section,
            page=passage.chunk.page,
            source_url=passage.chunk.source_url,
            snippet=passage.chunk.body[:300],
        )
        for passage in result.passages
    ]


def build_section(
    spec: SectionSpec,
    index,
    company_name: str,
    *,
    latest_fiscal_year: int | None = None,
) -> Section:
    """Retrieve evidence for one section and write it."""
    section = Section(key=spec.key, title=spec.title)

    # Bias retrieval toward the most recent filing year. Without it the sections
    # drew heavily on the FY2024 and FY2025 filings even though FY2026 results
    # were indexed, because three years of near-identical 10-K language compete
    # on equal terms. The year filter is soft (see retrieve.pipeline), so
    # comparatives are still reachable when the current year is thin.
    query = spec.query
    if latest_fiscal_year:
        query = f"{query} FY{latest_fiscal_year}"

    result = retrieve(query, index, top_k=config.FINAL_TOP_K)
    if not result:
        log.info("no evidence retrieved for section %s", spec.key)
        return section

    payload = complete_json(
        _PROMPT.format(
            company=company_name,
            title=spec.title,
            brief=spec.brief,
            context=result.context_block,
        ),
        system=_SYSTEM,
        temperature=config.TEMPERATURE_PROSE,
        max_tokens=1100,
    )

    if not isinstance(payload, dict):
        log.warning("section %s could not be generated", spec.key)
        return section

    section.evidence = _evidence_from(result)
    count = len(section.evidence)

    section.summary = _normalise_citations(str(payload.get("summary") or "").strip(), count)
    section.paragraphs = [
        _normalise_citations(str(p).strip(), count)
        for p in (payload.get("paragraphs") or [])
        if str(p).strip()
    ]
    section.bullets = [
        _normalise_citations(str(b).strip(), count)
        for b in (payload.get("bullets") or [])
        if str(b).strip()
    ]

    return section


def build_all(
    index,
    company_name: str,
    *,
    latest_fiscal_year: int | None = None,
    progress=None,
) -> list[Section]:
    """Generate every narrative section."""
    sections: list[Section] = []
    for position, spec in enumerate(SPECS, start=1):
        if progress:
            progress("section", f"[{position}/{len(SPECS)}] {spec.title}")
        section = build_section(
            spec, index, company_name, latest_fiscal_year=latest_fiscal_year
        )
        if not section.is_empty:
            sections.append(section)
    return sections
