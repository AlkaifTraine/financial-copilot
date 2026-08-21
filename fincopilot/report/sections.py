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
        title="Outlook & Capital Allocation",
        query=(
            "outlook guidance next quarter next fiscal year expectations "
            "capital allocation dividend buyback capital expenditure plans"
        ),
        brief=(
            "Management's stated forward expectations AND capital allocation (buybacks, "
            "dividends, capex). Name the title's two halves accurately: if the content is "
            "almost entirely capital allocation, say so. Be explicit about what is guidance "
            "versus inference, and state the period of every guidance figure."
        ),
        bullet_label="What to watch",
    ),
]

_SYSTEM = """You are an equity research analyst writing one section of a research report.

The difference between research and a summary is interpretation. A summary says what happened; research says what it means and whether it lasts. Every paragraph must move past the fact to its consequence.

Structure each paragraph as: FACT (the number or disclosure) -> INTERPRETATION (what it tells us about the business) -> SUSTAINABILITY (whether it persists, and what would break it). The section then closes on the INVESTMENT IMPLICATION — the "so what" for owning the stock.

Rules:
- Use ONLY the numbered sources provided. No outside knowledge.
- Cite inline as [n] after each factual claim.
- Quote figures exactly, with their period.
- Guidance discipline: state the PERIOD of every guidance figure — a named quarter, a full fiscal year, a calendar year, or multi-year. Many companies (NVIDIA included) guide only ONE QUARTER ahead; NEVER present a single-quarter guidance number as annual or full-year, and NEVER derive an annual growth rate by annualizing one quarter (a ~$78B quarter is a ~$312B run-rate, not "annual guidance", and comparing it to a full prior year is meaningless). Compare a quarter to the year-ago QUARTER (quarter YoY), or label a 4x figure a "run-rate" explicitly. If a figure's period is ambiguous in the source, say so rather than assuming it is annual.
- Segment vs end-market discipline: a REPORTABLE SEGMENT (e.g. NVIDIA's "Compute & Networking" and "Graphics") is not the same thing as an END-MARKET or platform (Data Center, Gaming, Professional Visualization, Automotive). Never substitute an end-market percentage for a reportable-segment percentage, or vice versa. Every percentage must state its DENOMINATOR and PERIOD — "X% of FY2026 revenue", making clear whether X is a segment or an end-market share. The same named line must not appear with two different percentages.
- Consensus vs our-scenario discipline: the analyst consensus / mean price target is an EXTERNAL view ("what the market believes"). Our bear/base/bull probabilities are OUR internal scenario framework. Never say the consensus "assumes" or "implies" our scenario probability, and never attribute our probability weighting to analysts — they are two separate things.
- Inference discipline: do not infer a company's INTENT or BELIEF from an action unless the source states it. A share buyback is capital returned to shareholders; it is NOT by itself evidence that "management believes the stock is undervalued" — that is an unsupported inference. Separate the observed action from a supported interpretation (e.g. it reduces share count and supports EPS) and never present the unsupported inference (why they did it) as fact.
- Source freshness: prefer the MOST RECENT filing as primary evidence for a current claim (latest 10-Q, then 10-K, then latest earnings release/call). Do not lean on an older filing where a newer one covers the same point. When you do cite an older statement for background, label it as historical context — do not present a dated figure as the current one.
- Interpret, do not narrate. Never write a sentence that only restates a source without drawing a conclusion from it.
- Do not repeat what other sections cover (you are told what they are). Stay in your lane.
- Banned: "underscores", "highlights", "demonstrates", "reflects", "showcases", "well-positioned", "remains committed", and any sentence that could appear in the company's own press release.
- If the sources are thin, write less. A short, sharp section beats a padded one."""

_PROMPT = """Company: {company}
Section: {title}

Brief: {brief}

{financial_facts}

Other sections of this report (do NOT duplicate their content):
{siblings}

Sources:
{context}

Write this section as interpretation, not description. Each paragraph: fact -> what it means -> whether it lasts. Close with the investment implication.

Return JSON:
{{"summary": "one sentence standfirst that states a view, not a topic, under 25 words",
  "paragraphs": ["2-4 analytical paragraphs, each ending on an interpretation rather than a bare fact"],
  "bullets": ["3-6 short specific points, each pairing a figure or named fact WITH the conclusion drawn from it"],
  "implication": "one sentence: what this section specifically means for the investment case"}}"""


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


def _sibling_brief(current: SectionSpec) -> str:
    """A one-line map of the other sections, so each stays in its lane (#24)."""
    others = [f"- {s.title}: {s.brief.split('.')[0]}." for s in SPECS if s.key != current.key]
    return "\n".join(others)


def build_section(
    spec: SectionSpec,
    index,
    company_name: str,
    *,
    latest_fiscal_year: int | None = None,
    financial_facts: str = "",
    corrections: list[str] | None = None,
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

    from .correction import correction_instruction
    payload = complete_json(
        _PROMPT.format(
            company=company_name,
            title=spec.title,
            brief=spec.brief,
            financial_facts=financial_facts,
            siblings=_sibling_brief(spec),
            context=result.context_block,
        ) + correction_instruction(corrections),
        system=_SYSTEM,
        model=config.WRITER_MODEL,
        # A correction re-draft runs deterministically (temp 0) so the fix is stable and
        # complies with the mandatory instruction rather than re-rolling the prose.
        temperature=0.0 if corrections else config.TEMPERATURE_PROSE,
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
    section.implication = _normalise_citations(str(payload.get("implication") or "").strip(), count)

    return section


def build_all(
    index,
    company_name: str,
    *,
    latest_fiscal_year: int | None = None,
    financial_facts: str = "",
    corrections: list[str] | None = None,
    progress=None,
) -> list[Section]:
    """Generate every narrative section.

    ``financial_facts`` is the canonical-figures block (see report/metrics.py):
    handed to every section as the authoritative source for headline metrics, so
    no section re-derives (and contradicts) a number that already has a canonical
    value.
    """
    sections: list[Section] = []
    for position, spec in enumerate(SPECS, start=1):
        if progress:
            progress("section", f"[{position}/{len(SPECS)}] {spec.title}")
        section = build_section(
            spec, index, company_name, latest_fiscal_year=latest_fiscal_year,
            financial_facts=financial_facts, corrections=corrections,
        )
        if not section.is_empty:
            sections.append(section)
    return sections
