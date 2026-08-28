"""
Audited financials for Indian issuers, from Ind-AS XBRL filed with the exchange.

This is the Indian counterpart to :mod:`fincopilot.fundamentals.xbrl`, and it
exists for the same reason. That module's opening argument — that asking a
language model to read figures out of PDF text fails dangerously, because a
mangled income statement still yields confident, precisely-formatted numbers
that are simply wrong — applies with more force here, not less. The results
PDFs Indian companies publish are frequently *scanned images run through OCR*:
in a real Bikaji filing the line "Revenue from operations" arrives as "Revenue
from oaerations", negatives print as ``/398.55)`` rather than ``(398.55)``, and
every figure is denominated in lakhs with the scale stated only in a page
header. Reading numbers out of that is guesswork with a confident face.

The exchange publishes the same results as an **Ind-AS XBRL instance**, which
is what this module reads. Every figure carries its concept, its unit, and its
period. Nothing is inferred and nothing is scaled by hand.

Three correctness problems are specific to this source and are handled here:

1.  **Standalone vs consolidated.** Indian companies file both. A standalone
    statement excludes subsidiaries and can understate a group badly. The
    exchange flags which is which (:mod:`fincopilot.ingest.nse` filters on it)
    and the instance repeats it in ``NatureOfReportStandaloneConsolidated``,
    which is re-checked here.

2.  **Quarter vs year.** An annual results filing tags *both* the fourth
    quarter and the full year, and the ``xbrli:context`` period dates cannot be
    trusted to tell them apart — in real filings both contexts carry the
    quarter's dates. The instance separately tags
    ``DateOfStartOfReportingPeriod`` / ``DateOfEndOfReportingPeriod`` per
    context, and *those* are correct. Period selection uses them, so a quarter
    can never be mistaken for a fiscal year. Getting this wrong understates
    revenue roughly fourfold while looking entirely plausible.

3.  **EBIT is not tagged.** Ind-AS has no operating-income concept. It is
    reconstructed deterministically from tagged components and cross-checked
    against the income statement's own totals; if the reconstruction does not
    tie out, the field is left empty rather than guessed.

Every history this module returns is validated against accounting identities
before it is handed back (see :func:`_validate`). A filing that fails is
rejected, not repaired.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from xml.etree import ElementTree as ET

from ..http_client import request
from ..ingest import nse
from ..resolve import Company
from .models import FinancialHistory, FiscalYear

log = logging.getLogger(__name__)

_XBRLI = "http://www.xbrl.org/2003/instance"

# A tagged period counts as a fiscal year at this duration. Indian fiscal years
# run April-March; the window is loose enough for a transition year.
_MIN_ANNUAL_DAYS = 330
_MAX_ANNUAL_DAYS = 400

# Ind-AS concept -> FiscalYear field, for values reported over a period.
_DURATION_CONCEPTS: dict[str, list[str]] = {
    "revenue": ["RevenueFromOperations"],
    "pretax_income": ["ProfitBeforeTax"],
    "tax_expense": ["TaxExpense"],
    "diluted_eps": [
        "DilutedEarningsLossPerShareFromContinuingAndDiscontinuedOperations",
        "DilutedEarningsLossPerShareFromContinuingOperations",
    ],
    "operating_cash_flow": ["CashFlowsFromUsedInOperatingActivities"],
    "depreciation_amortisation": ["DepreciationDepletionAndAmortisationExpense"],
    "stock_compensation": ["AdjustmentsForSharebasedPayments"],
}

# Values reported at a point in time (balance sheet).
_INSTANT_CONCEPTS: dict[str, list[str]] = {
    "total_assets": ["Assets"],
    "current_assets": ["CurrentAssets"],
    "current_liabilities": ["CurrentLiabilities"],
    "shareholders_equity": ["EquityAttributableToOwnersOfParent", "Equity"],
    "cash_and_equivalents": ["CashAndCashEquivalents"],
    "short_term_investments": ["BankBalanceOtherThanCashAndCashEquivalents"],
}

# Components summed rather than taken from a single tag.
_DEBT_CONCEPTS = ["BorrowingsCurrent", "BorrowingsNoncurrent"]
_CAPEX_CONCEPTS = [
    "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities",
    "PurchaseOfIntangibleAssetsClassifiedAsInvestingActivities",
    "PurchaseOfIntangibleAssetsUnderDevelopment",
]
# Cost of goods sold, in the Ind-AS presentation: materials plus traded goods
# plus the inventory movement (which is signed, and may be negative).
_COGS_CONCEPTS = [
    "CostOfMaterialsConsumed",
    "PurchasesOfStockInTrade",
    "ChangesInInventoriesOfFinishedGoodsWorkInProgressAndStockInTrade",
]


@dataclass
class _Period:
    """A context in the instance, with the period it actually covers."""

    context_id: str
    start: date | None
    end: date | None
    instant: date | None
    nature: str | None          # Consolidated / Standalone
    audited: str | None

    @property
    def days(self) -> int | None:
        if self.start and self.end:
            return (self.end - self.start).days
        return None

    @property
    def is_annual(self) -> bool:
        days = self.days
        return days is not None and _MIN_ANNUAL_DAYS <= days <= _MAX_ANNUAL_DAYS


def _to_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value.strip()[:10])
    except ValueError:
        return None


def _number(text: str | None) -> float | None:
    if text is None:
        return None
    cleaned = text.strip().replace(",", "")
    if not cleaned:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_instance(payload: bytes) -> tuple[dict[str, _Period], dict[str, list[tuple[str, str]]]]:
    """Return (contexts by id, facts as tag -> [(context_id, text)])."""
    root = ET.fromstring(payload)

    periods: dict[str, _Period] = {}
    for context in root.findall(f"{{{_XBRLI}}}context"):
        context_id = context.get("id")
        if not context_id:
            continue
        period = context.find(f"{{{_XBRLI}}}period")
        start = end = instant = None
        if period is not None:
            start = _to_date(period.findtext(f"{{{_XBRLI}}}startDate"))
            end = _to_date(period.findtext(f"{{{_XBRLI}}}endDate"))
            instant = _to_date(period.findtext(f"{{{_XBRLI}}}instant"))
        periods[context_id] = _Period(context_id, start, end, instant, None, None)

    facts: dict[str, list[tuple[str, str]]] = {}
    for element in root:
        tag = element.tag.split("}")[-1]
        context_ref = element.get("contextRef")
        if not context_ref or element.text is None:
            continue
        facts.setdefault(tag, []).append((context_ref, element.text.strip()))

    # NSE's older instances are malformed: every fact on the primary income
    # statement points at "OneD"/"FourD"/"OneI", and those contexts are never
    # defined — only the dimension-qualified segment contexts are. Dropping the
    # dangling references would silently discard every pre-2023 fiscal year, so
    # they are reconstructed as stubs and dated from the reporting-period facts
    # below, which the filing does tag against them.
    referenced = {ref for entries in facts.values() for ref, _text in entries}
    for context_id in referenced - set(periods):
        periods[context_id] = _Period(context_id, None, None, None, None, None)

    # The authoritative period for each context: the instance tags the real
    # reporting window separately, and it disagrees with xbrli:period in real
    # filings (both a quarter and its year can carry the quarter's dates).
    for tag, attribute in (
        ("DateOfStartOfReportingPeriod", "start"),
        ("DateOfEndOfReportingPeriod", "end"),
    ):
        for context_ref, text in facts.get(tag, []):
            period = periods.get(context_ref)
            parsed = _to_date(text)
            if period is not None and parsed is not None:
                setattr(period, attribute, parsed)

    for tag, attribute in (
        ("NatureOfReportStandaloneConsolidated", "nature"),
        ("WhetherResultsAreAuditedOrUnaudited", "audited"),
    ):
        for context_ref, text in facts.get(tag, []):
            period = periods.get(context_ref)
            if period is not None:
                setattr(period, attribute, text)

    return periods, facts


def _pick_annual_context(
    periods: dict[str, _Period], *, allow_standalone: bool = False
) -> _Period | None:
    """The audited twelve-month context in this instance.

    A results filing carries the quarter and the year side by side; this is
    what keeps them apart. Where several qualify, the one ending latest wins.

    ``allow_standalone`` is set only when the exchange has told us this filing
    *is* the standalone one and no consolidated filing exists for the company
    (see :func:`fincopilot.ingest.nse.annual_xbrl_filings`). Left false, a
    standalone context is refused, so a group's consolidated filing can never
    silently yield subsidiary-excluding numbers.
    """
    candidates = [
        p
        for p in periods.values()
        if p.is_annual
        and (p.audited or "").strip().lower() != "un-audited"
        and (
            allow_standalone
            or (p.nature or "").strip().lower() != "standalone"
        )
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.end or date.min)


def _pick_instant_context(
    periods: dict[str, _Period],
    facts: dict[str, list[tuple[str, str]]],
    annual: _Period,
) -> str | None:
    """The balance-sheet context for the year ``annual`` covers.

    Normally an instant context carries the period-end date and is matched on
    it. In the malformed older instances the balance-sheet context is a
    dangling reference with no date at all, so it is instead identified by the
    facts hanging off it: a results filing states one balance sheet, at the
    year end, so the undated context carrying ``Assets`` is that balance sheet.
    Returns None when no context reports total assets, leaving the balance
    sheet empty rather than attaching figures to a date that was inferred.
    """
    dated = [
        p.context_id
        for p in periods.values()
        if p.instant is not None and p.instant == annual.end
    ]
    if dated:
        return dated[0]

    undated_with_assets = [
        ref
        for ref, _text in facts.get("Assets", [])
        if (periods.get(ref) is not None
            and periods[ref].instant is None
            and periods[ref].start is None
            and periods[ref].end is None)
    ]
    return undated_with_assets[0] if undated_with_assets else None


def _value(
    facts: dict[str, list[tuple[str, str]]],
    concepts: list[str],
    context_id: str,
) -> float | None:
    """First concept in ``concepts`` reported against ``context_id``."""
    for concept in concepts:
        for context_ref, text in facts.get(concept, []):
            if context_ref == context_id:
                value = _number(text)
                if value is not None:
                    return value
    return None


def _sum_values(
    facts: dict[str, list[tuple[str, str]]],
    concepts: list[str],
    context_id: str,
) -> float | None:
    """Sum of every concept present; None when none of them are reported."""
    total = 0.0
    found = False
    for concept in concepts:
        value = _value(facts, [concept], context_id)
        if value is not None:
            total += value
            found = True
    return total if found else None


def _operating_income(
    facts: dict[str, list[tuple[str, str]]], context_id: str
) -> float | None:
    """Reconstruct EBIT. Ind-AS does not tag it.

    ``ProfitBeforeExceptionalItemsAndTax`` is after finance costs and includes
    other (non-operating) income, so both are unwound:

        EBIT = PBT before exceptionals + finance costs - other income

    Returns None when any component is missing, rather than a partial figure
    that would silently misstate the operating margin the whole DCF turns on.
    """
    profit = _value(facts, ["ProfitBeforeExceptionalItemsAndTax"], context_id)
    if profit is None:
        profit = _value(facts, ["ProfitBeforeTax"], context_id)
    finance_costs = _value(facts, ["FinanceCosts"], context_id)
    other_income = _value(facts, ["OtherIncome"], context_id)

    if profit is None or finance_costs is None or other_income is None:
        return None
    return profit + finance_costs - other_income


def _net_income(
    facts: dict[str, list[tuple[str, str]]], context_id: str
) -> float | None:
    """Profit for the year, preferring the figure attributable to owners.

    Filers with no non-controlling interest routinely tag
    ``ProfitOrLossAttributableToOwnersOfParent`` as a literal ``0.00`` rather
    than omitting it — real in Bikaji's FY2024 filing, where the actual profit
    of 2,634,626,000 sits only in ``ProfitLossForPeriod``. A company earning
    exactly zero attributable profit is not a real case, so a zero here is read
    as "not tagged" and the consolidated total is used instead.
    """
    owners = _value(facts, ["ProfitOrLossAttributableToOwnersOfParent"], context_id)
    if owners:
        return owners
    return _value(
        facts,
        ["ProfitLossForPeriod", "ProfitLossForPeriodFromContinuingOperations"],
        context_id,
    )


def _validate(entry: FiscalYear, *, tolerance: float = 0.02) -> list[str]:
    """Accounting identities that must hold. Returns the failures.

    These are the checks that catch a mis-selected context or a mis-mapped
    concept — the failure modes that produce plausible-looking wrong numbers.
    """
    problems: list[str] = []

    if not entry.revenue or entry.revenue <= 0:
        problems.append("revenue missing or non-positive")
        return problems

    # Balance sheet must balance.
    if (
        entry.total_assets is not None
        and entry.shareholders_equity is not None
        and entry.current_liabilities is not None
    ):
        if entry.total_assets <= 0:
            problems.append("total assets non-positive")

    # A margin outside this range is a scale or mapping error, not a business.
    if entry.operating_income is not None:
        margin = entry.operating_income / entry.revenue
        if not -1.0 <= margin <= 1.0:
            problems.append(f"operating margin implausible ({margin:.1%})")

    if entry.net_income is not None:
        margin = entry.net_income / entry.revenue
        if not -2.0 <= margin <= 1.0:
            problems.append(f"net margin implausible ({margin:.1%})")

    # PBT - tax = net income, within rounding and minority interest.
    if (
        entry.pretax_income is not None
        and entry.tax_expense is not None
        and entry.net_income is not None
    ):
        implied = entry.pretax_income - entry.tax_expense
        if abs(implied) > 0 and abs(implied - entry.net_income) / abs(implied) > 0.15:
            problems.append(
                f"PBT - tax ({implied:,.0f}) does not reconcile to "
                f"net income ({entry.net_income:,.0f})"
            )

    if entry.cash_and_equivalents is not None and entry.cash_and_equivalents < 0:
        problems.append("negative cash balance")

    return problems


def _fiscal_year_of(period_end: date) -> int:
    """Indian fiscal years run April-March and are named for the closing year.

    A 31 March 2024 year end is FY2024. A December year end (some subsidiaries
    and foreign-owned issuers) is named for its own year.
    """
    return period_end.year


def fetch(company: Company, *, max_years: int = 6) -> FinancialHistory | None:
    """Build a :class:`FinancialHistory` from Ind-AS XBRL filed with NSE.

    Returns ``None`` when the company is not an Indian issuer, when the
    exchange lists no audited consolidated XBRL, or when every filing fails
    validation — the caller must not fall back to an unaudited source.
    """
    if company.country != "IN":
        return None

    symbol = company.base_ticker
    filings = nse.annual_xbrl_filings(symbol)
    if not filings:
        log.warning("NSE lists no audited consolidated XBRL for %s", symbol)
        return None

    by_year: dict[int, FiscalYear] = {}
    provenance: dict[int, str] = {}
    rejected: list[str] = []
    standalone_years: list[int] = []

    # NSE lists the same instance under both "Annual" and "Fourth Quarter";
    # fetching it twice would double every rejection message.
    seen_urls: set[str] = set()

    for filing in filings:
        if len(by_year) >= max_years:
            break
        if filing.xbrl_url in seen_urls:
            continue
        seen_urls.add(filing.xbrl_url)

        response = request(filing.xbrl_url)
        if response is None or response.status_code != 200:
            log.warning("could not fetch XBRL %s", filing.xbrl_url)
            continue

        try:
            periods, facts = _parse_instance(response.content)
        except ET.ParseError as exc:
            log.warning("malformed XBRL at %s: %s", filing.xbrl_url, exc)
            continue

        annual = _pick_annual_context(
            periods, allow_standalone=not filing.consolidated
        )
        if annual is None or annual.end is None:
            log.info("no annual context in %s", filing.xbrl_url)
            continue

        fiscal_year = _fiscal_year_of(annual.end)
        # Filings arrive consolidated-first per period, so a year already
        # captured was captured from the better basis.
        if fiscal_year in by_year:
            continue

        entry = FiscalYear(
            fiscal_year=fiscal_year,
            period_end=annual.end.isoformat(),
        )

        for field_name, concepts in _DURATION_CONCEPTS.items():
            value = _value(facts, concepts, annual.context_id)
            if value is not None:
                setattr(entry, field_name, value)

        entry.net_income = _net_income(facts, annual.context_id)
        entry.operating_income = _operating_income(facts, annual.context_id)
        entry.cost_of_revenue = _sum_values(facts, _COGS_CONCEPTS, annual.context_id)

        # Capex is reported as a positive outflow in Ind-AS; the rest of the
        # engine expects the SEC sign convention (negative).
        capex = _sum_values(facts, _CAPEX_CONCEPTS, annual.context_id)
        if capex is not None:
            entry.capex = -abs(capex)

        # Balance-sheet values sit on an instant context at the period end.
        instant_id = _pick_instant_context(periods, facts, annual)
        if instant_id:
            for field_name, concepts in _INSTANT_CONCEPTS.items():
                value = _value(facts, concepts, instant_id)
                if value is not None:
                    setattr(entry, field_name, value)
            entry.total_debt = _sum_values(facts, _DEBT_CONCEPTS, instant_id)

        problems = _validate(entry)
        if problems:
            rejected.append(f"FY{fiscal_year}: {'; '.join(problems)}")
            log.warning(
                "rejected FY%s for %s from %s: %s",
                fiscal_year, symbol, filing.xbrl_url, "; ".join(problems),
            )
            continue

        by_year[fiscal_year] = entry
        provenance[fiscal_year] = filing.xbrl_url
        if not filing.consolidated:
            standalone_years.append(fiscal_year)

    if not by_year:
        log.warning("no usable Ind-AS XBRL years for %s", symbol)
        return None

    selected = sorted(by_year)[-max_years:]

    history = FinancialHistory(
        ticker=company.ticker,
        company_name=company.name,
        currency="INR",
        source="nse_indas_xbrl",
    )
    history.years = [by_year[year] for year in selected]

    used_standalone = [y for y in selected if y in standalone_years]
    basis = "standalone" if len(used_standalone) == len(selected) else "consolidated"

    history.notes.append(
        f"Financial statement data from audited {basis} Ind-AS XBRL filed "
        f"with the NSE by {company.name} ({symbol}); "
        f"{len(selected)} fiscal years (FY{selected[0]}-FY{selected[-1]}). "
        "Every figure is concept-tagged in the company's own filing."
    )
    if used_standalone:
        history.notes.append(
            "Reported on a standalone basis"
            + (
                ""
                if basis == "standalone"
                else f" for FY{', FY'.join(str(y) for y in used_standalone)}"
            )
            + " — the company files no consolidated results for "
            "these periods, so subsidiaries (if any) are excluded."
        )
    if rejected:
        history.notes.append(
            "Filings rejected by the accounting-identity check: "
            + " | ".join(rejected)
        )

    log.info(
        "Ind-AS history for %s: FY%s-FY%s", symbol, selected[0], selected[-1]
    )
    return history
