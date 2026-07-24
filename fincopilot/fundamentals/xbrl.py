"""
Audited financials from the SEC's XBRL `companyfacts` API.

This is the single most important correctness decision in the project.

The previous pipeline obtained its numbers by asking a language model to read
figures out of PDF text (`utils/financial_data_extractor.py`). That approach
fails in a particularly dangerous way: a mangled income statement still yields
confident, plausible, precisely-formatted numbers that are simply wrong, and
nothing downstream can detect it.

`companyfacts` returns every value the company itself tagged in its filings —
the same structured data the SEC indexes. Each figure carries its concept, unit,
period, source form and accession number. Nothing is inferred, nothing is read
out of prose, and every number traces to a specific filing.

Concept selection needs fallback lists because US GAAP offers several tags for
the same line. Revenue alone is reported as `Revenues`,
`RevenueFromContractWithCustomerExcludingAssessedTax`, or `SalesRevenueNet`
depending on the filer and the year, so each field tries its candidates in
order of specificity and takes the first with usable data.
"""

from __future__ import annotations

import logging
from datetime import date

from .. import config
from ..http_client import get_json_cached
from ..resolve import Company
from .models import FinancialHistory, FiscalYear

log = logging.getLogger(__name__)

# Forms carrying audited annual figures. Quarterly forms are excluded so that
# a Q3 balance sheet never masquerades as a fiscal year end.
_ANNUAL_FORMS = {"10-K", "10-K/A", "20-F", "20-F/A", "40-F"}

# A tagged period counts as a fiscal year at this duration. Filers' years run
# 52 or 53 weeks, so the window is deliberately loose.
_MIN_ANNUAL_DAYS = 330
_MAX_ANNUAL_DAYS = 400

DURATION = "duration"   # flows: revenue, income, cash flow
INSTANT = "instant"     # stocks: cash, debt, equity

# Ordered by specificity: the first concept with data wins.
_CONCEPTS: dict[str, tuple[list[str], str, str]] = {
    # field: (candidate tags, unit, period kind)
    "revenue": (
        [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "RevenueFromContractWithCustomerIncludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
        ],
        "USD",
        DURATION,
    ),
    "cost_of_revenue": (
        ["CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfServices"],
        "USD",
        DURATION,
    ),
    "gross_profit": (["GrossProfit"], "USD", DURATION),
    "operating_income": (["OperatingIncomeLoss"], "USD", DURATION),
    "pretax_income": (
        [
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
            "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
        ],
        "USD",
        DURATION,
    ),
    "tax_expense": (["IncomeTaxExpenseBenefit"], "USD", DURATION),
    "net_income": (["NetIncomeLoss", "ProfitLoss"], "USD", DURATION),
    "operating_cash_flow": (
        [
            "NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations",
        ],
        "USD",
        DURATION,
    ),
    "capex": (
        [
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "PaymentsToAcquireProductiveAssets",
        ],
        "USD",
        DURATION,
    ),
    "depreciation_amortisation": (
        [
            "DepreciationDepletionAndAmortization",
            "DepreciationAmortizationAndAccretionNet",
            "DepreciationAndAmortization",
        ],
        "USD",
        DURATION,
    ),
    "stock_compensation": (["ShareBasedCompensation"], "USD", DURATION),
    "cash_and_equivalents": (
        [
            "CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        ],
        "USD",
        INSTANT,
    ),
    "short_term_investments": (
        ["ShortTermInvestments", "MarketableSecuritiesCurrent",
         "AvailableForSaleSecuritiesDebtSecuritiesCurrent"],
        "USD",
        INSTANT,
    ),
    "total_assets": (["Assets"], "USD", INSTANT),
    "current_assets": (["AssetsCurrent"], "USD", INSTANT),
    "current_liabilities": (["LiabilitiesCurrent"], "USD", INSTANT),
    "shareholders_equity": (
        ["StockholdersEquity",
         "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"],
        "USD",
        INSTANT,
    ),
    "diluted_shares": (
        ["WeightedAverageNumberOfDilutedSharesOutstanding"],
        "shares",
        DURATION,
    ),
    "diluted_eps": (["EarningsPerShareDiluted"], "USD/shares", DURATION),
}

# Total debt is assembled from components; filers split it differently, so the
# parts are summed rather than relying on a single tag.
_DEBT_CURRENT = ["LongTermDebtCurrent", "DebtCurrent", "ShortTermBorrowings"]
_DEBT_NONCURRENT = ["LongTermDebtNoncurrent", "LongTermDebt"]


def _companyfacts_url(cik: str) -> str:
    return f"{config.SEC_DATA_URL}/api/xbrl/companyfacts/CIK{cik}.json"


def _values_for_concept(
    facts: dict,
    concept: str,
    unit: str,
    kind: str,
) -> dict[int, tuple[float, str]]:
    """Fiscal-year values for one XBRL concept: ``{year: (value, period_end)}``."""
    entries = facts.get("us-gaap", {}).get(concept, {}).get("units", {}).get(unit)
    if not entries:
        return {}

    best: dict[int, tuple[str, float, str]] = {}  # year -> (filed, value, end)

    for entry in entries:
        if entry.get("form") not in _ANNUAL_FORMS:
            continue

        end = entry.get("end")
        start = entry.get("start")
        if not end:
            continue

        if kind == DURATION:
            if not start:
                continue
            try:
                span = (date.fromisoformat(end) - date.fromisoformat(start)).days
            except ValueError:
                continue
            if not _MIN_ANNUAL_DAYS <= span <= _MAX_ANNUAL_DAYS:
                continue
        else:
            # Instant facts carry no start date; a start means it is a duration
            # fact and belongs to a different line entirely.
            if start:
                continue

        value = entry.get("val")
        if value is None:
            continue

        # Fiscal years are named for the calendar year they end in, matching
        # how filers label them (NVIDIA's FY2026 ends Jan 2026).
        fiscal_year = int(end[:4])
        filed = entry.get("filed", "")

        # The same period is restated across successive filings; the most
        # recently filed value is the current one.
        existing = best.get(fiscal_year)
        if existing is None or filed > existing[0]:
            best[fiscal_year] = (filed, float(value), end)

    return {year: (value, end) for year, (_filed, value, end) in best.items()}


def _annual_values(
    facts: dict,
    concepts: list[str],
    unit: str,
    kind: str,
) -> dict[int, tuple[float, str]]:
    """Fiscal-year values for a field, combining its candidate concepts.

    Taking the first concept that returns *any* data is wrong, because filers
    migrate between tags and leave the old one populated with stale history.
    NVIDIA reports revenue under ``RevenueFromContractWithCustomerExcludingAssessedTax``
    for FY2017-FY2022 and under ``Revenues`` for FY2008-FY2026; first-match
    selection returned the FY2017-FY2022 series and silently omitted every
    recent year. Its capex tag behaves the same way — the old
    ``PaymentsToAcquirePropertyPlantAndEquipment`` stops at FY2012, while the
    live figure sits in ``PaymentsToAcquireProductiveAssets``.

    So concepts are scored on how much of the *recent* history they cover: the
    best-covering concept supplies the series, keeping it internally
    consistent, and remaining gaps are filled from the other candidates in
    priority order.
    """
    per_concept = {
        concept: values
        for concept in concepts
        if (values := _values_for_concept(facts, concept, unit, kind))
    }
    if not per_concept:
        return {}

    all_years = sorted({year for values in per_concept.values() for year in values})
    recent = set(all_years[-6:])

    primary, merged = max(
        per_concept.items(),
        key=lambda item: (len(set(item[1]) & recent), len(item[1])),
    )
    merged = dict(merged)

    for concept in concepts:
        if concept == primary or concept not in per_concept:
            continue
        for year, payload in per_concept[concept].items():
            merged.setdefault(year, payload)

    if len(per_concept) > 1:
        log.debug("field resolved primarily via %s (%d years)", primary, len(merged))

    return merged


def _sum_concepts(
    facts: dict,
    concepts: list[str],
    unit: str = "USD",
) -> dict[int, float]:
    """Best available value per year across a set of alternative tags."""
    combined: dict[int, float] = {}
    for concept in concepts:
        values = _annual_values(facts, [concept], unit, INSTANT)
        for year, (value, _end) in values.items():
            combined.setdefault(year, value)
    return combined


def fetch(company: Company, *, max_years: int = 6) -> FinancialHistory | None:
    """Build a :class:`FinancialHistory` from SEC XBRL data.

    Returns ``None`` when the company is not an SEC filer or the API is
    unavailable, so the caller can fall back to another source.
    """
    if not company.is_sec_filer:
        return None

    if not config.is_sec_configured():
        log.warning("SEC_USER_AGENT is not configured; XBRL financials unavailable")
        return None

    payload = get_json_cached(
        _companyfacts_url(company.cik), sec=True, ttl_seconds=21_600
    )
    if not payload:
        log.warning("companyfacts unavailable for CIK %s", company.cik)
        return None

    facts = payload.get("facts", {})
    if not facts:
        return None

    # Pull every field, then pivot from field-major to year-major.
    by_field: dict[str, dict[int, tuple[float, str]]] = {}
    for field_name, (concepts, unit, kind) in _CONCEPTS.items():
        by_field[field_name] = _annual_values(facts, concepts, unit, kind)

    debt_current = _sum_concepts(facts, _DEBT_CURRENT)
    debt_noncurrent = _sum_concepts(facts, _DEBT_NONCURRENT)

    revenue_years = set(by_field.get("revenue", {}))
    if not revenue_years:
        log.warning("no annual revenue found in XBRL for %s", company.ticker)
        return None

    selected = sorted(revenue_years)[-max_years:]

    history = FinancialHistory(
        ticker=company.ticker,
        company_name=company.name,
        currency="USD",
        source="sec_xbrl",
    )

    for fiscal_year in selected:
        entry = FiscalYear(
            fiscal_year=fiscal_year,
            period_end=by_field["revenue"][fiscal_year][1],
        )

        for field_name, values in by_field.items():
            found = values.get(fiscal_year)
            if found is not None:
                setattr(entry, field_name, found[0])

        total_debt = debt_current.get(fiscal_year, 0.0) + debt_noncurrent.get(fiscal_year, 0.0)
        entry.total_debt = total_debt if total_debt else None

        history.years.append(entry)

    history.notes.append(
        f"Financial statement data from SEC XBRL companyfacts (CIK {company.cik}); "
        f"{len(history.years)} fiscal years."
    )
    log.info(
        "XBRL history for %s: FY%s",
        company.ticker,
        "-FY".join(str(y) for y in (selected[0], selected[-1])),
    )
    return history
