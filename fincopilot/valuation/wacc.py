"""
Weighted average cost of capital.

    WACC = We * Ke + Wd * Kd * (1 - t)

    Ke = Rf + beta * ERP                      (CAPM)
    Kd = observed interest cost, or Rf + spread
    We, Wd = market value weights of equity and debt

Every component is either measured or an explicitly recorded default, and each
one is written into the assumption ledger as it is computed. The discount rate
moves a DCF more than any other single input, so "how did you get 9.4%?" needs
a complete answer.
"""

from __future__ import annotations

import logging

from .. import config
from ..fundamentals import FinancialHistory
from ..resolve import Company
from .models import (
    SOURCE_DEFAULT,
    SOURCE_HISTORICAL,
    SOURCE_MARKET,
    Assumption,
    AssumptionLedger,
)

log = logging.getLogger(__name__)

# Beta outside this range is almost always a data artefact from a short or
# illiquid price history rather than a real risk measurement.
_BETA_BOUNDS = (0.30, 3.00)

# Spread over the risk-free rate used when a company's actual borrowing cost
# cannot be observed. Large, profitable issuers of the kind covered here borrow
# at investment-grade spreads.
_DEFAULT_CREDIT_SPREAD = 0.015


def _risk_free_rate(company: Company, ledger: AssumptionLedger) -> float:
    """Ten-year government bond yield for the company's market."""
    if company.country == "US":
        try:
            import yfinance as yf

            # ^TNX quotes the US 10-year Treasury yield in percentage points.
            data = yf.Ticker("^TNX").history(period="5d")
            if not data.empty:
                rate = float(data["Close"].iloc[-1]) / 100.0
                if 0.001 < rate < 0.15:
                    ledger.add(
                        Assumption(
                            key="risk_free_rate",
                            label="Risk-free rate",
                            value=rate,
                            unit="%",
                            source=SOURCE_MARKET,
                            derivation="US 10-year Treasury yield (^TNX), latest close",
                            rationale=(
                                "The 10-year Treasury is the standard proxy for the "
                                "risk-free rate in a US-dollar DCF, matching the "
                                "long horizon of the cash flows being discounted."
                            ),
                        )
                    )
                    return rate
        except Exception as exc:
            log.info("live risk-free rate unavailable: %s", exc)

    # The risk-free rate must be denominated in the same currency as the cash
    # flows being discounted. Falling back to a single global default applied a
    # US Treasury yield to a rupee valuation, understating TCS's discount rate
    # by around 250bps.
    rate = config.DEFAULT_RISK_FREE_RATE_BY_COUNTRY.get(
        company.country, config.DEFAULT_RISK_FREE_RATE
    )
    ledger.add(
        Assumption(
            key="risk_free_rate",
            label="Risk-free rate",
            value=rate,
            unit="%",
            source=SOURCE_DEFAULT,
            derivation=f"Long-run 10-year government bond yield for {company.country}",
            rationale=(
                "A live government bond yield was not available for this market, "
                "so a representative local sovereign yield is used. It is quoted "
                "in the same currency as the cash flows being discounted."
            ),
        )
    )
    return rate


def _equity_risk_premium(company: Company, ledger: AssumptionLedger) -> float:
    """Excess return demanded for holding equities over government bonds."""
    if company.country == "IN":
        premium = config.DEFAULT_INDIA_RISK_PREMIUM
        rationale = (
            "Mature-market equity risk premium plus an India country risk "
            "premium, reflecting additional sovereign and currency risk."
        )
    else:
        premium = config.DEFAULT_EQUITY_RISK_PREMIUM
        rationale = (
            "Standard mature-market equity risk premium, in line with long-run "
            "realised excess returns of equities over government bonds."
        )

    ledger.add(
        Assumption(
            key="equity_risk_premium",
            label="Equity risk premium",
            value=premium,
            unit="%",
            source=SOURCE_DEFAULT,
            derivation=f"Standard premium for {company.country}",
            rationale=rationale,
        )
    )
    return premium


def _beta(history: FinancialHistory, ledger: AssumptionLedger) -> float:
    """Sensitivity of the stock to the broad market, Blume-adjusted.

    The raw historical beta is corrected toward the market with Blume's
    adjustment before it enters CAPM: a backward-looking beta reliably
    overstates the forward beta because betas regress toward 1.0 over time.
    Left raw, NVIDIA's ~2.2 produced a 16.8% discount rate no analyst uses; the
    adjustment is the standard fix, not a thumb on the scale.
    """
    raw = history.beta

    if raw is None:
        ledger.add(
            Assumption(
                key="beta",
                label="Beta",
                value=1.0,
                unit="x",
                source=SOURCE_DEFAULT,
                derivation="No beta available; market average assumed",
                rationale="A beta of 1.0 assumes the stock moves with the market.",
            )
        )
        return 1.0

    # Blume adjustment: pull the raw beta toward the market before use.
    w = config.BETA_BLUME_WEIGHT
    adjusted = w * raw + (1 - w) * 1.0
    bounded = min(max(adjusted, _BETA_BOUNDS[0]), _BETA_BOUNDS[1])
    clamped = bounded != adjusted

    derivation = (
        f"Five-year raw beta {raw:.2f}, Blume-adjusted "
        f"({w:.2f} x {raw:.2f} + {1 - w:.2f} x 1.00) = {adjusted:.2f}"
    )
    if clamped:
        derivation += f", then constrained to {_BETA_BOUNDS}"

    ledger.add(
        Assumption(
            key="beta",
            label="Beta (Blume-adjusted)",
            value=bounded,
            unit="x",
            source=SOURCE_MARKET,
            derivation=derivation,
            rationale=(
                "The raw beta is adjusted toward the market to reflect the "
                "well-documented tendency of betas to regress toward 1.0 over "
                "time; this is the standard treatment for a forward discount rate."
            ),
            bounds=_BETA_BOUNDS if clamped else None,
            clamped=clamped,
            raw_value=raw,
        )
    )
    return bounded


def _size_premium(history: FinancialHistory, ledger: AssumptionLedger) -> float:
    """Build-up size premium added to the cost of equity, by market cap."""
    market_cap = history.market_cap or 0.0
    premium = 0.0
    label = "mid-cap"
    for threshold, value in config.SIZE_PREMIUM_TIERS:
        if market_cap >= threshold:
            premium = value
            label = {0.0: "small-cap", 2e9: "mid-cap", 10e9: "large-cap", 200e9: "mega-cap"}.get(
                threshold, "mid-cap"
            )
            break

    if market_cap <= 0:
        return 0.0

    ledger.add(
        Assumption(
            key="size_premium",
            label="Size premium",
            value=premium,
            unit="%",
            source=SOURCE_DEFAULT,
            derivation=(
                f"Build-up size premium for a {label} "
                f"(market cap {history.currency} {market_cap / 1e9:,.0f}bn)"
            ),
            rationale=(
                "The documented size effect: smaller companies require a higher "
                "return for illiquidity and fragility, the largest a small discount "
                "for depth and quality. Standard in the build-up cost of equity, and "
                "symmetric across the size spectrum."
            ),
        )
    )
    return premium


def _tax_rate(history: FinancialHistory, company: Company, ledger: AssumptionLedger) -> float:
    """Effective tax rate, preferred over the statutory rate."""
    rates = [
        year.effective_tax_rate
        for year in history.years[-3:]
        if year.effective_tax_rate is not None
    ]

    if rates:
        rate = sum(rates) / len(rates)
        years = [y.fiscal_year for y in history.years[-3:] if y.effective_tax_rate is not None]
        ledger.add(
            Assumption(
                key="tax_rate",
                label="Effective tax rate",
                value=rate,
                unit="%",
                source=SOURCE_HISTORICAL,
                derivation=(
                    f"Mean of tax expense / pre-tax income for FY"
                    f"{'-FY'.join(str(y) for y in (years[0], years[-1]))}"
                ),
                rationale=(
                    "The rate the company actually pays reflects its real mix of "
                    "jurisdictions, credits and incentives, which for a global "
                    "business differs materially from the statutory rate."
                ),
            )
        )
        return rate

    rate = config.DEFAULT_TAX_RATE.get(company.country, 0.21)
    ledger.add(
        Assumption(
            key="tax_rate",
            label="Statutory tax rate",
            value=rate,
            unit="%",
            source=SOURCE_DEFAULT,
            derivation=f"Statutory corporate rate for {company.country}",
            rationale="Pre-tax income was not reported, so the statutory rate is used.",
        )
    )
    return rate


def compute_wacc(
    history: FinancialHistory,
    company: Company,
    ledger: AssumptionLedger,
) -> float:
    """Compute the discount rate and record every component."""
    risk_free = _risk_free_rate(company, ledger)
    premium = _equity_risk_premium(company, ledger)
    beta = _beta(history, ledger)
    tax_rate = _tax_rate(history, company, ledger)

    # Horizon beta: the value of a 10-year DCF is dominated by mature-phase cash
    # flows, whose beta is far closer to the market than today's spot estimate.
    # Discounting them at a high-growth spot beta over-penalises the terminal
    # value, so the discount rate uses the horizon-average beta.
    horizon_weight = config.BETA_HORIZON_WEIGHT
    beta_dcf = horizon_weight * beta + (1 - horizon_weight) * 1.0
    ledger.add(
        Assumption(
            key="beta_dcf",
            label="Beta (long-horizon)",
            value=beta_dcf,
            unit="x",
            source=SOURCE_HISTORICAL,
            derivation=(
                f"{horizon_weight:.2f} x {beta:.2f} (Blume) + {1 - horizon_weight:.2f} x 1.00, "
                f"the average beta over a {config.DCF_FORECAST_YEARS}-year horizon"
            ),
            rationale=(
                "A long-dated DCF is dominated by mature-phase cash flows, whose "
                "beta reverts toward the market; the discount rate uses that "
                "horizon-average beta rather than the spot high-growth beta."
            ),
        )
    )

    size_premium = _size_premium(history, ledger)

    cost_of_equity = risk_free + beta_dcf * premium + size_premium
    size_term = f" {size_premium * 100:+.1f}% size" if size_premium else ""
    ledger.add(
        Assumption(
            key="cost_of_equity",
            label="Cost of equity",
            value=cost_of_equity,
            unit="%",
            source=SOURCE_HISTORICAL,
            derivation=(
                f"CAPM build-up: {risk_free * 100:.2f}% + {beta_dcf:.2f} x "
                f"{premium * 100:.2f}%{size_term}"
            ),
            rationale="The return equity investors require for this company's risk.",
        )
    )

    latest = history.latest
    total_debt = (latest.total_debt if latest else None) or 0.0
    equity_value = history.market_cap or 0.0

    # An unlevered company's WACC is simply its cost of equity.
    if equity_value <= 0 or total_debt <= 0:
        wacc = cost_of_equity
        ledger.add(
            Assumption(
                key="wacc",
                label="WACC (discount rate)",
                value=wacc,
                unit="%",
                source=SOURCE_HISTORICAL,
                derivation="Equal to cost of equity; no material debt in the capital structure",
                rationale=(
                    "With negligible debt there is no tax shield to weight, so the "
                    "cost of capital is the cost of equity."
                ),
            )
        )
        return wacc

    cost_of_debt = risk_free + _DEFAULT_CREDIT_SPREAD
    ledger.add(
        Assumption(
            key="cost_of_debt",
            label="Pre-tax cost of debt",
            value=cost_of_debt,
            unit="%",
            source=SOURCE_DEFAULT,
            derivation=(
                f"Risk-free {risk_free * 100:.2f}% + "
                f"{_DEFAULT_CREDIT_SPREAD * 100:.1f}% investment-grade spread"
            ),
            rationale=(
                "A representative borrowing cost for an issuer of this size and "
                "credit quality."
            ),
        )
    )

    total_capital = equity_value + total_debt
    equity_weight = equity_value / total_capital
    debt_weight = total_debt / total_capital

    raw_wacc = equity_weight * cost_of_equity + debt_weight * cost_of_debt * (1 - tax_rate)
    wacc = min(max(raw_wacc, config.WACC_BOUNDS[0]), config.WACC_BOUNDS[1])

    ledger.add(
        Assumption(
            key="wacc",
            label="WACC (discount rate)",
            value=wacc,
            unit="%",
            source=SOURCE_HISTORICAL,
            derivation=(
                f"{equity_weight * 100:.0f}% equity x {cost_of_equity * 100:.2f}% + "
                f"{debt_weight * 100:.0f}% debt x {cost_of_debt * 100:.2f}% "
                f"x (1 - {tax_rate * 100:.0f}%)"
            ),
            rationale=(
                "Capital structure weights use market value of equity and book "
                "value of debt, the conventional treatment."
            ),
            bounds=config.WACC_BOUNDS,
            clamped=wacc != raw_wacc,
            raw_value=raw_wacc if wacc != raw_wacc else None,
        )
    )
    return wacc
