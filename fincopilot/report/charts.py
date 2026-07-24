"""
Report charts.

Design rules followed here, and why each one matters for a document that is
meant to look like professional research:

* **No dual-axis charts.** Revenue and margin are different scales, so they get
  separate panels. Overlaying them on two y-axes invents a visual correlation
  that is an artefact of where the two scales happen to be aligned.
* **History and forecast share one hue at two steps**, not two categorical
  colours. They are the same quantity at different levels of certainty, and an
  ordinal ramp says that; two unrelated hues would say they are different
  things. The pair `#2a78d6` / `#86b6ef` is validated as an ordinal ramp
  (monotone lightness, visible step gap, light end clears the surface).
* **The sensitivity grid is diverging, centred on the current share price.**
  The reader's question there is polarity — is this cell above or below what
  the market charges today — so blue/red around a neutral grey answers it
  directly, and a sequential ramp would not.
* **Recessive chrome.** Solid hairline gridlines one shade off the surface, no
  top or right spine, muted tick labels. The data is the ink.
* **Selective direct labels.** The latest actual and final forecast values are
  labelled; the axis carries the rest. A number on every bar is unreadable.
"""

from __future__ import annotations

import logging
from pathlib import Path

from .. import config

log = logging.getLogger(__name__)

# --- palette ---------------------------------------------------------------
# Validated reference values. Kept together so a rebrand is a single edit.
SURFACE = "#ffffff"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES_ACTUAL = "#2a78d6"      # blue, step 450
SERIES_FORECAST = "#86b6ef"    # blue, step 250 - lightest step that stays visible
DIVERGE_HIGH = "#2a78d6"
DIVERGE_MID = "#f0efec"
DIVERGE_LOW = "#e34948"

_DPI = 200


def _style_axes(ax, *, ylabel: str = "") -> None:
    """Apply the shared recessive chrome."""
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.6, linestyle="-")
    ax.set_axisbelow(True)

    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS)
        ax.spines[side].set_linewidth(0.8)

    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=8)


def _new_figure(width: float = 6.4, height: float = 2.8):
    import matplotlib

    matplotlib.use("Agg")           # headless: no display required
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots(figsize=(width, height))
    figure.patch.set_facecolor(SURFACE)
    return figure, ax


def _save(figure, name: str, slug: str) -> str:
    path = config.CHARTS_DIR / f"{slug}_{name}.png"
    figure.tight_layout(pad=0.6)
    figure.savefig(path, dpi=_DPI, facecolor=SURFACE, bbox_inches="tight")

    import matplotlib.pyplot as plt

    plt.close(figure)
    return str(path)


def _scale(values: list[float]) -> tuple[float, str]:
    """Pick a display divisor and unit so axis labels stay readable."""
    peak = max((abs(v) for v in values if v is not None), default=0.0)
    if peak >= 1e12:
        return 1e12, "tn"
    if peak >= 1e9:
        return 1e9, "bn"
    if peak >= 1e6:
        return 1e6, "m"
    return 1.0, ""


def _label(value: float) -> str:
    """Format a scaled value with enough precision to stay meaningful.

    A fixed ``,.0f`` rendered $215.9bn as "0" once the axis divisor moved to
    trillions, and $3.6tn as "4" — the two labels the reader most needs.
    Decimals scale with magnitude instead.
    """
    magnitude = abs(value)
    if magnitude >= 100:
        return f"{value:,.0f}"
    if magnitude >= 10:
        return f"{value:,.1f}"
    return f"{value:,.2f}"


# ---------------------------------------------------------------------------
# Charts
# ---------------------------------------------------------------------------

def revenue_chart(
    history_years: list[int],
    history_values: list[float],
    forecast_years: list[int],
    forecast_values: list[float],
    *,
    slug: str,
    currency: str = "USD",
) -> str | None:
    """Reported revenue beside the forecast path."""
    if not history_values:
        return None

    divisor, unit = _scale(history_values + forecast_values)
    figure, ax = _new_figure()

    years = history_years + forecast_years
    positions = range(len(years))

    actual = [v / divisor for v in history_values]
    projected = [v / divisor for v in forecast_values]

    ax.bar(
        list(positions)[: len(actual)],
        actual,
        width=0.72,
        color=SERIES_ACTUAL,
        label="Reported",
    )
    ax.bar(
        list(positions)[len(actual) :],
        projected,
        width=0.72,
        color=SERIES_FORECAST,
        label="Forecast",
    )

    ax.set_xticks(list(positions))
    ax.set_xticklabels([f"FY{y % 100:02d}" for y in years])
    _style_axes(ax, ylabel=f"Revenue ({currency} {unit})")

    # Label only the last reported year and the end of the forecast.
    for index, value in ((len(actual) - 1, actual[-1] if actual else None),
                         (len(years) - 1, projected[-1] if projected else None)):
        if value is None:
            continue
        ax.text(
            index,
            value,
            _label(value),
            ha="center",
            va="bottom",
            fontsize=8,
            color=INK_SECONDARY,
        )

    if projected:
        legend = ax.legend(frameon=False, fontsize=8, loc="upper left")
        for text in legend.get_texts():
            text.set_color(INK_SECONDARY)

    ax.set_title("Revenue: reported and forecast", color=INK_PRIMARY,
                 fontsize=10, loc="left", pad=10)
    return _save(figure, "revenue", slug)


def margin_chart(
    history_years: list[int],
    history_margins: list[float],
    forecast_years: list[int],
    forecast_margins: list[float],
    *,
    slug: str,
) -> str | None:
    """Operating margin over time.

    A separate panel from revenue rather than a second axis on it.
    """
    if not history_margins:
        return None

    figure, ax = _new_figure(height=2.4)

    years = history_years + forecast_years
    values = [m * 100 for m in history_margins + forecast_margins]
    split = len(history_margins)

    ax.plot(range(split), values[:split], color=SERIES_ACTUAL,
            linewidth=2.0, marker="o", markersize=4, label="Reported")
    if forecast_margins:
        # Joined at the boundary so the line reads as one continuous series.
        ax.plot(
            range(split - 1, len(values)),
            values[split - 1 :],
            color=SERIES_FORECAST,
            linewidth=2.0,
            linestyle="-",
            marker="o",
            markersize=4,
            label="Forecast",
        )

    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([f"FY{y % 100:02d}" for y in years])
    _style_axes(ax, ylabel="Operating margin (%)")
    ax.set_ylim(0, max(values) * 1.25)

    ax.text(split - 1, values[split - 1], f" {values[split - 1]:.1f}%",
            ha="left", va="bottom", fontsize=8, color=INK_SECONDARY)

    if forecast_margins:
        legend = ax.legend(frameon=False, fontsize=8, loc="lower left")
        for text in legend.get_texts():
            text.set_color(INK_SECONDARY)

    ax.set_title("Operating margin", color=INK_PRIMARY, fontsize=10,
                 loc="left", pad=10)
    return _save(figure, "margin", slug)


def cash_flow_chart(
    years: list[int],
    values: list[float],
    *,
    slug: str,
    currency: str = "USD",
) -> str | None:
    """Reported free cash flow."""
    if not values:
        return None

    divisor, unit = _scale(values)
    figure, ax = _new_figure(height=2.4)

    scaled = [v / divisor for v in values]
    ax.bar(range(len(years)), scaled, width=0.72, color=SERIES_ACTUAL)

    ax.set_xticks(range(len(years)))
    ax.set_xticklabels([f"FY{y % 100:02d}" for y in years])
    _style_axes(ax, ylabel=f"Free cash flow ({currency} {unit})")

    ax.text(len(years) - 1, scaled[-1], _label(scaled[-1]), ha="center",
            va="bottom", fontsize=8, color=INK_SECONDARY)

    ax.set_title("Free cash flow", color=INK_PRIMARY, fontsize=10,
                 loc="left", pad=10)
    return _save(figure, "fcf", slug)


def sensitivity_chart(
    wacc_values: list[float],
    growth_values: list[float],
    grid: list[list[float]],
    *,
    slug: str,
    share_price: float | None = None,
) -> str | None:
    """Fair value across WACC and terminal growth.

    Diverging around the current share price: the reader's question is whether
    a cell sits above or below what the market charges today, which is a
    polarity question, not a magnitude one.
    """
    if not grid or not grid[0]:
        return None

    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

    data = np.array(grid, dtype=float)
    if not np.isfinite(data).any():
        return None

    figure, ax = _new_figure(width=6.4, height=3.2)

    colormap = LinearSegmentedColormap.from_list(
        "value", [DIVERGE_LOW, DIVERGE_MID, DIVERGE_HIGH]
    )

    finite = data[np.isfinite(data)]
    centre = share_price if share_price else float(np.nanmedian(finite))
    low, high = float(np.nanmin(finite)), float(np.nanmax(finite))

    # TwoSlopeNorm requires the centre to sit strictly inside the range; when
    # every cell falls on one side of the price, fall back to a plain scale so
    # the chart still renders rather than raising.
    if low < centre < high:
        norm = TwoSlopeNorm(vmin=low, vcenter=centre, vmax=high)
    else:
        norm = None

    image = ax.imshow(data, cmap=colormap, norm=norm, aspect="auto",
                      origin="lower")

    ax.set_xticks(range(len(growth_values)))
    ax.set_xticklabels([f"{g * 100:.2f}%" for g in growth_values])
    ax.set_yticks(range(len(wacc_values)))
    ax.set_yticklabels([f"{w * 100:.1f}%" for w in wacc_values])

    ax.set_xlabel("Terminal growth", color=INK_SECONDARY, fontsize=8)
    ax.set_ylabel("WACC", color=INK_SECONDARY, fontsize=8)
    ax.tick_params(colors=INK_MUTED, labelsize=8, length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.grid(False)

    # Every cell carries its value, so the chart is readable without relying on
    # colour alone — the table-view requirement, met inline.
    for row in range(data.shape[0]):
        for column in range(data.shape[1]):
            value = data[row, column]
            if not np.isfinite(value):
                continue
            ax.text(column, row, f"{value:,.0f}", ha="center", va="center",
                    fontsize=7.5, color=INK_PRIMARY)

    bar = figure.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    bar.ax.tick_params(colors=INK_MUTED, labelsize=7, length=0)
    bar.outline.set_visible(False)
    if share_price:
        bar.set_label(f"Fair value (price {share_price:,.0f})",
                      color=INK_SECONDARY, fontsize=7.5)

    ax.set_title("Fair value sensitivity", color=INK_PRIMARY, fontsize=10,
                 loc="left", pad=10)
    return _save(figure, "sensitivity", slug)


def build_all(valuation, history, *, slug: str) -> dict[str, str]:
    """Render every chart the report uses. Missing data skips a chart, never raises."""
    charts: dict[str, str] = {}
    config.CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    revenue_series = history.series("revenue")
    margin_series = history.series("operating_margin")
    fcf_series = history.series("free_cash_flow")

    forecast = valuation.dcf.forecast if valuation.dcf else []
    forecast_years = [f.year for f in forecast]

    try:
        chart = revenue_chart(
            [y for y, _v in revenue_series],
            [v for _y, v in revenue_series],
            forecast_years,
            [f.revenue for f in forecast],
            slug=slug,
            currency=history.currency,
        )
        if chart:
            charts["revenue"] = chart
    except Exception as exc:
        log.warning("revenue chart failed: %s", exc)

    try:
        chart = margin_chart(
            [y for y, _v in margin_series],
            [v for _y, v in margin_series],
            forecast_years,
            [f.operating_margin for f in forecast],
            slug=slug,
        )
        if chart:
            charts["margin"] = chart
    except Exception as exc:
        log.warning("margin chart failed: %s", exc)

    try:
        chart = cash_flow_chart(
            [y for y, _v in fcf_series],
            [v for _y, v in fcf_series],
            slug=slug,
            currency=history.currency,
        )
        if chart:
            charts["fcf"] = chart
    except Exception as exc:
        log.warning("cash flow chart failed: %s", exc)

    if valuation.sensitivity:
        try:
            chart = sensitivity_chart(
                valuation.sensitivity.wacc_values,
                valuation.sensitivity.growth_values,
                valuation.sensitivity.values,
                slug=slug,
                share_price=valuation.share_price,
            )
            if chart:
                charts["sensitivity"] = chart
        except Exception as exc:
            log.warning("sensitivity chart failed: %s", exc)

    return charts
