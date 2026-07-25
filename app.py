"""
Financial Copilot - Streamlit interface.

Deliberately thin. Every computation lives in ``fincopilot``; this file resolves
input, holds session state, and renders. That separation is what lets the same
pipeline be driven from the evaluation harness and the tests without a browser.

The interface is organised around the order a user actually works in:

    Overview   what was loaded, and the documents it is grounded in
    Chat       ask questions, see the evidence behind each answer
    Valuation  the model, and every assumption behind it
    Report     the one-click research document

The Source Documents panel on Overview is not decoration. Being able to open
the exact filing a number came from is the difference between a demo and
something a finance user would trust twice.
"""

from __future__ import annotations

import logging
from pathlib import Path

import streamlit as st

from fincopilot import config, ui
from fincopilot.ratelimit import RateLimitExceeded, enforce
from fincopilot.chat import ask
from fincopilot.fundamentals import load_financials
from fincopilot.index import build_index
from fincopilot.report import build_report, render_document, render_html, render_pdf
from fincopilot.resolve import resolve_company
from fincopilot.retrieve import retrieve
from fincopilot.valuation import value_company

logging.basicConfig(level=logging.WARNING)

st.set_page_config(
    page_title="Financial Copilot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

ui.inject_css()


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

_DEFAULTS = {
    "company": None,
    "index": None,
    "ingest": None,
    "history": None,
    "valuation": None,
    "report": None,
    "messages": [],
    "load_error": None,
}

for key, default in _DEFAULTS.items():
    st.session_state.setdefault(key, default)


def reset_company_state() -> None:
    """Clear everything derived from the previously loaded company.

    Without this, switching companies leaves the prior chat history and
    valuation on screen under the new company's name — which would be a
    correctness problem, not a cosmetic one.
    """
    for key in ("index", "ingest", "history", "valuation", "report", "load_error"):
        st.session_state[key] = _DEFAULTS[key]
    st.session_state["messages"] = []


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_company(query: str, *, refresh: bool = False) -> None:
    """Run the full pipeline for ``query``, reporting progress as it goes."""
    try:
        enforce("load")
    except RateLimitExceeded as limit:
        st.session_state["load_error"] = str(limit)
        return

    reset_company_state()

    with st.status("Loading company...", expanded=True) as status:
        try:
            status.write("Resolving company identity...")
            company = resolve_company(query)
            st.session_state["company"] = company
            status.write(
                f"**{company.name}** ({company.ticker} · {company.exchange})"
                + (f" · SEC CIK {company.cik}" if company.cik else " · not an SEC filer")
            )

            def progress(stage: str, detail: str) -> None:
                if detail:
                    status.write(f"`{stage}` {detail}")

            status.write("Finding, downloading and indexing filings...")
            index, ingest = build_index(company, refresh=refresh, progress=progress)

            if index is None:
                st.session_state["load_error"] = (
                    "No usable documents could be indexed for this company."
                    + ("\n\n" + "\n\n".join(ingest.notes) if ingest.notes else "")
                )
                status.update(label="Could not load company", state="error")
                return

            st.session_state["index"] = index
            st.session_state["ingest"] = ingest

            status.write("Loading audited financials...")
            history = load_financials(company)
            st.session_state["history"] = history

            if history and history.is_sufficient_for_dcf:
                status.write("Building the valuation...")
                context = retrieve(
                    "management outlook guidance revenue growth margin expectations",
                    index,
                    top_k=6,
                ).context_block
                st.session_state["valuation"] = value_company(
                    company, history, qualitative_context=context
                )

            status.update(
                label=f"{company.name} ready · {len(index.chunks):,} passages indexed",
                state="complete",
                expanded=False,
            )

        except LookupError as exc:
            st.session_state["load_error"] = str(exc)
            status.update(label="Company not found", state="error")
        except Exception as exc:  # surfaced in the UI rather than swallowed
            logging.exception("load failed")
            st.session_state["load_error"] = f"{type(exc).__name__}: {exc}"
            status.update(label="Loading failed", state="error")


# ---------------------------------------------------------------------------
# Search (main page)
# ---------------------------------------------------------------------------

def render_search(*, key: str) -> None:
    """Company search on the main page. A form, so Enter submits."""
    with st.form(key=key, clear_on_submit=False, border=False):
        cols = st.columns([5, 1])
        query = cols[0].text_input(
            "Company",
            placeholder="Enter a company name or ticker — e.g. NVIDIA, Microsoft, TCS",
            label_visibility="collapsed",
            key=f"{key}_q",
        )
        submitted = cols[1].form_submit_button(
            "Analyze", type="primary", width="stretch"
        )
        refresh = st.checkbox("Re-fetch documents (ignore cache, slower)", key=f"{key}_r")

    if submitted:
        if query.strip():
            load_company(query.strip(), refresh=refresh)
            # load_company populates session state. Rerun so the freshly loaded
            # company renders its dashboard: the landing page (where this search
            # lives) ends in st.stop(), so without a rerun the script halts here
            # and the user is left staring at a completed status on the landing
            # screen instead of advancing to the results.
            st.rerun()
        else:
            st.warning("Enter a company name or ticker.")


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    ui.sidebar_brand()
    st.caption("Equity research from primary filings, with citations you can check.")

    company = st.session_state["company"]
    index = st.session_state["index"]

    if company and index:
        st.divider()
        st.markdown(f"**{company.name}**")
        st.caption(
            f"{company.ticker} · {company.exchange} · {company.country}\n\n"
            f"{len(index.chunks):,} passages · "
            f"{len(st.session_state['ingest'].accepted)} documents"
        )

    if not config.is_sec_configured():
        st.divider()
        st.warning(
            "`SEC_USER_AGENT` has no contact email, so SEC EDGAR is disabled. "
            "Filings fall back to web search and audited XBRL financials are "
            "unavailable.",
            icon="⚠️",
        )

    st.divider()
    st.caption(
        "Not investment advice. Figures come from public filings and market data "
        "and may contain errors."
    )


# ---------------------------------------------------------------------------
# Empty / error states
# ---------------------------------------------------------------------------

if st.session_state["load_error"]:
    st.error(st.session_state["load_error"])

if not st.session_state["index"]:
    ui.hero(
        title="Read the filings. Value the company. Cite every number.",
        subtitle=(
            "Enter a company and Financial Copilot fetches its SEC filings, answers "
            "questions with verifiable citations, and builds a discounted-cash-flow "
            "valuation where every assumption is stated and justified."
        ),
    )

    st.markdown("<div style='height:22px'></div>", unsafe_allow_html=True)
    render_search(key="search_landing")
    st.caption(
        "Try **NVIDIA**, **Microsoft**, or **TCS**. The first load of a company takes a "
        "few minutes while filings are downloaded and embedded; after that it is instant "
        "from cache."
    )

    st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
    ui.feature_grid([
        ("🔎", "Grounded answers", "Every response cites the exact document, section and page — open the source and check it yourself."),
        ("🧩", "Advanced retrieval", "Table-aware chunking, hybrid dense + sparse search, reranking. Not tutorial RAG."),
        ("🧮", "Real valuation", "Deterministic DCF on audited XBRL data. The model proposes assumptions; the math is never left to it."),
        ("📄", "One-click report", "A full equity research document — analysis, valuation, sensitivity, sources — as HTML and PDF."),
    ])
    st.stop()


# ---------------------------------------------------------------------------
# Main tabs
# ---------------------------------------------------------------------------

company = st.session_state["company"]
index = st.session_state["index"]
ingest = st.session_state["ingest"]
history = st.session_state["history"]
valuation = st.session_state["valuation"]

with st.expander("🔎  Analyze a different company"):
    render_search(key="search_loaded")

overview_tab, chat_tab, valuation_tab, report_tab = st.tabs(
    ["Overview", "Chat", "Valuation", "Report"]
)


# -- Overview ---------------------------------------------------------------

with overview_tab:
    meta = f"{company.ticker} · {company.exchange} · {company.sector or 'sector n/a'}" + (
        f" · SEC CIK {company.cik}" if company.cik else ""
    )
    ui.company_header(
        company.name,
        meta,
        rating=valuation.rating if valuation else None,
        upside=valuation.upside if valuation else None,
    )

    if history:
        latest = history.latest
        period = f"FY{latest.fiscal_year}"

        # `delta` renders as a coloured arrow, so only a genuine change belongs
        # in it. Passing the fiscal year there drew a green up-arrow beside
        # "FY2026", which reads as growth of an unspecified quantity.
        revenue_growth = dict(history.growth_rates("revenue")).get(latest.fiscal_year)

        def _bn(value):
            """Compact money that fits a metric card: 'USD 215.9B'."""
            if value is None:
                return "-"
            unit, div = ("T", 1e12) if abs(value) >= 1e12 else ("B", 1e9)
            return f"{history.currency} {value / div:,.1f}{unit}"

        # Four wide cards, not five narrow ones — five truncated the values
        # ("USD 21..."). The data source moves to a caption below.
        cols = st.columns(4)
        cols[0].metric(
            f"Revenue · {period}",
            _bn(latest.revenue),
            f"{revenue_growth * 100:+.1f}% YoY" if revenue_growth is not None else None,
        )
        cols[1].metric(
            f"Op. margin · {period}",
            f"{latest.operating_margin * 100:.1f}%"
            if latest.operating_margin is not None else "-",
        )
        cols[2].metric(f"Free cash flow · {period}", _bn(latest.free_cash_flow))
        cols[3].metric(
            "Share price",
            f"{history.currency} {history.share_price:,.2f}"
            if history.share_price else "-",
        )

        source_label = (
            "SEC XBRL — the company's own audited, tagged filing data"
            if history.source == "sec_xbrl"
            else "a structured market-data provider"
        )
        st.caption(f"Financial figures sourced from {source_label}.")

        if history.source != "sec_xbrl":
            st.info(
                f"{company.name} does not file with the SEC, so financial figures come "
                f"from a structured market data provider rather than audited XBRL. "
                f"Verify against the company's own reports before relying on them.",
                icon="ℹ️",
            )

    st.divider()
    st.markdown("#### Source documents")
    st.caption(
        "Every answer and figure is grounded in these documents. Open the original "
        "or download the copy that was indexed."
    )

    for document in ingest.accepted:
        row = st.columns([5, 2, 2])
        with row[0]:
            st.markdown(f"**{document.label}**")
            meta = []
            if document.filed_date:
                meta.append(f"filed {document.filed_date}")
            if document.page_count:
                meta.append(f"{document.page_count} pages")
            meta.append("SEC EDGAR" if document.origin == "sec_edgar" else "investor relations")
            st.caption(" · ".join(meta))
        with row[1]:
            st.link_button("Open original", document.url, width='stretch')
        with row[2]:
            path = Path(document.local_path) if document.local_path else None
            if path and path.exists():
                st.download_button(
                    "Download",
                    data=path.read_bytes(),
                    file_name=path.name,
                    mime="application/pdf" if path.suffix == ".pdf" else "text/html",
                    key=f"dl_{document.sha256}",
                    width='stretch',
                )

    if ingest.rejected:
        with st.expander(f"{len(ingest.rejected)} documents were rejected — and why"):
            st.caption(
                "Shown deliberately. A search engine returns plenty of documents that "
                "look right and are not; these were filtered before indexing."
            )
            for document in ingest.rejected:
                st.markdown(f"- **{document.label}** — {document.rejection_reason}")

    for note in ingest.notes:
        st.info(note, icon="ℹ️")


# -- Chat -------------------------------------------------------------------

with chat_tab:
    st.markdown("#### Ask the filings")
    st.caption(
        "Answers are drawn only from the indexed documents. Each claim carries a "
        "citation you can expand to see the passage it came from."
    )

    if not st.session_state["messages"]:
        st.markdown("**Try:**")
        examples = [
            "What drove the change in gross margin?",
            "What are the biggest customer concentration risks?",
            "How has free cash flow trended, and why?",
        ]
        example_cols = st.columns(len(examples))
        for column, example in zip(example_cols, examples):
            if column.button(example, width='stretch', key=f"ex_{example[:12]}"):
                st.session_state["messages"].append({"role": "user", "content": example})
                st.rerun()

    # Render the whole conversation, then answer any unanswered turn, then the
    # input. Generating the answer inline *before* the input placed the reply
    # below the text box, so the transcript read question / input box / answer.
    # Appending to state and rerunning keeps the ordering correct and means the
    # answer is rendered by exactly one code path.
    messages = st.session_state["messages"]

    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            for citation in message.get("citations", []):
                with st.expander(f"[{citation['number']}] {citation['label']}"):
                    st.markdown(f"> {citation['snippet']}")
                    st.link_button("Open source document", citation["url"])
            if message["role"] == "assistant" and not message.get("citations"):
                st.caption(
                    "No citations — the filings did not contain an answer to this."
                )

    if messages and messages[-1]["role"] == "user":
        try:
            enforce("chat")
        except RateLimitExceeded as limit:
            messages.append({"role": "assistant", "content": str(limit), "citations": []})
            st.rerun()

        with st.spinner("Searching the filings..."):
            answer = ask(messages[-1]["content"], index, history=messages[:-1])

        messages.append(
            {
                "role": "assistant",
                "content": answer.text,
                "citations": [
                    {
                        "number": c.number,
                        "label": f"{c.doc_title}"
                        + (f" · {c.section}" if c.section else "")
                        + (f" · p.{c.page}" if c.page else ""),
                        "snippet": c.snippet,
                        "url": c.source_url,
                    }
                    for c in answer.citations
                ],
            }
        )
        st.rerun()

    if prompt := st.chat_input("Ask a question about the filings..."):
        messages.append({"role": "user", "content": prompt})
        st.rerun()


# -- Valuation --------------------------------------------------------------

with valuation_tab:
    if not valuation or not valuation.dcf:
        st.warning(
            "A valuation could not be built for this company."
            + ("\n\n" + "\n\n".join(valuation.warnings) if valuation else "")
        )
    else:
        dcf = valuation.dcf
        currency = valuation.currency

        top = st.columns(4)
        top[0].metric("Market price", f"{currency} {valuation.share_price:,.2f}"
                      if valuation.share_price else "-")
        top[1].metric("DCF fair value", f"{currency} {valuation.fair_value:,.2f}")
        top[2].metric("Upside", valuation.upside_display
                      if hasattr(valuation, "upside_display")
                      else f"{valuation.upside * 100:+.1f}%")
        top[3].metric("Rating", valuation.rating)

        if valuation.market_implied_growth is not None:
            st.info(
                f"**What the price implies.** Holding every other assumption fixed, "
                f"today's price of {currency} {valuation.share_price:,.2f} implies "
                f"first-year revenue growth of about "
                f"**{valuation.market_implied_growth * 100:.0f}%**, decaying toward the "
                f"terminal rate. Compare that with the reported history before reading "
                f"the rating as a forecast.",
                icon="🔍",
            )

        st.markdown("#### Assumptions")
        st.caption(
            "Every input, where it came from, and why. Inputs tagged *bounded* were "
            "proposed by the language model and constrained to a range supported by "
            "reported history before entering the calculation."
        )

        for item in valuation.assumptions.items:
            with st.container(border=True):
                head = st.columns([3, 1])
                head[0].markdown(f"**{item.label}**")
                head[1].markdown(f"### {item.display}")
                tags = f"`{item.source}`"
                if item.clamped:
                    tags += f" `bounded from {item.raw_display}`"
                st.caption(f"{tags} — {item.derivation}")
                if item.rationale:
                    st.caption(item.rationale)

        if valuation.sensitivity:
            st.markdown("#### Sensitivity")
            st.caption(
                "Fair value per share across the discount rate and terminal growth "
                "rate — the two assumptions that move a DCF most."
            )
            grid = valuation.sensitivity
            import pandas as pd

            frame = pd.DataFrame(
                grid.values,
                index=[f"WACC {w * 100:.1f}%" for w in grid.wacc_values],
                columns=[f"g {g * 100:.2f}%" for g in grid.growth_values],
            )
            st.dataframe(
                frame.style.format("{:,.0f}").background_gradient(cmap="RdYlBu"),
                width='stretch',
            )

        st.markdown("#### Forecast")
        import pandas as pd

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Year": f"FY{f.year}",
                        "Revenue": f.revenue,
                        "Growth": f"{f.revenue_growth * 100:.1f}%",
                        "Op. margin": f"{f.operating_margin * 100:.1f}%",
                        "Free cash flow": f.free_cash_flow,
                        "Present value": f.present_value,
                    }
                    for f in dcf.forecast
                ]
            ),
            width='stretch',
            hide_index=True,
        )

        with st.expander("Model notes and limitations"):
            st.caption(
                f"Terminal value is {dcf.terminal_value_share * 100:.0f}% of "
                f"enterprise value."
            )
            for warning in valuation.warnings:
                st.markdown(f"- {warning}")


# -- Report -----------------------------------------------------------------

with report_tab:
    st.markdown("#### One-click equity research report")
    st.caption(
        "Generates a full research document: business analysis, financial performance, "
        "growth drivers, competitive position, risks, outlook, the valuation, every "
        "assumption, and a source appendix."
    )

    if not valuation or not valuation.dcf:
        st.warning("A valuation is required before a report can be generated.")
    elif st.button("Generate report", type="primary"):
        try:
            enforce("report")
            with st.status("Writing report...", expanded=True) as status:
                def progress(stage: str, detail: str) -> None:
                    status.write(f"`{stage}` {detail}")

                report = build_report(
                    company, history, valuation, ingest, index, progress=progress
                )
                st.session_state["report"] = report
                status.update(label="Report ready", state="complete", expanded=False)
        except RateLimitExceeded as limit:
            st.warning(str(limit))

    report = st.session_state["report"]
    if report:
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        stem = config.REPORTS_DIR / f"{company.slug}_equity_research"

        pdf_path = render_pdf(report, f"{stem}.pdf")
        html_document = render_document(report)

        downloads = st.columns(2)
        downloads[0].download_button(
            "⬇ Download PDF",
            data=Path(pdf_path).read_bytes(),
            file_name=f"{company.slug}_equity_research.pdf",
            mime="application/pdf",
            width='stretch',
            type="primary",
        )
        downloads[1].download_button(
            "⬇ Download HTML",
            data=html_document,
            file_name=f"{company.slug}_equity_research.html",
            mime="text/html",
            width='stretch',
        )

        st.divider()
        st.components.v1.html(render_html(report), height=1400, scrolling=True)
