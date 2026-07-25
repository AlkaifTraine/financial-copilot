"""
Presentation layer: a custom theme that lifts the Streamlit app out of its
default look.

Streamlit's defaults read as "a data script with widgets." For a portfolio
piece the interface has to look considered, so this module injects a cohesive
design system — dark branded sidebar, light content canvas, card-based
surfaces, real typography — and provides a few HTML component helpers (hero,
feature cards, section headers) that Streamlit does not offer natively.

Selectors target Streamlit's stable ``data-testid`` / ``data-baseweb`` hooks
rather than hashed class names, so the theme survives minor version bumps.
"""

from __future__ import annotations

import streamlit as st

# --- design tokens ---------------------------------------------------------
BRAND_1 = "#2563eb"      # primary blue
BRAND_2 = "#4f46e5"      # indigo, for gradients
ACCENT = "#06b6d4"       # cyan, used sparingly
INK = "#0f172a"
INK_2 = "#475569"
MUTED = "#94a3b8"
BG = "#f4f6fb"
SURFACE = "#ffffff"
SURFACE_2 = "#eef2f9"
BORDER = "#e3e8f2"
POSITIVE = "#059669"
NEGATIVE = "#dc2626"


_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {{
  --brand-1: {BRAND_1};
  --brand-2: {BRAND_2};
  --accent: {ACCENT};
  --ink: {INK};
  --ink-2: {INK_2};
  --muted: {MUTED};
  --bg: {BG};
  --surface: {SURFACE};
  --surface-2: {SURFACE_2};
  --border: {BORDER};
  --positive: {POSITIVE};
  --negative: {NEGATIVE};
  --radius: 14px;
  --shadow: 0 1px 2px rgba(16,24,40,.04), 0 8px 24px rgba(16,24,40,.06);
}}

/* ---- base canvas ---- */
html, body, [data-testid="stAppViewContainer"] {{
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  color: var(--ink);
}}
[data-testid="stAppViewContainer"] {{
  background:
    radial-gradient(1200px 600px at 100% -10%, rgba(79,70,229,.06), transparent 60%),
    radial-gradient(900px 500px at -10% 0%, rgba(37,99,235,.05), transparent 55%),
    var(--bg);
}}
.block-container {{ padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1180px; }}

/* ---- slim, transparent top chrome ---- */
[data-testid="stHeader"] {{ background: transparent; }}
[data-testid="stToolbar"] {{ right: 1rem; }}

/* ---- typography ---- */
h1, h2, h3 {{ font-family: 'Space Grotesk', 'Inter', sans-serif; letter-spacing: -.02em; color: var(--ink); }}
h1 {{ font-weight: 700; }}
[data-testid="stMarkdownContainer"] p {{ color: var(--ink-2); }}

/* ---- sidebar: dark, branded ---- */
[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #0b1730 0%, #101f3e 40%, #14264d 100%);
  border-right: 1px solid rgba(255,255,255,.06);
}}
[data-testid="stSidebar"] > div {{ background: linear-gradient(180deg, #0b1730 0%, #101f3e 55%, #13233f 100%); }}
[data-testid="stSidebar"] * {{ color: #e6edf7 !important; }}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{ color: #ffffff !important; }}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .st-emotion-cache-1 {{ color: #aebbd0 !important; }}
[data-testid="stSidebar"] .stTextInput input {{
  background: rgba(255,255,255,.06) !important;
  color: #ffffff !important;
  border: 1px solid rgba(255,255,255,.14) !important;
}}
[data-testid="stSidebar"] .stTextInput input::placeholder {{ color: #7f8db0 !important; }}
/* Alerts (info/warning) inside the dark sidebar keep their own light background,
   so their text must stay dark — otherwise the blanket light-text rule above
   makes them near-invisible. */
[data-testid="stSidebar"] [data-testid="stAlert"] * {{ color: #1f2937 !important; }}

/* Captions on the light main canvas: readable secondary ink, not pale grey.
   No !important, so the dark-sidebar rule still wins inside the sidebar. */
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ color: var(--ink-2); }}

/* ---- buttons ---- */
/* `kind*="primary"` matches both a normal primary button (kind="primary") and
   a form submit button (kind="primaryFormSubmit"), so the search's Analyze
   button gets the gradient too. */
.stButton > button, .stDownloadButton > button, [data-testid="stFormSubmitButton"] > button {{
  border-radius: 10px;
  font-weight: 600;
  border: 1px solid var(--border);
  transition: transform .06s ease, box-shadow .2s ease, background .2s ease;
}}
.stButton > button:hover, .stDownloadButton > button:hover,
[data-testid="stFormSubmitButton"] > button:hover {{ transform: translateY(-1px); box-shadow: var(--shadow); }}
.stButton > button[kind*="primary"], .stDownloadButton > button[kind*="primary"],
[data-testid="stFormSubmitButton"] > button[kind*="primary"] {{
  background: linear-gradient(135deg, var(--brand-1), var(--brand-2));
  color: #fff; border: none;
  box-shadow: 0 6px 16px rgba(37,99,235,.28);
}}
/* Button labels render inside a markdown <p>, which the paragraph colour rule
   below would otherwise paint dark — making primary-button text unreadable on
   the gradient. Force the label to inherit the button's own colour. */
.stButton button *, .stDownloadButton button *,
[data-testid="stFormSubmitButton"] button * {{ color: inherit !important; }}

/* ---- metrics as cards ---- */
[data-testid="stMetric"] {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px 18px;
  box-shadow: var(--shadow);
}}
/* ink-2 (not muted) so labels clear WCAG AA on a white card — muted grey was ~2.6:1 */
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] * {{ color: var(--ink-2) !important; font-weight: 600; font-size: .74rem; text-transform: uppercase; letter-spacing: .04em; }}
[data-testid="stMetricValue"] {{ font-family: 'Space Grotesk', sans-serif; color: var(--ink); font-weight: 600; }}

/* ---- tabs: pill style ---- */
[data-baseweb="tab-list"] {{ gap: 6px; background: var(--surface-2); padding: 5px; border-radius: 12px; border: 1px solid var(--border); }}
[data-baseweb="tab"] {{ border-radius: 9px; padding: 6px 16px; color: var(--ink-2); font-weight: 600; }}
[data-baseweb="tab"][aria-selected="true"] {{ background: var(--surface); color: var(--brand-1); box-shadow: var(--shadow); }}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] {{ background: transparent !important; height: 0 !important; }}

/* ---- inputs ---- */
/* Main-page inputs sit on a light surface, so force dark, visible text and a
   clear focus ring. (The sidebar rules above intentionally override these for
   the dark sidebar; the search input now lives on the main page.) */
.stTextInput input {{
  border-radius: 10px;
  padding: 12px 15px !important;
  font-size: .96rem;
  color: var(--ink) !important;
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
}}
.stTextInput input::placeholder {{ color: var(--muted) !important; }}
.stTextInput input:focus {{
  border-color: var(--brand-1) !important;
  box-shadow: 0 0 0 3px rgba(37,99,235,.14) !important;
}}
[data-testid="stSidebar"] .stTextInput input {{ color: #fff !important; background: rgba(255,255,255,.06) !important; }}

/* ---- chat ---- */
[data-testid="stChatMessage"] {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  box-shadow: var(--shadow);
  padding: 4px 6px;
}}

/* ---- expanders ---- */
[data-testid="stExpander"] {{ border: 1px solid var(--border); border-radius: 12px; background: var(--surface); box-shadow: var(--shadow); }}
[data-testid="stExpander"] summary {{ font-weight: 600; color: var(--ink); }}

/* ---- dataframes ---- */
[data-testid="stDataFrame"] {{ border-radius: 12px; overflow: hidden; border: 1px solid var(--border); }}

/* ---- status widget ---- */
[data-testid="stStatusWidget"], [data-testid="stStatus"] {{ border-radius: 12px; }}

/* ---- custom components ---- */
.fc-hero {{
  position: relative;
  border-radius: 22px;
  padding: 46px 44px;
  color: #fff;
  background:
    radial-gradient(600px 300px at 100% 0%, rgba(6,182,212,.35), transparent 55%),
    linear-gradient(135deg, #0b1730 0%, #1e3a8a 55%, #4f46e5 120%);
  box-shadow: 0 20px 50px rgba(15,23,42,.25);
  overflow: hidden;
}}
.fc-hero::after {{
  content: ""; position: absolute; inset: 0;
  background-image: radial-gradient(rgba(255,255,255,.10) 1px, transparent 1px);
  background-size: 22px 22px; opacity: .4; pointer-events: none;
}}
.fc-hero .eyebrow {{
  display: inline-block; font-size: .72rem; font-weight: 700; letter-spacing: .14em;
  text-transform: uppercase; color: #a9c2ff; margin-bottom: 14px;
  padding: 5px 12px; border: 1px solid rgba(255,255,255,.18); border-radius: 999px;
  background: rgba(255,255,255,.06);
}}
.fc-hero h1 {{ color: #fff !important; font-size: 2.5rem; line-height: 1.08; margin: 0 0 12px; max-width: 20ch; }}
.fc-hero p {{ color: #cdd8ee !important; font-size: 1.05rem; max-width: 60ch; margin: 0; }}

.fc-features {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 16px; margin-top: 22px; }}
.fc-feature {{
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 20px; box-shadow: var(--shadow); transition: transform .12s ease, box-shadow .2s ease;
}}
.fc-feature:hover {{ transform: translateY(-3px); box-shadow: 0 12px 30px rgba(16,24,40,.12); }}
.fc-feature .ic {{
  width: 40px; height: 40px; border-radius: 11px; display: grid; place-items: center;
  font-size: 1.2rem; margin-bottom: 12px;
  background: linear-gradient(135deg, rgba(37,99,235,.12), rgba(79,70,229,.12)); color: var(--brand-1);
}}
.fc-feature h4 {{ font-family: 'Space Grotesk', sans-serif; margin: 0 0 6px; font-size: 1rem; color: var(--ink); }}
.fc-feature p {{ margin: 0; font-size: .86rem; color: var(--ink-2); line-height: 1.5; }}

.fc-brand {{ display: flex; align-items: center; gap: 11px; margin-bottom: 4px; }}
.fc-brand .logo {{
  width: 38px; height: 38px; border-radius: 11px; display: grid; place-items: center; font-size: 1.2rem;
  background: linear-gradient(135deg, var(--brand-1), var(--brand-2)); box-shadow: 0 6px 16px rgba(37,99,235,.4);
}}
.fc-brand .name {{ font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.18rem; color: #fff; }}

.fc-doc {{
  display: flex; align-items: center; gap: 12px;
  background: var(--surface); border: 1px solid var(--border); border-radius: 12px;
  padding: 12px 14px; box-shadow: var(--shadow);
}}
.fc-doc .badge {{
  font-size: .64rem; font-weight: 700; letter-spacing: .05em; text-transform: uppercase;
  padding: 3px 8px; border-radius: 6px; background: var(--surface-2); color: var(--ink-2); border: 1px solid var(--border);
}}
.fc-doc .badge.edgar {{ background: rgba(5,150,105,.1); color: var(--positive); border-color: rgba(5,150,105,.2); }}

.fc-rating-pill {{
  display: inline-flex; align-items: center; gap: 8px; font-family: 'Space Grotesk', sans-serif;
  font-weight: 700; font-size: 1.05rem; padding: 8px 18px; border-radius: 999px; color: #fff;
}}
/* Darkened so white pill text clears WCAG AA (the lightest stop still gives
   >=4.5:1). The vivid amber/green ends were ~2-3:1 against white. */
.fc-rating-pill.buy {{ background: linear-gradient(135deg, #065f46, #047857); }}
.fc-rating-pill.sell {{ background: linear-gradient(135deg, #b91c1c, #dc2626); }}
.fc-rating-pill.hold {{ background: linear-gradient(135deg, #92400e, #b45309); }}
.fc-rating-pill.na {{ background: linear-gradient(135deg, #475569, #64748b); }}

.fc-section-title {{
  font-family: 'Space Grotesk', sans-serif; font-weight: 600; font-size: 1.15rem; color: var(--ink);
  display: flex; align-items: center; gap: 10px; margin: 6px 0 2px;
}}
.fc-section-title::before {{ content: ""; width: 4px; height: 20px; border-radius: 3px; background: linear-gradient(180deg, var(--brand-1), var(--brand-2)); }}

.fc-cohead {{
  display: flex; align-items: flex-start; justify-content: space-between; gap: 20px;
  background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius);
  padding: 22px 26px; box-shadow: var(--shadow); margin-bottom: 18px;
}}
.fc-cohead-name {{ font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.7rem; color: var(--ink); letter-spacing: -.02em; }}
.fc-cohead-meta {{ color: var(--ink-2); font-size: .9rem; margin-top: 3px; }}
</style>
"""


def inject_css() -> None:
    """Apply the theme. Call once, right after ``st.set_page_config``."""
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, eyebrow: str = "AI-Powered Equity Research") -> None:
    st.markdown(
        f"""
        <div class="fc-hero">
          <span class="eyebrow">{eyebrow}</span>
          <h1>{title}</h1>
          <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def feature_grid(features: list[tuple[str, str, str]]) -> None:
    """features: list of (emoji, title, description)."""
    cards = "".join(
        f"""<div class="fc-feature"><div class="ic">{icon}</div>
             <h4>{title}</h4><p>{body}</p></div>"""
        for icon, title, body in features
    )
    st.markdown(f'<div class="fc-features">{cards}</div>', unsafe_allow_html=True)


def sidebar_brand() -> None:
    st.markdown(
        """
        <div class="fc-brand">
          <div class="logo">📊</div>
          <div class="name">Financial Copilot</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def rating_pill(rating: str) -> str:
    """Return HTML for a coloured rating pill."""
    cls = {"BUY": "buy", "SELL": "sell", "HOLD": "hold"}.get(rating, "na")
    return f'<span class="fc-rating-pill {cls}">{rating}</span>'


def section_title(text: str) -> None:
    st.markdown(f'<div class="fc-section-title">{text}</div>', unsafe_allow_html=True)


def company_header(
    name: str,
    meta: str,
    *,
    rating: str | None = None,
    upside: float | None = None,
) -> None:
    """A research-style header: company on the left, rating pill on the right."""
    right = ""
    if rating and rating != "NOT RATED":
        upside_txt = ""
        if upside is not None:
            colour = "var(--positive)" if upside > 0 else "var(--negative)"
            upside_txt = (
                f'<div style="text-align:right;font-weight:600;color:{colour};'
                f'font-family:Space Grotesk,sans-serif;margin-top:6px">'
                f'{upside * 100:+.1f}% to fair value</div>'
            )
        right = f'<div>{rating_pill(rating)}{upside_txt}</div>'

    st.markdown(
        f"""
        <div class="fc-cohead">
          <div>
            <div class="fc-cohead-name">{name}</div>
            <div class="fc-cohead-meta">{meta}</div>
          </div>
          {right}
        </div>
        """,
        unsafe_allow_html=True,
    )
