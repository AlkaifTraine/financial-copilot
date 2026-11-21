#!/usr/bin/env bash
#
# Rebrand Streamlit's static HTML shell.
#
# st.set_page_config(page_title=...) only takes effect after the JavaScript
# bundle loads and rewrites document.title. Until then the browser shows what
# is in the shipped index.html, which is "<title>Streamlit</title>".
#
# Three places that matters:
#   - the tab reads "Streamlit" for a beat on every cold load
#   - link previews (LinkedIn, WhatsApp, Slack) read the static HTML and never
#     run the JS, so a shared CV link is captioned "Streamlit"
#   - so do crawlers
#
# This is a self-hosted deployment, so patching the vendored asset is ours to
# do. It is NOT something to do on Streamlit Community Cloud, where the badge
# and branding are part of the free hosting terms.
#
# Idempotent. Re-run after any `pip install` that reinstalls streamlit; setup.sh
# calls it for exactly that reason.
#
set -euo pipefail

TITLE="${1:-Financial Copilot}"
APP_DIR="${APP_DIR:-$HOME/financial-copilot}"

SHELL_HTML="$(find "$APP_DIR/venv" -path '*/streamlit/static/index.html' -print -quit 2>/dev/null || true)"

if [ -z "$SHELL_HTML" ]; then
    echo "Could not find streamlit/static/index.html under $APP_DIR/venv" >&2
    echo "Is the virtualenv built? Nothing changed." >&2
    exit 1
fi

# Keep one pristine copy, so a later Streamlit upgrade can be diffed against
# what we started from rather than against our own edit.
if [ ! -f "$SHELL_HTML.orig" ]; then
    cp "$SHELL_HTML" "$SHELL_HTML.orig"
fi

current="$(grep -o '<title>[^<]*</title>' "$SHELL_HTML" | head -1 || true)"

if [ "$current" = "<title>$TITLE</title>" ]; then
    echo "Already branded: $current"
    exit 0
fi

if [ -z "$current" ]; then
    echo "No <title> found in $SHELL_HTML - Streamlit's shell has changed shape." >&2
    echo "Nothing changed; patch it by hand or update this script." >&2
    exit 1
fi

# Restore from the pristine copy first, so repeated runs never nest edits.
cp "$SHELL_HTML.orig" "$SHELL_HTML"
sed -i "s|<title>[^<]*</title>|<title>${TITLE}</title>|" "$SHELL_HTML"

# A description for link previews. Streamlit ships none, so this is an insert
# rather than a replace, and only if it is not already there.
if ! grep -q 'name="description"' "$SHELL_HTML"; then
    sed -i "s|<title>${TITLE}</title>|<title>${TITLE}</title><meta name=\"description\" content=\"An equity-research assistant that reads a company's own filings, answers questions with verifiable citations, and produces a one-click DCF valuation and research report.\"/>|" "$SHELL_HTML"
fi

echo "Branded: $(grep -o '<title>[^<]*</title>' "$SHELL_HTML" | head -1)"
echo "Pristine copy kept at: $SHELL_HTML.orig"
