#!/usr/bin/env bash
#
# HTTPS for fincopilot.site via Caddy + Let's Encrypt.
#
# Caddy is used instead of nginx for the same reason as on EC2: it obtains and
# renews the certificate itself, and it handles Streamlit's WebSocket upgrade
# without the manual proxy headers nginx needed.
#
# PREREQUISITES - check both, or the certificate request will fail and
# Let's Encrypt will rate-limit you for an hour:
#   1. fincopilot.site and www.fincopilot.site must already resolve to THIS
#      VM's static IP. Verify from your laptop:  nslookup fincopilot.site
#   2. The VM must have the http-server and https-server network tags, and the
#      VPC firewall must allow tcp:80 and tcp:443.
#
# Usage:
#   bash deploy/gcp/setup_https.sh fincopilot.site www.fincopilot.site
#
set -euo pipefail

if [ "$#" -lt 1 ]; then
    echo "Usage: bash deploy/gcp/setup_https.sh <domain> [extra-domain ...]" >&2
    exit 1
fi

DOMAINS="$*"
MY_IP="$(curl -s -H 'Metadata-Flavor: Google' \
    'http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/access-configs/0/external-ip' || echo '')"

echo "==> This VM's external IP: ${MY_IP:-unknown}"
for d in $DOMAINS; do
    resolved="$(getent hosts "$d" | awk '{print $1}' | head -1 || true)"
    echo "    $d resolves to: ${resolved:-NOTHING}"
    if [ -n "$MY_IP" ] && [ "$resolved" != "$MY_IP" ]; then
        echo
        echo "    !! $d does not point at this VM yet." >&2
        echo "    !! Fix the DNS A record and wait, or Let's Encrypt will fail." >&2
        echo
        read -r -p "    Continue anyway? [y/N] " ans
        [ "$ans" = "y" ] || exit 1
    fi
done

echo "==> Installing Caddy"
sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
    | sudo gpg --batch --yes --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
    | sudo tee /etc/apt/sources.list.d/caddy-stable.list >/dev/null
sudo apt-get update
sudo apt-get install -y caddy

echo "==> Writing Caddy config for: $DOMAINS"
# Caddy serves every listed name off one certificate and proxies to Streamlit.
# The long timeouts matter: report generation holds the connection for minutes.
sudo tee /etc/caddy/Caddyfile >/dev/null <<EOF
${DOMAINS// /, } {
    reverse_proxy 127.0.0.1:8501 {
        transport http {
            read_timeout 300s
            write_timeout 300s
        }
    }

    request_body {
        max_size 25MB
    }

    encode gzip
}
EOF

sudo systemctl restart caddy
sudo systemctl enable caddy

echo
echo "============================================================"
echo "Done. In ~30 seconds:  https://fincopilot.site"
echo "The first load waits on the certificate; after that it is instant."
echo "If it does not come up:  sudo journalctl -u caddy -n 40 --no-pager"
echo "============================================================"
