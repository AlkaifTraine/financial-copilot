#!/usr/bin/env bash
#
# Add a real HTTPS certificate and a domain, using Caddy.
#
# Caddy fetches a free Let's Encrypt certificate automatically and renews it
# forever, with a single line of config. It replaces nginx as the reverse proxy
# (it also handles Streamlit's WebSocket upgrade without the manual headers
# nginx needed).
#
# Prerequisites, done before running this:
#   1. A domain (e.g. a free DuckDNS subdomain) pointing at this server's IP.
#   2. The security group allows inbound 80 AND 443 from anywhere.
#
# Usage:
#   bash deploy/setup_https.sh your-subdomain.duckdns.org
#
set -euo pipefail

DOMAIN="${1:?Usage: bash deploy/setup_https.sh <your-domain>}"

echo "==> Installing Caddy"
sudo apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' \
    | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' \
    | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt-get update
sudo apt-get install -y caddy

echo "==> Writing Caddy config for ${DOMAIN}"
# One block: terminate HTTPS on the public domain, proxy to Streamlit on
# localhost. Caddy obtains and renews the certificate on its own.
sudo tee /etc/caddy/Caddyfile >/dev/null <<EOF
${DOMAIN} {
    reverse_proxy 127.0.0.1:8501
}
EOF

# nginx and Caddy both want port 80; hand it over to Caddy.
echo "==> Handing port 80 from nginx to Caddy"
sudo systemctl stop nginx 2>/dev/null || true
sudo systemctl disable nginx 2>/dev/null || true

sudo systemctl restart caddy

echo
echo "============================================================"
echo "Done. In ~30 seconds, visit:  https://${DOMAIN}"
echo "The first load waits for the certificate; after that it is instant."
echo "If it doesn't come up, check:  sudo journalctl -u caddy -n 30 --no-pager"
echo "============================================================"
