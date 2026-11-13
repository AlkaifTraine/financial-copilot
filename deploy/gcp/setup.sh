#!/usr/bin/env bash
#
# One-time bootstrap for a fresh Compute Engine VM (Ubuntu 24.04 LTS).
#
# Differs from the EC2 version in one way that matters: nothing is hardcoded to
# the user "ubuntu". On GCE your login name is derived from your Google account
# (romanempirealkaif@gmail.com -> romanempirealkaif), so every path and the
# systemd unit are generated from whoever runs this.
#
# Run once, by hand, after `gcloud compute ssh`. Idempotent - safe to re-run.
#
#   gcloud compute ssh fincopilot --zone=us-central1-a
#   git clone https://github.com/AlkaifTraine/financial-copilot.git
#   cd financial-copilot
#   bash deploy/gcp/setup.sh
#
set -euo pipefail

APP_USER="$(whoami)"
APP_DIR="$HOME/financial-copilot"
ENV_FILE="$HOME/financial-copilot.env"

if [ "$APP_USER" = "root" ]; then
    echo "Run this as your normal login user, not root." >&2
    exit 1
fi

if [ ! -f "$APP_DIR/app.py" ]; then
    echo "Expected the repo at $APP_DIR (app.py not found there)." >&2
    echo "Clone it to your home directory first, then re-run." >&2
    exit 1
fi

echo "==> System packages"
sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-pip git curl

# --- Swap -------------------------------------------------------------------
# e2-micro has 1 GiB of RAM, the same squeeze as the t3.micro this came from.
# Parsing a 200-page 10-K peaks above that, so 2 GiB of swap turns an
# out-of-memory kill into a brief slowdown. GCE images ship with no swap at all.
if ! sudo swapon --show | grep -q '/swapfile'; then
    echo "==> Creating 2G swap"
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    grep -q '^/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
    # e2-micro is slow and small; lean on swap a little earlier than the default.
    sudo sysctl -w vm.swappiness=20
    grep -q '^vm.swappiness' /etc/sysctl.conf || echo 'vm.swappiness=20' | sudo tee -a /etc/sysctl.conf
fi

# --- Python environment -----------------------------------------------------
echo "==> Python virtualenv"
cd "$APP_DIR"
[ -d venv ] || python3 -m venv venv
./venv/bin/pip install --upgrade pip
# --no-cache-dir: the wheel cache is dead weight on this disk, and on a 1 GiB
# VM the cache write competes for memory with the build that is already the
# peak of this script. This is the step most likely to be OOM-killed.
./venv/bin/pip install --no-cache-dir -r requirements.txt

# --- Secrets ----------------------------------------------------------------
# Lives outside the repo, so `git reset --hard` during a deploy never touches it.
if [ ! -f "$ENV_FILE" ]; then
    echo "==> Creating secrets file at $ENV_FILE (EDIT IT NEXT)"
    cat > "$ENV_FILE" <<'EOF'
OPENAI_API_KEY=sk-REPLACE_ME
SEC_USER_AGENT=Financial Copilot your-email@example.com
GEMINI_API_KEY=
DEMO_MODE=1
MAX_USD_PER_PROCESS=25
# ACCESS_CODE=pick-something-and-keep-it-off-github
EOF
    chmod 600 "$ENV_FILE"
fi

# --- systemd service --------------------------------------------------------
# Generated rather than copied, because User= and the paths depend on $APP_USER.
echo "==> Installing systemd service for user $APP_USER"
sudo tee /etc/systemd/system/financial-copilot.service >/dev/null <<EOF
[Unit]
Description=Financial Copilot (Streamlit)
After=network.target

[Service]
Type=simple
User=${APP_USER}
WorkingDirectory=${APP_DIR}
EnvironmentFile=${ENV_FILE}
ExecStart=${APP_DIR}/venv/bin/streamlit run app.py \
    --server.port 8501 \
    --server.address 127.0.0.1 \
    --server.headless true \
    --server.enableCORS false \
    --server.enableXsrfProtection true \
    --browser.gatherUsageStats false
Restart=always
RestartSec=5

# Give the process room; a large filing parse is the memory peak.
MemoryHigh=900M

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable financial-copilot

echo
echo "============================================================"
echo "Base setup complete. Next:"
echo "  1. Edit $ENV_FILE - real OPENAI_API_KEY and SEC_USER_AGENT."
echo "  2. sudo systemctl start financial-copilot"
echo "  3. Point fincopilot.site at this VM's static IP, wait for DNS,"
echo "     then:  bash deploy/gcp/setup_https.sh fincopilot.site www.fincopilot.site"
echo
echo "Until HTTPS is set up the app is on 127.0.0.1:8501 only - not public."
echo "============================================================"
