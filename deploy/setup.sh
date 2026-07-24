#!/usr/bin/env bash
#
# One-time bootstrap for a fresh Ubuntu 22.04/24.04 EC2 instance.
#
# Run it once, by hand, after SSHing into the instance. It is idempotent - safe
# to re-run. After it completes, ongoing deploys are automatic via GitHub
# Actions (see .github/workflows/deploy.yml); this script is only for the
# initial setup.
#
#   ssh ubuntu@<your-ip>
#   git clone https://github.com/<you>/financial-copilot.git
#   cd financial-copilot
#   bash deploy/setup.sh
#
set -euo pipefail

APP_DIR="/home/ubuntu/financial-copilot"
ENV_FILE="/home/ubuntu/financial-copilot.env"

echo "==> System packages"
sudo apt-get update -y
sudo apt-get install -y python3 python3-venv python3-pip nginx git

# --- Swap -------------------------------------------------------------------
# A t3.micro has 1 GiB of RAM. Parsing a 200-page 10-K peaks above that, so a
# 2 GiB swap file turns an out-of-memory crash into a brief slowdown. Without
# it the first load of a large company can kill the process.
if [ ! -f /swapfile ]; then
    echo "==> Creating 2G swap"
    sudo fallocate -l 2G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

# --- Python environment -----------------------------------------------------
echo "==> Python virtualenv"
cd "$APP_DIR"
python3 -m venv venv
./venv/bin/pip install --upgrade pip
./venv/bin/pip install -r requirements.txt

# --- Secrets ----------------------------------------------------------------
# Written outside the repo so `git pull` never touches it and it is never
# committed. Fill in the real values after this script finishes.
if [ ! -f "$ENV_FILE" ]; then
    echo "==> Creating secrets file at $ENV_FILE (EDIT IT NEXT)"
    cat > "$ENV_FILE" <<'EOF'
OPENAI_API_KEY=sk-REPLACE_ME
SEC_USER_AGENT=Financial Copilot your-email@example.com
DEMO_MODE=1
EOF
    chmod 600 "$ENV_FILE"
fi

# --- systemd service --------------------------------------------------------
echo "==> Installing systemd service"
sudo cp deploy/financial-copilot.service /etc/systemd/system/financial-copilot.service
sudo systemctl daemon-reload
sudo systemctl enable financial-copilot

# --- nginx ------------------------------------------------------------------
echo "==> Configuring nginx"
sudo cp deploy/nginx.conf /etc/nginx/sites-available/financial-copilot
sudo ln -sf /etc/nginx/sites-available/financial-copilot /etc/nginx/sites-enabled/financial-copilot
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t
sudo systemctl restart nginx

echo
echo "============================================================"
echo "Setup complete. Two things left:"
echo "  1. Edit $ENV_FILE and put in your real OPENAI_API_KEY"
echo "     and SEC_USER_AGENT (with your email)."
echo "  2. Start the app:  sudo systemctl start financial-copilot"
echo
echo "Then open  http://<this-instance-public-ip>/"
echo "============================================================"
