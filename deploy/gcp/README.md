# Moving Financial Copilot from AWS EC2 to Google Compute Engine

Target: `fincopilot.site` served over HTTPS from a free-tier `e2-micro` VM,
with the same push-to-main CI/CD you had on EC2.

Everything below runs from **your Windows machine** unless the step is marked
`[ON THE VM]`.

---

## What this replaces

| EC2 | GCE |
|---|---|
| t3.micro, Ubuntu | e2-micro, Ubuntu 24.04 LTS |
| Security group rules | VPC firewall rules + network tags |
| Elastic IP | Reserved regional static IP |
| user `ubuntu`, `/home/ubuntu` | user derived from your Google account |
| `deploy/setup.sh` | `deploy/gcp/setup.sh` |
| `deploy/setup_https.sh` | `deploy/gcp/setup_https.sh` (multi-domain) |
| secrets `EC2_HOST/USER/SSH_KEY` | secrets `GCP_HOST/USER/SSH_KEY` |

`deploy/nginx.conf` and `deploy/financial-copilot.service` are not reused.
Caddy replaces nginx outright, and the systemd unit is now **generated** by the
setup script because `User=` and every path depend on your GCE login name —
the single most common thing that breaks a straight copy of the EC2 setup.

---

## Step 0 — Accounts, and the honest cost picture

Create a Google Cloud account at <https://console.cloud.google.com>. A card is
required even for the free tier; new accounts get **$300 of credit valid for 90
days**.

What the Compute Engine free tier actually covers, per Google:

- 1 non-preemptible **e2-micro** per month, in **us-west1, us-central1 or
  us-east1 only**
- **30 GB-months of standard persistent disk** — `pd-standard`, *not* the
  `pd-balanced` the console picks by default
- **1 GB/month** outbound North America egress

What it does **not** cover, and which nothing warns you about until the bill:

- **The external IPv4 address: roughly $3.50/month**, billed whether the VM is
  running or stopped. Unavoidable for a public site. Your $300 credit absorbs
  it for the first 90 days.
- Egress past 1 GB/month, ~$0.12/GB. A Streamlit app serving PDF reports can
  pass 1 GB on modest traffic.

Realistic steady state is **~$3.50–6/month**, not zero — still well under an
always-on EC2 instance once credits lapse. Set a budget alert (Step 10) anyway.

---

## Step 1 — Install the gcloud CLI

In PowerShell:

    winget install Google.CloudSDK

Close and reopen PowerShell, then confirm:

    gcloud version

If `winget` is unavailable, use the installer at
<https://cloud.google.com/sdk/docs/install>.

Then sign in:

    gcloud auth login

---

## Steps 2–5, in one command

`provision.ps1` does everything from here to the end of Step 5 — project,
billing, API, static IP, firewall, VM — with the two defaults the console gets
wrong already set (`pd-standard`, and the `http-server`/`https-server` tags).
It is idempotent: every step checks before it creates, so re-running after a
failure is safe, and it asks before creating the VM.

    .\provision.ps1

It prints the static IP and the exact next commands when it finishes. Skip to
**Step 6**.

Steps 2–5 below are the same thing by hand, if you would rather see each
command or the script fails partway.

---

## Step 2 — Create the project and enable Compute Engine

    gcloud auth login
    gcloud projects create fincopilot-prod --name="Financial Copilot"
    gcloud config set project fincopilot-prod

Project IDs are globally unique. If `fincopilot-prod` is taken, append digits
and use that ID everywhere below.

Link billing — the project cannot create VMs without it:

    gcloud billing accounts list
    gcloud billing projects link fincopilot-prod --billing-account=XXXXXX-XXXXXX-XXXXXX

Then enable the API (~30 seconds):

    gcloud services enable compute.googleapis.com

---

## Step 3 — Reserve a static IP

Do this **before** creating the VM. An ephemeral IP changes on every VM
restart, which would silently break `fincopilot.site` at the worst moment.

    gcloud compute addresses create fincopilot-ip --region=us-central1

Print it and keep it — you need it for DNS and for GitHub secrets:

    gcloud compute addresses describe fincopilot-ip --region=us-central1 --format="value(address)"

---

## Step 4 — Create the VM

Note `--boot-disk-type=pd-standard`. The console defaults to `pd-balanced`,
which is **not** in the free tier and quietly costs ~$3/month by itself.

    gcloud compute instances create fincopilot --zone=us-central1-a --machine-type=e2-micro --image-family=ubuntu-2404-lts-amd64 --image-project=ubuntu-os-cloud --boot-disk-size=30GB --boot-disk-type=pd-standard --address=fincopilot-ip --tags=http-server,https-server

The `--tags` are what the firewall rules in the next step attach to. Without
them the VM is unreachable on 80/443 no matter what Caddy does.

---

## Step 5 — Open the firewall

GCP's default VPC allows SSH but **not** HTTP or HTTPS.

    gcloud compute firewall-rules create allow-http --allow=tcp:80 --target-tags=http-server --direction=INGRESS

    gcloud compute firewall-rules create allow-https --allow=tcp:443 --target-tags=https-server --direction=INGRESS

If either reports "already exists", that is fine — carry on.

---

## Step 6 — SSH in and bootstrap

    gcloud compute ssh fincopilot --zone=us-central1-a

The first run generates a key pair and uploads the public half. An empty
passphrase is fine.

`[ON THE VM]` — note your username, needed for GitHub secrets later:

    whoami

`[ON THE VM]` — clone and bootstrap:

    git clone https://github.com/AlkaifTraine/financial-copilot.git

    cd financial-copilot && bash deploy/gcp/setup.sh

This installs Python, creates a 2 GB swap file (e2-micro has 1 GB of RAM and
parsing a 200-page 10-K peaks above that — without swap the process is killed),
builds the venv, writes a secrets file, and installs a systemd unit generated
for *your* username.

The `pip install` takes several minutes on an e2-micro. That is expected.

---

## Step 7 — Secrets, then start it

`[ON THE VM]`:

    nano ~/financial-copilot.env

Fill in at minimum:

    OPENAI_API_KEY=sk-...
    SEC_USER_AGENT=Financial Copilot your-real-email@example.com
    DEMO_MODE=1

`SEC_USER_AGENT` must contain a real contact email — the SEC rejects requests
without one and IP-bans repeat offenders. Set `GEMINI_API_KEY` too if you want
the LiteLLM router's cross-provider fallback; without it the app runs
OpenAI-only and says so. Leave `ACCESS_CODE` unset unless you want visitors
spending your credits — unset means bring-your-own-key, the safe default.

Save (Ctrl+O, Enter, Ctrl+X), then:

    sudo systemctl start financial-copilot

    sudo systemctl status financial-copilot --no-pager

Confirm it is serving before touching DNS:

    curl -I http://127.0.0.1:8501

`200 OK` means the app is up. It is still private — Streamlit is bound to
localhost and nothing is listening on 80 or 443 yet.

---

## Step 8 — Repoint fincopilot.site

`fincopilot.site` currently resolves to `13.205.11.254`, your AWS instance in
Mumbai. At whichever registrar holds the domain, edit the DNS records:

| Type | Name | Value |
|---|---|---|
| A | `@` | *(the static IP from Step 3)* |
| CNAME | `www` | `fincopilot.site` |

The `www` CNAME already exists and is correct — only the A record changes.
Lower the TTL to 300 first if the registrar allows it.

Wait for propagation, then verify **from your Windows machine**:

    nslookup fincopilot.site

Do not move on until that returns the GCP IP. Running the next step early
burns a Let's Encrypt rate limit and locks you out for an hour.

---

## Step 9 — HTTPS

`[ON THE VM]`:

    cd ~/financial-copilot && bash deploy/gcp/setup_https.sh fincopilot.site www.fincopilot.site

The script re-checks DNS itself before installing anything, then installs
Caddy, which fetches a Let's Encrypt certificate and renews it forever with no
cron job. It handles Streamlit's WebSocket upgrade natively — the reason nginx
is not carried over.

Give it ~30 seconds, then open <https://fincopilot.site>.

A blank page is almost always the WebSocket — check
`sudo journalctl -u caddy -n 40 --no-pager`.

---

## Step 10 — CI/CD, and a budget alert

### The deploy key

GitHub Actions needs its own key, separate from your personal gcloud one. On
your Windows machine:

    ssh-keygen -t ed25519 -f "$env:USERPROFILE\gha_deploy" -N '""' -C "github-actions"

Add the public half to the VM, replacing `YOURUSER` with what `whoami` printed
in Step 6. The `YOURUSER:` prefix is part of the value — GCE rejects the key
without it:

    gcloud compute instances add-metadata fincopilot --zone=us-central1-a --metadata="ssh-keys=YOURUSER:$(Get-Content $env:USERPROFILE\gha_deploy.pub)"

### The repository secrets

In GitHub → Settings → Secrets and variables → Actions, add three:

| Secret | Value |
|---|---|
| `GCP_HOST` | the static IP from Step 3 |
| `GCP_USER` | your GCE username from Step 6 |
| `GCP_SSH_KEY` | full contents of `gha_deploy` (the private one, including the BEGIN/END lines) |

Delete the old `EC2_HOST`, `EC2_USER` and `EC2_SSH_KEY`.

Print the private key to copy it:

    Get-Content "$env:USERPROFILE\gha_deploy"

Then replace `.github/workflows/deploy.yml` with the GCP version, commit and
push. That push is itself the first deploy — watch it in the Actions tab.

The workflow is a plain SSH deploy, so no GCP service-account credential ever
sits in GitHub. The `sudo systemctl restart` works without a password because
GCE puts your login user in `google-sudoers` with NOPASSWD.

### Budget alert

Worth two minutes, given why you left AWS. Console → Billing → Budgets & alerts
→ Create budget, scoped to this project, $10, with alerts at 50/90/100%.

---

## Step 11 — Decommission AWS

Only once <https://fincopilot.site> works end to end: terminate the EC2
instance and **release the Elastic IP**. An unattached Elastic IP is billed
hourly and is the classic way an "unused" AWS account keeps charging.

---

## Known rough edges on e2-micro

- **1 GB RAM, 0.25 vCPU baseline.** Swap keeps large filings from OOM-killing
  the process, but the first load of a big 10-K will be noticeably slower than
  on t3.micro. If you hit real OOM kills (`sudo dmesg | grep -i oom`), resize
  to `e2-small` — 2 GB RAM, ~$13/month, no longer free:

      gcloud compute instances stop fincopilot --zone=us-central1-a

      gcloud compute instances set-machine-type fincopilot --zone=us-central1-a --machine-type=e2-small

      gcloud compute instances start fincopilot --zone=us-central1-a

- **us-central1 is in Iowa** — considerably further from Indian users than the
  Mumbai EC2 box. The free tier does not extend to `asia-south1`; this is the
  price of free.

- **State lives on the boot disk.** `data/reports.db` and `data/usage.db`
  survive restarts and redeploys, but not disk deletion. If the report cache
  matters, snapshot occasionally:

      gcloud compute disks snapshot fincopilot --zone=us-central1-a --snapshot-names=fincopilot-backup-1

---

## Troubleshooting

| Symptom | Check |
|---|---|
| Page blank / spins forever | WebSocket. `sudo journalctl -u caddy -n 40 --no-pager` |
| 502 from Caddy | App is down. `sudo systemctl status financial-copilot` |
| Certificate never issued | DNS not propagated. `nslookup fincopilot.site` must return the GCP IP |
| Connection times out | Missing firewall rule or network tag (Steps 4–5) |
| App dies on big filings | OOM. `sudo dmesg \| grep -i oom`, then resize to e2-small |
| Actions deploy fails at SSH | `GCP_USER` wrong, or `ssh-keys` metadata missing the `user:` prefix |
| SSH key in metadata ignored | OS Login is on. It is off by default; if it was turned on, `gcloud compute instances add-metadata fincopilot --zone=us-central1-a --metadata=enable-oslogin=FALSE` |
| SEC requests rejected | `SEC_USER_AGENT` missing a real email |

Live log:

    sudo journalctl -u financial-copilot -f
