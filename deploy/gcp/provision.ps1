<#
    Provisions the GCP side of the Financial Copilot deployment.

    Replaces Steps 2-5 of GCP_DEPLOY.md: project, billing, API, static IP, VM,
    firewall. It does NOT touch the VM's contents - that is setup.sh, over SSH.

    Idempotent: every step checks for what it is about to create and skips it if
    it is already there, so re-running after a failure is safe.

    Usage (PowerShell, from anywhere):

        .\provision.ps1
        .\provision.ps1 -ProjectId fincopilot-prod-2 -Zone us-east1-b

    Prerequisites: gcloud installed (winget install Google.CloudSDK) and
    `gcloud auth login` already done.
#>

[CmdletBinding()]
param(
    [string]$ProjectId   = "fincopilot-prod",
    [string]$InstanceName = "fincopilot",
    [string]$Zone        = "us-central1-a",
    [string]$AddressName = "fincopilot-ip"
)

# gcloud writes progress to stderr as a matter of course. Letting that abort the
# script would make every successful step look like a failure.
$ErrorActionPreference = "Continue"

# The region is the zone minus its trailing letter; a regional static IP must
# match the zone's region or the VM cannot attach it.
$Region = $Zone -replace '-[a-z]$', ''

function Say   ($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Ok    ($m) { Write-Host "    $m"  -ForegroundColor Green }
function Warn  ($m) { Write-Host "    $m"  -ForegroundColor Yellow }
function Die   ($m) { Write-Host "`nFAILED: $m" -ForegroundColor Red; exit 1 }

# Run gcloud and return its stdout, or $null on a non-zero exit.
function RunGcloud {
    $out = & gcloud @args
    if ($LASTEXITCODE -ne 0) { return $null }
    return $out
}

# ---------------------------------------------------------------------------
Say "Checking gcloud"
$gcloud = Get-Command gcloud -ErrorAction SilentlyContinue
if (-not $gcloud) {
    Die "gcloud not found. Install it with:  winget install Google.CloudSDK`n        Then close and reopen PowerShell, and run:  gcloud auth login"
}
$account = RunGcloud auth list --filter=status:ACTIVE --format="value(account)"
if (-not $account) { Die "Not signed in. Run:  gcloud auth login" }
Ok "Signed in as $account"

# ---------------------------------------------------------------------------
Say "Project: $ProjectId"
$exists = RunGcloud projects describe $ProjectId --format="value(projectId)"
if ($exists) {
    Ok "Already exists."
} else {
    # Output is deliberately not swallowed: gcloud's own error is the useful
    # diagnostic here, and guessing at the cause sends people down the wrong path.
    & gcloud projects create $ProjectId --name="Financial Copilot"
    if ($LASTEXITCODE -ne 0) {
        Die @"
Could not create '$ProjectId'. gcloud's error is printed above. The two causes
that actually come up:

  * 'Callers must accept Terms of Service'
    This Google account has never completed Cloud onboarding. Open
    https://console.cloud.google.com, sign in as the account above, accept the
    Terms of Service and activate the free trial. Then re-run this script.

  * 'already exists' / permission denied on create
    Project IDs are globally unique and someone has taken this one. Re-run as:
      .\deploy\gcp\provision.ps1 -ProjectId $ProjectId-2
"@
    }
    Ok "Created."
}
& gcloud config set project $ProjectId | Out-Null

# ---------------------------------------------------------------------------
Say "Billing"
# A project with no billing account cannot create a VM, and the error it gives
# when you try is unhelpfully generic. Resolve it here instead.
$linked = RunGcloud billing projects describe $ProjectId --format="value(billingEnabled)"
if ($linked -eq "True") {
    Ok "Already linked."
} else {
    $accounts = @(RunGcloud billing accounts list --filter=open=true --format="value(name,displayName)")
    if (-not $accounts -or $accounts.Count -eq 0) {
        Die "No open billing account. Create one at https://console.cloud.google.com/billing then re-run."
    }
    if ($accounts.Count -eq 1) {
        $billingId = ($accounts[0] -split "\s+")[0]
        Ok "Using billing account $billingId"
    } else {
        Write-Host "    Several billing accounts:"
        for ($i = 0; $i -lt $accounts.Count; $i++) { Write-Host "      [$i] $($accounts[$i])" }
        $pick = Read-Host "    Which one? (number)"
        $billingId = ($accounts[[int]$pick] -split "\s+")[0]
    }
    & gcloud billing projects link $ProjectId --billing-account=$billingId | Out-Null
    if ($LASTEXITCODE -ne 0) { Die "Could not link billing." }
    Ok "Linked."
}

# ---------------------------------------------------------------------------
Say "Enabling Compute Engine API (can take ~30s)"
& gcloud services enable compute.googleapis.com --project=$ProjectId | Out-Null
if ($LASTEXITCODE -ne 0) { Die "Could not enable compute.googleapis.com" }
Ok "Enabled."

# ---------------------------------------------------------------------------
Say "Static IP: $AddressName ($Region)"
# Reserved BEFORE the VM, so the address survives a VM restart. An ephemeral IP
# changes on every stop/start, which would silently break fincopilot.site.
$ip = RunGcloud compute addresses describe $AddressName --region=$Region --format="value(address)"
if ($ip) {
    Ok "Already reserved: $ip"
} else {
    & gcloud compute addresses create $AddressName --region=$Region --project=$ProjectId | Out-Null
    if ($LASTEXITCODE -ne 0) { Die "Could not reserve the static IP." }
    $ip = RunGcloud compute addresses describe $AddressName --region=$Region --format="value(address)"
    Ok "Reserved: $ip"
}

# ---------------------------------------------------------------------------
Say "Firewall"
# The default VPC permits SSH but not HTTP/HTTPS. These rules attach to the
# network tags the VM is given below.
foreach ($r in @(@{n="allow-http"; p="tcp:80"; t="http-server"},
                 @{n="allow-https"; p="tcp:443"; t="https-server"})) {
    $have = RunGcloud compute firewall-rules describe $r.n --format="value(name)"
    if ($have) {
        Ok "$($r.n) already present."
    } else {
        & gcloud compute firewall-rules create $r.n --allow=$($r.p) --target-tags=$($r.t) --direction=INGRESS --project=$ProjectId | Out-Null
        if ($LASTEXITCODE -eq 0) { Ok "$($r.n) created." } else { Warn "$($r.n) not created - check manually." }
    }
}

# ---------------------------------------------------------------------------
Say "VM: $InstanceName ($Zone)"
$have = RunGcloud compute instances describe $InstanceName --zone=$Zone --format="value(name)"
if ($have) {
    Ok "Already exists - leaving it alone."
} else {
    Write-Host ""
    Write-Host "    About to create:" -ForegroundColor Yellow
    Write-Host "      e2-micro, 30GB pd-standard, Ubuntu 24.04, $Zone"
    Write-Host "      The VM and disk are inside GCP's free tier."
    Write-Host "      The external IPv4 address is NOT - about `$3.50/month," -ForegroundColor Yellow
    Write-Host "      covered by the `$300 / 90-day credit on a new account." -ForegroundColor Yellow
    Write-Host ""
    $go = Read-Host "    Create it? (y/N)"
    if ($go -ne "y") { Die "Stopped at your request. Nothing was created in this step." }

    # pd-standard is deliberate: the console defaults to pd-balanced, which is
    # not in the free tier and quietly costs ~$3/month on its own.
    & gcloud compute instances create $InstanceName `
        --project=$ProjectId `
        --zone=$Zone `
        --machine-type=e2-micro `
        --image-family=ubuntu-2404-lts-amd64 `
        --image-project=ubuntu-os-cloud `
        --boot-disk-size=30GB `
        --boot-disk-type=pd-standard `
        --address=$AddressName `
        --tags=http-server,https-server | Out-Null

    if ($LASTEXITCODE -ne 0) { Die "Could not create the VM." }
    Ok "Created."
}

# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " Provisioned." -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host " Static IP:  $ip"
Write-Host " Project:    $ProjectId"
Write-Host " Zone:       $Zone"
Write-Host ""
Write-Host " NEXT - connect (first run uploads an SSH key):"
Write-Host "   gcloud compute ssh $InstanceName --zone=$Zone"
Write-Host ""
Write-Host " THEN - on the VM:"
Write-Host "   whoami                                  # note this for GitHub secrets"
Write-Host "   git clone https://github.com/AlkaifTraine/financial-copilot.git"
Write-Host "   cd financial-copilot && bash deploy/gcp/setup.sh"
Write-Host ""
Write-Host " AND - at your registrar, repoint the domain:"
Write-Host "   A      @      $ip"
Write-Host "   CNAME  www    fincopilot.site"
Write-Host "   (delete the old A record for 13.205.11.254)"
Write-Host ""
Write-Host " Verify DNS from here before setting up HTTPS:"
Write-Host "   nslookup fincopilot.site"
Write-Host ""
