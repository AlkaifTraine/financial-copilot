<#
    Writes the runtime secrets to the VM, starts the service, and VERIFIES the
    result.

    Three things here are deliberate, each because the obvious version failed:

    1. It does not use `gcloud compute ssh`. On Windows that shells out to
       plink, whose "Store key in cache?" prompt reads from the same stdin the
       secrets are piped on - so the prompt swallows the payload and the remote
       `cat` writes an empty file. Native OpenSSH over an explicit IAP tunnel
       has no such prompt.

    2. It transfers base64, not text. Piping a PowerShell string to a native
       command prepends a UTF-8 BOM, and systemd then reads the first key as
       "﻿OPENAI_API_KEY" and silently ignores it. base64 -d -i discards the
       BOM bytes as non-alphabet garbage.

    3. It verifies what landed. The first version of this script reported
       success on the SSH exit code, which was zero while the file was empty -
       and the service still started, and still answered HTTP 200, because
       Streamlit boots fine without a key and only fails when a visitor uses it.
       A deployment that looks green and breaks on a recruiter's first click is
       the worst possible outcome, so the checks below are the point of the
       script, not decoration.

    Usage:
        powershell -ExecutionPolicy Bypass -File ".\deploy\gcp\set-secrets.ps1"
#>

[CmdletBinding()]
param(
    [string]$InstanceName = "fincopilot",
    [string]$Zone         = "us-central1-a",
    [string]$ProjectId    = "fincopilot-prod",
    [string]$RemoteUser   = "VICTUS",
    [int]$TunnelPort      = 2222
)

$ErrorActionPreference = "Continue"

function Say  ($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Ok   ($m) { Write-Host "    $m"  -ForegroundColor Green }
function Warn ($m) { Write-Host "    $m"  -ForegroundColor Yellow }
function Die  ($m) { Write-Host "`nFAILED: $m" -ForegroundColor Red; exit 1 }

$KeyFile = "$env:USERPROFILE\.ssh\google_compute_engine"
if (-not (Test-Path $KeyFile)) {
    Die "No SSH key at $KeyFile. Run this once to generate it:`n  gcloud compute ssh $InstanceName --zone=$Zone --tunnel-through-iap --command=true"
}

$gcloud = (Get-Command gcloud -ErrorAction SilentlyContinue).Source
if (-not $gcloud) {
    $fallback = "$env:LOCALAPPDATA\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
    if (Test-Path $fallback) { $gcloud = $fallback } else { Die "gcloud not found." }
}

# Test-NetConnection also does DNS and ICMP work and takes ~20 seconds to tell
# you about a local port, with a progress bar over the prompt. A raw TCP connect
# answers the only question being asked, immediately.
function PortOpen {
    param([int]$Port)
    $c = New-Object Net.Sockets.TcpClient
    try {
        $r = $c.BeginConnect("127.0.0.1", $Port, $null, $null)
        if (-not $r.AsyncWaitHandle.WaitOne(1500, $false)) { return $false }
        $c.EndConnect($r)
        return $true
    } catch {
        return $false
    } finally {
        $c.Close()
    }
}

# Run a command on the VM, optionally feeding it stdin. Returns the output.
function Remote {
    param([string]$Cmd, [string]$StdIn)
    if ($PSBoundParameters.ContainsKey("StdIn")) {
        return ($StdIn | ssh -p $TunnelPort -i $KeyFile -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 "$RemoteUser@127.0.0.1" $Cmd)
    }
    return (ssh -p $TunnelPort -i $KeyFile -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 "$RemoteUser@127.0.0.1" $Cmd)
}

# ---------------------------------------------------------------------------
Say "IAP tunnel on 127.0.0.1:$TunnelPort"
# Your network blocks outbound 22, so SSH goes through Google's Identity-Aware
# Proxy over 443 instead. The tunnel is a local listener that forwards into it.
$listening = PortOpen -Port $TunnelPort
if ($listening) {
    Ok "Already open."
    $startedTunnel = $false
} else {
    Write-Host "    Opening it..."
    Start-Process -FilePath $gcloud `
        -ArgumentList @("compute","start-iap-tunnel",$InstanceName,"22","--local-host-port=localhost:$TunnelPort","--zone=$Zone","--project=$ProjectId") `
        -WindowStyle Hidden
    $startedTunnel = $true
    $up = $false
    foreach ($i in 1..25) {
        Start-Sleep -Seconds 2
        if (PortOpen -Port $TunnelPort) { $up = $true; break }
    }
    if (-not $up) { Die "The tunnel did not come up on port $TunnelPort." }
    Ok "Open."
}

Say "Checking the connection"
$who = Remote "whoami"
if ($who -ne $RemoteUser) { Die "Expected to log in as '$RemoteUser' but got '$who'." }
Ok "Logged in as $who"

# ---------------------------------------------------------------------------
Say "OpenAI API key"
Write-Host "    Paste it and press Enter. NOTHING WILL APPEAR as you type -"
Write-Host "    that is deliberate. Paste once; do not paste repeatedly."
Write-Host "    Get one at https://platform.openai.com/api-keys"
$secure = Read-Host "    OPENAI_API_KEY" -AsSecureString
$bstr   = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
$key    = [Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)
[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)

if (-not $key) { Die "No key entered." }
$key = $key.Trim()
# Shape check only; whether it works is OpenAI's call on the first request. But
# a truncated paste caught here saves a baffling failure much later.
if ($key -notmatch '^sk-[A-Za-z0-9_\-]{20,}$') {
    Die "That does not look like an OpenAI key (expected sk- then 20+ characters). Nothing was sent."
}
$keyLen = $key.Length
Ok "Accepted: $keyLen characters, starts sk-$($key.Substring(3,4))..."

Say "SEC contact email"
Write-Host "    The SEC rejects requests without a real contact address and"
Write-Host "    IP-bans repeat offenders. Use an address you own."
$email = (Read-Host "    Email").Trim()
if ($email -notmatch '^[^@\s]+@[^@\s]+\.[^@\s]+$') { Die "'$email' is not a valid email address. Nothing was sent." }
Ok "Will send as: Financial Copilot $email"

Say "Gemini API key (optional - press Enter to skip)"
Write-Host "    Cross-provider fallback, so an OpenAI incident does not take"
Write-Host "    the demo down mid-report. https://aistudio.google.com"
$gemini = (Read-Host "    GEMINI_API_KEY").Trim()

# LF line endings: systemd's EnvironmentFile parser keeps a trailing CR as part
# of the value, so a CRLF file yields a key with an invisible CR on the end.
$envBody = (@(
    "OPENAI_API_KEY=$key",
    "SEC_USER_AGENT=Financial Copilot $email",
    "GEMINI_API_KEY=$gemini",
    "DEMO_MODE=1",
    "MAX_USD_PER_PROCESS=5",
    "REQUIRE_ACCESS=0",
    "ANALYTICS_ENABLED=1"
) -join "`n") + "`n"

$remotePath = "/home/$RemoteUser/financial-copilot.env"
$b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($envBody))

# ---------------------------------------------------------------------------
Say "Writing $remotePath"
# umask first, so the file is never briefly world-readable between create and chmod.
$null = Remote "umask 077 && base64 -d -i > $remotePath && chmod 600 $remotePath" -StdIn $b64

# Scrub the plaintext from this process before the verification round-trip.
$key = $null; $envBody = $null; $b64 = $null; [GC]::Collect()

# --- verification: the whole reason this script exists ----------------------
Say "Verifying what actually landed"

$size = [int](Remote "wc -c < $remotePath").Trim()
if ($size -lt 100) { Die "$remotePath is $size bytes - the write did not land. The service was NOT restarted, so nothing changed." }
Ok "$size bytes"

$names = Remote "cut -d= -f1 $remotePath"
foreach ($need in @("OPENAI_API_KEY","SEC_USER_AGENT","DEMO_MODE","REQUIRE_ACCESS")) {
    if ($names -notcontains $need) { Die "$need is missing from the file. The service was NOT restarted." }
}
Ok "All expected keys present"

# Length is compared rather than the value, so the key is never echoed back.
$remoteLen = [int](Remote "grep '^OPENAI_API_KEY=' $remotePath | head -1 | cut -d= -f2- | tr -d '\n' | wc -c").Trim()
if ($remoteLen -ne $keyLen) { Die "The key arrived as $remoteLen characters but you typed $keyLen - it was corrupted in transit. The service was NOT restarted." }
Ok "Key intact: $remoteLen characters, matches what you typed"

$perm = (Remote "stat -c %a $remotePath").Trim()
if ($perm -ne "600") { Warn "Permissions are $perm, expected 600" } else { Ok "Mode 600" }

# ---------------------------------------------------------------------------
Say "Restarting the service"
$null = Remote "sudo systemctl restart financial-copilot"
Start-Sleep -Seconds 12

$state = (Remote "systemctl is-active financial-copilot").Trim()
if ($state -ne "active") {
    Write-Host ""
    Remote "journalctl -u financial-copilot -n 30 --no-pager"
    Die "The service is '$state', not 'active'. The log is above."
}
Ok "Service active"

# Proves Streamlit is serving. It does NOT prove the key works - that is what
# the verification above is for.
$probe = (Remote "curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8501/").Trim()
if ($probe -ne "200") { Die "The app answered HTTP $probe on 127.0.0.1:8501, expected 200." }
Ok "HTTP 200 on 127.0.0.1:8501"

# Confirm the running process actually inherited the key, rather than trusting
# that the file on disk is what systemd loaded.
$inEnv = (Remote "sudo tr '\0' '\n' < /proc/`$(systemctl show -p MainPID --value financial-copilot)/environ | grep -c '^OPENAI_API_KEY=sk-'").Trim()
if ($inEnv -eq "0") { Die "The service is running but its process has no OPENAI_API_KEY. systemd did not load the file." }
Ok "The running process has the key in its environment"

# ---------------------------------------------------------------------------
Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " Secrets written and the app is running with them." -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host " Still private - bound to 127.0.0.1, nothing on 80/443 yet."
Write-Host ""
Write-Host " Next: point fincopilot.site at the VM's static IP, wait for DNS,"
Write-Host " then on the VM:"
Write-Host "   bash deploy/gcp/setup_https.sh fincopilot.site www.fincopilot.site"
Write-Host ""
if ($startedTunnel) {
    Write-Host " A background IAP tunnel is left open on port $TunnelPort. Harmless;"
    Write-Host " it closes when you log out, or kill the gcloud process."
    Write-Host ""
}
