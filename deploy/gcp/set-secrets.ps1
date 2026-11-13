<#
    Writes the runtime secrets to the VM and starts the service.

    The key is typed at a prompt, never passed as an argument and never written
    to a file on this machine. It goes to the VM over the SSH tunnel on stdin,
    so it does not appear in PowerShell history, in the VM's shell history, or
    in the remote process list.

    Usage:
        powershell -ExecutionPolicy Bypass -File ".\deploy\gcp\set-secrets.ps1"
#>

[CmdletBinding()]
param(
    [string]$InstanceName = "fincopilot",
    [string]$Zone         = "us-central1-a",
    [string]$ProjectId    = "fincopilot-prod",
    [string]$RemoteUser   = "VICTUS"
)

$ErrorActionPreference = "Continue"

function Say  ($m) { Write-Host "`n==> $m" -ForegroundColor Cyan }
function Ok   ($m) { Write-Host "    $m"  -ForegroundColor Green }
function Die  ($m) { Write-Host "`nFAILED: $m" -ForegroundColor Red; exit 1 }

# gcloud may not be on PATH in a shell that predates its install.
$gcloud = (Get-Command gcloud -ErrorAction SilentlyContinue).Source
if (-not $gcloud) {
    $fallback = "$env:LOCALAPPDATA\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd"
    if (Test-Path $fallback) { $gcloud = $fallback } else { Die "gcloud not found." }
}

Say "OpenAI API key"
Write-Host "    Paste it and press Enter. It will not be shown as you type."
Write-Host "    Get one at https://platform.openai.com/api-keys"
$secure = Read-Host "    OPENAI_API_KEY" -AsSecureString
$bstr   = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
$key    = [Runtime.InteropServices.Marshal]::PtrToStringAuto($bstr)
[Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)

if (-not $key) { Die "No key entered." }
$key = $key.Trim()
# Shape check only. Whether it actually works is decided by OpenAI on the first
# call - but catching a truncated paste here saves a confusing 500 later.
if ($key -notmatch '^sk-[A-Za-z0-9_\-]{20,}$') {
    Die "That does not look like an OpenAI key (expected sk- followed by 20+ characters). Nothing was sent."
}
Ok "Key accepted ($($key.Length) characters, starts sk-$($key.Substring(3,4))...)"

Say "SEC contact email"
Write-Host "    The SEC rejects requests without a real contact address and"
Write-Host "    IP-bans repeat offenders. This must be an address you own."
$email = (Read-Host "    Email").Trim()
if ($email -notmatch '^[^@\s]+@[^@\s]+\.[^@\s]+$') { Die "'$email' is not a valid email address. Nothing was sent." }
Ok "Will send as: Financial Copilot $email"

Say "Gemini API key (optional - press Enter to skip)"
Write-Host "    Enables cross-provider fallback, so an OpenAI incident does not"
Write-Host "    take the demo down mid-report. https://aistudio.google.com"
$gemini = (Read-Host "    GEMINI_API_KEY").Trim()

# Built as one document and piped to the VM, rather than sed-ing line by line:
# a partial write here leaves a service that starts and then fails on a missing
# key, which is harder to diagnose than a file that is simply correct or absent.
$envBody = @"
OPENAI_API_KEY=$key
SEC_USER_AGENT=Financial Copilot $email
GEMINI_API_KEY=$gemini
DEMO_MODE=1
MAX_USD_PER_PROCESS=5
REQUIRE_ACCESS=0
ANALYTICS_ENABLED=1
"@

$remotePath = "/home/$RemoteUser/financial-copilot.env"

Say "Writing to the VM"
# umask 077 so the file is never briefly world-readable between create and chmod.
$remoteCmd = "umask 077 && cat > $remotePath && chmod 600 $remotePath && echo WROTE && wc -l < $remotePath"
$envBody | & $gcloud compute ssh $InstanceName --zone=$Zone --project=$ProjectId --tunnel-through-iap --quiet --command=$remoteCmd
if ($LASTEXITCODE -ne 0) { Die "Could not write the secrets file. The service was not restarted, so nothing changed." }

# Scrub the plaintext from this process before doing anything else.
$key = $null; $envBody = $null; [GC]::Collect()
Ok "Written, mode 600."

Say "Starting the service"
& $gcloud compute ssh $InstanceName --zone=$Zone --project=$ProjectId --tunnel-through-iap --quiet --command="sudo systemctl restart financial-copilot && sleep 10 && systemctl is-active financial-copilot && echo '--- local probe ---' && curl -s -o /dev/null -w 'HTTP %{http_code}\n' http://127.0.0.1:8501/"

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host " If that printed 'active' and 'HTTP 200', the app is running." -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host " It is still private - bound to 127.0.0.1 with nothing on 80/443."
Write-Host " Next: point fincopilot.site at 34.28.30.75, wait for DNS, then run"
Write-Host " setup_https.sh on the VM to get the certificate."
Write-Host ""
Write-Host " If it printed 'activating' or 'failed', read the log with:"
Write-Host "   gcloud compute ssh fincopilot --zone=us-central1-a --tunnel-through-iap --command=""journalctl -u financial-copilot -n 40 --no-pager"""
Write-Host ""
