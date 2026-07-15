# set_github_secrets.ps1
# Reads your .env file and pushes all secrets to GitHub repo.
# Run from the project root: .\scripts\set_github_secrets.ps1

$repo = "keerthana-nc/audible-frames-azure"
$envFile = ".env"

Write-Host "Reading .env file..." -ForegroundColor Cyan

# Parse .env file into a hashtable
$envVars = @{}
Get-Content $envFile | ForEach-Object {
    $line = $_.Trim()
    # Skip comments and empty lines
    if ($line -and -not $line.StartsWith("#")) {
        $parts = $line -split "=", 2
        if ($parts.Length -eq 2) {
            $key = $parts[0].Trim()
            $value = $parts[1].Trim()
            $envVars[$key] = $value
        }
    }
}

# Set secrets from .env
$envSecrets = @(
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_DEPLOYMENT_NAME",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_VISION_ENDPOINT",
    "AZURE_VISION_KEY",
    "AZURE_SPEECH_KEY",
    "AZURE_SPEECH_REGION",
    "AZURE_CONTENT_SAFETY_ENDPOINT",
    "AZURE_CONTENT_SAFETY_KEY",
    "APPLICATIONINSIGHTS_CONNECTION_STRING"
)

foreach ($key in $envSecrets) {
    if ($envVars.ContainsKey($key)) {
        Write-Host "Setting $key..." -ForegroundColor Yellow
        $envVars[$key] | gh secret set $key --repo $repo
    } else {
        Write-Host "WARNING: $key not found in .env -- skipping" -ForegroundColor Red
    }
}

# Static infrastructure secrets
Write-Host "Setting infrastructure secrets..." -ForegroundColor Cyan
"audibleframesacr.azurecr.io"              | gh secret set ACR_LOGIN_SERVER          --repo $repo
"Audible-Frames-azure"                      | gh secret set AZURE_RG                  --repo $repo
"audible-frames-app"                        | gh secret set CONTAINER_APP_NAME        --repo $repo
"managedEnvironment-AudibleFramesaz-8cfe"  | gh secret set CONTAINER_APP_ENV_NAME   --repo $repo

# Prompt for secrets that aren't in .env
Write-Host "`nNow enter the 3 remaining secrets:" -ForegroundColor Cyan

Write-Host "Paste AZURE_CREDENTIALS JSON (press Enter twice when done):" -ForegroundColor Yellow
$lines = @()
while ($true) {
    $line = Read-Host
    if ($line -eq "") { break }
    $lines += $line
}
$azureCreds = $lines -join "`n"
$azureCreds | gh secret set AZURE_CREDENTIALS --repo $repo

$acrUser = Read-Host "Enter ACR_USERNAME"
$acrUser | gh secret set ACR_USERNAME --repo $repo

$acrPass = Read-Host "Enter ACR_PASSWORD"
$acrPass | gh secret set ACR_PASSWORD --repo $repo

Write-Host "`nAll secrets set! Check: https://github.com/$repo/settings/secrets/actions" -ForegroundColor Green
