# Quick training monitor — tails the active run and shows progress.
# Auto-detects which run is active (C or D) by looking at running processes
# and log mtimes. Use Ctrl+C to stop watching.
#
# Usage:
#   .\scripts\watch-training.ps1
#   .\scripts\watch-training.ps1 -Run D       # force a specific log
#   .\scripts\watch-training.ps1 -Lines 50    # show more history

[CmdletBinding()]
param(
    [string] $Run     = "auto",
    [int]    $Lines   = 30,
    [string] $DataDir = "./data/cortex-auto"
)

Set-Location $PSScriptRoot/..

# ── 1. Pick log file ────────────────────────────────────────────────
$logFile = $null
if ($Run -eq "auto") {
    # Newest .out file wins.
    $candidates = Get-ChildItem "$DataDir/longrun-*.out" -ErrorAction SilentlyContinue |
                  Sort-Object LastWriteTime -Descending
    if (-not $candidates) {
        Write-Host "[!] No longrun-*.out files in $DataDir" -ForegroundColor Red
        exit 1
    }
    $logFile = $candidates[0].FullName
    Write-Host "[auto] tailing: $logFile" -ForegroundColor DarkGray
} else {
    $logFile = "$DataDir/longrun-$Run.out"
    if (-not (Test-Path $logFile)) {
        Write-Host "[!] Log not found: $logFile" -ForegroundColor Red
        exit 1
    }
}

# ── 2. Show process status ──────────────────────────────────────────
$proc = Get-Process cortex-broca-train -ErrorAction SilentlyContinue
if ($proc) {
    Write-Host "Process alive:" -ForegroundColor Green
    $proc | Select-Object Id,
        @{N='CPU_min';E={[math]::Round($_.CPU/60,1)}},
        @{N='MemMB';E={[math]::Round($_.WorkingSet64/1MB,0)}},
        StartTime | Format-Table -AutoSize
} else {
    Write-Host "[!] No cortex-broca-train process running." -ForegroundColor Yellow
}

# ── 3. Tail log ─────────────────────────────────────────────────────
Write-Host "── Last $Lines lines (Ctrl+C to stop watching) ──"
Get-Content $logFile -Tail $Lines -Wait
