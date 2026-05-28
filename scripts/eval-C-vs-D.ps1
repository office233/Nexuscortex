# eval-C-vs-D.ps1 — run broca-eval against C's best vs D's best.
#
# Workflow (run AFTER both Cursa C and Cursa D have finished):
#   1. Locate the most recent C-era best checkpoint (a .bak made by run-D.ps1).
#   2. Locate the current best (which is D's, after run-D.ps1 trained).
#   3. Run broca-eval --compare and print A vs B delta.
#
# Optional: pass -WithCoT to also run a second comparison with
# Self-Consistency voting enabled (slower, but shows CoT's marginal gain).

[CmdletBinding()]
param(
    [string] $DataDir   = "./data/cortex-auto",
    [int]    $MaxTokens = 40,
    [double] $Temp      = 0.5,
    [int]    $TopK      = 40,
    [int]    $Seed      = 1,
    [switch] $WithCoT,
    [int]    $CoTSamples = 3
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot/..

# Locale-safe float formatting (same as run-D.ps1).
[System.Threading.Thread]::CurrentThread.CurrentCulture = [System.Globalization.CultureInfo]::InvariantCulture
function FmtFloat([double]$v) {
    return $v.ToString("G15", [System.Globalization.CultureInfo]::InvariantCulture)
}

# ── 1. Find C-era best snapshot (created by run-D.ps1 backup step) ──
$cBest = Get-ChildItem "$DataDir/transformer.best.nxtf.pre-D-*.bak" -ErrorAction SilentlyContinue |
         Sort-Object LastWriteTime -Descending |
         Select-Object -First 1
if (-not $cBest) {
    Write-Host "[!] No 'transformer.best.nxtf.pre-D-*.bak' found." -ForegroundColor Red
    Write-Host "    run-D.ps1 must have been launched (it makes the backup) before this script can compare C vs D." -ForegroundColor Red
    exit 1
}

# ── 2. Current best = D's best ──────────────────────────────────────
$dBest = "$DataDir/transformer.best.nxtf"
if (-not (Test-Path $dBest)) {
    Write-Host "[!] $dBest not found." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "============================================================"
Write-Host "  Cursa C vs Cursa D — broca-eval comparison"
Write-Host "============================================================"
Write-Host "  A (Cursa C best):  $($cBest.Name)" -ForegroundColor Cyan
Write-Host "  B (Cursa D best):  $(Split-Path $dBest -Leaf)"          -ForegroundColor Cyan
Write-Host ""

# ── 3. Build the eval binary fresh ──────────────────────────────────
Write-Host "Building broca-eval..."
go build -o bin/broca-eval.exe ./cmd/broca-eval
if ($LASTEXITCODE -ne 0) { Write-Host "Build failed." -ForegroundColor Red; exit 1 }

# ── 4. Greedy comparison run ────────────────────────────────────────
$args = @(
    "--data-dir",   $DataDir,
    "--compare",
    "--max-tokens", "$MaxTokens",
    "--temp",       (FmtFloat $Temp),
    "--top-k",      "$TopK",
    "--seed",       "$Seed",
    $cBest.FullName,
    $dBest
)
Write-Host ""
Write-Host "Pass 1: greedy / no CoT" -ForegroundColor Yellow
& ".\bin\broca-eval.exe" @args
if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] broca-eval --compare exited with code $LASTEXITCODE" -ForegroundColor Red
    exit $LASTEXITCODE
}

# ── 5. Optional CoT pass ────────────────────────────────────────────
if ($WithCoT) {
    $cotArgs = $args + @("--cot", "--cot-samples", "$CoTSamples")
    Write-Host ""
    Write-Host "Pass 2: Self-Consistency ($CoTSamples samples)" -ForegroundColor Yellow
    & ".\bin\broca-eval.exe" @cotArgs
}

Write-Host ""
Write-Host "Done. JSON reports in $DataDir/evals/" -ForegroundColor Green
