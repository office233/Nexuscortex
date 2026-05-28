# ─────────────────────────────────────────────────────────────────────
# Cursa D — Broca 2.0 extended training on Wikipedia simple + dolly + alpaca + gsm8k
# ─────────────────────────────────────────────────────────────────────
#
# Plan:
#   - 4.2× more corpus than Cursa C (1.08M lines vs ~257k)
#   - Resume from C's best checkpoint (transformer.best.nxtf)
#   - Lower peak LR (1e-4 vs 2e-4) for fine-tuning on new distribution
#   - Longer total horizon (20000 steps; C ran to 10000)
#   - Larger eval pool (per-corpus 200 lines → 1200 total)
#
# Safety:
#   - Auto-backup of transformer.nxtf, transformer.best.nxtf, optimizer.nxto
#   - Refuses to start if Cursa C process still alive (avoid GPU contention)
#   - Dry-run mode prints command without launching
#
# Usage:
#   .\scripts\run-D.ps1                  # interactive: requires confirmation
#   .\scripts\run-D.ps1 -DryRun          # print command only
#   .\scripts\run-D.ps1 -Force           # skip Cursa C alive check (DANGEROUS)
#   .\scripts\run-D.ps1 -TotalSteps 15000 # override step budget

[CmdletBinding()]
param(
    [int]    $TotalSteps      = 20000,
    [double] $PeakLR          = 1e-4,
    [double] $MinLR           = 1e-5,
    [int]    $Warmup          = 300,
    [int]    $BatchSize       = 16,
    [int]    $EvalEvery       = 500,
    [int]    $EvalLines       = 200,
    [int]    $CheckpointEvery = 500,
    [int]    $EarlyStop       = 0,
    [int]    $Seed            = 42,
    [string] $DataDir         = "./data/cortex-auto",
    [switch] $DryRun,
    [switch] $Force,
    [switch] $NoBackup
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot/..

# Force invariant culture so float formatting uses '.' as decimal point.
# Without this, ro-RO locale renders 0.0001 as "0,0001" and the Go flag
# parser rejects it, killing the run before it begins.
[System.Threading.Thread]::CurrentThread.CurrentCulture = [System.Globalization.CultureInfo]::InvariantCulture

function FmtFloat([double]$v) {
    # G15 keeps full precision for typical LRs (1e-3, 1e-4, 1e-5) without
    # the float64 round-trip noise that G17 introduces. InvariantCulture
    # guarantees '.' as the decimal separator regardless of system locale.
    return $v.ToString("G15", [System.Globalization.CultureInfo]::InvariantCulture)
}

Write-Host ""
Write-Host "═══════════════════════════════════════════════════════════════"
Write-Host "  CURSA D — Broca 2.0 extended training"
Write-Host "═══════════════════════════════════════════════════════════════"
Write-Host ""

# ── 1. Safety: check no other training is running ───────────────────
$running = Get-Process cortex-broca-train -ErrorAction SilentlyContinue
if ($running -and -not $Force) {
    Write-Host "[!] Cursa C (or another training) is STILL RUNNING:" -ForegroundColor Yellow
    $running | Select-Object Id, @{N='CPU_min';E={[math]::Round($_.CPU/60,1)}}, @{N='MemMB';E={[math]::Round($_.WorkingSet64/1MB,0)}}, StartTime | Format-Table -AutoSize
    Write-Host "[!] Refusing to start Cursa D to avoid GPU contention." -ForegroundColor Yellow
    Write-Host "    Wait for it to finish, or re-run with -Force (NOT recommended)." -ForegroundColor Yellow
    exit 1
}

# ── 2. Define corpus list (all verified 0% UNK on current tokenizer) ─
$corpora = @(
    "./data/corpus/wikipedia_simple.jsonl",
    "./data/corpus/dolly.jsonl",
    "./data/corpus/alpaca.jsonl",
    "./data/corpus/gsm8k_train.jsonl",
    "./data/corpus/reasoning.jsonl",
    "./data/corpus/general.jsonl"
)
$missing = $corpora | Where-Object { -not (Test-Path $_) }
if ($missing) {
    Write-Host "[!] Missing corpus files:" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "    $_" -ForegroundColor Red }
    exit 1
}
$corpusArg = ($corpora -join ",")

# ── 3. Pre-flight: print configuration ──────────────────────────────
Write-Host "Configuration:"
Write-Host "  Corpora:           $($corpora.Count) files"
$corpora | ForEach-Object {
    $sz = (Get-Item $_).Length / 1MB
    Write-Host ("    {0,-45} {1,8:F1} MB" -f $_, $sz)
}
Write-Host ""
Write-Host "  Total steps:       $TotalSteps"
Write-Host "  Peak LR:           $PeakLR"
Write-Host "  Min LR:            $MinLR"
Write-Host "  Warmup:            $Warmup"
Write-Host "  Batch size:        $BatchSize"
Write-Host "  Eval every:        $EvalEvery steps"
Write-Host "  Eval lines:        $EvalLines per corpus  ($($EvalLines * $corpora.Count) total)"
Write-Host "  Checkpoint every:  $CheckpointEvery steps"
Write-Host "  Early stop:        $EarlyStop  (0 = disabled)"
Write-Host "  Seed:              $Seed"
Write-Host "  Data dir:          $DataDir"
Write-Host ""

# ── 4. Backup checkpoints ───────────────────────────────────────────
$backupTag = "pre-D-$(Get-Date -Format 'yyyyMMdd-HHmmss')"
if (-not $NoBackup -and -not $DryRun) {
    Write-Host "Backing up checkpoints with tag: $backupTag"
    $toBackup = @(
        "$DataDir/transformer.nxtf",
        "$DataDir/transformer.best.nxtf",
        "$DataDir/optimizer.nxto",
        "$DataDir/training.log"
    )
    foreach ($src in $toBackup) {
        if (Test-Path $src) {
            $dst = "$src.$backupTag.bak"
            Copy-Item $src $dst -Force
            Write-Host "  [bak] $src -> $(Split-Path $dst -Leaf)" -ForegroundColor DarkGray
        }
    }
    Write-Host ""
} elseif ($NoBackup) {
    Write-Host "[!] -NoBackup specified: skipping backup step." -ForegroundColor Yellow
    Write-Host ""
}

# ── 5. Build command ────────────────────────────────────────────────
$logOut = "$DataDir/longrun-D.out"
$logErr = "$DataDir/longrun-D.err"

$peakLRStr = FmtFloat $PeakLR
$minLRStr  = FmtFloat $MinLR

$args = @(
    "--data-dir",         $DataDir,
    "--corpus",           $corpusArg,
    "--peak-lr",          $peakLRStr,
    "--min-lr",           $minLRStr,
    "--warmup",           "$Warmup",
    "--total-steps",      "$TotalSteps",
    "--batch-size",       "$BatchSize",
    "--eval-every",       "$EvalEvery",
    "--eval-lines",       "$EvalLines",
    "--checkpoint-every", "$CheckpointEvery",
    "--early-stop",       "$EarlyStop",
    "--seed",             "$Seed"
)

$cmd = ".\bin\cortex-broca-train.exe " + ($args -join " ")
Write-Host "Command:"
Write-Host "  $cmd" -ForegroundColor Cyan
Write-Host ""
Write-Host "Stdout log:  $logOut"
Write-Host "Stderr log:  $logErr"
Write-Host ""

# ── 6. Dry run exit ─────────────────────────────────────────────────
if ($DryRun) {
    Write-Host "[DryRun] Not launching. Review configuration above." -ForegroundColor Green
    exit 0
}

# ── 7. Ensure binary is built and fresh ─────────────────────────────
Write-Host "Building cortex-broca-train..."
go build -o bin/cortex-broca-train.exe ./cmd/cortex-broca-train
if ($LASTEXITCODE -ne 0) {
    Write-Host "[!] Build failed." -ForegroundColor Red
    exit 1
}
Write-Host "  [ok] binary fresh" -ForegroundColor Green
Write-Host ""

# ── 8. Confirm before launch ────────────────────────────────────────
if (-not $Force) {
    $answer = Read-Host "Launch Cursa D now? [y/N]"
    if ($answer -notmatch '^[Yy]') {
        Write-Host "Aborted by user." -ForegroundColor Yellow
        exit 0
    }
}

# ── 9. Launch (detached, logs to file) ──────────────────────────────
Write-Host ""
Write-Host "Launching Cursa D..." -ForegroundColor Green
$p = Start-Process -FilePath ".\bin\cortex-broca-train.exe" `
    -ArgumentList $args `
    -RedirectStandardOutput $logOut `
    -RedirectStandardError  $logErr `
    -WindowStyle Hidden `
    -PassThru
Start-Sleep -Seconds 3
Write-Host "  PID: $($p.Id)" -ForegroundColor Green
Write-Host ""
Write-Host "Tail the log:"
Write-Host "  Get-Content $logOut -Tail 30 -Wait"
Write-Host ""
Write-Host "Check status:"
Write-Host "  Get-Process cortex-broca-train | Format-List"
Write-Host ""
