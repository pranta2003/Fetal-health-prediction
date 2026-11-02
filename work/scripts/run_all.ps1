param (
    [switch]$Eval,
    [switch]$Train,
    [switch]$Slow
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkDir = Split-Path -Parent $ScriptDir
Set-Location $WorkDir

Write-Host "=========================================================="
Write-Host "run_all.ps1: Automated test and pipeline runner"
Write-Host "Working directory: $WorkDir"
Write-Host "Flags: Eval=$Eval | Train=$Train | Slow=$Slow"
Write-Host "=========================================================="

# Check and activate virtual environment if present
$VenvPy = "$WorkDir\.venv\Scripts\python.exe"
$VenvPytest = "$WorkDir\.venv\Scripts\pytest.exe"

if (-not (Test-Path $VenvPytest)) {
    # Check parent directory or fetal-health-multimodal venv
    if (Test-Path "C:\fetal-health-multimodal\.venv\Scripts\pytest.exe") {
        $VenvPy = "C:\fetal-health-multimodal\.venv\Scripts\python.exe"
        $VenvPytest = "C:\fetal-health-multimodal\.venv\Scripts\pytest.exe"
    } else {
        $VenvPy = "python"
        $VenvPytest = "pytest"
    }
}

Write-Host "`n[Step 1] Running fast pytest suite..."
& $VenvPytest -v
if ($LASTEXITCODE -ne 0) {
    Write-Error "Fast pytest suite failed with exit code $LASTEXITCODE"
    exit $LASTEXITCODE
}

if ($Slow) {
    Write-Host "`n[Step 2: -Slow] Running slow regression test suite..."
    & $VenvPytest -m slow -v -s
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Slow pytest suite failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

if ($Eval) {
    Write-Host "`n[Step 3: -Eval] Running 06_eval_ac.py on existing checkpoints..."
    & $VenvPy (Join-Path $ScriptDir "06_eval_ac.py")
    if ($LASTEXITCODE -ne 0) {
        Write-Error "06_eval_ac.py failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

if ($Train) {
    Write-Host "`n[Step 4: -Train] Running 04_train_ac.py..."
    & $VenvPy (Join-Path $ScriptDir "04_train_ac.py")
    if ($LASTEXITCODE -ne 0) {
        Write-Error "04_train_ac.py failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }

    Write-Host "`n[Step 5: -Train] Running 05_train_ensemble.py..."
    & $VenvPy (Join-Path $ScriptDir "05_train_ensemble.py")
    if ($LASTEXITCODE -ne 0) {
        Write-Error "05_train_ensemble.py failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

Write-Host "`n=========================================================="
Write-Host "run_all.ps1: All requested steps completed successfully."
Write-Host "=========================================================="

