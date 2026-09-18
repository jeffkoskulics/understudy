# Understudy installer for Windows (PowerShell 5.1+).
#
# Run it from a clone:
#     .\install.ps1
#
# or straight from the web, which clones into .\understudy first:
#     iwr -useb https://raw.githubusercontent.com/jeffkoskulics/understudy/main/install.ps1 | iex

$ErrorActionPreference = 'Stop'

function Find-Python {
    # The Microsoft Store stub named python.exe exits 9009 and opens the Store,
    # so a plain Get-Command is not enough -- each candidate is actually run.
    foreach ($candidate in @(@('py', '-3'), @('python'), @('python3'))) {
        $exe = $candidate[0]
        $prefix = @($candidate | Select-Object -Skip 1)
        if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) { continue }
        $v = $null
        try {
            $v = & $exe @prefix -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        } catch { continue }
        if ($LASTEXITCODE -ne 0 -or -not $v) { continue }
        try { $parsed = [version]([string]$v).Trim() } catch { continue }
        if ($parsed -ge [version]'3.9') { return ,$candidate }
    }
    return $null
}

$python = Find-Python
if (-not $python) {
    Write-Host "Python 3.9 or newer was not found." -ForegroundColor Red
    Write-Host "Install it from https://www.python.org/downloads/windows/ and tick"
    Write-Host '"Add python.exe to PATH" in the installer, then run this script again.'
    exit 1
}

# Working from a clone if understudy\__main__.py is beside this script; if the
# script was piped in from the web there is no clone yet, so make one.
$root = $PSScriptRoot
if (-not $root -or -not (Test-Path (Join-Path $root 'understudy\__main__.py'))) {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Write-Host "git was not found. Install it from https://git-scm.com/download/win" -ForegroundColor Red
        exit 1
    }
    $root = Join-Path (Get-Location) 'understudy'
    if (-not (Test-Path $root)) {
        Write-Host "Cloning understudy into $root"
        git clone https://github.com/jeffkoskulics/understudy $root
        if ($LASTEXITCODE -ne 0) { exit 1 }
    } else {
        Write-Host "Using existing clone at $root"
    }
}

Set-Location $root

$venv = Join-Path $root '.venv'
$venvPython = Join-Path $venv 'Scripts\python.exe'
if (-not (Test-Path $venvPython)) {
    Write-Host "Creating virtual environment in $venv"
    $pythonExe = $python[0]
    $pythonArgs = @($python | Select-Object -Skip 1)
    & $pythonExe @pythonArgs -m venv $venv
    if ($LASTEXITCODE -ne 0) { exit 1 }
}

Write-Host "Installing dependencies"
& $venvPython -m pip install --upgrade pip -q
if ($LASTEXITCODE -ne 0) { exit 1 }
& $venvPython -m pip install -r (Join-Path $root 'requirements.txt') -q
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host ""
Write-Host "Understudy is installed." -ForegroundColor Green
Write-Host "Record a workflow with:"
Write-Host "    $root\bin\understudy.cmd record --out $env:USERPROFILE\Recordings"
Write-Host ""
Write-Host "Note: OCR is macOS-only for now (the Windows OCR helper is not written"
Write-Host "yet), so 'record' works but 'pack' cannot read text off the frames."
