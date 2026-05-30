Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "  AI Vision Upscaler Pro - Virtual Env Setup" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check if Python is installed
try {
    $pythonVer = python --version
    Write-Host "[*] Detected Python: $pythonVer"
} catch {
    Write-Host "[ERROR] Python is not installed or not in PATH. Please install Python 3.10+." -ForegroundColor Red
    Read-Host "Press Enter to exit..."
    exit 1
}

# 2. Create virtual environment
if (-not (Test-Path "venv")) {
    Write-Host "[*] Creating virtual environment 'venv'..."
    python -m venv venv
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[ERROR] Failed to create virtual environment." -ForegroundColor Red
        Read-Host "Press Enter to exit..."
        exit 1
    }
    Write-Host "[SUCCESS] Virtual environment created." -ForegroundColor Green
} else {
    Write-Host "[*] 'venv' directory already exists. Skipping creation."
}

# 3. Upgrade pip and install requirements
Write-Host ""
Write-Host "[*] Upgrading pip inside virtual environment..."
& .\venv\Scripts\python.exe -m pip install --upgrade pip

Write-Host ""
Write-Host "[*] Installing dependencies from requirements.txt..."
& .\venv\Scripts\pip.exe install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to install dependencies." -ForegroundColor Red
    Read-Host "Press Enter to exit..."
    exit 1
}

Write-Host ""
Write-Host "===================================================" -ForegroundColor Green
Write-Host "  [SUCCESS] Setup complete!" -ForegroundColor Green
Write-Host "  To run the app: .\venv\Scripts\python.exe upscaler_app.py" -ForegroundColor Green
Write-Host "===================================================" -ForegroundColor Green
Write-Host ""
Read-Host "Press Enter to exit..."
