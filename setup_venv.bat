@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo   AI Vision Upscaler Pro - Virtual Env Setup
echo ===================================================
echo.

:: 1. Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to your PATH.
    echo Please install Python 3.10+ and check "Add Python to PATH".
    pause
    exit /b 1
)

:: 2. Create virtual environment
if not exist "venv" (
    echo [*] Creating virtual environment 'venv'...
    python -m venv venv
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [SUCCESS] Virtual environment created.
) else (
    echo [*] 'venv' directory already exists. Skipping creation.
)

:: 3. Upgrade pip and install requirements
echo.
echo [*] Upgrading pip inside virtual environment...
call venv\Scripts\python.exe -m pip install --upgrade pip

echo.
echo [*] Installing dependencies from requirements.txt...
call venv\Scripts\pip.exe install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

echo.
echo ===================================================
echo   [SUCCESS] Setup complete!
echo   To run the app: venv\Scripts\python.exe upscaler_app.py
echo ===================================================
echo.
pause
