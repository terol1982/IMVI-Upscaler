@echo off
setlocal enabledelayedexpansion

echo ===================================================
echo   AI Vision Upscaler Pro - Standalone Build Script
echo ===================================================
echo.

:: 1. Run setup_venv.bat to ensure the environment is fully up-to-date
echo [*] Step 1: Running virtual environment setup...
call setup_venv.bat --no-pause
if %errorlevel% neq 0 (
    echo [ERROR] Virtual environment setup failed. Aborting build.
    pause
    exit /b 1
)

:: 2. Check if pyinstaller exists in the venv
if not exist "venv\Scripts\pyinstaller.exe" (
    echo [ERROR] PyInstaller was not found in 'venv\Scripts\'.
    echo Please make sure pyinstaller is listed in requirements.txt and setup finished successfully.
    pause
    exit /b 1
)

:: 3. Run PyInstaller build
echo.
echo [*] Step 2: Compiling standalone executable via PyInstaller...
call venv\Scripts\pyinstaller.exe AI_Upscaler.spec --noconfirm
if %errorlevel% neq 0 (
    echo [ERROR] PyInstaller compilation failed.
    pause
    exit /b 1
)

:: 4. Copy models folder to the dist folder
echo.
echo [*] Step 3: Copying 'models/' directory to 'dist/models/'...
if not exist "dist" (
    mkdir "dist"
)

xcopy "models" "dist\models" /E /I /Y
if %errorlevel% neq 0 (
    echo [WARNING] Failed to copy some model files. Please verify 'dist\models' directory manually.
) else (
    echo [SUCCESS] Models copied successfully to 'dist\models'.
)

echo.
echo ===================================================
echo   [SUCCESS] Standalone Build Complete!
echo   Executable location: dist\AI_Upscaler.exe
echo ===================================================
echo.
pause
