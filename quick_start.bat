@echo off
REM ============================================================================
REM quick_start.bat - Windows Quick Start Script for Trading Agent
REM ============================================================================

setlocal enabledelayedexpansion

echo.
echo ============================================================================
echo              TRADING AGENT - QUICK START WIZARD (Windows)
echo ============================================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed or not in PATH
    echo.
    echo Please install Python 3.10+ from: https://www.python.org
    echo Make sure to check "Add Python to PATH" during installation
    pause
    exit /b 1
)

echo ✅ Python found:
python --version
echo.

REM Menu
echo Select an option:
echo.
echo 1) Install dependencies (requirements.txt - standard)
echo 2) Install dependencies (requirements-python313.txt - Python 3.13 optimized)
echo 3) Validate environment (check installed packages)
echo 4) Train ML models
echo 5) Run trading agent
echo 6) Launch dashboard
echo 7) Run tests
echo 8) Exit
echo.

set /p choice="Enter choice (1-8): "

if "%choice%"=="1" (
    goto install_standard
) else if "%choice%"=="2" (
    goto install_python313
) else if "%choice%"=="3" (
    goto validate
) else if "%choice%"=="4" (
    goto train_models
) else if "%choice%"=="5" (
    goto run_agent
) else if "%choice%"=="6" (
    goto dashboard
) else if "%choice%"=="7" (
    goto run_tests
) else if "%choice%"=="8" (
    goto end
) else (
    echo Invalid choice
    goto menu
)

:install_standard
echo.
echo Installing dependencies from requirements.txt...
echo.
pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ❌ Installation failed
    echo.
    echo Try alternative: Install Python 3.13 optimized version?
    set /p retry="Enter 'y' to try Python 3.13 version: "
    if "!retry!"=="y" goto install_python313
)
echo.
echo ✅ Installation complete!
echo.
pause
goto menu

:install_python313
echo.
echo Installing dependencies (Python 3.13 optimized)...
echo.
pip install --upgrade pip
pip install -r requirements-python313.txt --prefer-binary --only-binary :all:
if errorlevel 1 (
    echo.
    echo ❌ Installation failed. See INSTALLATION_GUIDE.md for troubleshooting
)
echo.
echo ✅ Installation attempt complete!
echo.
pause
goto menu

:validate
echo.
echo Validating environment...
echo.
python validate_environment.py
echo.
pause
goto menu

:train_models
echo.
echo Training ML models...
echo This may take 2-5 minutes
echo.
python train_models.py --verify
if errorlevel 1 (
    echo.
    echo ❌ Training failed
) else (
    echo.
    echo ✅ ML models trained successfully!
)
echo.
pause
goto menu

:run_agent
echo.
echo Starting trading agent...
echo Press Ctrl+C to stop
echo.
python main.py
echo.
pause
goto menu

:dashboard
echo.
echo Launching dashboard...
echo Opening browser at http://localhost:8501
echo Press Ctrl+C to stop
echo.
streamlit run dashboard/app.py
echo.
pause
goto menu

:run_tests
echo.
echo Running test suite...
echo.
pytest tests/ -v --tb=short
echo.
pause
goto menu

:end
echo.
echo Thank you for using Trading Agent!
echo.
exit /b 0
